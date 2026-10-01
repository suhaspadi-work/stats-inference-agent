from __future__ import annotations
from typing import Callable
from langgraph.types import interrupt

from core.agent_state import AgentState
from core.classifier import classify_request
from core.plan import ClassificationDecision
from core.dataset import DatasetStore
from core.ingestion import profile_dataset
from core.classifier import draft_plan_fields, validate_draft_plan_fields
from core.plan import AnalysisPlan
from core.testing import execute_two_group_test, select_two_group_method
from core.report import generate_report
from core.report import generate_report, check_for_causal_language
from core.detection import detect_all_issues
from core.classifier import propose_wrangling_operations
from core.operation import Operation, ApprovalStatus, WranglingRationale, AlternativeApproachConsidered
from core.plan import UnaddressedIssue
from core.wrangling import (
    propose_cast, execute_cast, propose_rename, execute_rename,
    propose_filter, execute_filter, propose_dedupe, execute_dedupe,
)
from dataclasses import replace

CONFIDENCE_THRESHOLD = 0.6


def classify_node(
    state: AgentState,
    classify_fn: Callable[[str], ClassificationDecision] = classify_request,
) -> dict:
    """
    Classify the request. If confidence is below CONFIDENCE_THRESHOLD, this
    is the #5 gap from our design review made real: interrupt and wait for
    a human to clarify, rather than silently guessing. The clarification
    interrupt resumes with the human's answer, which is appended to the
    original request and re-classified -- not blindly trusted as a new
    classification itself.

    `classify_fn` is injectable so the interrupt/routing LOGIC can be tested
    deterministically without a live model call; production use relies on
    the default (the real classify_request).
    """
    decision = classify_fn(state["request_text"])

    if decision.confidence < CONFIDENCE_THRESHOLD:
        clarification_answer = interrupt({
            "type": "clarification_needed",
            "question": (
                f"Your request could reasonably be read more than one way "
                f"(confidence: {decision.confidence:.2f}). {decision.rationale} "
                f"Could you clarify what you're asking?"
            ),
            "original_request": state["request_text"],
            "tentative_classification": decision.question_type.value,
        })
        combined_request = f"{state['request_text']}\n\nClarification: {clarification_answer}"
        decision = classify_fn(combined_request)

    return {"classification": decision}

def make_profile_node(store: DatasetStore) -> Callable[[AgentState], dict]:
    """
    Returns a node function closed over the store. The store is genuine
    infrastructure (a filesystem/cloud connection), not data that belongs
    in AgentState -- this keeps that distinction clean rather than smuggling
    the store through the state dict.
    """
    def profile_node(state: AgentState) -> dict:
        view = profile_dataset(state["dataset"], store=store)
        return {"profile": view}
    return profile_node

_EXECUTE_FNS = {"cast": execute_cast, "rename": execute_rename, "filter": execute_filter, "dedupe": execute_dedupe}

def _normalize_params(op_type: str, params: dict) -> dict:
    """
    Translate model-proposed params into the shape Phase 4's execute_*
    functions actually expect. The model naturally produces an empty list
    for "dedupe across all columns," but core.wrangling's dedupe treats
    that differently from the correct None -- this guard closes that gap
    rather than trusting the model's literal output shape.
    """
    if op_type == "dedupe":
        subset = params.get("subset")
        if subset == [] or subset is None:
            return {**params, "subset": None}
    return params



def _build_operation_from_proposal(dataset, proposal, issue) -> Operation:
    """Build a pending Operation from one model proposal, using Phase 1's WranglingRationale."""
    rationale = WranglingRationale(
        rationale=proposal.rationale,
        alternatives=(AlternativeApproachConsidered(
            approach=proposal.alternative_approach, rejected_because=proposal.alternative_rejected_because,
        ),),
        evidence=issue.evidence,
    )
    return Operation(
        op_type=proposal.op_type, session_id=dataset.session_id,
        input_handles=(dataset.handle,), params=_normalize_params(proposal.op_type, proposal.params),
        approval_status=ApprovalStatus.PENDING, rationale=rationale,
    )


# Issue types that stakeholder mode auto-approves without an interrupt --
# deliberately narrow: only exact-duplicate removal, since that rarely
# changes analytical conclusions. Anything affecting missingness (which
# drops rows and can change group sizes/balance) always interrupts,
# regardless of mode.
_STAKEHOLDER_AUTO_APPROVE_ISSUE_TYPES = {"duplicate_rows"}


def make_wrangle_node(store: DatasetStore) -> Callable[[AgentState], dict]:
    """
    Detect real data-quality issues (7a), propose fixes for all of them in
    one batched model call (7b), present the whole list as a checklist
    interrupt, then execute each approved/edited operation in sequence
    (producing a chained Dataset version per operation) and record any
    rejection as an UnaddressedIssue -- via pending_unaddressed_issues,
    since no AnalysisPlan exists yet at this point in the graph -- so the
    caveat survives into method selection and the final report.

    If no issues are detected, this node is a no-op: no model call, no
    interrupt, the dataset passes through unchanged.

    autonomy_mode (from state, defaults to "analyst" if absent) controls
    which issue types require human approval. In "stakeholder" mode, issue
    types in _STAKEHOLDER_AUTO_APPROVE_ISSUE_TYPES are approved
    automatically and recorded in the SAME audit trail (Operation,
    WranglingRationale) as a human approval would produce -- the only
    difference is who/what made the approval decision, which is itself
    worth recording for honesty.
    """
    def wrangle_node(state: AgentState) -> dict:
        dataset = state["dataset"]
        profile = state["profile"]
        autonomy_mode = state.get("autonomy_mode", "analyst")

        issues = detect_all_issues(dataset, profile, store)
        if not issues:
            return {"dataset": dataset}

        proposals = propose_wrangling_operations(issues)
        operations = [_build_operation_from_proposal(dataset, p, issues[p.issue_index]) for p in proposals]

        auto_indices = set()
        if autonomy_mode == "stakeholder":
            for i, p in enumerate(proposals):
                if issues[p.issue_index].issue_type in _STAKEHOLDER_AUTO_APPROVE_ISSUE_TYPES:
                    auto_indices.add(i)

        needs_review_indices = [i for i in range(len(operations)) if i not in auto_indices]

        decisions_by_index = {}
        for i in auto_indices:
            decisions_by_index[i] = {"index": i, "action": "approve"}

        if needs_review_indices:
            review_payload = {
                "type": "wrangling_approval",
                "question": "Review the proposed data-cleaning operations before they run.",
                "proposed_operations": [
                    {
                        "index": i, "op_type": operations[i].op_type, "params": operations[i].params,
                        "rationale": operations[i].rationale.rationale,
                        "alternative": operations[i].rationale.alternatives[0].approach,
                    }
                    for i in needs_review_indices
                ],
                "allowed_decisions": ["approve", "edit", "reject"],
            }
            if auto_indices:
                review_payload["auto_approved_note"] = (
                    f"{len(auto_indices)} additional operation(s) were auto-approved under "
                    f"'{autonomy_mode}' autonomy mode and are not shown here for review."
                )
            human_decisions = interrupt(review_payload)
            for decision in human_decisions.get("decisions", []):
                decisions_by_index[decision["index"]] = decision

        current_dataset = dataset
        unaddressed = ()

        for idx in sorted(decisions_by_index):
            decision = decisions_by_index[idx]
            op = operations[idx]
            issue = issues[proposals[idx].issue_index]
            action = decision.get("action")

            if action == "reject":
                unaddressed += (UnaddressedIssue(
                    issue_type=issue.issue_type, description=issue.description,
                    proposed_operation=f"{op.op_type}({op.params})",
                    rejection_reason=decision.get("reason", "no reason given"),
                    affected_columns=issue.affected_columns,
                ),)
                continue

            raw_params = decision.get("params", op.params) if action == "edit" else op.params
            params = _normalize_params(op.op_type, raw_params)
            op = Operation(
                op_type=op.op_type, session_id=op.session_id,
                input_handles=(current_dataset.handle,), params=params,
                approval_status=ApprovalStatus.PENDING, rationale=op.rationale,
            )
            approved_op = op.approve()
            execute_fn = _EXECUTE_FNS[op.op_type]
            current_dataset, _finished_op = execute_fn(current_dataset, store, approved_op)

        result = {"dataset": current_dataset}
        if unaddressed:
            result["pending_unaddressed_issues"] = unaddressed
        return result

    return wrangle_node

def make_draft_plan_node(draft_fn=draft_plan_fields) -> Callable[[AgentState], dict]:
    """
    Injectable draft_fn, same pattern as classify_node, for deterministic
    testing. Also folds in any pending_unaddressed_issues left by
    wrangle_node -- since that node runs before a plan exists, this is the
    first point where a rejected data-quality fix can actually be attached
    to the AnalysisPlan it needs to travel with (R2).
    """
    def draft_plan_node(state: AgentState) -> dict:
        fields = draft_fn(state["request_text"], state["profile"])
        validate_draft_plan_fields(fields, state["profile"])

        plan = AnalysisPlan.from_classification(
            request_text=state["request_text"],
            session_id=state["session_id"],
            classification=state["classification"],
            outcome_name=fields.outcome_name,
            outcome_type=fields.outcome_type,
            unit_of_analysis=fields.unit_of_analysis,
            data_sources=(state["dataset"].handle,),
        )
        plan = plan.refine(candidate_predictors=tuple(fields.candidate_predictors))

        pending = state.get("pending_unaddressed_issues")
        if pending:
            for issue in pending:
                plan = plan.record_unaddressed_issue(issue)

        return {"plan": plan}
    return draft_plan_node

def make_select_method_node(store: DatasetStore) -> Callable[[AgentState], dict]:
    """
    Selects the test method and records the MethodDecision on the plan
    BEFORE freeze -- this is what makes the human-approval screen able to
    show which test will run and why, rather than the method being decided
    silently during execution after approval has already happened.

    If the plan carries any unaddressed_issues (a data-quality fix was
    proposed and rejected), that caveat is folded into this MethodDecision's
    evidence -- so the rationale an approver sees at freeze already reflects
    known, acknowledged problems in the data, not just the test statistics.
    """
    def select_method_node(state: AgentState) -> dict:
        plan = state["plan"]
        predictors = plan.candidate_predictors or ()
        if len(predictors) != 1:
            raise ValueError(
                f"select_two_group_method requires exactly one grouping predictor; "
                f"got {len(predictors)}: {predictors}."
            )
        choice = select_two_group_method(
            dataset=state["dataset"], store=store,
            outcome_col=plan.outcome_name, group_col=predictors[0],
        )
        decision = choice.decision
        if plan.unaddressed_issues:
            decision = replace(decision, evidence={
                **decision.evidence,
                "unaddressed_issues": [
                    {"issue_type": i.issue_type, "description": i.description, "rejection_reason": i.rejection_reason}
                    for i in plan.unaddressed_issues
                ],
            })
        updated_plan = plan.record_method_decision(decision)
        return {"plan": updated_plan}
    return select_method_node

def freeze_node(state: AgentState) -> dict:
    """
    The single most consequential interrupt in the system: nothing runs a
    hypothesis test until a human has seen the actual plan and approved it.
    This is freeze() from Phase 1 finally being gated by a real human
    decision, not just a method call anyone could skip.
    """
    plan = state["plan"]

    decision = interrupt({
        "type": "plan_approval",
        "question": "Review this analysis plan before it is frozen and executed.",
        "request_text": plan.request_text,
        "question_type": plan.question_type.value,
        "causal_status": plan.causal_status.value,
        "outcome_name": plan.outcome_name,
        "outcome_type": plan.outcome_type.value,
        "candidate_predictors": list(plan.candidate_predictors or ()),
        "classification_rationale": plan.classification_decision.rationale if plan.classification_decision else None,
        "allowed_decisions": ["approve", "edit", "reject"],
    })

    action = decision.get("action")

    if action == "reject":
        # A rejected plan is never frozen -- the graph should route to a
        # clean stop, not silently proceed. We signal this the same way
        # classify_node signals "needs clarification": a distinct state key.
        return {"plan": plan, "clarification_needed": f"Plan rejected: {decision.get('reason', 'no reason given')}"}

    if action == "edit":
        new_predictors = tuple(decision.get("candidate_predictors", plan.candidate_predictors or ()))
        plan = plan.refine(candidate_predictors=new_predictors)

    frozen_plan = plan.freeze()
    return {"plan": frozen_plan}
def make_execute_node(store: DatasetStore) -> Callable[[AgentState], dict]:
    """
    Runs the actual hypothesis test -- using the method ALREADY selected and
    approved via select_method_node/freeze_node, never re-deciding it here.
    Refuses to run unless the plan is FROZEN, structurally enforcing the
    pre-registration discipline rather than trusting the graph's edges
    alone to guarantee correct ordering.
    """
    def execute_node(state: AgentState) -> dict:
        plan = state["plan"]
        if plan.status.value != "frozen":
            raise ValueError(f"Cannot execute a plan that is not FROZEN (status: {plan.status.value}).")

        predictors = plan.candidate_predictors or ()
        if len(predictors) != 1:
            raise ValueError(
                f"execute_two_group_test requires exactly one grouping predictor; "
                f"got {len(predictors)}: {predictors}."
            )
        method = plan.primary_method
        if method is None:
            raise ValueError("Cannot execute: no method was selected on the plan before freeze.")

        result = execute_two_group_test(
            dataset=state["dataset"], store=store,
            outcome_col=plan.outcome_name, group_col=predictors[0], method=method,
        )

        updated_plan = plan.mark_executed()  # no new MethodDecision -- already recorded pre-freeze

        return {
            "plan": updated_plan,
            "test_result": {
                "method": result.method,
                "statistic": result.statistic,
                "p_value": result.p_value,
                "effect_size": result.effect_size,
                "confidence_interval": result.confidence_interval,
                "group_a_n": result.group_a_n,
                "group_b_n": result.group_b_n,
                "group_a_mean": result.group_a_mean,
                "group_b_mean": result.group_b_mean,
            },
        }
    return execute_node

def make_report_node(report_fn=generate_report) -> Callable[[AgentState], dict]:
    """
    Injectable report_fn, same testing pattern as classify_node and
    draft_plan_node. The causal-language guard is enforced HERE, at the
    node level -- not only inside generate_report -- so it applies to
    whatever report_fn produces, whether that's the real model call or a
    test stub standing in for it.
    """
    def report_node(state: AgentState) -> dict:
        plan = state["plan"]
        report_text = report_fn(plan, state["test_result"])
        check_for_causal_language(report_text, plan)
        return {"report": report_text}
    return report_node