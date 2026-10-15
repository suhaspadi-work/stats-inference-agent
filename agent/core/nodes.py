from __future__ import annotations
from typing import Callable
from langgraph.types import interrupt

from core.agent_state import AgentState
from core.classifier import classify_request
from core.plan import ClassificationDecision
from core.dataset import DatasetStore
from core.ingestion import profile_dataset
from core.classifier import draft_plan_fields, validate_draft_plan_fields
from core.plan import AnalysisPlan, OutcomeType, QuestionType
from core.testing import execute_two_group_test, select_two_group_method
from core.multi_group import execute_multi_group_test, select_multi_group_method
from core.regression import execute_regression, select_regression_method
from dataclasses import asdict
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
from core.descriptive import compute_descriptive_stats
from core.report import generate_report, check_for_causal_language, generate_descriptive_report, generate_multi_group_report, generate_regression_report
from core.capabilities import DESCRIPTIVE, capability_for_plan, capability_named, analysis_for_result
from core.routing import route_for_plan
from core.model_boundary import prepare_for_model
from core.classification_records import (
    CAUSAL_STATUS_POINT, QUESTION_TYPE_POINT, causal_status_record,
    clarification_gate_record, clarification_record, question_type_record,
)
from core.decision_log import log_update
from core.draft_plan_records import draft_plan_records, PREDICTORS_POINT
from core.cleaning_records import scan_record, fix_record, approval_record, execution_record
from core.freeze_records import plan_approval_record, edited_predictors_record
from core.routing_records import skipped_draft_record, route_record
from core.selection_records import TEST_SELECTION_POINT, REGRESSION_METHOD_POINT
from core.stage_plan import build_stage_plan, plan_steps
from core.stage_plan_records import stage_plan_record, step_records
from core.describe_records import describe_execution_record

CONFIDENCE_THRESHOLD = 0.6


def _capability_for_state(state: AgentState):
    """The capability from the route draft_plan_node chose; the question-type lookup when no route is in state."""
    route = state.get("route")
    if route and route.get("capability"):
        return capability_named(route["capability"])
    return capability_for_plan(state["plan"])


def classify_node(
    state: AgentState,
    classify_fn: Callable[[str], ClassificationDecision] = classify_request,
) -> dict:
    """
    First-pass classification. If confidence is below CONFIDENCE_THRESHOLD,
    "classification" is NOT set: the decision is held in
    pending_classification and route_after_classify sends the run to
    clarify_node, rather than silently guessing.

    The clarification interrupt lives in its own node so that resuming it
    re-runs only clarify_node, never this one: the first-pass model call
    happens exactly once.

    `classify_fn` is injectable so the routing LOGIC can be tested
    deterministically without a live model call; production use relies on
    the default (the real classify_request).
    """
    decision = classify_fn(state["request_text"])
    records = [question_type_record(decision, pass_number=1),
               causal_status_record(decision, pass_number=1)]
    gate = clarification_gate_record(decision.confidence, CONFIDENCE_THRESHOLD)

    if gate.chosen == "ask_human":
        return {"pending_classification": decision, **log_update(*records, gate)}
    return {"classification": decision, "pending_classification": None, **log_update(*records, gate)}


def route_after_classify(state: AgentState) -> str:
    return "clarify" if state.get("pending_classification") is not None else "profile"


def clarify_node(state: AgentState) -> dict:
    """
    Ask the human to clarify a low-confidence classification. interrupt() is
    the first thing this node does, so the re-run on resume has no side
    effects before it.
    """
    decision = state["pending_classification"]
    answer = interrupt({
        "type": "clarification_needed",
        "question": (
            f"Your request could reasonably be read more than one way "
            f"(confidence: {decision.confidence:.2f}). {decision.rationale} "
            f"Could you clarify what you're asking?"
        ),
        "original_request": state["request_text"],
        "tentative_classification": decision.question_type.value,
    })
    return {"clarification_answer": answer, **log_update(clarification_record(answer))}


def _latest_record_id(state: AgentState, decision_point: str):
    for entry in reversed(state.get("decision_log") or []):
        if entry.get("decision_point") == decision_point:
            return entry.get("id")
    return None


def reclassify_node(
    state: AgentState,
    classify_fn: Callable[[str], ClassificationDecision] = classify_request,
) -> dict:
    """
    Second pass: the human's answer is appended to the original request and
    re-classified -- not blindly trusted as a new classification itself.
    Each second-pass record supersedes the matching first-pass record.
    """
    combined_request = f"{state['request_text']}\n\nClarification: {state['clarification_answer']}"
    decision = classify_fn(combined_request)
    records = (
        question_type_record(decision, pass_number=2,
                             supersedes=_latest_record_id(state, QUESTION_TYPE_POINT)),
        causal_status_record(decision, pass_number=2,
                             supersedes=_latest_record_id(state, CAUSAL_STATUS_POINT)),
    )
    return {"classification": decision, "pending_classification": None, **log_update(*records)}

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


def make_plan_stages_node(store: DatasetStore) -> Callable[[AgentState], dict]:
    """
    M1.1: decide which steps of the three stages this run takes, by code from the classification, the
    profile and the detected data-quality issues (no model call). Read-only: the classification, the
    profile and the dataset are not changed. An unsupported question ends the run here (route_after_plan_stages).
    """
    def plan_stages_node(state: AgentState) -> dict:
        issues = detect_all_issues(state["dataset"], state["profile"], store)
        plan = build_stage_plan(state["classification"], state["profile"], issues)
        records = (stage_plan_record(plan, state["classification"], issues), *step_records(plan))
        update = {"stage_plan": plan, **log_update(*records)}
        if plan["status"] == "unsupported":
            update["clarification_needed"] = plan["message"]
        return update
    return plan_stages_node

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


def make_wrangle_propose_node(store: DatasetStore, propose_fn=propose_wrangling_operations) -> Callable[[AgentState], dict]:
    """
    Detect real data-quality issues (7a) and propose fixes for all of them in
    one batched model call (7b). The proposals are held in pending_wrangling
    for make_wrangle_review_node, which owns the checklist interrupt. Keeping
    the model call in its own node means resuming the interrupt never re-runs
    it, so the operations a human approves are exactly the ones shown (G26).

    If no issues are detected, this node is a no-op: no model call, and
    route_after_wrangle_propose skips the review node entirely.

    autonomy_mode (from state, defaults to "analyst" if absent) controls
    which issue types require human approval. In "stakeholder" mode, issue
    types in _STAKEHOLDER_AUTO_APPROVE_ISSUE_TYPES are approved
    automatically and recorded in the SAME audit trail (Operation,
    WranglingRationale) as a human approval would produce -- the only
    difference is who/what made the approval decision, which is itself
    worth recording for honesty.
    """
    def wrangle_propose_node(state: AgentState) -> dict:
        dataset = state["dataset"]
        profile = state["profile"]
        autonomy_mode = state.get("autonomy_mode", "analyst")

        issues = detect_all_issues(dataset, profile, store)
        scan = scan_record(profile, issues)
        if not issues:
            return {"dataset": dataset, "pending_wrangling": None, **log_update(scan)}

        proposals = propose_fn(issues)
        operations = [_build_operation_from_proposal(dataset, p, issues[p.issue_index]) for p in proposals]

        auto_indices = set()
        if autonomy_mode == "stakeholder":
            for i, p in enumerate(proposals):
                if issues[p.issue_index].issue_type in _STAKEHOLDER_AUTO_APPROVE_ISSUE_TYPES:
                    auto_indices.add(i)

        fixes = [fix_record(issues[p.issue_index], op, p.issue_index) for p, op in zip(proposals, operations)]

        return {"pending_wrangling": {
            "issues": issues,
            "operations": operations,
            "issue_indices": [p.issue_index for p in proposals],
            "auto_indices": sorted(auto_indices),
            "fix_record_ids": [f.id for f in fixes],
        }, **log_update(scan, *fixes)}

    return wrangle_propose_node


def route_after_cleaning(state: AgentState) -> str:
    """
    After cleaning (or when there was nothing to clean): the describe step when the stage plan says run
    and the question is not descriptive (a descriptive question is described by eda), otherwise
    draft_plan. A state without a stage plan goes to draft_plan, as before M1.2.
    """
    plan = state.get("stage_plan")
    if plan is not None and plan["question_type"] != "descriptive":
        describe = next(s for s in plan_steps(plan) if s["step"] == "describe")
        if describe["run"]:
            return "describe"
    return "draft_plan"


def route_after_wrangle_propose(state: AgentState) -> str:
    return "wrangle_review" if state.get("pending_wrangling") is not None else route_after_cleaning(state)


def make_describe_node(store: DatasetStore) -> Callable[[AgentState], dict]:
    """
    M1.2: describe the finished dataset as the last step of stage 1 (deterministic, no model call, no
    report yet: M1.4). Stores descriptive_result; the dataset, plan, profile and classification are untouched.
    """
    def describe_node(state: AgentState) -> dict:
        result = DESCRIPTIVE.compute(state["dataset"], store)
        step = next(s for s in plan_steps(state["stage_plan"]) if s["step"] == "describe")
        return {"descriptive_result": result, **log_update(describe_execution_record(result, step))}
    return describe_node


def make_wrangle_review_node(store: DatasetStore) -> Callable[[AgentState], dict]:
    """
    Present the proposed operations as a checklist interrupt, then execute
    each approved/edited operation in sequence (producing a chained Dataset
    version per operation) and record any rejection as an UnaddressedIssue
    -- via pending_unaddressed_issues, since no AnalysisPlan exists yet at
    this point in the graph -- so the caveat survives into method selection
    and the final report.

    interrupt() is the first side-effecting call in this node, so the re-run
    on resume repeats nothing before it.
    """
    def wrangle_review_node(state: AgentState) -> dict:
        dataset = state["dataset"]
        autonomy_mode = state.get("autonomy_mode", "analyst")
        pending = state["pending_wrangling"]
        issues = pending["issues"]
        operations = pending["operations"]
        issue_indices = pending["issue_indices"]
        auto_indices = set(pending["auto_indices"])

        needs_review_indices = [i for i in range(len(operations)) if i not in auto_indices]

        decisions_by_index = {}
        for i in auto_indices:
            decisions_by_index[i] = {"index": i, "action": "approve"}
        auto_decisions = {i: decisions_by_index[i] for i in auto_indices}

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
        records = []
        fix_ids = pending.get("fix_record_ids") or [None] * len(operations)
        auto_rule = (f"stakeholder mode auto-approves "
                     f"{', '.join(sorted(_STAKEHOLDER_AUTO_APPROVE_ISSUE_TYPES))} only")

        # Every operation index gets an approval record; indices without a decision are
        # skipped exactly as before. Decision indices run in the same ascending order.
        for idx in sorted(set(range(len(operations))) | set(decisions_by_index)):
            if idx not in decisions_by_index:
                records.append(approval_record(operations[idx], issues[issue_indices[idx]], None, None,
                                               fix_ids[idx]))
                continue
            decision = decisions_by_index[idx]
            op = operations[idx]
            issue = issues[issue_indices[idx]]
            action = decision.get("action")
            is_auto = auto_decisions.get(idx) is decision

            if action == "reject":
                records.append(approval_record(op, issue, decision, None, fix_ids[idx]))
                unaddressed += (UnaddressedIssue(
                    issue_type=issue.issue_type, description=issue.description,
                    proposed_operation=f"{op.op_type}({op.params})",
                    rejection_reason=decision.get("reason", "no reason given"),
                    affected_columns=issue.affected_columns,
                ),)
                continue

            raw_params = decision.get("params", op.params) if action == "edit" else op.params
            params = _normalize_params(op.op_type, raw_params)
            approval = approval_record(op, issue, decision, auto_rule if is_auto else None, fix_ids[idx],
                                       edited_params=params if action == "edit" else None)
            records.append(approval)
            op = Operation(
                op_type=op.op_type, session_id=op.session_id,
                input_handles=(current_dataset.handle,), params=params,
                approval_status=ApprovalStatus.PENDING, rationale=op.rationale,
            )
            approved_op = op.approve()
            execute_fn = _EXECUTE_FNS[op.op_type]
            current_dataset, finished_op = execute_fn(current_dataset, store, approved_op)
            records.append(execution_record(finished_op, approval.id))

        result = {"dataset": current_dataset, "pending_wrangling": None, **log_update(*records)}
        if unaddressed:
            result["pending_unaddressed_issues"] = unaddressed
        return result

    return wrangle_review_node

def make_draft_plan_node(draft_fn=draft_plan_fields) -> Callable[[AgentState], dict]:
    """
    Injectable draft_fn, same pattern as classify_node, for deterministic
    testing. Also folds in any pending_unaddressed_issues left by
    wrangle_node -- since that node runs before a plan exists, this is the
    first point where a rejected data-quality fix can actually be attached
    to the AnalysisPlan it needs to travel with (R2).
    """
    def draft_plan_node(state: AgentState) -> dict:
        classification = state["classification"]

        if classification.question_type.value == "descriptive":
            # EDA describes the whole dataset -- there is no single outcome
            # column or set of predictors to draft, so skip the LLM call
            # entirely rather than forcing the model to invent values for
            # fields that don't apply to this question type.
            plan = AnalysisPlan.from_classification(
                request_text=state["request_text"],
                session_id=state["session_id"],
                classification=classification,
                outcome_name="(whole dataset)",
                outcome_type=OutcomeType.CONTINUOUS,  # placeholder; eda_node never reads this
                unit_of_analysis="row",
                data_sources=(state["dataset"].handle,),
            )
            route = route_for_plan(plan, state["profile"])
            records = (skipped_draft_record(plan), route_record(plan, state["profile"], route))
        else:
            fields = draft_fn(state["request_text"], state["profile"])
            validate_draft_plan_fields(fields, state["profile"])

            plan = AnalysisPlan.from_classification(
                request_text=state["request_text"],
                session_id=state["session_id"],
                classification=classification,
                outcome_name=fields.outcome_name,
                outcome_type=fields.outcome_type,
                unit_of_analysis=fields.unit_of_analysis,
                data_sources=(state["dataset"].handle,),
            )
            plan = plan.refine(candidate_predictors=tuple(fields.candidate_predictors))
            route = route_for_plan(plan, state["profile"])
            if route["status"] == "reconciled":
                # The data supports a different comparison than the question type routes to (G2):
                # the plan carries the routed question type; the classification stays as the model gave it.
                plan = plan.refine(question_type=QuestionType(route["question_type"]))
            records = (*draft_plan_records(fields, state["profile"]), route_record(plan, state["profile"], route))

        pending = state.get("pending_unaddressed_issues")
        if pending:
            for issue in pending:
                plan = plan.record_unaddressed_issue(issue)

        update = {"plan": plan, "route": route, **log_update(*records)}
        if route["status"] == "unsupported":
            update["clarification_needed"] = route["message"]
        return update
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
        capability = _capability_for_state(state)
        if capability.kind != "analysis":
            raise ValueError(
                f"select_method_node called on a '{capability.name}' plan "
                f"(question_type={plan.question_type.value}). "
                f"This indicates a routing bug in the graph, not a user input problem."
            )
        capability.validate_predictors(plan, "Method selection")

        decision = capability.select(state["dataset"], store, plan)
        if plan.unaddressed_issues:
            decision = replace(decision, evidence={
                **decision.evidence,
                "unaddressed_issues": [
                    {"issue_type": i.issue_type, "description": i.description, "rejection_reason": i.rejection_reason}
                    for i in plan.unaddressed_issues
                ],
            })
        updated_plan = plan.record_method_decision(decision)
        if capability.record is not None:
            result = capability.record(decision)
            records = result if isinstance(result, (tuple, list)) else (result,)
            return {"plan": updated_plan, **log_update(*records)}
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
    selection_record_id = next(
        (e.get("id") for e in reversed(state.get("decision_log") or [])
         if e.get("decision_point") in (TEST_SELECTION_POINT, REGRESSION_METHOD_POINT)),
        None,
    )

    if action == "reject":
        # A rejected plan is never frozen -- the graph should route to a
        # clean stop, not silently proceed. We signal this the same way
        # classify_node signals "needs clarification": a distinct state key.
        approval = plan_approval_record(plan, decision, selection_record_id)
        return {"plan": plan, "clarification_needed": f"Plan rejected: {decision.get('reason', 'no reason given')}",
                **log_update(approval)}

    if action == "edit":
        new_predictors = tuple(decision.get("candidate_predictors", plan.candidate_predictors or ()))
        valid_names = {c.name for c in state["profile"].columns}
        invalid = [name for name in new_predictors if name not in valid_names]
        if invalid:
            raise ValueError(
                f"Edited predictor(s) not present in the dataset: {invalid}. "
                f"Available columns: {sorted(valid_names)}"
            )
        edited = edited_predictors_record(plan, new_predictors, state["profile"],
                                          _latest_record_id(state, PREDICTORS_POINT), selection_record_id)
        approval = plan_approval_record(plan, decision, selection_record_id, predictors_record_id=edited.id)
        records = (edited, approval)
        plan = plan.refine(candidate_predictors=new_predictors)
    else:
        records = (plan_approval_record(plan, decision, selection_record_id),)

    frozen_plan = plan.freeze()
    return {"plan": frozen_plan, **log_update(*records)}


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

        capability = _capability_for_state(state)
        if capability.kind != "analysis":
            raise ValueError(
                f"execute_node called on a '{capability.name}' plan "
                f"(question_type={plan.question_type.value}). "
                f"This indicates a routing bug in the graph, not a user input problem."
            )
        capability.validate_predictors(plan, "Execution")
        if plan.primary_method is None:
            raise ValueError("Cannot execute: no method was selected on the plan before freeze.")

        updated_plan = plan.mark_executed()  # no new MethodDecision -- already recorded pre-freeze
        result = capability.execute(state["dataset"], store, plan)
        selection_entry = next(
            (e for e in reversed(state.get("decision_log") or [])
             if e.get("decision_point") in (TEST_SELECTION_POINT, REGRESSION_METHOD_POINT)),
            None,
        )
        records = capability.execution_records(result, selection_entry) if capability.execution_records is not None else ()
        return {"plan": updated_plan, "test_result": result, **log_update(*records)}
    return execute_node

def make_eda_node(store: DatasetStore) -> Callable[[AgentState], dict]:
    """
    Runs descriptive statistics -- the EDA capability's execution step.
    Unlike execute_node, this never requires the plan to be FROZEN first:
    EDA makes no inferential method choice for a human to approve, so
    there is nothing for freeze's interrupt to meaningfully gate. It does
    still require a diagnostic capability, structurally enforced so
    this node can never silently run in place of a real hypothesis test.
    """
    def eda_node(state: AgentState) -> dict:
        plan = state["plan"]
        capability = _capability_for_state(state)
        if capability.kind != "diagnostic":
            raise ValueError(
                f"eda_node called on a non-descriptive plan (question_type={plan.question_type.value}). "
                f"This indicates a routing bug in the graph, not a user input problem."
            )

        result = capability.compute(state["dataset"], store)
        updated_plan = plan.mark_executed_descriptive()
        return {"plan": updated_plan, capability.result_key: result}
    return eda_node

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
        capability = _capability_for_state(state)
        if capability.kind == "diagnostic":
            if capability.report is None:
                raise ValueError(f"Capability '{capability.name}' has no report.")
            safe_plan, safe_result = prepare_for_model(plan, state[capability.result_key])
            report_text = capability.report(safe_plan, safe_result)
            # No check_for_causal_language call here: a pure description makes
            # no inferential claim for that guard to meaningfully check.
        else:
            result = state["test_result"]
            analysis = analysis_for_result(result)
            safe_plan, safe_result = prepare_for_model(plan, result)  # no chart data reaches a prompt (M0.5)
            if analysis.result_type == "two_group":
                report_text = report_fn(safe_plan, safe_result)  # the injectable stub applies here only, as before
            else:
                report_text = analysis.report(safe_plan, safe_result)
            check_for_causal_language(report_text, plan)
        return {"report": report_text}
    return report_node