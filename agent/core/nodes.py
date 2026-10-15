from __future__ import annotations
from typing import Callable, Optional
from langgraph.graph import END
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
from core.report import CausalLanguageViolation, find_causal_violation, violation_feedback
from core.report_fallback import MINIMAL_FALLBACK_TEXT, facts_fallback_report
from core.report_records import report_narrative_record
from core.capabilities import DESCRIPTIVE, capability_for_plan, capability_named, analysis_for_result
from core.routing import route_for_plan
from core.model_boundary import prepare_for_model
from core.rounds import make_round_entry, with_exploratory_note
from core.round_two import (
    MAX_OTHER_PREDICTORS, build_exploratory_plan, exploratory_offer, propose_exploratory_predictors,
    stage_gate_payload,
)
from core.predictor_rules import validate_edit, validate_predictor_choice
from core.round_records import round_start_record, stage_gate_record
from core.routing import COMPARISON_TYPES
from core.classification_records import (
    CAUSAL_STATUS_POINT, QUESTION_TYPE_POINT, causal_status_record,
    clarification_gate_record, clarification_record, question_type_record,
)
from core.decision_log import log_update
from core.draft_plan_records import draft_plan_records, PREDICTORS_POINT
from core.cleaning_records import scan_record, fix_record, approval_record, execution_record
from core.freeze_records import (
    PLAN_APPROVAL_POINT, edited_predictors_record, plan_approval_record, refused_plan_approval_record,
)
from core.routing_records import skipped_draft_record, route_record
from core.selection_records import TEST_SELECTION_POINT, REGRESSION_METHOD_POINT
from core.stage_plan import build_stage_plan, plan_steps
from core.stage_plan_records import stage_plan_approval_record, stage_plan_record, step_records
from core.stage_plan_gate import apply_gate_decision, changed_steps, check_run_overrides, gate_needed, gate_payload
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
        # M1.3b: a new question on the same thread starts outside any exploratory round.
        update = {"stage_plan": plan, "exploratory_round": False, "next_round": None, "stage_gate_refusal": None,
                  "stage_plan_refusal": None, "plan_approval_refusal": None, "reselect_after_edit": None,
                  **log_update(*records)}
        if plan["status"] == "unsupported":
            update["clarification_needed"] = plan["message"]
        return update
    return plan_stages_node


# Refusal loops (M1.3b step 2b, M1.3c). At the stage-plan gate, at the plan-approval edit and at the boundary
# gate, a choice that breaks a rule is a recorded human decision ("refused") and the node returns normally; a
# self-loop in the graph re-opens the same pause, which then shows the refusal. A node that raises loses the
# records it would have returned (M0.2 step 13), so a refusal must never be an exception.

def _refused_count(state: AgentState, point: str, after_points: tuple) -> int:
    """Refused records of point since the last record of any of after_points (the current plan)."""
    log = state.get("decision_log") or []
    start = max((i for i, e in enumerate(log) if e.get("decision_point") in after_points), default=-1)
    return sum(1 for e in log[start + 1:] if e.get("decision_point") == point and e.get("chosen") == "refused")


def route_after_plan_stages(state: AgentState) -> str:
    """An unsupported question ends the run after the stage plan, before any cleaning (M1.1); a plan with
    an offered step goes to the stage-plan gate (M1.3a); otherwise straight to cleaning."""
    plan = state["stage_plan"]
    if plan["status"] == "unsupported":
        return END
    if gate_needed(plan):
        return "stage_plan_gate"
    return "wrangle_propose"


def stage_plan_gate_node(state: AgentState) -> dict:
    """
    M1.3a: the user reviews the stage plan before anything is changed. interrupt() comes first, so the
    resumed re-run has no side effect before it. Describe can be switched on or off; rejecting ends the run
    with the reason. The approved run flags are written back into the stage plan, so route_after_cleaning
    acts on them. M1.3c: run choices that break a rule are refused and recorded, and the gate re-opens.
    """
    decision = interrupt(gate_payload(state["stage_plan"], state["request_text"],
                                      refused=state.get("stage_plan_refusal"))) or {}
    plan = state["stage_plan"]
    action = decision.get("action")

    if action == "reject":
        reason = decision.get("reason")
        record = stage_plan_approval_record(plan, decision, {}, "rejected")
        return {"stage_plan": apply_gate_decision(plan, {}, "rejected", reason), "stage_plan_refusal": None,
                "clarification_needed": f"Stage plan rejected: {reason or 'no reason given'}", **log_update(record)}

    # An unrecognised action approves the plan as proposed (G36), so its run choices are not applied.
    overrides = {}
    if action == "approve":
        checked = check_run_overrides(plan, decision.get("run"))
        if not checked["ok"]:
            submitted = decision.get("run")
            refusal = {"submitted": dict(submitted) if isinstance(submitted, dict) else {},
                       "refusals": checked["refusals"],
                       "attempt": 1 + _refused_count(state, "planner.stage_plan_approval", ("planner.stage_plan",))}
            record = stage_plan_approval_record(plan, decision, {}, "refused", refusal=refusal)
            return {"stage_plan_refusal": refusal, **log_update(record)}
        overrides = checked["overrides"]
    status = "approved_with_changes" if changed_steps(plan, overrides) else "approved"
    record = stage_plan_approval_record(plan, decision, overrides, status)
    return {"stage_plan": apply_gate_decision(plan, overrides, status), "stage_plan_refusal": None,
            **log_update(record)}


def route_after_stage_plan_gate(state: AgentState) -> str:
    if state.get("stage_plan_refusal") is not None:
        return "stage_plan_gate"
    if state["stage_plan"]["approval"]["status"] == "rejected":
        return END
    return "wrangle_propose"

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
        reselect = state.get("reselect_after_edit") is not None  # M1.3c: after an accepted edit at plan approval
        extra = {"reselect_after_edit": None} if reselect else {}
        if capability.record is not None:
            result = capability.record(decision)
            records = result if isinstance(result, (tuple, list)) else (result,)
            if reselect:
                records = tuple(replace(r, supersedes=_latest_record_id(state, r.decision_point)) for r in records)
            return {"plan": updated_plan, **extra, **log_update(*records)}
        return {"plan": updated_plan, **extra}
    return select_method_node

def make_freeze_node(store: DatasetStore) -> Callable[[AgentState], dict]:
    """
    The single most consequential interrupt in the system: nothing runs a hypothesis test until a human has
    seen the actual plan and approved it. M1.3c: an edit is checked against the predictor rules and re-routed
    by code; a refused edit is recorded and the pause re-opens; an accepted edit is not frozen: the plan goes
    back through select_method (new selection records supersede the old ones) and this pause shows the
    recomputed method.
    """
    def freeze_node(state: AgentState) -> dict:
        plan = state["plan"]
        selection = next((e for e in reversed(state.get("decision_log") or [])
                          if e.get("decision_point") in (TEST_SELECTION_POINT, REGRESSION_METHOD_POINT)), None)

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
            "refused": state.get("plan_approval_refusal"),
            "method": plan.primary_method,
            # True when the method shown was selected again after an accepted edit (its record supersedes one)
            "reselected": bool(selection and selection.get("supersedes")),
        })

        action = decision.get("action")
        selection_record_id = selection.get("id") if selection else None
        cleared = {"plan_approval_refusal": None, "reselect_after_edit": None}

        if action == "reject":
            # A rejected plan is never frozen -- the graph should route to a
            # clean stop, not silently proceed. We signal this the same way
            # classify_node signals "needs clarification": a distinct state key.
            approval = plan_approval_record(plan, decision, selection_record_id)
            return {"plan": plan, "clarification_needed": f"Plan rejected: {decision.get('reason', 'no reason given')}",
                    **cleared, **log_update(approval)}

        if action == "edit":
            # A fresh profile of the (cleaned) dataset, so the identifier and level rules see the data as it is.
            profile = profile_dataset(state["dataset"], store=store)
            checked = validate_edit(profile, plan, decision.get("candidate_predictors", plan.candidate_predictors or ()),
                                    plan.outcome_name)
            if not checked["ok"]:
                submitted = decision.get("candidate_predictors")
                attempt = 1 + _refused_count(state, PLAN_APPROVAL_POINT, ("planner.stage_plan", "planner.round_start"))
                refusal = {"submitted": list(submitted) if isinstance(submitted, (list, tuple)) else [],
                           "refusals": checked["refusals"], "attempt": attempt}
                record = refused_plan_approval_record(plan, decision, refusal, selection_record_id)
                return {"plan_approval_refusal": refusal, **log_update(record)}
            new_predictors = tuple(checked["predictors"])
            route = checked["route"]
            edited = edited_predictors_record(plan, new_predictors, profile,
                                              _latest_record_id(state, PREDICTORS_POINT), selection_record_id)
            approval = plan_approval_record(plan, decision, selection_record_id, predictors_record_id=edited.id)
            refined = plan.refine(candidate_predictors=new_predictors,
                                  question_type=QuestionType(route["question_type"]), method_decisions=())
            return {"plan": refined, "route": route, "plan_approval_refusal": None,
                    "reselect_after_edit": {"supersedes": selection_record_id}, **log_update(edited, approval)}

        records = (plan_approval_record(plan, decision, selection_record_id),)
        return {"plan": plan.freeze(), **cleared, **log_update(*records)}
    return freeze_node


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

def _attempt(n: int, violation: Optional[CausalLanguageViolation]) -> dict:
    return {"attempt": n, "passed": violation is None,
            "violation": None if violation is None else {"matched": violation.matched, "sentence": violation.sentence}}


def _guarded_narrative(generate, plan, safe_plan, safe_result, capability_name: str):
    """
    M1.4a (G52): the model's text, checked against the causal-language guard; on a violation ONE retry with the
    violation fed back; then a report written by code from the computed facts (and, if even that trips the
    guard, a minimal text). A report can no longer fail the run. Errors other than a violation propagate.
    Returns (text, attempts, chosen).
    """
    attempts = []
    try:
        text = generate(safe_plan, safe_result)
        check_for_causal_language(text, plan)
        attempts.append(_attempt(1, None))
        return text, attempts, "model"
    except CausalLanguageViolation as first:
        attempts.append(_attempt(1, first))
        feedback = violation_feedback(first)
    try:
        text = generate(safe_plan, safe_result, feedback=feedback)
        check_for_causal_language(text, plan)
        attempts.append(_attempt(2, None))
        return text, attempts, "model_retry"
    except CausalLanguageViolation as second:
        attempts.append(_attempt(2, second))
    text = facts_fallback_report(plan, safe_result, capability_name)
    if plan.allowed_claims == "associations only" and find_causal_violation(text) is not None:
        return MINIMAL_FALLBACK_TEXT, attempts, "minimal_fallback"
    return text, attempts, "facts_fallback"


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
            original_result = state[capability.result_key]
        else:
            result = state["test_result"]
            analysis = analysis_for_result(result)
            safe_plan, safe_result = prepare_for_model(plan, result)  # no chart data reaches a prompt (M0.5)
            # the injectable stub applies to two-group results only, as before
            generate = report_fn if analysis.result_type == "two_group" else analysis.report
            report_text, attempts, chosen = _guarded_narrative(generate, plan, safe_plan, safe_result, capability.name)
            original_result = result

        # M1.3b: one entry per finished round, holding the original result (charts included), not the
        # chart-free copy the model saw. The exploratory note is added after the causal check.
        exploratory = bool(state.get("exploratory_round"))
        if exploratory:
            report_text = with_exploratory_note(report_text)
        round_index = len(state.get("rounds") or ()) + 1
        entry = make_round_entry(round_index, plan, capability.name, original_result, report_text, exploratory)
        if capability.kind == "diagnostic":
            return {"report": report_text, "rounds": [entry]}  # the descriptive report is not guarded: no record
        record = report_narrative_record(round_index, capability.name, exploratory, attempts, chosen)
        return {"report": report_text, "rounds": [entry], **log_update(record)}
    return report_node


# --- M1.3b: the boundary gate and the exploratory round ---------------------------------------------------

def _round_two_proposal(state: AgentState, store: DatasetStore):
    """Re-profile the cleaned dataset (read-only) and propose the exploratory predictors."""
    plan = state["plan"]
    profile = profile_dataset(state["dataset"], store=store)
    group_column = (plan.candidate_predictors or ())[0]
    return profile, propose_exploratory_predictors(profile, group_column, plan.outcome_name)


def route_after_report(state: AgentState) -> str:
    """After a report: an exploratory round ends the run; a comparison whose stage plan offered modeling goes
    to the boundary gate; everything else ends."""
    if state.get("exploratory_round"):
        return END
    stage_plan, plan = state.get("stage_plan"), state.get("plan")
    if stage_plan is None or plan is None or plan.question_type.value not in COMPARISON_TYPES:
        return END
    modeling = next(s for s in plan_steps(stage_plan) if s["step"] == "modeling")
    return "stage_gate" if modeling["state"] == "offered" else END


def _refused_attempts(state: AgentState) -> int:
    """Refused stage-gate answers since the last stage plan (the current question on this thread)."""
    return _refused_count(state, "planner.stage_gate", ("planner.stage_plan",))


def make_stage_gate_node(store: DatasetStore) -> Callable[[AgentState], dict]:
    """
    M1.3b: after the main report, offer the exploratory regression when the stage plan offered it and the data
    has a usable extra predictor; otherwise record that it was not offered and end. The proposal is computed
    before interrupt() from a read-only re-profile, so the resumed re-run repeats it harmlessly.

    Step 2b: the user may choose the predictors ({"action": "run", "predictors": [...]}). The choice goes
    through the same rules as the proposal (core.predictor_rules). A choice that breaks a rule is refused: the
    node returns normally with a refused record (so the refusal is saved), and route_after_stage_gate re-opens
    the gate, whose payload then shows the refusal.
    """
    def stage_gate_node(state: AgentState) -> dict:
        plan = state["plan"]
        rounds = state.get("rounds") or []
        capability = rounds[-1]["capability"] if rounds else None
        profile, proposal = _round_two_proposal(state, store)
        offered = exploratory_offer(state.get("stage_plan"), plan, proposal)
        if not offered:
            record = stage_gate_record(plan, proposal, None, "not_offered", False, capability)
            return {"next_round": {"run": False, "predictors": []}, "stage_gate_refusal": None, **log_update(record)}

        decision = interrupt(stage_gate_payload(plan, state.get("test_result"), capability, len(rounds),
                                                state["stage_plan"], proposal,
                                                refused=state.get("stage_gate_refusal"))) or {}
        attempts = _refused_attempts(state)
        if decision.get("action") == "run":
            predictors = list(proposal["predictors"])
            if "predictors" in decision:
                submitted = decision["predictors"]
                checked = validate_predictor_choice(profile, submitted, plan.outcome_name, proposal["predictors"][0],
                                                    MAX_OTHER_PREDICTORS)
                if not checked["ok"]:
                    refusal = {"submitted": list(submitted) if isinstance(submitted, list) else [],
                               "refusals": checked["refusals"], "attempt": attempts + 1}
                    record = stage_gate_record(plan, proposal, decision, "refused", True, capability,
                                               refused_attempts=attempts, refusal=refusal)
                    return {"stage_gate_refusal": refusal, "next_round": None, **log_update(record)}
                predictors = checked["predictors"]
            record = stage_gate_record(plan, proposal, decision, "run", True, capability,
                                       chosen_predictors=predictors, refused_attempts=attempts)
            return {"next_round": {"run": True, "predictors": predictors}, "stage_gate_refusal": None,
                    **log_update(record)}
        record = stage_gate_record(plan, proposal, decision, "skip", True, capability, refused_attempts=attempts)
        return {"next_round": {"run": False, "predictors": []}, "stage_gate_refusal": None, **log_update(record)}
    return stage_gate_node


def route_after_stage_gate(state: AgentState) -> str:
    """A refused choice re-opens the gate; run starts the exploratory round; skip ends the run."""
    if state.get("stage_gate_refusal") is not None:
        return "stage_gate"
    return "start_round" if (state.get("next_round") or {}).get("run") else END


def make_start_round_node(store: DatasetStore) -> Callable[[AgentState], dict]:
    """
    M1.3b: set up the exploratory round by code: a new regression plan on the same outcome with the predictors
    chosen at the gate. The existing select_method, freeze (the user's plan approval), execute and report run it.
    """
    def start_round_node(state: AgentState) -> dict:
        predictors = state["next_round"]["predictors"]
        profile, proposal = _round_two_proposal(state, store)
        plan2 = build_exploratory_plan(state["plan"], state["dataset"].handle, predictors)
        route = route_for_plan(plan2, profile)
        if route["status"] != "ok":
            raise ValueError(f"The exploratory round cannot run: {route['reason']}")
        record = round_start_record(plan2, profile, proposal, from_round=len(state.get("rounds") or []))
        return {"plan": plan2, "route": route, "profile": profile, "exploratory_round": True,
                "plan_approval_refusal": None, "reselect_after_edit": None, **log_update(record)}
    return start_round_node
