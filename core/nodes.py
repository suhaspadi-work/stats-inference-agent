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
from core.testing import run_two_group_test
from core.report import generate_report
from core.report import generate_report, check_for_causal_language

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

def make_draft_plan_node(draft_fn=draft_plan_fields) -> Callable[[AgentState], dict]:
    """Injectable draft_fn, same pattern as classify_node, for deterministic testing."""
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
        return {"plan": plan}
    return draft_plan_node

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
    Runs the actual hypothesis test -- the only node so far that produces a
    real statistical result. Refuses to run unless the plan is FROZEN,
    structurally enforcing the pre-registration discipline rather than
    trusting the graph's edges alone to guarantee correct ordering.
    """
    def execute_node(state: AgentState) -> dict:
        plan = state["plan"]
        if plan.status.value != "frozen":
            raise ValueError(f"Cannot execute a plan that is not FROZEN (status: {plan.status.value}).")

        predictors = plan.candidate_predictors or ()
        if len(predictors) != 1:
            raise ValueError(
                f"run_two_group_test requires exactly one grouping predictor; "
                f"got {len(predictors)}: {predictors}."
            )
        group_col = predictors[0]

        result, method_decision = run_two_group_test(
            dataset=state["dataset"], store=store,
            outcome_col=plan.outcome_name, group_col=group_col, alpha=plan.alpha,
        )

        updated_plan = plan.record_method_decision(method_decision).mark_executed()

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