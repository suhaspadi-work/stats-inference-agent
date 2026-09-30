import pytest
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from core.agent_state import AgentState
from core.nodes import freeze_node
from core.plan import AnalysisPlan, MethodDecision, AlternativeConsidered, QuestionType, CausalStatus, OutcomeType


def make_test_plan():
    plan = AnalysisPlan(
        request_text="Did West spend more than East?", session_id="session-abc",
        question_type=QuestionType.TWO_GROUP_COMPARISON, causal_status=CausalStatus.OBSERVATIONAL,
        outcome_name="total_spend", outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="customer", data_sources=("customers@v1",),
        candidate_predictors=("region",),
    )
    decision = MethodDecision(
        stage="test_selection", chosen_method="welch_t_test",
        rationale="test", alternatives=(AlternativeConsidered(method="mann_whitney_u", rejected_because="test"),),
    )
    return plan.record_method_decision(decision)


def build_freeze_graph():
    graph = StateGraph(AgentState)
    graph.add_node("freeze", freeze_node)
    graph.add_edge(START, "freeze")
    graph.add_edge("freeze", END)
    return graph.compile(checkpointer=InMemorySaver())


def test_freeze_interrupts_and_waits_for_human():
    app = build_freeze_graph()
    config = {"configurable": {"thread_id": "t1"}}
    result = app.invoke({"plan": make_test_plan(), "request_text": "x", "session_id": "session-abc"}, config)

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["type"] == "plan_approval"
    assert payload["candidate_predictors"] == ["region"]
    assert payload["allowed_decisions"] == ["approve", "edit", "reject"]
    # Critically: the plan must NOT be frozen while waiting -- freeze_node
    # paused before ever calling plan.freeze(), so status is still DRAFT
    assert result["plan"].status.value == "draft"


def test_approve_freezes_the_plan():
    app = build_freeze_graph()
    config = {"configurable": {"thread_id": "t2"}}
    app.invoke({"plan": make_test_plan(), "request_text": "x", "session_id": "session-abc"}, config)

    result = app.invoke(Command(resume={"action": "approve"}), config)
    assert "__interrupt__" not in result
    assert result["plan"].status.value == "frozen"
    # Note: candidate_predictors round-trips through the checkpointer as a
    # list, not a tuple -- JSON has no native tuple type. This is a real,
    # documented behavior of checkpointed state, not a bug in freeze_node.
    assert list(result["plan"].candidate_predictors) == ["region"]


def test_edit_changes_predictors_before_freezing():
    app = build_freeze_graph()
    config = {"configurable": {"thread_id": "t3"}}
    app.invoke({"plan": make_test_plan(), "request_text": "x", "session_id": "session-abc"}, config)

    result = app.invoke(
        Command(resume={"action": "edit", "candidate_predictors": ["region", "plan_type"]}), config,
    )
    assert result["plan"].status.value == "frozen"
    assert list(result["plan"].candidate_predictors) == ["region", "plan_type"]


def test_reject_never_freezes_and_signals_clarification_needed():
    app = build_freeze_graph()
    config = {"configurable": {"thread_id": "t4"}}
    app.invoke({"plan": make_test_plan(), "request_text": "x", "session_id": "session-abc"}, config)

    result = app.invoke(Command(resume={"action": "reject", "reason": "Wrong predictors"}), config)

    assert result["plan"].status.value != "frozen"
    assert "clarification_needed" in result
    assert "rejected" in result["clarification_needed"].lower()
    assert "Wrong predictors" in result["clarification_needed"]