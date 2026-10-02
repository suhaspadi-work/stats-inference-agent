import pytest
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from core.agent_state import AgentState
from core.nodes import classify_node
from core.plan import ClassificationDecision, QuestionType, CausalStatus


def fake_classify(request_text: str) -> ClassificationDecision:
    """
    Deterministic stub: reports LOW confidence unless the request contains
    a clarification (simulating what a re-classification after human input
    would look like). No API call -- this tests the NODE's routing logic,
    not the real classifier (already covered by tests/test_classifier.py).
    """
    if "Clarification:" in request_text:
        return ClassificationDecision(
            question_type=QuestionType.ASSOCIATION, causal_status=CausalStatus.OBSERVATIONAL,
            rationale="Clarified by human input", alternatives=(), confidence=0.9, evidence={},
        )
    return ClassificationDecision(
        question_type=QuestionType.ASSOCIATION, causal_status=CausalStatus.OBSERVATIONAL,
        rationale="Ambiguous without more context", alternatives=(), confidence=0.3, evidence={},
    )


def build_test_graph():
    """A minimal single-node graph, just to exercise classify_node's interrupt mechanics in isolation."""
    graph = StateGraph(AgentState)
    graph.add_node("classify", lambda state: classify_node(state, classify_fn=fake_classify))
    graph.add_edge(START, "classify")
    graph.add_edge("classify", END)
    return graph.compile(checkpointer=InMemorySaver())


def test_low_confidence_triggers_interrupt_not_silent_proceed():
    app = build_test_graph()
    config = {"configurable": {"thread_id": "test-1"}}

    result = app.invoke({"request_text": "Is satisfaction related to response time?", "session_id": "s1"}, config)

    assert "__interrupt__" in result
    interrupt_payload = result["__interrupt__"][0].value
    assert interrupt_payload["type"] == "clarification_needed"
    assert "confidence" in interrupt_payload["question"].lower() or "0.3" in interrupt_payload["question"]
    # Critically: classification was NOT silently set on low confidence
    assert "classification" not in result


def test_resuming_after_clarification_reclassifies_with_higher_confidence():
    app = build_test_graph()
    config = {"configurable": {"thread_id": "test-2"}}

    first = app.invoke({"request_text": "Is satisfaction related to response time?", "session_id": "s1"}, config)
    assert "__interrupt__" in first

    resumed = app.invoke(Command(resume="I mean does higher satisfaction cause faster response, or vice versa?"), config)

    assert "__interrupt__" not in resumed
    assert resumed["classification"].confidence == 0.9
    assert resumed["classification"].rationale == "Clarified by human input"


def test_high_confidence_never_interrupts():
    def always_confident(request_text: str) -> ClassificationDecision:
        return ClassificationDecision(
            question_type=QuestionType.TWO_GROUP_COMPARISON, causal_status=CausalStatus.OBSERVATIONAL,
            rationale="Clear two-group comparison", alternatives=(), confidence=0.95, evidence={},
        )

    graph = StateGraph(AgentState)
    graph.add_node("classify", lambda state: classify_node(state, classify_fn=always_confident))
    graph.add_edge(START, "classify")
    graph.add_edge("classify", END)
    app = graph.compile(checkpointer=InMemorySaver())

    result = app.invoke(
        {"request_text": "Did West spend more than East?", "session_id": "s1"},
        {"configurable": {"thread_id": "test-3"}},
    )
    assert "__interrupt__" not in result
    assert result["classification"].confidence == 0.95


@pytest.mark.live
def test_live_classify_node_on_a_clear_request_does_not_interrupt():
    """One real, live check: confirm the node works end-to-end against the actual model, not just the stub."""
    graph = StateGraph(AgentState)
    graph.add_node("classify", classify_node)  # uses the real classify_request by default
    graph.add_edge(START, "classify")
    graph.add_edge("classify", END)
    app = graph.compile(checkpointer=InMemorySaver())

    result = app.invoke(
        {"request_text": "Did customers in the West region spend more than the East region?", "session_id": "s1"},
        {"configurable": {"thread_id": "test-live-1"}},
    )
    assert "__interrupt__" not in result
    assert result["classification"].question_type.value == "two_group_comparison"