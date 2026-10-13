import pytest
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from core.agent_state import AgentState
from core.dataset import LocalDiskStore
from core.graph import build_agent_graph
from core.nodes import classify_node, clarify_node, reclassify_node, route_after_classify
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


def build_test_graph(classify_fn=fake_classify):
    """
    The classify -> (clarify -> reclassify) -> profile wiring from build_agent_graph,
    with a stub profile node, to exercise the clarification mechanics in isolation.
    """
    graph = StateGraph(AgentState)
    graph.add_node("classify", lambda state: classify_node(state, classify_fn=classify_fn))
    graph.add_node("clarify", clarify_node)
    graph.add_node("reclassify", lambda state: reclassify_node(state, classify_fn=classify_fn))
    graph.add_node("profile", lambda state: {})
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_after_classify, {"profile": "profile", "clarify": "clarify"})
    graph.add_edge("clarify", "reclassify")
    graph.add_edge("reclassify", "profile")
    graph.add_edge("profile", END)
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

    app = build_test_graph(always_confident)

    result = app.invoke(
        {"request_text": "Did West spend more than East?", "session_id": "s1"},
        {"configurable": {"thread_id": "test-3"}},
    )
    assert "__interrupt__" not in result
    assert result["classification"].confidence == 0.95


def _counting(fn):
    calls = []

    def wrapped(request_text):
        calls.append(request_text)
        return fn(request_text)
    return wrapped, calls


def _confident(request_text: str) -> ClassificationDecision:
    return ClassificationDecision(
        question_type=QuestionType.TWO_GROUP_COMPARISON, causal_status=CausalStatus.OBSERVATIONAL,
        rationale="Clear two-group comparison", alternatives=(), confidence=0.95, evidence={},
    )


def _points(log):
    return [(e["decision_point"], e["chosen"]) for e in log]


def test_high_confidence_logs_exactly_three_records():
    app = build_test_graph(_confident)
    result = app.invoke({"request_text": "Did West spend more than East?", "session_id": "s1"},
                        {"configurable": {"thread_id": "test-records-high"}})
    assert result["classification"].confidence == 0.95
    assert result["pending_classification"] is None
    assert _points(result["decision_log"]) == [
        ("intake.question_type", "two_group_comparison"),
        ("intake.causal_status", "observational"),
        ("intake.clarification_gate", "proceed"),
    ]
    assert all(e["evidence"].get("pass", 1) == 1 for e in result["decision_log"])


def test_low_confidence_pauses_with_first_pass_records_and_no_classification():
    app = build_test_graph()
    paused = app.invoke({"request_text": "Is satisfaction related to response time?", "session_id": "s1"},
                        {"configurable": {"thread_id": "test-records-low"}})
    assert "__interrupt__" in paused
    assert "classification" not in paused
    assert paused["pending_classification"].confidence == 0.3
    assert _points(paused["decision_log"]) == [
        ("intake.question_type", "association"),
        ("intake.causal_status", "observational"),
        ("intake.clarification_gate", "ask_human"),
    ]


def test_interrupt_payload_unchanged():
    app = build_test_graph()
    paused = app.invoke({"request_text": "Is satisfaction related to response time?", "session_id": "s1"},
                        {"configurable": {"thread_id": "test-payload"}})
    payload = paused["__interrupt__"][0].value
    assert payload == {
        "type": "clarification_needed",
        "question": ("Your request could reasonably be read more than one way "
                     "(confidence: 0.30). Ambiguous without more context "
                     "Could you clarify what you're asking?"),
        "original_request": "Is satisfaction related to response time?",
        "tentative_classification": "association",
    }


def test_resume_logs_clarification_and_superseding_second_pass_with_two_model_calls():
    fn, calls = _counting(fake_classify)
    app = build_test_graph(fn)
    config = {"configurable": {"thread_id": "test-records-resume"}}
    paused = app.invoke({"request_text": "Is satisfaction related to response time?", "session_id": "s1"}, config)
    assert len(calls) == 1
    first_ids = {e["decision_point"]: e["id"] for e in paused["decision_log"]}

    resumed = app.invoke(Command(resume="Just whether they move together."), config)
    assert "__interrupt__" not in resumed
    assert resumed["classification"].confidence == 0.9
    assert resumed["pending_classification"] is None
    assert resumed["clarification_answer"] == "Just whether they move together."

    log = resumed["decision_log"]
    assert _points(log) == [
        ("intake.question_type", "association"),
        ("intake.causal_status", "observational"),
        ("intake.clarification_gate", "ask_human"),
        ("intake.clarification", "clarification_provided"),
        ("intake.question_type", "association"),
        ("intake.causal_status", "observational"),
    ]
    assert log[3]["decided_by"] == "human"
    assert log[3]["rationale"] == "Just whether they move together."
    assert log[4]["supersedes"] == first_ids["intake.question_type"]
    assert log[5]["supersedes"] == first_ids["intake.causal_status"]
    assert log[4]["evidence"]["pass"] == 2 and log[5]["evidence"]["pass"] == 2
    assert len({e["id"] for e in log}) == 6  # nothing duplicated

    assert len(calls) == 2  # first pass once, second pass once; resume does not re-run classify
    assert calls[1] == "Is satisfaction related to response time?\n\nClarification: Just whether they move together."


def test_stale_pending_classification_does_not_misroute_new_confident_request():
    def by_text(request_text):
        return _confident(request_text) if "West" in request_text else fake_classify(request_text)

    app = build_test_graph(by_text)
    config = {"configurable": {"thread_id": "test-stale"}}
    paused = app.invoke({"request_text": "Is satisfaction related to response time?", "session_id": "s1"}, config)
    assert paused["pending_classification"] is not None  # abandoned: never resumed

    result = app.invoke({"request_text": "Did West spend more than East?", "session_id": "s1"}, config)
    assert "__interrupt__" not in result
    assert result["pending_classification"] is None
    assert result["classification"].question_type == QuestionType.TWO_GROUP_COMPARISON
    assert _points(result["decision_log"])[-1] == ("intake.clarification_gate", "proceed")


def test_route_after_classify():
    assert route_after_classify({}) == "profile"
    assert route_after_classify({"pending_classification": None}) == "profile"
    assert route_after_classify({"pending_classification": fake_classify("x")}) == "clarify"


def test_full_graph_has_classify_clarify_reclassify(tmp_path):
    app = build_agent_graph(LocalDiskStore(base_dir=tmp_path))
    nodes = set(app.get_graph().nodes)
    assert {"classify", "clarify", "reclassify", "profile"} <= nodes


def test_full_graph_classification_edges(tmp_path):
    app = build_agent_graph(LocalDiskStore(base_dir=tmp_path))
    edges = {(e.source, e.target): e.conditional for e in app.get_graph().edges}
    assert edges[("classify", "clarify")] is True
    assert edges[("classify", "profile")] is True
    assert edges[("clarify", "reclassify")] is False
    assert edges[("reclassify", "profile")] is False
    assert ("classify", "reclassify") not in edges


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