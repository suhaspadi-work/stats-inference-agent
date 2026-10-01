import pytest
from langgraph.graph import StateGraph, START, END

from core.agent_state import AgentState
from core.nodes import make_report_node
from core.report import check_for_causal_language, CausalLanguageViolation
from core.plan import AnalysisPlan, QuestionType, CausalStatus, OutcomeType
from core.plan import UnaddressedIssue


def make_observational_plan():
    return AnalysisPlan(
        request_text="Did West spend more than East?", session_id="session-abc",
        question_type=QuestionType.TWO_GROUP_COMPARISON, causal_status=CausalStatus.OBSERVATIONAL,
        outcome_name="total_spend", outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="customer", data_sources=("customers@v1",),
        candidate_predictors=("region",),
    )


def fake_test_result():
    return {
        "method": "welch_t_test", "statistic": 2.5, "p_value": 0.012,
        "effect_size": 0.35, "confidence_interval": (1.2, 8.4),
        "group_a_n": 200, "group_b_n": 200, "group_a_mean": 50.1, "group_b_mean": 55.3,
    }


def test_guard_catches_causal_language_directly():
    plan = make_observational_plan()
    with pytest.raises(CausalLanguageViolation):
        check_for_causal_language("Being in the West region causes higher spending.", plan)


def test_guard_allows_association_language():
    plan = make_observational_plan()
    check_for_causal_language("Being in the West region is associated with higher spending.", plan)
    # No exception -- this is the success case


def test_report_node_blocks_a_report_that_violates_the_guard():
    plan = make_observational_plan()

    def violating_report_fn(plan, test_result):
        return "Living in the West region drives higher customer spending."

    graph = StateGraph(AgentState)
    graph.add_node("report", make_report_node(report_fn=violating_report_fn))
    graph.add_edge(START, "report")
    graph.add_edge("report", END)
    app = graph.compile()

    with pytest.raises(CausalLanguageViolation):
        app.invoke({"plan": plan, "test_result": fake_test_result(), "request_text": "x", "session_id": "s"})


def test_report_node_passes_through_a_compliant_report():
    plan = make_observational_plan()

    def compliant_report_fn(plan, test_result):
        return "Customers in the West region show higher average spending, associated with region."

    graph = StateGraph(AgentState)
    graph.add_node("report", make_report_node(report_fn=compliant_report_fn))
    graph.add_edge(START, "report")
    graph.add_edge("report", END)
    app = graph.compile()

    result = app.invoke({"plan": plan, "test_result": fake_test_result(), "request_text": "x", "session_id": "s"})
    assert "associated" in result["report"]


@pytest.mark.live
def test_live_report_node_on_observational_data_avoids_causal_language():
    plan = make_observational_plan()

    graph = StateGraph(AgentState)
    graph.add_node("report", make_report_node())  # real generate_report
    graph.add_edge(START, "report")
    graph.add_edge("report", END)
    app = graph.compile()

    result = app.invoke({"plan": plan, "test_result": fake_test_result(), "request_text": "x", "session_id": "s"})
    # If this ever fails, generate_report's own guard would have already
    # raised -- this additionally confirms the happy path produces real text
    assert len(result["report"]) > 0
    assert "55.3" in result["report"] or "50.1" in result["report"] or "0.012" in result["report"]

def test_guard_allows_correct_negated_causal_disclaimer():
    """
    The exact case the live test surfaced: a model correctly writing
    'not that it causes the difference' should NOT be flagged -- it's
    disclaiming causation, not asserting it.
    """
    plan = make_observational_plan()
    check_for_causal_language(
        "Region is associated with total spend; we cannot say that it causes the difference.",
        plan,
    )
    # No exception -- this is the success case


def test_guard_still_catches_unhedged_causal_claims():
    """Confirms the negation-window fix didn't accidentally weaken the guard."""
    plan = make_observational_plan()
    with pytest.raises(CausalLanguageViolation):
        check_for_causal_language("Living in the West region causes higher spending.", plan)

def test_guard_allows_rather_than_caused_by_phrasing():
    """
    Regression test for a real false positive found via the live test: the
    model correctly wrote 'associated with, rather than caused by, the
    regional difference' -- a safe disclaimer using 'rather than' instead
    of a 'not'-style negation. Must not be flagged.
    """
    plan = make_observational_plan()
    check_for_causal_language(
        "The higher spend in the West is associated with, rather than caused by, the regional difference.",
        plan,
    )

def make_plan_with_unaddressed_issue():
    plan = make_observational_plan()
    issue = UnaddressedIssue(
        issue_type="duplicate_rows", description="3 exact duplicate rows detected",
        proposed_operation="dedupe(subset=None)", rejection_reason="User wanted to investigate manually",
        affected_columns=("customer_id",),
    )
    return plan.record_unaddressed_issue(issue)


def test_report_node_with_unaddressed_issue_includes_the_caveat():
    plan = make_plan_with_unaddressed_issue()

    def report_fn_mentioning_caveat(plan, test_result):
        return (
            "Customers in the West region show higher average spending, associated with region. "
            "Note: 3 exact duplicate rows were identified but not removed, per the user's decision "
            "to investigate manually -- results should be interpreted with this caveat in mind."
        )

    graph = StateGraph(AgentState)
    graph.add_node("report", make_report_node(report_fn=report_fn_mentioning_caveat))
    graph.add_edge(START, "report")
    graph.add_edge("report", END)
    app = graph.compile()

    result = app.invoke({"plan": plan, "test_result": fake_test_result(), "request_text": "x", "session_id": "s"})
    assert "duplicate" in result["report"].lower()
    assert "caveat" in result["report"].lower() or "not removed" in result["report"].lower()


@pytest.mark.live
def test_live_report_mentions_unaddressed_issue_when_present():
    plan = make_plan_with_unaddressed_issue()

    graph = StateGraph(AgentState)
    graph.add_node("report", make_report_node())
    graph.add_edge(START, "report")
    graph.add_edge("report", END)
    app = graph.compile()

    result = app.invoke({"plan": plan, "test_result": fake_test_result(), "request_text": "x", "session_id": "s"})
    report_lower = result["report"].lower()
    assert "duplicate" in report_lower
    assert any(phrase in report_lower for phrase in (
        "investigate", "not removed", "not addressed", "unaddressed", "left in the data", "caveat",
    ))


@pytest.mark.live
def test_live_report_without_unaddressed_issues_does_not_mention_any():
    """Confirms the conditional prompt addition is genuinely conditional -- clean plans get no caveat text."""
    plan = make_observational_plan()  # no unaddressed_issues

    graph = StateGraph(AgentState)
    graph.add_node("report", make_report_node())
    graph.add_edge(START, "report")
    graph.add_edge("report", END)
    app = graph.compile()

    result = app.invoke({"plan": plan, "test_result": fake_test_result(), "request_text": "x", "session_id": "s"})
    assert "duplicate" not in result["report"].lower()