import pandas as pd
import pytest
from langgraph.graph import StateGraph, START, END

from core.agent_state import AgentState
from core.dataset import LocalDiskStore
from core.ingestion import load_dataset, profile_dataset
from core.nodes import make_draft_plan_node, make_select_method_node
from core.classifier import _DraftPlanOut
from core.plan import ClassificationDecision, QuestionType, CausalStatus, OutcomeType, UnaddressedIssue
from core.synthetic import generate_two_group_continuous


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


@pytest.fixture
def dataset_and_profile(tmp_path, store):
    df, spec = generate_two_group_continuous(n_per_group=200, mean_a=50, effect=5.0, std=10.0, seed=1)
    df = df.rename(columns={"outcome": "total_spend", "group": "region"})
    path = tmp_path / "customers.csv"
    df.to_csv(path, index=False)
    ds = load_dataset(path, session_id="session-abc", name="customers", store=store)
    return ds, profile_dataset(ds, store=store)


def fake_classification():
    return ClassificationDecision(
        question_type=QuestionType.TWO_GROUP_COMPARISON, causal_status=CausalStatus.OBSERVATIONAL,
        rationale="test", alternatives=(), confidence=0.9, evidence={},
    )


def fake_draft_fn(request_text, profile):
    return _DraftPlanOut(
        outcome_name="total_spend", outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="customer", candidate_predictors=["region"], rationale="test",
    )


def test_draft_plan_node_folds_in_pending_unaddressed_issues(dataset_and_profile):
    ds, profile = dataset_and_profile
    issue = UnaddressedIssue(
        issue_type="duplicate_rows", description="3 duplicates detected",
        proposed_operation="dedupe(subset=None)", rejection_reason="Investigate first",
        affected_columns=("customer_id",),
    )

    graph = StateGraph(AgentState)
    graph.add_node("draft_plan", make_draft_plan_node(draft_fn=fake_draft_fn))
    graph.add_edge(START, "draft_plan")
    graph.add_edge("draft_plan", END)
    app = graph.compile()

    result = app.invoke({
        "request_text": "x", "session_id": "session-abc", "dataset": ds, "profile": profile,
        "classification": fake_classification(), "pending_unaddressed_issues": (issue,),
    })

    plan = result["plan"]
    assert len(plan.unaddressed_issues) == 1
    assert plan.unaddressed_issues[0].rejection_reason == "Investigate first"


def test_draft_plan_node_handles_no_pending_issues_gracefully(dataset_and_profile):
    """The common path: no wrangling happened, no pending_unaddressed_issues key at all."""
    ds, profile = dataset_and_profile

    graph = StateGraph(AgentState)
    graph.add_node("draft_plan", make_draft_plan_node(draft_fn=fake_draft_fn))
    graph.add_edge(START, "draft_plan")
    graph.add_edge("draft_plan", END)
    app = graph.compile()

    result = app.invoke({
        "request_text": "x", "session_id": "session-abc", "dataset": ds, "profile": profile,
        "classification": fake_classification(),
    })  # no pending_unaddressed_issues key -- must not raise

    assert len(result["plan"].unaddressed_issues) == 0


def test_select_method_node_surfaces_unaddressed_issues_in_evidence(dataset_and_profile, store):
    ds, profile = dataset_and_profile
    issue = UnaddressedIssue(
        issue_type="duplicate_rows", description="3 duplicates detected",
        proposed_operation="dedupe(subset=None)", rejection_reason="Investigate first",
        affected_columns=("customer_id",),
    )

    draft_graph = StateGraph(AgentState)
    draft_graph.add_node("draft_plan", make_draft_plan_node(draft_fn=fake_draft_fn))
    draft_graph.add_edge(START, "draft_plan")
    draft_graph.add_edge("draft_plan", END)
    draft_app = draft_graph.compile()

    draft_result = draft_app.invoke({
        "request_text": "x", "session_id": "session-abc", "dataset": ds, "profile": profile,
        "classification": fake_classification(), "pending_unaddressed_issues": (issue,),
    })

    select_graph = StateGraph(AgentState)
    select_graph.add_node("select_method", make_select_method_node(store))
    select_graph.add_edge(START, "select_method")
    select_graph.add_edge("select_method", END)
    select_app = select_graph.compile()

    final = select_app.invoke({
        "plan": draft_result["plan"], "dataset": ds, "request_text": "x", "session_id": "session-abc",
    })

    decision = final["plan"].latest_decision
    assert "unaddressed_issues" in decision.evidence
    assert decision.evidence["unaddressed_issues"][0]["rejection_reason"] == "Investigate first"


def test_select_method_node_has_no_unaddressed_issues_key_when_plan_is_clean(dataset_and_profile, store):
    ds, profile = dataset_and_profile

    draft_graph = StateGraph(AgentState)
    draft_graph.add_node("draft_plan", make_draft_plan_node(draft_fn=fake_draft_fn))
    draft_graph.add_edge(START, "draft_plan")
    draft_graph.add_edge("draft_plan", END)
    draft_app = draft_graph.compile()

    draft_result = draft_app.invoke({
        "request_text": "x", "session_id": "session-abc", "dataset": ds, "profile": profile,
        "classification": fake_classification(),
    })

    select_graph = StateGraph(AgentState)
    select_graph.add_node("select_method", make_select_method_node(store))
    select_graph.add_edge(START, "select_method")
    select_graph.add_edge("select_method", END)
    select_app = select_graph.compile()

    final = select_app.invoke({
        "plan": draft_result["plan"], "dataset": ds, "request_text": "x", "session_id": "session-abc",
    })

    assert "unaddressed_issues" not in final["plan"].latest_decision.evidence