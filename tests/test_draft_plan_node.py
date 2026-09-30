import pandas as pd
import pytest
from langgraph.graph import StateGraph, START, END

from core.agent_state import AgentState
from core.dataset import LocalDiskStore
from core.ingestion import load_dataset, profile_dataset
from core.nodes import make_draft_plan_node
from core.plan import ClassificationDecision, QuestionType, CausalStatus, OutcomeType
from core.classifier import _DraftPlanOut


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


@pytest.fixture
def dataset_and_profile(tmp_path, store):
    df = pd.DataFrame({
        "customer_id": [1, 2, 3, 4],
        "region": ["East", "West", "East", "West"],
        "total_spend": [100.0, 150.0, 90.0, 200.0],
    })
    path = tmp_path / "customers.csv"
    df.to_csv(path, index=False)
    dataset = load_dataset(path, session_id="session-abc", name="customers", store=store)
    profile = profile_dataset(dataset, store=store)
    return dataset, profile


def fake_classification():
    return ClassificationDecision(
        question_type=QuestionType.TWO_GROUP_COMPARISON, causal_status=CausalStatus.OBSERVATIONAL,
        rationale="test", alternatives=(), confidence=0.9, evidence={},
    )


def test_draft_plan_node_builds_a_valid_plan(dataset_and_profile):
    dataset, profile = dataset_and_profile

    def fake_draft_fn(request_text, profile):
        return _DraftPlanOut(
            outcome_name="total_spend", outcome_type=OutcomeType.CONTINUOUS,
            unit_of_analysis="customer", candidate_predictors=["region"],
            rationale="Comparing spend across two named regions",
        )

    graph = StateGraph(AgentState)
    graph.add_node("draft_plan", make_draft_plan_node(draft_fn=fake_draft_fn))
    graph.add_edge(START, "draft_plan")
    graph.add_edge("draft_plan", END)
    app = graph.compile()

    result = app.invoke({
        "request_text": "Did West spend more than East?", "session_id": "session-abc",
        "dataset": dataset, "profile": profile, "classification": fake_classification(),
    })

    plan = result["plan"]
    assert plan.outcome_name == "total_spend"
    assert plan.outcome_type == OutcomeType.CONTINUOUS
    assert plan.candidate_predictors == ("region",)
    assert plan.classification_decision.confidence == 0.9
    assert plan.status.value in ("draft", "refined")


def test_draft_plan_node_rejects_hallucinated_column(dataset_and_profile):
    dataset, profile = dataset_and_profile

    def hallucinating_draft_fn(request_text, profile):
        return _DraftPlanOut(
            outcome_name="revenue_that_does_not_exist", outcome_type=OutcomeType.CONTINUOUS,
            unit_of_analysis="customer", candidate_predictors=["region"],
            rationale="fabricated",
        )

    graph = StateGraph(AgentState)
    graph.add_node("draft_plan", make_draft_plan_node(draft_fn=hallucinating_draft_fn))
    graph.add_edge(START, "draft_plan")
    graph.add_edge("draft_plan", END)
    app = graph.compile()

    with pytest.raises(ValueError, match="not present in the dataset"):
        app.invoke({
            "request_text": "x", "session_id": "session-abc",
            "dataset": dataset, "profile": profile, "classification": fake_classification(),
        })


@pytest.mark.live
def test_live_draft_plan_node_picks_real_columns(dataset_and_profile):
    dataset, profile = dataset_and_profile

    graph = StateGraph(AgentState)
    graph.add_node("draft_plan", make_draft_plan_node())  # real draft_plan_fields
    graph.add_edge(START, "draft_plan")
    graph.add_edge("draft_plan", END)
    app = graph.compile()

    result = app.invoke({
        "request_text": "Did customers in the West region spend more than the East region?",
        "session_id": "session-abc", "dataset": dataset, "profile": profile,
        "classification": fake_classification(),
    })

    plan = result["plan"]
    assert plan.outcome_name == "total_spend"
    assert "region" in plan.candidate_predictors