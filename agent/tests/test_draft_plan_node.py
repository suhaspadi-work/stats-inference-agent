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


def _run_draft(dataset, profile, classification, draft_fn):
    graph = StateGraph(AgentState)
    graph.add_node("draft_plan", make_draft_plan_node(draft_fn=draft_fn))
    graph.add_edge(START, "draft_plan")
    graph.add_edge("draft_plan", END)
    return graph.compile().invoke({
        "request_text": "Did West spend more than East?", "session_id": "session-abc",
        "dataset": dataset, "profile": profile, "classification": classification,
    })


def test_draft_plan_node_logs_four_records_and_plan_is_unchanged(dataset_and_profile):
    dataset, profile = dataset_and_profile

    def fake_draft_fn(request_text, profile):
        return _DraftPlanOut(
            outcome_name="total_spend", outcome_type=OutcomeType.CONTINUOUS,
            unit_of_analysis="customer", candidate_predictors=["region"],
            rationale="Comparing spend across two named regions",
        )

    result = _run_draft(dataset, profile, fake_classification(), fake_draft_fn)

    log = result["decision_log"]
    assert [(e["decision_point"], e["chosen"]) for e in log] == [
        ("planner.outcome", "total_spend"),
        ("planner.outcome_type", "continuous"),
        ("planner.predictors", "region"),
        ("planner.unit_of_analysis", "customer"),
    ]
    assert log[0]["rationale"] == "Comparing spend across two named regions"
    assert [c["option"] for c in log[0]["candidates"]] == [c.name for c in profile.columns]

    plan = result["plan"]
    assert plan.outcome_name == "total_spend"
    assert plan.outcome_type == OutcomeType.CONTINUOUS
    assert plan.unit_of_analysis == "customer"
    assert plan.candidate_predictors == ("region",)


def test_descriptive_classification_adds_no_records_and_keeps_placeholders(dataset_and_profile):
    dataset, profile = dataset_and_profile
    descriptive = ClassificationDecision(
        question_type=QuestionType.DESCRIPTIVE, causal_status=CausalStatus.OBSERVATIONAL,
        rationale="describe", alternatives=(), confidence=0.9, evidence={},
    )

    def must_not_be_called(request_text, profile):
        raise AssertionError("descriptive plans must not call the draft model")

    result = _run_draft(dataset, profile, descriptive, must_not_be_called)
    assert "decision_log" not in result or result["decision_log"] == []
    plan = result["plan"]
    assert plan.outcome_name == "(whole dataset)"
    assert plan.outcome_type == OutcomeType.CONTINUOUS
    assert plan.unit_of_analysis == "row"


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