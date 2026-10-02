import pandas as pd
import pytest
from langgraph.graph import StateGraph, START, END

from core.agent_state import AgentState
from core.dataset import LocalDiskStore
from core.ingestion import load_dataset
from core.nodes import make_execute_node
from core.plan import AnalysisPlan, MethodDecision, QuestionType, CausalStatus, OutcomeType
from core.synthetic import generate_two_group_continuous
from core.testing import select_two_group_method


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


@pytest.fixture
def dataset(tmp_path, store):
    df, spec = generate_two_group_continuous(n_per_group=200, mean_a=50, effect=5.0, std=10.0, seed=1)
    df = df.rename(columns={"outcome": "total_spend", "group": "region"})
    path = tmp_path / "customers.csv"
    df.to_csv(path, index=False)
    ds = load_dataset(path, session_id="session-abc", name="customers", store=store)
    return ds, spec


def make_frozen_plan(dataset, store, predictors=("region",)):
    plan = AnalysisPlan(
        request_text="x", session_id="session-abc",
        question_type=QuestionType.TWO_GROUP_COMPARISON, causal_status=CausalStatus.OBSERVATIONAL,
        outcome_name="total_spend", outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="customer", data_sources=(dataset.handle,),
        candidate_predictors=predictors,
    )
    if not predictors:
        placeholder = MethodDecision(stage="test_selection", chosen_method="placeholder", rationale="x", alternatives=())
        return plan.record_method_decision(placeholder).freeze()
    choice = select_two_group_method(dataset, store, outcome_col="total_spend", group_col=predictors[0])
    return plan.record_method_decision(choice.decision).freeze()


def test_execute_node_runs_real_test_and_marks_plan_executed(dataset, store):
    ds, spec = dataset
    plan = make_frozen_plan(ds, store)

    graph = StateGraph(AgentState)
    graph.add_node("execute", make_execute_node(store))
    graph.add_edge(START, "execute")
    graph.add_edge("execute", END)
    app = graph.compile()

    result = app.invoke({"plan": plan, "dataset": ds, "request_text": "x", "session_id": "session-abc"})

    assert result["plan"].status.value == "executed"
    assert result["test_result"]["p_value"] < 0.05
    # Exactly ONE method decision should exist -- the one made before freeze,
    # never a second, silent one made during execution
    assert len(result["plan"].method_decisions) == 1
    assert result["plan"].latest_decision.chosen_method in ("student_t_test", "welch_t_test")


def test_execute_node_refuses_to_run_on_an_unfrozen_plan(dataset, store):
    ds, spec = dataset
    unfrozen_plan = AnalysisPlan(
        request_text="x", session_id="session-abc",
        question_type=QuestionType.TWO_GROUP_COMPARISON, causal_status=CausalStatus.OBSERVATIONAL,
        outcome_name="total_spend", outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="customer", data_sources=(ds.handle,),
        candidate_predictors=("region",),
    )  # never frozen

    graph = StateGraph(AgentState)
    graph.add_node("execute", make_execute_node(store))
    graph.add_edge(START, "execute")
    graph.add_edge("execute", END)
    app = graph.compile()

    with pytest.raises(ValueError, match="not FROZEN"):
        app.invoke({"plan": unfrozen_plan, "dataset": ds, "request_text": "x", "session_id": "session-abc"})


def test_execute_node_refuses_wrong_predictor_count(dataset, store):
    ds, spec = dataset
    plan = make_frozen_plan(ds, store, predictors=())  # zero predictors

    graph = StateGraph(AgentState)
    graph.add_node("execute", make_execute_node(store))
    graph.add_edge(START, "execute")
    graph.add_edge("execute", END)
    app = graph.compile()

    with pytest.raises(ValueError, match="exactly one grouping predictor"):
        app.invoke({"plan": plan, "dataset": ds, "request_text": "x", "session_id": "session-abc"})


def test_execute_node_result_matches_spec_effect_direction(dataset, store):
    """Cross-check against the synthetic generator's own known ground truth."""
    ds, spec = dataset
    plan = make_frozen_plan(ds, store)

    graph = StateGraph(AgentState)
    graph.add_node("execute", make_execute_node(store))
    graph.add_edge(START, "execute")
    graph.add_edge("execute", END)
    app = graph.compile()

    result = app.invoke({"plan": plan, "dataset": ds, "request_text": "x", "session_id": "session-abc"})

    observed_effect = result["test_result"]["group_b_mean"] - result["test_result"]["group_a_mean"]
    assert abs(observed_effect - spec.true_effect) < 1.5  # same tolerance as Phase 3's own generator test