import json
import numpy as np
import pandas as pd
import pytest
from langgraph.graph import StateGraph, START, END

from core.agent_state import AgentState
from core.dataset import LocalDiskStore
from core.ingestion import load_dataset
from core.nodes import make_execute_node, make_select_method_node
from core.plan import AnalysisPlan, QuestionType, CausalStatus, OutcomeType


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


@pytest.fixture
def dataset(tmp_path, store):
    rng = np.random.default_rng(11)
    n = 400
    region = rng.choice(["East", "West", "North"], n, p=[0.5, 0.3, 0.2])
    tenure = rng.normal(24, 8, n)
    visits = rng.normal(10, 3, n)
    shift = {"East": 0.0, "West": 8.0, "North": 15.0}
    spend = 20 + 2.0 * tenure + 1.5 * visits + np.array([shift[r] for r in region]) + rng.normal(0, 5, n)
    df = pd.DataFrame({
        "customer_id": range(1, n + 1), "region": region,
        "tenure": tenure, "visits": visits, "total_spend": spend,
    })
    path = tmp_path / "customers.csv"
    df.to_csv(path, index=False)
    return load_dataset(path, session_id="session-abc", name="customers", store=store)


def make_plan(dataset, question_type, predictors):
    return AnalysisPlan(
        request_text="x", session_id="session-abc",
        question_type=question_type, causal_status=CausalStatus.OBSERVATIONAL,
        outcome_name="total_spend", outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="customer", data_sources=(dataset.handle,),
        candidate_predictors=tuple(predictors),
    )


def run_node(node, plan, dataset):
    graph = StateGraph(AgentState)
    graph.add_node("n", node)
    graph.add_edge(START, "n")
    graph.add_edge("n", END)
    return graph.compile().invoke(
        {"plan": plan, "dataset": dataset, "request_text": "x", "session_id": "session-abc"}
    )


def select_then_freeze(dataset, store, predictors=("region", "tenure", "visits")):
    plan = make_plan(dataset, QuestionType.DRIVER_ANALYSIS, predictors)
    selected = run_node(make_select_method_node(store), plan, dataset)["plan"]
    return selected.freeze()


def test_select_node_accepts_multiple_predictors_for_driver_analysis(dataset, store):
    plan = make_plan(dataset, QuestionType.DRIVER_ANALYSIS, ("region", "tenure", "visits"))
    result = run_node(make_select_method_node(store), plan, dataset)
    assert result["plan"].primary_method in ("ols", "ols_hc3_robust")
    assert len(result["plan"].method_decisions) == 1


def test_select_node_still_requires_exactly_one_predictor_for_two_group(dataset, store):
    plan = make_plan(dataset, QuestionType.TWO_GROUP_COMPARISON, ("region", "tenure"))
    with pytest.raises(ValueError, match="exactly one grouping predictor"):
        run_node(make_select_method_node(store), plan, dataset)


def test_execute_node_runs_regression_and_marks_plan_executed(dataset, store):
    frozen = select_then_freeze(dataset, store)
    result = run_node(make_execute_node(store), frozen, dataset)

    assert result["plan"].status.value == "executed"
    tr = result["test_result"]
    assert tr["result_type"] == "regression"
    assert tr["r_squared"] > 0.8
    coefs = {c["name"]: c for c in tr["coefficients"]}
    assert abs(coefs["tenure"]["estimate"] - 2.0) < 0.3
    assert abs(coefs["visits"]["estimate"] - 1.5) < 0.3
    # exactly one method decision: the one approved before freeze
    assert len(result["plan"].method_decisions) == 1


def test_execute_node_uses_the_encodings_the_approver_saw(dataset, store):
    """Reference level is the largest group (East), carried via the MethodDecision evidence."""
    frozen = select_then_freeze(dataset, store)
    tr = run_node(make_execute_node(store), frozen, dataset)["test_result"]
    names = {c["name"] for c in tr["coefficients"]}
    assert "region[North vs East]" in names
    assert "region[West vs East]" in names
    assert tr["encodings"]["region"]["reference"] == "East"
    assert [f["label"] for f in tr["alternative_fits"]] == ["effect_coding"]


def test_regression_test_result_is_plain_json(dataset, store):
    """Checkpoint-safety: nothing but plain dicts/lists/numbers may reach state."""
    frozen = select_then_freeze(dataset, store)
    tr = run_node(make_execute_node(store), frozen, dataset)["test_result"]
    json.dumps(tr)


def test_execute_node_refuses_an_unfrozen_regression_plan(dataset, store):
    plan = make_plan(dataset, QuestionType.DRIVER_ANALYSIS, ("region", "tenure", "visits"))
    with pytest.raises(ValueError, match="not FROZEN"):
        run_node(make_execute_node(store), plan, dataset)