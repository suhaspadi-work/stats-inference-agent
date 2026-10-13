import numpy as np
import pandas as pd
import pytest
from langgraph.graph import StateGraph, START, END

from core.agent_state import AgentState
from core.capabilities import MULTI_GROUP, REGRESSION
from core.dataset import LocalDiskStore
from core.decision_log import log_update
from core.ingestion import load_dataset
from core.nodes import make_execute_node
from core.plan import AnalysisPlan, MethodDecision, QuestionType, CausalStatus, OutcomeType
from core.selection_records import multi_group_selection_record, two_group_selection_record
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


# --- run-time records and charts (M0.2 step 11) ------------------------------------

def _run_execute(store, plan, ds, decision_log=None):
    graph = StateGraph(AgentState)
    graph.add_node("execute", make_execute_node(store))
    graph.add_edge(START, "execute")
    graph.add_edge("execute", END)
    state = {"plan": plan, "dataset": ds, "request_text": "x", "session_id": "session-abc"}
    if decision_log is not None:
        state["decision_log"] = decision_log
    return graph.compile().invoke(state)


def _multi_group_dataset(tmp_path, store, name, means):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "region": [g for g in ("East", "North", "West") for _ in range(60)],
        "total_spend": np.concatenate([rng.normal(m, 5, 60) for m in means]),
    })
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    return load_dataset(path, session_id="session-abc", name=name, store=store)


def _frozen(ds, store, question_type, capability, predictors=("region",)):
    plan = AnalysisPlan(
        request_text="x", session_id="session-abc", question_type=question_type,
        causal_status=CausalStatus.OBSERVATIONAL, outcome_name="total_spend", outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="customer", data_sources=(ds.handle,), candidate_predictors=predictors,
    )
    decision = capability.select(ds, store, plan)
    return plan.record_method_decision(decision).freeze(), decision


@pytest.mark.parametrize("means, expect_post_hoc", [((50, 55, 65), True), ((50, 50, 50), False)])
def test_multi_group_execute_logs_one_post_hoc_record(tmp_path, store, means, expect_post_hoc):
    ds = _multi_group_dataset(tmp_path, store, f"mg_{expect_post_hoc}", means)
    plan, decision = _frozen(ds, store, QuestionType.MULTI_GROUP_COMPARISON, MULTI_GROUP)
    seed = log_update(multi_group_selection_record(decision))["decision_log"]
    result = _run_execute(store, plan, ds, decision_log=seed)

    new = result["decision_log"][len(seed):]
    assert [r["decision_point"] for r in new] == ["stage2.post_hoc"]
    tr = result["test_result"]
    assert new[0]["chosen"] == tr["post_hoc_method"]
    assert new[0]["evidence"]["selection_record_id"] == seed[0]["id"]
    if expect_post_hoc:
        assert tr["post_hoc_method"] in ("tukey_hsd", "dunn")
        assert len(tr["chart_data"]) == 1
        assert tr["chart_data"][0]["chart_type"] == "post_hoc_comparison_chart"
    else:
        assert tr["post_hoc_method"] == "none"
        assert tr["chart_data"] == []
    assert result["plan"].status.value == "executed"
    assert len(result["plan"].method_decisions) == 1


def test_two_group_execute_logs_no_records_and_has_no_chart_data(dataset, store):
    ds, _ = dataset
    plan = make_frozen_plan(ds, store)
    seed = log_update(two_group_selection_record(plan.latest_decision))["decision_log"]
    result = _run_execute(store, plan, ds, decision_log=seed)
    assert result["decision_log"] == seed
    assert "chart_data" not in result["test_result"]
    assert result["plan"].status.value == "executed"
    assert len(result["plan"].method_decisions) == 1


def test_regression_execute_logs_no_records_and_has_one_coefficient_plot(tmp_path, store):
    rng = np.random.default_rng(4)
    n = 200
    tenure = rng.normal(24, 8, n)
    df = pd.DataFrame({"tenure": tenure, "total_spend": 20 + 2.0 * tenure + rng.normal(0, 5, n)})
    path = tmp_path / "reg.csv"
    df.to_csv(path, index=False)
    ds = load_dataset(path, session_id="session-abc", name="reg", store=store)
    plan, _ = _frozen(ds, store, QuestionType.DRIVER_ANALYSIS, REGRESSION, predictors=("tenure",))

    result = _run_execute(store, plan, ds)
    assert "decision_log" not in result or result["decision_log"] == []
    charts = result["test_result"]["chart_data"]
    assert [c["chart_type"] for c in charts] == ["coefficient_plot"]
    assert result["plan"].status.value == "executed"
    assert len(result["plan"].method_decisions) == 1
