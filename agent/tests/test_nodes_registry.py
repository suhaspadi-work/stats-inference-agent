import numpy as np
import pandas as pd
import pytest
from langgraph.graph import StateGraph, START, END
from types import SimpleNamespace

import core.nodes as nodes_module
from core.agent_state import AgentState
from core.dataset import LocalDiskStore
from core.ingestion import load_dataset
from core.nodes import make_select_method_node, make_execute_node, make_eda_node, make_report_node
from core.plan import AnalysisPlan, MethodDecision, QuestionType, CausalStatus, OutcomeType


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


@pytest.fixture
def dataset(tmp_path, store):
    rng = np.random.default_rng(21)
    df = pd.DataFrame({
        "region": ["East"] * 60 + ["West"] * 60 + ["North"] * 60,
        "total_spend": np.concatenate([rng.normal(50, 10, 60), rng.normal(50, 10, 60), rng.normal(70, 10, 60)]),
    })
    path = tmp_path / "customers.csv"
    df.to_csv(path, index=False)
    return load_dataset(path, session_id="session-abc", name="customers", store=store)


def make_plan(dataset, question_type, predictors=("region",)):
    return AnalysisPlan(
        request_text="x", session_id="session-abc",
        question_type=question_type, causal_status=CausalStatus.OBSERVATIONAL,
        outcome_name="total_spend", outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="customer", data_sources=(dataset.handle,),
        candidate_predictors=tuple(predictors),
    )


def run_node(node, plan, dataset, **extra):
    graph = StateGraph(AgentState)
    graph.add_node("n", node)
    graph.add_edge(START, "n")
    graph.add_edge("n", END)
    state = {"plan": plan, "dataset": dataset, "request_text": "x", "session_id": "session-abc"}
    state.update(extra)
    return graph.compile().invoke(state)


def placeholder_decision():
    return MethodDecision(stage="test_selection", chosen_method="placeholder", rationale="x", alternatives=())


# ---------------------------------------------------------- select and execute

def test_select_node_chooses_a_multi_group_method(dataset, store):
    plan = make_plan(dataset, QuestionType.MULTI_GROUP_COMPARISON)
    result = run_node(make_select_method_node(store), plan, dataset)
    assert result["plan"].primary_method in ("one_way_anova", "kruskal_wallis")
    assert len(result["plan"].method_decisions) == 1


def test_execute_node_runs_multi_group_and_tags_the_result(dataset, store):
    plan = make_plan(dataset, QuestionType.MULTI_GROUP_COMPARISON)
    frozen = run_node(make_select_method_node(store), plan, dataset)["plan"].freeze()
    result = run_node(make_execute_node(store), frozen, dataset)
    assert result["plan"].status.value == "executed"
    assert result["test_result"]["result_type"] == "multi_group"
    assert result["test_result"]["p_value"] < 0.05  # North is planted 20 units higher
    assert len(result["plan"].method_decisions) == 1


def test_select_node_treats_a_descriptive_plan_as_a_routing_bug(dataset, store):
    plan = make_plan(dataset, QuestionType.DESCRIPTIVE, predictors=())
    with pytest.raises(ValueError, match="routing bug"):
        run_node(make_select_method_node(store), plan, dataset)


def test_execute_node_treats_a_frozen_descriptive_plan_as_a_routing_bug(dataset, store):
    plan = make_plan(dataset, QuestionType.DESCRIPTIVE, predictors=()).record_method_decision(placeholder_decision()).freeze()
    with pytest.raises(ValueError, match="routing bug"):
        run_node(make_execute_node(store), plan, dataset)


# ------------------------------------------------------------------------- eda

def test_eda_node_runs_descriptive_stats_and_stores_the_result(dataset, store):
    plan = make_plan(dataset, QuestionType.DESCRIPTIVE, predictors=())
    result = run_node(make_eda_node(store), plan, dataset)
    assert result["plan"].status.value == "executed"
    assert result["descriptive_result"]["row_count"] == 180
    assert "test_result" not in result


def test_eda_node_rejects_non_descriptive_plans(dataset, store):
    plan = make_plan(dataset, QuestionType.MULTI_GROUP_COMPARISON)
    with pytest.raises(ValueError, match="non-descriptive"):
        run_node(make_eda_node(store), plan, dataset)


# ---------------------------------------------------------------------- report

def test_report_node_uses_the_injected_stub_for_two_group_results(dataset):
    plan = make_plan(dataset, QuestionType.TWO_GROUP_COMPARISON)
    node = make_report_node(report_fn=lambda p, r: "two-group stub")
    result = run_node(node, plan, dataset, test_result={"p_value": 0.01})
    assert result["report"] == "two-group stub"


def test_report_node_dispatches_to_the_report_matching_the_result_tag(dataset, monkeypatch):
    fake = SimpleNamespace(result_type="multi_group", report=lambda p, r: "multi-group report")
    monkeypatch.setattr(nodes_module, "analysis_for_result", lambda result: fake)
    plan = make_plan(dataset, QuestionType.MULTI_GROUP_COMPARISON)
    node = make_report_node(report_fn=lambda p, r: "should not be used")
    result = run_node(node, plan, dataset, test_result={"result_type": "multi_group"})
    assert result["report"] == "multi-group report"


def test_report_node_reads_a_diagnostic_result_from_its_declared_key(dataset, monkeypatch):
    fake = SimpleNamespace(kind="diagnostic", result_key="descriptive_result", report=lambda p, r: f"described {r['row_count']} rows")
    monkeypatch.setattr(nodes_module, "capability_for_plan", lambda plan: fake)
    plan = make_plan(dataset, QuestionType.DESCRIPTIVE, predictors=())
    result = run_node(make_report_node(), plan, dataset, descriptive_result={"row_count": 180})
    assert result["report"] == "described 180 rows"