"""
M0.5 step 1: the chart-data boundary. Raw-value charts are for the UI only; no chart payload reaches a
prompt. report_node hands the generators copies without chart data, and assert_model_safe rejects any
chart payload or non-empty chart_data it finds.
"""
import copy
import re
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import core.classifier as classifier_module
import core.nodes as nodes_module
import core.visualization as viz
from core.capabilities import DESCRIPTIVE, MULTI_GROUP, REGRESSION, TWO_GROUP
from core.classifier import draft_plan_fields, propose_wrangling_operations
from core.dataset import LocalDiskStore
from core.decision_points import ALLOWED_RECORD_CHART_TYPES
from core.detection import CandidateIssue
from core.ingestion import load_dataset, profile_dataset
from core.model_boundary import plan_for_model, prepare_for_model, result_for_model
from core.model_safe_view import ColumnSummary, ModelSafeView, assert_model_safe
from core.nodes import make_report_node
from core.plan import AnalysisPlan, CausalStatus, MethodDecision, OutcomeType, QuestionType
from core.visualization import RAW_VALUE_CHART_TYPES, SUMMARY_CHART_TYPES, is_raw_value_chart

RAW = [101.25, 202.5, 303.75]  # distinctive values fed only to the raw-value builders


def _numbers(value):
    """Every number anywhere inside a chart's data."""
    if isinstance(value, dict):
        return [n for v in value.values() for n in _numbers(v)]
    if isinstance(value, (list, tuple)):
        return [n for v in value for n in _numbers(v)]
    return [value] if isinstance(value, (int, float)) and not isinstance(value, bool) else []


def _has_key(value, key):
    if isinstance(value, dict):
        return key in value or any(_has_key(v, key) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_key(v, key) for v in value)
    return False


# --- 1. classification ---------------------------------------------------------------

def test_every_chart_type_in_the_source_is_classified():
    source = Path(viz.__file__).read_text()
    literals = set(re.findall(r'"chart_type":\s*"([a-z_]+)"', source))
    assert len(RAW_VALUE_CHART_TYPES) == 9 and len(SUMMARY_CHART_TYPES) == 6
    assert set(RAW_VALUE_CHART_TYPES).isdisjoint(SUMMARY_CHART_TYPES)
    assert set(RAW_VALUE_CHART_TYPES) | set(SUMMARY_CHART_TYPES) == literals
    assert set(ALLOWED_RECORD_CHART_TYPES) <= set(SUMMARY_CHART_TYPES)


def test_is_raw_value_chart():
    assert is_raw_value_chart(viz.boxplot_by_group({"A": np.array(RAW)}, "y")) is True
    assert is_raw_value_chart(viz.post_hoc_comparison_chart(["A-B"], [1.0], [0.5], [1.5], [True])) is False


# --- 2. real charts from every builder ---------------------------------------------------

def _raw_charts():
    a = np.array(RAW)
    return {
        "histogram": viz.histogram(a, "y"),
        "boxplot_by_group": viz.boxplot_by_group({"A": a}, "y"),
        "boxplot_single_column": viz.boxplot_single_column(a, "y"),
        "qq_plot": viz.qq_plot(a, "A"),
        "scatter_with_fit": viz.scatter_with_fit(a, a, a[:2], a[:2], "x", "y"),
        "residual_plot": viz.residual_plot(a, a),
        "residual_qq_plot": viz.residual_qq_plot(a),
        "residuals_vs_leverage": viz.residuals_vs_leverage(a, a),
        "scatter_matrix": viz.scatter_matrix(["x", "y"], {("x", "y"): (list(RAW), list(RAW))}),
    }


def _summary_charts():
    return {
        "bar_chart": viz.bar_chart(["East", "West"], [3, 4], "region"),
        "coefficient_plot": viz.coefficient_plot(["x"], [1.5], [1.0], [2.0]),
        "correlation_heatmap": viz.correlation_heatmap(["x", "y"], np.array([[1.0, 0.4], [0.4, 1.0]])),
        "missingness_chart": viz.missingness_chart(["x"], [0.1]),
        "pairwise_categorical_chart": viz.pairwise_categorical_chart("a", "b", ["p", "q"], ["r", "s"], [[1, 2], [3, 4]]),
        "post_hoc_comparison_chart": viz.post_hoc_comparison_chart(["A-B"], [1.0], [0.5], [1.5], [True]),
    }


def test_every_builder_returns_a_classified_chart_with_its_own_type():
    raw, summary = _raw_charts(), _summary_charts()
    assert len(raw) + len(summary) == 15
    for name, chart in {**raw, **summary}.items():
        assert chart["chart_type"] == name
        assert (name in RAW_VALUE_CHART_TYPES) != (name in SUMMARY_CHART_TYPES)


@pytest.mark.parametrize("name", RAW_VALUE_CHART_TYPES)
def test_raw_value_charts_hold_the_input_values(name):
    numbers = _numbers(_raw_charts()[name]["data"])
    assert all(any(abs(n - r) < 1e-9 for n in numbers) for r in RAW), name


@pytest.mark.parametrize("name", SUMMARY_CHART_TYPES)
def test_summary_charts_hold_none_of_the_raw_values(name):
    numbers = _numbers(_summary_charts()[name]["data"])
    assert not any(abs(n - r) < 1e-9 for n in numbers for r in RAW), name


# --- 3. assert_model_safe -------------------------------------------------------------------

BOX = viz.boxplot_by_group({"A": np.array(RAW)}, "y")
POST_HOC = viz.post_hoc_comparison_chart(["A-B"], [1.0], [0.5], [1.5], [True])


def _plan(decision_charts=()):
    plan = AnalysisPlan(
        request_text="x", session_id="s", question_type=QuestionType.TWO_GROUP_COMPARISON,
        causal_status=CausalStatus.OBSERVATIONAL, outcome_name="total_spend", outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="customer", data_sources=("d@v1",), candidate_predictors=("region",),
    )
    return plan.record_method_decision(MethodDecision(stage="test_selection", chosen_method="welch_t_test",
                                                      rationale="r", alternatives=(), chart_data=tuple(decision_charts)))


@pytest.mark.parametrize("chart", [BOX, POST_HOC], ids=["raw", "summary"])
@pytest.mark.parametrize("wrap, path", [
    (lambda c: [c], r"items\[0\]"),
    (lambda c: {"inner": {"chart": c}}, r"items\.inner\.chart"),
])
def test_chart_payloads_are_rejected_with_their_path(chart, wrap, path):
    with pytest.raises(ValueError, match=path):
        assert_model_safe(wrap(chart), "items")


def test_chart_data_on_a_plan_is_rejected_with_its_path():
    with pytest.raises(ValueError, match=r"plan\.method_decisions\[0\]\.chart_data"):
        assert_model_safe(_plan([BOX]), "plan")


def test_non_empty_chart_data_key_is_rejected():
    with pytest.raises(ValueError, match=r"result\.chart_data"):
        assert_model_safe({"chart_data": [{"chart_type": "x"}]}, "result")


@pytest.mark.parametrize("value", [{}, {"chart_data": ()}, {"chart_data": []}, {"chart_data": None}])
def test_empty_chart_data_passes(value):
    assert_model_safe({"p_value": 0.01, **value})


def test_normal_objects_pass():
    assert_model_safe({"result_type": "two_group", "method": "welch_t_test", "p_value": 0.01,
                       "confidence_interval": (1.0, 2.0), "group_a_label": "East"})
    assert_model_safe(ModelSafeView(dataset_handle="d@v1", row_count=4, column_count=1, columns=(
        ColumnSummary(name="region", dtype="object", missing_count=0, missing_pct=0.0, distinct_count=2,
                      top_values=(("East", 2), ("West", 2))),)))
    assert_model_safe([CandidateIssue(issue_type="duplicate_rows", description="d", affected_columns=("id",),
                                      evidence={"duplicate_row_count": 1})])
    assert_model_safe(_plan())


@pytest.mark.parametrize("raw", [pd.DataFrame({"x": [1]}), pd.Series([1]), np.array([1.0])])
def test_raw_data_still_raises_type_error(raw):
    with pytest.raises(TypeError, match="Refusing to pass"):
        assert_model_safe(raw)
    with pytest.raises(TypeError, match=r"at result\.rows"):
        assert_model_safe({"rows": raw}, "result")


def test_a_self_referencing_dict_does_not_loop():
    loop = {"a": 1}
    loop["self"] = loop
    assert_model_safe(loop)


# --- 4. model_boundary on real objects --------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _dataset(df, tmp_path, store, name):
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    return load_dataset(path, session_id="s", name=name, store=store)


def _real_plan(ds, question_type, predictors):
    return AnalysisPlan(request_text="x", session_id="s", question_type=question_type,
                        causal_status=CausalStatus.OBSERVATIONAL, outcome_name="total_spend",
                        outcome_type=OutcomeType.CONTINUOUS, unit_of_analysis="row",
                        data_sources=(ds.handle,), candidate_predictors=tuple(predictors))


@pytest.fixture
def real(tmp_path, store):
    rng = np.random.default_rng(1)
    n = 60
    mg = pd.DataFrame({"region": [g for g in ("East", "North", "West") for _ in range(n)],
                       "tenure": rng.normal(24, 8, 3 * n),
                       "total_spend": np.concatenate([rng.normal(m, 5, n) for m in (50, 55, 65)])})
    ds3 = _dataset(mg, tmp_path, store, "mg")
    ds2 = _dataset(mg[mg.region != "North"].reset_index(drop=True), tmp_path, store, "tg")

    mg_plan = _real_plan(ds3, QuestionType.MULTI_GROUP_COMPARISON, ["region"])
    mg_plan = mg_plan.record_method_decision(MULTI_GROUP.select(ds3, store, mg_plan)).freeze()
    reg_plan = _real_plan(ds3, QuestionType.DRIVER_ANALYSIS, ["tenure", "region"])
    reg_plan = reg_plan.record_method_decision(REGRESSION.select(ds3, store, reg_plan)).freeze()
    tg_plan = _real_plan(ds2, QuestionType.TWO_GROUP_COMPARISON, ["region"])
    tg_plan = tg_plan.record_method_decision(TWO_GROUP.select(ds2, store, tg_plan)).freeze()
    return SimpleNamespace(
        multi_group=MULTI_GROUP.execute(ds3, store, mg_plan),
        regression=REGRESSION.execute(ds3, store, reg_plan),
        descriptive=DESCRIPTIVE.compute(ds3, store),
        two_group_plan=tg_plan, two_group_result=TWO_GROUP.execute(ds2, store, tg_plan),
        multi_group_plan=mg_plan,
    )


def test_real_objects_carry_charts(real):
    assert [c["chart_type"] for c in real.multi_group["chart_data"]] == ["post_hoc_comparison_chart"]
    assert [c["chart_type"] for c in real.regression["chart_data"]] == ["coefficient_plot"]
    assert any(is_raw_value_chart(c) for c in real.descriptive["chart_data"])
    decision_types = {c["chart_type"] for c in real.two_group_plan.method_decisions[0].chart_data}
    assert {"boxplot_by_group", "qq_plot"} <= decision_types


@pytest.mark.parametrize("name", ["multi_group", "regression", "descriptive"])
def test_result_for_model_removes_chart_data_without_mutating(real, name):
    result = getattr(real, name)
    before = copy.deepcopy(result)
    safe = result_for_model(result)
    assert not _has_key(safe, "chart_data")
    assert repr(result) == repr(before)  # repr, because NaN != NaN in the descriptive result
    assert result["chart_data"]
    with pytest.raises(ValueError):
        assert_model_safe(result, "result")
    assert_model_safe(safe, "result")


def test_result_for_model_removes_chart_data_at_depth_three():
    nested = {"a": {"b": {"chart_data": [BOX], "keep": 1}}, "chart_data": [POST_HOC]}
    safe = result_for_model(nested)
    assert safe == {"a": {"b": {"keep": 1}}}
    assert nested["a"]["b"]["chart_data"] == [BOX]


def test_plan_for_model_empties_charts_and_keeps_everything_else(real):
    plan = real.two_group_plan
    charts_before = plan.method_decisions[0].chart_data
    safe = plan_for_model(plan)
    assert all(d.chart_data == () for d in safe.method_decisions)
    assert plan.method_decisions[0].chart_data == charts_before and charts_before

    def without_charts(p):
        d = asdict(p)
        for md in d["method_decisions"]:
            md["chart_data"] = ()
        return d
    assert without_charts(safe) == without_charts(plan)


def test_prepare_for_model_output_is_safe_where_the_inputs_are_not(real):
    with pytest.raises(ValueError):
        assert_model_safe(real.two_group_plan, "plan")
    with pytest.raises(ValueError):
        assert_model_safe(real.descriptive, "result")
    safe_plan, safe_result = prepare_for_model(real.two_group_plan, real.descriptive)
    assert_model_safe(safe_plan, "plan")
    assert_model_safe(safe_result, "result")


# --- 5. report_node hands the generators chart-free copies ------------------------------------

class _Recorder:
    def __init__(self, text):
        self.text, self.calls = text, []

    def __call__(self, plan, result):
        self.calls.append((plan, result))
        return self.text


def _assert_received_safe(recorder):
    (plan, result), = recorder.calls
    assert all(d.chart_data == () for d in plan.method_decisions)
    assert not _has_key(result, "chart_data")
    assert_model_safe(plan, "plan")
    assert_model_safe(result, "result")


def _state(plan, **extra):
    return {"plan": plan, "request_text": "x", "session_id": "s", **extra}


def test_report_node_two_group_path(real):
    stub = _Recorder("two-group report")
    result = {**real.two_group_result, "chart_data": [BOX]}  # make sure the result carries a chart too
    update = make_report_node(report_fn=stub)(_state(real.two_group_plan, test_result=result))
    assert update["report"] == "two-group report"
    _assert_received_safe(stub)
    assert real.two_group_plan.method_decisions[0].chart_data and result["chart_data"] == [BOX]


def test_report_node_non_two_group_path(real, monkeypatch):
    stub = _Recorder("multi-group report")
    monkeypatch.setattr(nodes_module, "analysis_for_result",
                        lambda result: SimpleNamespace(result_type="multi_group", report=stub))
    state = _state(real.multi_group_plan, test_result=real.multi_group)
    update = make_report_node(report_fn=lambda p, r: "must not be used")(state)
    assert update["report"] == "multi-group report"
    _assert_received_safe(stub)
    assert real.multi_group["chart_data"] and real.multi_group_plan.method_decisions[0].chart_data


def test_report_node_diagnostic_path(real, monkeypatch):
    stub = _Recorder("descriptive report")
    monkeypatch.setattr(nodes_module, "_capability_for_state",
                        lambda state: SimpleNamespace(kind="diagnostic", result_key="descriptive_result", report=stub,
                                                      name="descriptive"))  # M1.3b: the round entry reads the name
    update = make_report_node()(_state(real.two_group_plan, descriptive_result=real.descriptive))
    assert update["report"] == "descriptive report"
    _assert_received_safe(stub)
    assert real.descriptive["chart_data"]


# --- 6. classifier tripwire ----------------------------------------------------------------------

@pytest.fixture
def no_model(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("model must not be created")
    monkeypatch.setattr(classifier_module, "init_chat_model", refuse)


def test_draft_plan_fields_rejects_a_chart_before_the_model(no_model):
    with pytest.raises(ValueError, match="chart payload"):
        draft_plan_fields("x", {"chart_type": "histogram", "data": {"values": [1.0]}})


def test_propose_wrangling_operations_rejects_a_chart_before_the_model(no_model):
    with pytest.raises(ValueError, match="chart payload"):
        propose_wrangling_operations([{"chart_type": "bar_chart", "data": {}}])


def test_real_inputs_pass_the_tripwire_and_reach_the_model(no_model, tmp_path, store):
    ds = _dataset(pd.DataFrame({"region": ["East", "West", "East"], "total_spend": [1.0, 2.0, 3.0]}),
                  tmp_path, store, "small")
    with pytest.raises(AssertionError, match="model must not be created"):
        draft_plan_fields("x", profile_dataset(ds, store=store))
    issues = [CandidateIssue(issue_type="duplicate_rows", description="d", affected_columns=("id",),
                             evidence={"duplicate_row_count": 1, "total_rows": 3})]
    with pytest.raises(AssertionError, match="model must not be created"):
        propose_wrangling_operations(issues)
