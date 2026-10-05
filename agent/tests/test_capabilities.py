import json
import numpy as np
import pandas as pd
import pytest

from core.capabilities import (
    TWO_GROUP, MULTI_GROUP, REGRESSION, DESCRIPTIVE, ANALYSES,
    capability_for_plan, analysis_for_result, evaluate_all, is_numeric_dtype_name,
)
from core.classifier import _CONTINUOUS_COMPATIBLE_DTYPES
from core.dataset import LocalDiskStore
from core.ingestion import load_dataset
from core.model_safe_view import ColumnSummary, ModelSafeView
from core.plan import AnalysisPlan, QuestionType, CausalStatus, OutcomeType
from core.testing import select_two_group_method, execute_two_group_test
from core.multi_group import select_multi_group_method, execute_multi_group_test
from core.regression import select_regression_method, execute_regression
from core.descriptive import compute_descriptive_stats


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def make_dataset(df, tmp_path, store, name="data"):
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    return load_dataset(path, session_id="session-abc", name=name, store=store)


def make_plan(question_type, predictors, outcome="total_spend"):
    return AnalysisPlan(
        request_text="x", session_id="session-abc",
        question_type=question_type, causal_status=CausalStatus.OBSERVATIONAL,
        outcome_name=outcome, outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="customer", data_sources=("x@v1",),
        candidate_predictors=tuple(predictors),
    )


def make_profile(columns):
    """columns: {name: (dtype, distinct_count)}"""
    return ModelSafeView(
        dataset_handle="x@v1", row_count=100, column_count=len(columns),
        columns=[
            ColumnSummary(
                name=n, dtype=d, missing_count=0, missing_pct=0.0,
                distinct_count=k, top_values=[], detected_format=None,
            )
            for n, (d, k) in columns.items()
        ],
    )


def two_group_df():
    rng = np.random.default_rng(5)
    return pd.DataFrame({
        "region": ["East"] * 100 + ["West"] * 100,
        "total_spend": np.concatenate([rng.normal(50, 10, 100), rng.normal(58, 10, 100)]),
    })


def multi_group_df():
    rng = np.random.default_rng(6)
    return pd.DataFrame({
        "region": ["East"] * 100 + ["West"] * 100 + ["North"] * 100,
        "total_spend": np.concatenate([rng.normal(50, 10, 100), rng.normal(50, 10, 100), rng.normal(70, 10, 100)]),
    })


def regression_df():
    rng = np.random.default_rng(7)
    n = 300
    region = rng.choice(["East", "West", "North"], n, p=[0.5, 0.3, 0.2])
    tenure = rng.normal(24, 8, n)
    shift = {"East": 0.0, "West": 8.0, "North": 15.0}
    spend = 20 + 2.0 * tenure + np.array([shift[r] for r in region]) + rng.normal(0, 5, n)
    return pd.DataFrame({"region": region, "tenure": tenure, "total_spend": spend})


# ------------------------------------------------------- routing and structure

def test_every_question_type_maps_to_the_expected_capability():
    expected = {
        QuestionType.DESCRIPTIVE: DESCRIPTIVE,
        QuestionType.TWO_GROUP_COMPARISON: TWO_GROUP,
        QuestionType.AB_TEST: TWO_GROUP,
        QuestionType.MULTI_GROUP_COMPARISON: MULTI_GROUP,
        QuestionType.DRIVER_ANALYSIS: REGRESSION,
    }
    for qt, cap in expected.items():
        assert capability_for_plan(make_plan(qt, ["region"])) is cap
    for qt in QuestionType:
        capability_for_plan(make_plan(qt, ["region"]))  # none may raise


def test_association_still_routes_to_two_group_known_gap_G1():
    """Documents gap G1. When M3.3 adds an association capability, this test must change."""
    assert capability_for_plan(make_plan(QuestionType.ASSOCIATION, ["tenure"])) is TWO_GROUP


def test_stage_kind_and_guard_attributes():
    assert (DESCRIPTIVE.kind, DESCRIPTIVE.stage, DESCRIPTIVE.guard_causal_language) == ("diagnostic", 1, False)
    assert (TWO_GROUP.stage, MULTI_GROUP.stage, REGRESSION.stage) == (2, 2, 3)
    for a in ANALYSES:
        assert a.kind == "analysis" and a.guard_causal_language


def test_result_dispatch_uses_the_result_tag_and_defaults_to_two_group():
    assert analysis_for_result({"result_type": "regression"}) is REGRESSION
    assert analysis_for_result({"result_type": "multi_group"}) is MULTI_GROUP
    assert analysis_for_result({"p_value": 0.5}) is TWO_GROUP
    assert analysis_for_result(None) is TWO_GROUP


def test_predictor_validation_keeps_the_existing_messages():
    plan = make_plan(QuestionType.TWO_GROUP_COMPARISON, ["region", "tenure"])
    with pytest.raises(ValueError, match="Method selection requires exactly one grouping predictor"):
        TWO_GROUP.validate_predictors(plan, "Method selection")
    with pytest.raises(ValueError, match="Execution requires exactly one grouping predictor"):
        MULTI_GROUP.validate_predictors(plan, "Execution")
    REGRESSION.validate_predictors(plan, "Method selection")  # many predictors are fine


def test_numeric_dtype_rule_agrees_with_the_classifier_constant():
    for dtype in _CONTINUOUS_COMPATIBLE_DTYPES:
        assert is_numeric_dtype_name(dtype)
    for dtype in ("str", "object", "bool"):
        assert not is_numeric_dtype_name(dtype)


# ------------------------------------------------------------- applicability

def test_two_group_applicability():
    profile = make_profile({"region": ("str", 2), "total_spend": ("float64", 100)})
    ok = TWO_GROUP.applicability(make_plan(QuestionType.TWO_GROUP_COMPARISON, ["region"]), profile)
    assert ok.eligible and ok.evidence["levels"] == 2

    three = make_profile({"region": ("str", 3), "total_spend": ("float64", 100)})
    bad = TWO_GROUP.applicability(make_plan(QuestionType.TWO_GROUP_COMPARISON, ["region"]), three)
    assert not bad.eligible and "exactly 2" in bad.reason and bad.evidence["levels"] == 3

    two_preds = TWO_GROUP.applicability(make_plan(QuestionType.TWO_GROUP_COMPARISON, ["region", "tenure"]),
                                        make_profile({"region": ("str", 2), "tenure": ("float64", 90), "total_spend": ("float64", 100)}))
    assert not two_preds.eligible and "exactly one grouping column" in two_preds.reason


def test_multi_group_applicability():
    plan = make_plan(QuestionType.MULTI_GROUP_COMPARISON, ["region"])
    ok = MULTI_GROUP.applicability(plan, make_profile({"region": ("str", 3), "total_spend": ("float64", 100)}))
    assert ok.eligible and ok.evidence["levels"] == 3
    bad = MULTI_GROUP.applicability(plan, make_profile({"region": ("str", 2), "total_spend": ("float64", 100)}))
    assert not bad.eligible and "3 or more" in bad.reason


def test_regression_applicability():
    profile = make_profile({"region": ("str", 3), "tenure": ("float64", 90), "total_spend": ("float64", 100)})
    ok = REGRESSION.applicability(make_plan(QuestionType.DRIVER_ANALYSIS, ["region", "tenure"]), profile)
    assert ok.eligible and ok.evidence["predictor_count"] == 2

    none = REGRESSION.applicability(make_plan(QuestionType.DRIVER_ANALYSIS, []), profile)
    assert not none.eligible and "at least one predictor" in none.reason

    self_ref = REGRESSION.applicability(make_plan(QuestionType.DRIVER_ANALYSIS, ["total_spend"]), profile)
    assert not self_ref.eligible and "cannot also be a predictor" in self_ref.reason


def test_applicability_reports_missing_columns_and_non_numeric_outcome():
    profile = make_profile({"region": ("str", 2), "label": ("str", 2), "total_spend": ("float64", 100)})
    missing = TWO_GROUP.applicability(make_plan(QuestionType.TWO_GROUP_COMPARISON, ["nope"]), profile)
    assert not missing.eligible and missing.evidence["missing_columns"] == ["nope"]

    text_outcome = REGRESSION.applicability(make_plan(QuestionType.DRIVER_ANALYSIS, ["region"], outcome="label"), profile)
    assert not text_outcome.eligible and "data type 'str'" in text_outcome.reason


def test_evaluate_all_returns_every_analysis_with_consistent_eligibility():
    profile = make_profile({"region": ("str", 3), "total_spend": ("float64", 100)})
    results = {a.name: app for a, app in evaluate_all(make_plan(QuestionType.DRIVER_ANALYSIS, ["region"]), profile)}
    assert set(results) == {"two_group", "multi_group", "regression"}
    assert not results["two_group"].eligible
    assert results["multi_group"].eligible
    assert results["regression"].eligible


# ---------------------------------------------- equivalence with direct calls

def test_execute_refuses_a_plan_with_no_selected_method(tmp_path, store):
    ds = make_dataset(two_group_df(), tmp_path, store)
    plan = make_plan(QuestionType.TWO_GROUP_COMPARISON, ["region"])
    with pytest.raises(ValueError, match="no method was selected"):
        TWO_GROUP.execute(ds, store, plan)


def test_two_group_matches_the_direct_functions(tmp_path, store):
    ds = make_dataset(two_group_df(), tmp_path, store)
    plan = make_plan(QuestionType.TWO_GROUP_COMPARISON, ["region"])
    decision = TWO_GROUP.select(ds, store, plan)
    direct = select_two_group_method(ds, store, "total_spend", "region").decision
    assert decision.chosen_method == direct.chosen_method
    assert decision.evidence == direct.evidence

    result = TWO_GROUP.execute(ds, store, plan.record_method_decision(decision))
    expected = execute_two_group_test(ds, store, "total_spend", "region", decision.chosen_method)
    assert result["result_type"] == "two_group"
    assert result["p_value"] == expected.p_value
    assert result["group_a_n"] == expected.group_a_n
    json.dumps(result)


def test_multi_group_matches_the_direct_functions(tmp_path, store):
    ds = make_dataset(multi_group_df(), tmp_path, store)
    plan = make_plan(QuestionType.MULTI_GROUP_COMPARISON, ["region"])
    decision = MULTI_GROUP.select(ds, store, plan)
    direct = select_multi_group_method(ds, store, "total_spend", "region").decision
    assert decision.chosen_method == direct.chosen_method

    result = MULTI_GROUP.execute(ds, store, plan.record_method_decision(decision))
    expected = execute_multi_group_test(ds, store, "total_spend", "region", decision.chosen_method)
    assert result["result_type"] == "multi_group"
    assert result["p_value"] == expected.p_value
    assert len(result["pairwise_comparisons"]) == len(expected.pairwise_comparisons)


def test_regression_matches_the_direct_functions_and_reuses_the_approved_encodings(tmp_path, store):
    ds = make_dataset(regression_df(), tmp_path, store)
    plan = make_plan(QuestionType.DRIVER_ANALYSIS, ["region", "tenure"])
    decision = REGRESSION.select(ds, store, plan)
    choice = select_regression_method(ds, store, "total_spend", ["region", "tenure"])
    assert decision.chosen_method == choice.method

    result = REGRESSION.execute(ds, store, plan.record_method_decision(decision))
    expected = execute_regression(ds, store, "total_spend", ["region", "tenure"], choice.method, choice.encodings)
    assert result["result_type"] == "regression"
    assert result["r_squared"] == expected.r_squared
    assert result["encodings"]["region"]["reference"] == "East"
    assert {c["name"] for c in result["coefficients"]} == {c.name for c in expected.coefficients}
    json.dumps(result)


def test_descriptive_matches_the_direct_function(tmp_path, store):
    df = two_group_df()
    ds = make_dataset(df, tmp_path, store)
    result = DESCRIPTIVE.compute(ds, store)
    expected = compute_descriptive_stats(ds, store)
    assert result["row_count"] == expected.row_count == len(df)
    assert [c["name"] for c in result["column_stats"]] == [c.name for c in expected.column_stats]
    json.dumps(result)