import copy
import json

import pytest

import core.routing as routing_module
from core.capabilities import (
    GROUPING_MAX_LEVELS, MULTI_GROUP, REGRESSION, TWO_GROUP, capability_for_question_type, is_grouping_column,
)
from core.model_safe_view import ColumnSummary, ModelSafeView
from core.plan import AnalysisPlan, CausalStatus, OutcomeType, QuestionType
from core.regression import MAX_CATEGORICAL_LEVELS
from core.routing import route_for_plan

KEYS = {"capability", "status", "question_type", "classifier_question_type", "reason", "message", "evidence"}


def make_plan(question_type, predictors, outcome="total_spend", outcome_type=OutcomeType.CONTINUOUS):
    return AnalysisPlan(
        request_text="x", session_id="session-abc",
        question_type=question_type, causal_status=CausalStatus.OBSERVATIONAL,
        outcome_name=outcome, outcome_type=outcome_type,
        unit_of_analysis="customer", data_sources=("x@v1",),
        candidate_predictors=tuple(predictors),
    )


def make_profile(columns):
    """columns: {name: (dtype, distinct_count)}"""
    return ModelSafeView(
        dataset_handle="x@v1", row_count=200, column_count=len(columns),
        columns=tuple(
            ColumnSummary(name=n, dtype=d, missing_count=0, missing_pct=0.0, distinct_count=k)
            for n, (d, k) in columns.items()
        ),
    )


PROFILE = make_profile({
    "region": ("object", 2), "tier": ("object", 3), "band": ("int64", 10), "store": ("object", 11),
    "tenure": ("float64", 80), "label": ("object", 2), "converted": ("int64", 2), "total_spend": ("float64", 150),
})


def _column(dtype, distinct):
    return ColumnSummary(name="c", dtype=dtype, missing_count=0, missing_pct=0.0, distinct_count=distinct)


def _route(question_type, predictors, **kwargs):
    result = route_for_plan(make_plan(question_type, predictors, **kwargs), PROFILE)
    json.dumps(result)
    assert set(result) == KEYS
    return result


# --- the grouping-column rule ------------------------------------------------------

@pytest.mark.parametrize("distinct, expected", [(1, False), (2, True), (10, True), (11, False)])
@pytest.mark.parametrize("dtype", ["object", "bool", "int64", "float64"])
def test_is_grouping_column_by_count_whatever_the_dtype(dtype, distinct, expected):
    assert is_grouping_column(_column(dtype, distinct)) is expected


def test_grouping_max_levels_matches_regression():
    assert GROUPING_MAX_LEVELS == MAX_CATEGORICAL_LEVELS == 10


# --- applicability -------------------------------------------------------------------

@pytest.mark.parametrize("analysis, question_type", [
    (TWO_GROUP, QuestionType.TWO_GROUP_COMPARISON), (MULTI_GROUP, QuestionType.MULTI_GROUP_COMPARISON),
])
@pytest.mark.parametrize("column, distinct", [("store", 11), ("tenure", 80)])
def test_too_many_levels_is_not_a_grouping_column(analysis, question_type, column, distinct):
    a = analysis.applicability(make_plan(question_type, [column]), PROFILE)
    assert a.eligible is False
    assert a.reason == f"'{column}' has {distinct} distinct values; a grouping column can have 2 to 10."
    assert a.evidence == {"group_column": column, "levels": distinct, "max_levels": 10}


def test_multi_group_with_exactly_ten_levels_is_eligible():
    a = MULTI_GROUP.applicability(make_plan(QuestionType.MULTI_GROUP_COMPARISON, ["band"]), PROFILE)
    assert a.eligible is True
    assert a.reason == "'band' has 10 levels and the outcome is numeric."


def test_two_and_three_levels_behave_as_before():
    two = TWO_GROUP.applicability(make_plan(QuestionType.TWO_GROUP_COMPARISON, ["region"]), PROFILE)
    assert (two.eligible, two.reason) == (True, "'region' has exactly 2 levels and the outcome is numeric.")
    assert two.evidence == {"group_column": "region", "levels": 2}
    three = MULTI_GROUP.applicability(make_plan(QuestionType.MULTI_GROUP_COMPARISON, ["tier"]), PROFILE)
    assert (three.eligible, three.reason) == (True, "'tier' has 3 levels and the outcome is numeric.")
    two_on_three = TWO_GROUP.applicability(make_plan(QuestionType.TWO_GROUP_COMPARISON, ["tier"]), PROFILE)
    assert two_on_three.reason == "'tier' has 3 levels; a two-group comparison needs exactly 2."


@pytest.mark.parametrize("outcome", ["converted", "label"])  # a numeric 0/1 column and a text column
@pytest.mark.parametrize("analysis", [TWO_GROUP, MULTI_GROUP, REGRESSION])
def test_binary_outcome_is_ineligible_for_every_analysis(analysis, outcome):
    plan = make_plan(QuestionType.TWO_GROUP_COMPARISON, ["region"], outcome=outcome,
                     outcome_type=OutcomeType.BINARY)
    a = analysis.applicability(plan, PROFILE)
    assert a.eligible is False
    assert a.reason == f"Outcome '{outcome}' is binary; binary outcomes are not supported yet (G3)."
    assert a.evidence == {"outcome": outcome, "outcome_type": "binary"}


# --- route_for_plan: ok and reconciled ----------------------------------------------

def test_two_group_question_with_two_levels_is_ok():
    r = _route(QuestionType.TWO_GROUP_COMPARISON, ["region"])
    assert (r["status"], r["capability"], r["question_type"], r["classifier_question_type"]) == \
        ("ok", "two_group", "two_group_comparison", "two_group_comparison")
    assert r["reason"] == ("question type 'two_group_comparison' routes to two_group, and the data supports it: "
                           "'region' has exactly 2 levels and the outcome is numeric.")
    assert r["message"] is None


def test_two_group_question_with_three_levels_is_reconciled_to_multi_group():
    r = _route(QuestionType.TWO_GROUP_COMPARISON, ["tier"])
    assert (r["status"], r["capability"], r["question_type"], r["classifier_question_type"]) == \
        ("reconciled", "multi_group", "multi_group_comparison", "two_group_comparison")
    assert r["reason"] == "the data supports multi_group ('tier' has 3 levels), not the two_group the question type routes to"
    assert r["message"] is None


def test_multi_group_question_with_two_levels_is_reconciled_to_two_group():
    r = _route(QuestionType.MULTI_GROUP_COMPARISON, ["region"])
    assert (r["status"], r["capability"], r["question_type"], r["classifier_question_type"]) == \
        ("reconciled", "two_group", "two_group_comparison", "multi_group_comparison")
    assert r["reason"] == "the data supports two_group ('region' has 2 levels), not the multi_group the question type routes to"


def test_ab_test_with_two_levels_stays_ab_test():
    r = _route(QuestionType.AB_TEST, ["region"])
    assert (r["status"], r["capability"], r["question_type"], r["classifier_question_type"]) == \
        ("ok", "two_group", "ab_test", "ab_test")
    assert r["reason"].startswith("question type 'ab_test' routes to two_group, and the data supports it:")


def test_ab_test_with_three_levels_is_reconciled_to_multi_group():
    r = _route(QuestionType.AB_TEST, ["tier"])
    assert (r["status"], r["capability"], r["question_type"], r["classifier_question_type"]) == \
        ("reconciled", "multi_group", "multi_group_comparison", "ab_test")


def test_driver_analysis_with_predictors_is_ok_regression():
    r = _route(QuestionType.DRIVER_ANALYSIS, ["region", "tenure"])
    assert (r["status"], r["capability"], r["question_type"], r["classifier_question_type"]) == \
        ("ok", "regression", "driver_analysis", "driver_analysis")
    assert r["reason"] == ("question type 'driver_analysis' routes to regression, and the data supports it: "
                           "Numeric outcome with 2 predictor(s).")
    assert r["message"] is None


def test_descriptive_is_ok_and_does_not_evaluate_analyses(monkeypatch):
    def must_not_be_called(*args, **kwargs):
        raise AssertionError("evaluate_all must not run for a descriptive plan")
    monkeypatch.setattr(routing_module, "evaluate_all", must_not_be_called)
    r = route_for_plan(make_plan(QuestionType.DESCRIPTIVE, [], outcome="(whole dataset)"), PROFILE)
    json.dumps(r)
    assert set(r) == KEYS
    assert (r["status"], r["capability"], r["question_type"], r["classifier_question_type"]) == \
        ("ok", "descriptive", "descriptive", "descriptive")
    assert r["reason"] == "question type 'descriptive' routes to the descriptive diagnostics"
    assert r["message"] is None
    assert r["evidence"] == {"applicability": {}, "grouping_max_levels": 10}


# --- route_for_plan: unsupported ----------------------------------------------------

def _assert_unsupported(r, question_type, reason, message):
    assert (r["status"], r["capability"]) == ("unsupported", None)
    assert r["question_type"] == r["classifier_question_type"] == question_type
    assert r["reason"] == reason
    assert r["message"] == message


def test_comparison_with_two_predictors_is_unsupported():
    r = _route(QuestionType.TWO_GROUP_COMPARISON, ["region", "tier"])
    _assert_unsupported(
        r, "two_group_comparison", "Needs exactly one grouping column; got 2.",
        "This comparison can't be run: Needs exactly one grouping column; got 2. A group comparison needs exactly "
        "one grouping column; for several predictors, ask what drives 'total_spend'.")


def test_comparison_with_a_continuous_predictor_is_unsupported():
    r = _route(QuestionType.MULTI_GROUP_COMPARISON, ["tenure"])
    _assert_unsupported(
        r, "multi_group_comparison", "'tenure' has 80 distinct values; a grouping column can have 2 to 10.",
        "This comparison can't be run: 'tenure' has 80 distinct values; a grouping column can have 2 to 10. "
        "Choose a column with 2 to 10 groups, or ask what drives 'total_spend' to use it as a predictor.")
    assert "80" in r["message"] and "10" in r["message"]


def test_comparison_with_no_predictor_is_unsupported():
    r = _route(QuestionType.TWO_GROUP_COMPARISON, [])
    _assert_unsupported(
        r, "two_group_comparison", "Needs exactly one grouping column; got 0.",
        "This comparison can't be run: Needs exactly one grouping column; got 0. A group comparison needs exactly "
        "one grouping column; for several predictors, ask what drives 'total_spend'.")


def test_driver_analysis_with_no_predictors_is_unsupported():
    r = _route(QuestionType.DRIVER_ANALYSIS, [])
    _assert_unsupported(r, "driver_analysis", "Needs at least one predictor column.",
                        "This analysis can't be run: Needs at least one predictor column. Name at least one predictor.")


def test_association_is_unsupported_even_with_a_valid_grouping_column():
    r = _route(QuestionType.ASSOCIATION, ["region"])
    _assert_unsupported(
        r, "association", "association has no analysis yet (G1)",
        "Association questions are not supported yet. To see which variables relate to 'total_spend', "
        "ask what drives 'total_spend'.")
    assert r["evidence"]["applicability"]["two_group"]["eligible"] is True  # the data would have allowed it


def test_binary_outcome_is_unsupported():
    r = _route(QuestionType.TWO_GROUP_COMPARISON, ["region"], outcome="converted", outcome_type=OutcomeType.BINARY)
    reason = "Outcome 'converted' is binary; binary outcomes are not supported yet (G3)."
    _assert_unsupported(r, "two_group_comparison", reason,
                        f"This comparison can't be run: {reason} Comparisons need a continuous numeric outcome; "
                        f"other outcomes are not supported yet.")


def test_text_outcome_is_unsupported():
    r = _route(QuestionType.TWO_GROUP_COMPARISON, ["region"], outcome="label")
    reason = "Outcome 'label' has data type 'object', which a numeric analysis cannot use."
    _assert_unsupported(r, "two_group_comparison", reason,
                        f"This comparison can't be run: {reason} Comparisons need a continuous numeric outcome; "
                        f"other outcomes are not supported yet.")


def test_missing_column_is_unsupported():
    r = _route(QuestionType.TWO_GROUP_COMPARISON, ["nope"])
    reason = "Columns not in the dataset: ['nope']."
    _assert_unsupported(r, "two_group_comparison", reason,
                        f"This comparison can't be run: {reason} Pick columns that exist in the dataset.")


def test_driver_analysis_with_the_outcome_as_a_predictor_is_unsupported():
    r = _route(QuestionType.DRIVER_ANALYSIS, ["total_spend", "tenure"])
    reason = "The outcome 'total_spend' cannot also be a predictor."
    _assert_unsupported(r, "driver_analysis", reason,
                        f"This analysis can't be run: {reason} Remove 'total_spend' from the predictors; "
                        f"the outcome can't also be a predictor.")


def test_driver_analysis_with_a_text_outcome_gets_the_regression_hint():
    r = _route(QuestionType.DRIVER_ANALYSIS, ["region"], outcome="label")
    assert r["status"] == "unsupported"
    assert r["message"].endswith("Regression needs a continuous numeric outcome; other outcomes are not supported yet.")


def test_driver_analysis_with_a_binary_outcome_gets_the_regression_hint():
    r = _route(QuestionType.DRIVER_ANALYSIS, ["region"], outcome="converted", outcome_type=OutcomeType.BINARY)
    assert r["status"] == "unsupported"
    assert "(G3)" in r["reason"]
    assert r["message"].endswith("Regression needs a continuous numeric outcome; other outcomes are not supported yet.")

def test_unknown_question_type_raises():
    plan = make_plan(QuestionType.TWO_GROUP_COMPARISON, ["region"])
    object.__setattr__(plan, "question_type", type("QT", (), {"value": "survival"})())
    with pytest.raises(ValueError, match="survival"):
        route_for_plan(plan, PROFILE)


# --- shape, evidence, no mutation -------------------------------------------------------

ALL_CASES = [
    (QuestionType.TWO_GROUP_COMPARISON, ["region"]), (QuestionType.TWO_GROUP_COMPARISON, ["tier"]),
    (QuestionType.MULTI_GROUP_COMPARISON, ["tenure"]), (QuestionType.AB_TEST, ["region"]),
    (QuestionType.DRIVER_ANALYSIS, ["region", "tenure"]), (QuestionType.DRIVER_ANALYSIS, []),
    (QuestionType.ASSOCIATION, ["region"]),
]


@pytest.mark.parametrize("question_type, predictors", ALL_CASES)
def test_evidence_lists_every_analysis_and_inputs_are_not_mutated(question_type, predictors):
    plan = make_plan(question_type, predictors)
    plan_before, profile_before = copy.deepcopy(plan), copy.deepcopy(PROFILE)
    r = route_for_plan(plan, PROFILE)
    json.dumps(r)
    assert set(r) == KEYS
    assert set(r["evidence"]["applicability"]) == {"two_group", "multi_group", "regression"}
    for entry in r["evidence"]["applicability"].values():
        assert set(entry) == {"eligible", "reason"} and type(entry["eligible"]) is bool
    assert r["evidence"]["grouping_max_levels"] == 10
    assert plan == plan_before and PROFILE == profile_before


# --- consistency guard -------------------------------------------------------------------

@pytest.mark.parametrize("question_type", [qt for qt in QuestionType if qt.value not in ("descriptive", "association")])
@pytest.mark.parametrize("predictors", [["region"], ["tier"], ["band"], ["store"], ["tenure"], ["region", "tier"], []])
def test_routed_capability_is_always_eligible(question_type, predictors):
    r = route_for_plan(make_plan(question_type, predictors), PROFILE)
    if r["status"] in ("ok", "reconciled"):
        assert r["evidence"]["applicability"][r["capability"]]["eligible"] is True
    else:
        assert r["capability"] is None and r["message"]


def test_association_still_maps_to_the_two_group_capability():
    assert capability_for_question_type("association") is capability_for_question_type("two_group_comparison")
