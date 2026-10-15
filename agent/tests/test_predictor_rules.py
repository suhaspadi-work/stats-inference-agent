"""
M1.3b step 2b: the predictor rules. One check (check_predictor) is used by the agent's proposal and by every
choice the user makes at the boundary gate; validate_predictor_choice collects every problem of a choice at once.
"""
import copy
import json

import pytest

from core.model_safe_view import ColumnSummary, ModelSafeView
from core.predictor_rules import check_predictor, validate_predictor_choice
from core.round_two import MAX_OTHER_PREDICTORS, propose_exploratory_predictors
from tests.test_routing_records import make_profile


def _profile(*columns, row_count=100):
    """columns: (name, dtype, distinct_count, missing_count)"""
    return ModelSafeView(dataset_handle="x@v1", row_count=row_count, column_count=len(columns), columns=tuple(
        ColumnSummary(name=n, dtype=d, missing_count=m, missing_pct=m / row_count, distinct_count=k)
        for n, d, k, m in columns))


PROFILE = _profile(("region", "object", 2, 0), ("total_spend", "float64", 100, 0), ("tenure", "float64", 80, 0),
                   ("tier", "object", 3, 0), ("customer_id", "int64", 100, 0), ("flag", "float64", 1, 0),
                   ("city", "object", 11, 0), ("state", "object", 10, 0), ("score", "float64", 100, 0))


# --- check_predictor -------------------------------------------------------------------------------------

def test_unknown_column():
    v = check_predictor(PROFILE, "colour", "total_spend")
    names = sorted(c.name for c in PROFILE.columns)
    assert v == {"column": "colour", "rule": "unknown_column", "reason": "not a column in the dataset",
                 "explanation": f"'colour' is not a column in this dataset. Available columns: {names}."}


def test_outcome():
    assert check_predictor(PROFILE, "total_spend", "total_spend") == {
        "column": "total_spend", "rule": "outcome", "reason": "the outcome cannot also be a predictor",
        "explanation": "it is the outcome being explained, so it cannot also be used to explain itself"}


def test_constant():
    assert check_predictor(PROFILE, "flag", "total_spend") == {
        "column": "flag", "rule": "constant", "reason": "constant or empty",
        "explanation": "a column with a single value (or no values) cannot explain any difference in the outcome"}


@pytest.mark.parametrize("name, dtype, distinct, missing", [("customer_id", "int64", 100, 0),
                                                            ("code", "object", 100, 0),
                                                            ("ref", "int64", 96, 4)],
                         ids=["int64", "object", "int64_with_missing"])
def test_identifier_like(name, dtype, distinct, missing):
    profile = _profile(("region", "object", 2, 0), ("total_spend", "float64", 100, 0), (name, dtype, distinct, missing))
    assert check_predictor(profile, name, "total_spend") == {
        "column": name, "rule": "identifier_like", "reason": "identifier-like: every value is distinct",
        "explanation": "every row has its own value in this column, so a model would fit each row exactly and tell "
                       "you nothing that carries over to other data"}


def test_an_all_distinct_float_column_fits():
    assert check_predictor(PROFILE, "score", "total_spend") is None


def test_too_many_levels():
    assert check_predictor(PROFILE, "city", "total_spend") == {
        "column": "city", "rule": "too_many_levels", "reason": "text column with more than 10 distinct values",
        "explanation": "each of its 11 levels becomes its own coefficient; with more than 10 levels the model "
                       "becomes hard to read and unstable"}
    assert check_predictor(PROFILE, "state", "total_spend") is None


def test_rule_order_outcome_before_constant():
    profile = _profile(("region", "object", 2, 0), ("y", "float64", 1, 0))
    assert check_predictor(profile, "y", "y")["rule"] == "outcome"


def test_fitting_columns_and_no_mutation():
    before = copy.deepcopy(PROFILE)
    assert [check_predictor(PROFILE, n, "total_spend") for n in ("tenure", "tier", "region")] == [None, None, None]
    json.dumps([check_predictor(PROFILE, c.name, "total_spend") for c in PROFILE.columns])
    assert PROFILE == before


# --- validate_predictor_choice -------------------------------------------------------------------------------

def _validate(chosen, max_others=MAX_OTHER_PREDICTORS):
    return validate_predictor_choice(PROFILE, chosen, "total_spend", "region", max_others)


def test_ok_keeps_the_order_with_the_group_first():
    assert _validate(["tenure", "tier", "region"]) == {"ok": True, "predictors": ["region", "tenure", "tier"],
                                                       "refusals": []}


def test_every_problem_is_collected():
    result = _validate(["region", "colour", "total_spend", "customer_id", "tenure", "tenure"])
    assert result["ok"] is False and result["predictors"] == []
    assert [(v["column"], v["rule"]) for v in result["refusals"]] == [
        ("tenure", "duplicate"), ("colour", "unknown_column"), ("total_spend", "outcome"),
        ("customer_id", "identifier_like")]
    json.dumps(result)


def test_group_missing():
    (v,) = _validate(["tenure"])["refusals"]
    assert v == {"column": "region", "rule": "group_missing", "reason": "the grouping column must stay",
                 "explanation": "the grouping column of the main analysis must stay in the model, because the "
                                "exploratory model adjusts that comparison"}


def test_a_duplicate_is_reported_once():
    refusals = _validate(["region", "tenure", "tenure", "tenure", "tier"])["refusals"]
    assert refusals == [{"column": "tenure", "rule": "duplicate", "reason": "listed twice",
                         "explanation": "it was listed more than once"}]


def test_cap():
    profile = _profile(("region", "object", 2, 0), ("total_spend", "float64", 100, 0),
                       *[(f"x{i}", "float64", 50, 0) for i in range(8)])
    result = validate_predictor_choice(profile, ["region", *[f"x{i}" for i in range(7)]], "total_spend", "region", 5)
    assert result["refusals"] == [{"column": None, "rule": "cap", "reason": "beyond the limit of 5 other predictors",
                                   "explanation": "you chose 7 other predictors; the limit is 5; remove 2"}]


def test_the_cap_counts_distinct_names():
    result = _validate(["region", "tenure", "tier", "tenure"], max_others=2)
    assert [v["rule"] for v in result["refusals"]] == ["duplicate"]


def test_none_chosen():
    assert _validate(["region"])["refusals"] == [{
        "column": None, "rule": "none_chosen", "reason": "no other predictor",
        "explanation": "choose at least one other predictor besides the grouping column"}]


@pytest.mark.parametrize("chosen", ["region", None, {"region": True}, ["region", 3]],
                         ids=["string", "none", "dict", "non_str_item"])
def test_not_a_list(chosen):
    assert _validate(chosen) == {"ok": False, "predictors": [], "refusals": [{
        "column": None, "rule": "not_a_list", "reason": "predictors must be a list of column names",
        "explanation": "predictors must be a list of column names"}]}


def test_the_group_column_is_never_checked_against_the_rules():
    profile = _profile(("code", "int64", 100, 0), ("total_spend", "float64", 100, 0), ("tenure", "float64", 80, 0))
    assert validate_predictor_choice(profile, ["code", "tenure"], "total_spend", "code", 5)["ok"] is True


# --- the proposal and the user check use the same rules ---------------------------------------------------------------

def test_the_proposal_and_the_check_agree():
    profile = _profile(("region", "object", 2, 0), ("total_spend", "float64", 100, 0), ("tenure", "float64", 80, 0),
                       ("tier", "object", 3, 0), ("customer_id", "int64", 100, 0), ("flag", "float64", 1, 0),
                       ("city", "object", 11, 0), *[(f"x{i}", "float64", 50, 0) for i in range(5)])
    p = propose_exploratory_predictors(profile, "region", "total_spend")
    for name in p["excluded"]:
        verdict = p["verdicts"][name]
        if verdict["rule"] == "cap":
            assert check_predictor(profile, name, "total_spend") is None
            assert verdict["explanation"] == ("it was left out only because the proposal is limited to 5 other "
                                              "predictors; you can swap it in for another")
        else:
            assert verdict == check_predictor(profile, name, "total_spend")
            assert p["reasons"][name] == verdict["reason"]
    assert set(p["verdicts"]) == set(p["excluded"])
    assert any(v["rule"] == "cap" for v in p["verdicts"].values())
    fitting = [c.name for c in profile.columns
               if c.name != "region" and check_predictor(profile, c.name, "total_spend") is None]
    assert p["selectable"] == fitting == ["tenure", "tier", "x0", "x1", "x2", "x3", "x4"]
    assert p["included"] == ["region", "tenure", "tier", "x0", "x1", "x2"]
    json.dumps(p)


def test_make_profile_helper_still_gives_the_same_proposal():
    profile = make_profile({"region": ("object", 2), "tier": ("object", 3), "tenure": ("float64", 80),
                            "total_spend": ("float64", 100)})
    p = propose_exploratory_predictors(profile, "region", "total_spend")
    assert p["predictors"] == ["region", "tier", "tenure"] and p["selectable"] == ["tier", "tenure"]
