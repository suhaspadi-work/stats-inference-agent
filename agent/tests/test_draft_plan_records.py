import json

import numpy as np
import pytest

from core.classifier import _DraftPlanOut
from core.decision_log import log_update
from core.draft_plan_records import (
    OUTCOME_POINT, OUTCOME_TYPE_POINT, PREDICTORS_POINT, UNIT_POINT, draft_plan_records,
)
from core.model_safe_view import ColumnSummary, ModelSafeView
from core.plan import OutcomeType

RATIONALE = "Spend is the outcome; region names the two groups being compared."


def _profile(columns=None):
    columns = columns or [
        ColumnSummary(name="customer_id", dtype="int64", missing_count=0, missing_pct=0.0, distinct_count=100),
        ColumnSummary(name="region", dtype="object", missing_count=2, missing_pct=0.02, distinct_count=2),
        ColumnSummary(name="total_spend", dtype="float64", missing_count=5, missing_pct=0.05, distinct_count=95),
    ]
    return ModelSafeView(dataset_handle="t@v1", row_count=100, column_count=len(columns), columns=tuple(columns))


def _fields(outcome="total_spend", outcome_type=OutcomeType.CONTINUOUS, predictors=("region",), unit="customer"):
    return _DraftPlanOut(outcome_name=outcome, outcome_type=outcome_type, unit_of_analysis=unit,
                         candidate_predictors=list(predictors), rationale=RATIONALE)


def _records(**kwargs):
    profile = kwargs.pop("profile", None) or _profile()
    return draft_plan_records(_fields(**kwargs), profile)


def _by_option(record):
    return {c.option: c for c in record.candidates}


# --- shape --------------------------------------------------------------------

def test_order_and_decided_by():
    records = _records()
    assert [r.decision_point for r in records] == [OUTCOME_POINT, OUTCOME_TYPE_POINT, PREDICTORS_POINT, UNIT_POINT]
    assert [r.decision_point for r in records] == [
        "planner.outcome", "planner.outcome_type", "planner.predictors", "planner.unit_of_analysis"]
    assert all(r.decided_by == "model" for r in records)


def test_all_four_pass_log_update():
    update = log_update(*_records())
    assert len(update["decision_log"]) == 4
    json.dumps(update)


def test_missing_outcome_column_raises():
    with pytest.raises(ValueError, match="revenue_made_up"):
        _records(outcome="revenue_made_up")


# --- outcome ----------------------------------------------------------------

def test_outcome_candidates_are_all_columns_in_profile_order_with_evidence():
    outcome = _records()[0]
    assert [c.option for c in outcome.candidates] == ["customer_id", "region", "total_spend"]
    assert all(c.generated_by == "code" for c in outcome.candidates)
    assert _by_option(outcome)["region"].evidence == {"dtype": "object", "distinct_count": 2, "missing_pct": 0.02}
    assert outcome.chosen == "total_spend"
    assert outcome.evidence == {"n_columns": 3, "outcome_type": "continuous"}


def test_outcome_eligibility_for_continuous():
    cands = _by_option(_records()[0])
    assert cands["total_spend"].eligible is True
    assert cands["total_spend"].reason == "chosen by the model; see rationale"
    assert cands["customer_id"].eligible is True
    assert cands["customer_id"].reason == "not chosen by the model"
    assert cands["region"].eligible is False
    assert cands["region"].reason == "type object cannot support a continuous outcome"


def test_outcome_eligibility_for_binary_is_not_determined():
    cands = _by_option(_records(outcome="region", outcome_type=OutcomeType.BINARY, predictors=())[0])
    assert all(c.eligible is None for c in cands.values())
    assert cands["region"].reason == "chosen by the model; see rationale"
    assert cands["total_spend"].reason == (
        "not chosen by the model; the agent does not check column types for binary outcomes (G3)")


def test_model_rationale_verbatim_in_outcome_and_predictors_closes_g14():
    outcome, _, predictors, _ = _records()
    assert outcome.rationale == RATIONALE
    assert predictors.rationale == RATIONALE


# --- outcome type --------------------------------------------------------------

def test_outcome_type_float64_continuous():
    r = _records()[1]
    assert r.chosen == "continuous"
    cands = _by_option(r)
    assert cands["continuous"].eligible is True
    assert cands["continuous"].reason == "outcome column 'total_spend' has type float64, which supports a continuous outcome"
    assert cands["binary"].eligible is None
    assert cands["binary"].reason == "the agent does not check column types for binary outcomes (G3)"
    assert r.rationale == "The model proposed continuous; code checked it against the outcome column's type (float64)."
    assert r.evidence == {"outcome_column": "total_spend", "outcome_dtype": "float64"}


def test_outcome_type_object_continuous_is_ineligible():
    r = _records(outcome="region", predictors=())[1]
    cands = _by_option(r)
    assert cands["continuous"].eligible is False
    assert cands["continuous"].reason == "outcome column 'region' has type object, which cannot support a continuous outcome"


def test_outcome_type_binary():
    r = _records(outcome="region", outcome_type=OutcomeType.BINARY, predictors=())[1]
    assert r.chosen == "binary"
    assert _by_option(r)["binary"].eligible is None
    assert "(object)" in r.rationale


# --- predictors -------------------------------------------------------------

def test_predictors_joined_and_outcome_ineligible():
    r = _records(predictors=("region", "customer_id"))[2]
    assert r.chosen == "region, customer_id"
    cands = _by_option(r)
    assert cands["total_spend"].eligible is False
    assert cands["total_spend"].reason == "the outcome cannot also be a predictor"
    assert cands["region"].eligible is True and cands["region"].reason == "chosen by the model; see rationale"
    assert cands["customer_id"].reason == "chosen by the model; see rationale"
    assert r.evidence == {"n_columns": 3, "n_chosen": 2}
    assert all(set(c.evidence) == {"dtype", "distinct_count", "missing_pct"} for c in r.candidates)


def test_unchosen_predictor_reason():
    cands = _by_option(_records()[2])
    assert cands["customer_id"].eligible is True
    assert cands["customer_id"].reason == "not chosen by the model"


def test_predictor_equal_to_outcome_shows_chosen_and_ineligible():
    r = _records(predictors=("region", "total_spend"))[2]
    assert r.chosen == "region, total_spend"
    assert _by_option(r)["total_spend"].eligible is False
    assert r.evidence["n_chosen"] == 2


def test_empty_predictors_gives_none():
    r = _records(predictors=())[2]
    assert r.chosen == "none"
    assert r.evidence["n_chosen"] == 0


# --- unit of analysis ---------------------------------------------------------

def test_unit_verbatim():
    r = _records(unit="Customer Account")[3]
    assert r.chosen == "Customer Account"
    assert r.candidates == ()
    assert "G12" in r.rationale


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_unit_is_not_stated(blank):
    assert _records(unit=blank)[3].chosen == "(not stated)"


# --- plain numbers ----------------------------------------------------------

def test_numpy_profile_numbers_become_plain():
    profile = _profile([
        ColumnSummary(name="y", dtype="float64", missing_count=0,
                      missing_pct=np.float64(0.1), distinct_count=np.int64(50)),
    ])
    r = draft_plan_records(_fields(outcome="y", predictors=()), profile)[0]
    ev = r.candidates[0].evidence
    assert type(ev["distinct_count"]) is int and type(ev["missing_pct"]) is float
    assert type(r.evidence["n_columns"]) is int
