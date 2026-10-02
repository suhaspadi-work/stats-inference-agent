import pytest
from core.classifier import validate_draft_plan_fields, _DraftPlanOut
from core.model_safe_view import ModelSafeView, ColumnSummary
from core.plan import OutcomeType


def make_profile(columns):
    return ModelSafeView(
        dataset_handle="test@v1", row_count=100, column_count=len(columns), columns=tuple(columns),
    )


def test_rejects_continuous_outcome_on_a_text_column():
    profile = make_profile([
        ColumnSummary(name="total_spend", dtype="object", missing_pct=0.0, missing_count=0, distinct_count=3),
        ColumnSummary(name="region", dtype="object", missing_pct=0.0, missing_count=0, distinct_count=2),
    ])
    fields = _DraftPlanOut(
        outcome_name="total_spend", outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="customer", candidate_predictors=["region"], rationale="test",
    )
    with pytest.raises(ValueError, match="cannot support a continuous"):
        validate_draft_plan_fields(fields, profile)


def test_accepts_continuous_outcome_on_a_numeric_column():
    profile = make_profile([
        ColumnSummary(name="total_spend", dtype="float64", missing_pct=0.0, missing_count=0, distinct_count=100),
        ColumnSummary(name="region", dtype="object", missing_pct=0.0, missing_count=0, distinct_count=2),
    ])
    fields = _DraftPlanOut(
        outcome_name="total_spend", outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="customer", candidate_predictors=["region"], rationale="test",
    )
    validate_draft_plan_fields(fields, profile)  # must not raise


def test_still_catches_hallucinated_columns_alongside_the_new_check():
    """Confirms the new type-validation code didn't accidentally weaken the existing hallucination guard."""
    profile = make_profile([
        ColumnSummary(name="total_spend", dtype="float64", missing_pct=0.0, missing_count=0, distinct_count=100),
    ])
    fields = _DraftPlanOut(
        outcome_name="revenue_made_up", outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="customer", candidate_predictors=[], rationale="test",
    )
    with pytest.raises(ValueError, match="not present in the dataset"):
        validate_draft_plan_fields(fields, profile)