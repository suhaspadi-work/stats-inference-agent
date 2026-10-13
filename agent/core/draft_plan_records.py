"""
Decision records for drafting the plan's columns (M0.2 step 4, G14). Pure
functions: no graph and no model calls. Candidate lists are every column in
the model-safe profile; eligibility and its reasons come from code checking
column types. The model supplies the choices and one shared rationale.
"""
from __future__ import annotations

from core.classifier import _CONTINUOUS_COMPATIBLE_DTYPES
from core.decision_record import Candidate, DecisionRecord
from core.model_safe_view import ColumnSummary, ModelSafeView
from core.plan import OutcomeType

OUTCOME_POINT = "planner.outcome"
OUTCOME_TYPE_POINT = "planner.outcome_type"
PREDICTORS_POINT = "planner.predictors"
UNIT_POINT = "planner.unit_of_analysis"

_CHOSEN_REASON = "chosen by the model; see rationale"
_NOT_CHOSEN_REASON = "not chosen by the model"
_BINARY_UNCHECKED = "the agent does not check column types for binary outcomes (G3)"


def _column_evidence(col: ColumnSummary) -> dict:
    return {
        "dtype": str(col.dtype),
        "distinct_count": int(col.distinct_count),
        "missing_pct": float(col.missing_pct),
    }


def _supports_continuous(dtype: str) -> bool:
    return str(dtype) in _CONTINUOUS_COMPATIBLE_DTYPES


def _outcome_record(fields, profile: ModelSafeView) -> DecisionRecord:
    continuous = fields.outcome_type == OutcomeType.CONTINUOUS
    candidates = []
    for col in profile.columns:
        dtype = str(col.dtype)
        eligible = _supports_continuous(dtype) if continuous else None
        if col.name == fields.outcome_name:
            reason = _CHOSEN_REASON
        elif eligible is False:
            reason = f"type {dtype} cannot support a continuous outcome"
        elif eligible is None:
            reason = f"{_NOT_CHOSEN_REASON}; {_BINARY_UNCHECKED}"
        else:
            reason = _NOT_CHOSEN_REASON
        candidates.append(Candidate(option=col.name, eligible=eligible, reason=reason,
                                    evidence=_column_evidence(col)))
    return DecisionRecord(
        decision_point=OUTCOME_POINT, chosen=fields.outcome_name, decided_by="model",
        candidates=tuple(candidates), rationale=fields.rationale,
        evidence={"n_columns": len(profile.columns), "outcome_type": fields.outcome_type.value},
    )


def _outcome_type_record(fields, outcome_col: ColumnSummary) -> DecisionRecord:
    dtype = str(outcome_col.dtype)
    supports = _supports_continuous(dtype)
    which = "supports" if supports else "cannot support"
    return DecisionRecord(
        decision_point=OUTCOME_TYPE_POINT, chosen=fields.outcome_type.value, decided_by="model",
        candidates=(
            Candidate(option=OutcomeType.CONTINUOUS.value, eligible=supports,
                      reason=f"outcome column '{outcome_col.name}' has type {dtype}, which {which} a continuous outcome",
                      evidence={"outcome_dtype": dtype}),
            Candidate(option=OutcomeType.BINARY.value, eligible=None, reason=_BINARY_UNCHECKED),
        ),
        rationale=(f"The model proposed {fields.outcome_type.value}; code checked it against "
                   f"the outcome column's type ({dtype})."),
        evidence={"outcome_column": outcome_col.name, "outcome_dtype": dtype},
    )


def predictor_candidates(profile: ModelSafeView, outcome_name: str, chosen_names) -> tuple[Candidate, ...]:
    """Every profile column as a predictor candidate; the outcome column is ineligible."""
    chosen = list(chosen_names)
    candidates = []
    for col in profile.columns:
        if col.name == outcome_name:
            eligible, reason = False, "the outcome cannot also be a predictor"
        else:
            eligible = True
            reason = _CHOSEN_REASON if col.name in chosen else _NOT_CHOSEN_REASON
        candidates.append(Candidate(option=col.name, eligible=eligible, reason=reason,
                                    evidence=_column_evidence(col)))
    return tuple(candidates)


def _predictors_record(fields, profile: ModelSafeView) -> DecisionRecord:
    chosen = list(fields.candidate_predictors)
    candidates = predictor_candidates(profile, fields.outcome_name, chosen)
    return DecisionRecord(
        decision_point=PREDICTORS_POINT, chosen=", ".join(chosen) if chosen else "none",
        decided_by="model", candidates=candidates, rationale=fields.rationale,
        evidence={"n_columns": len(profile.columns), "n_chosen": len(chosen)},
    )


def _unit_record(fields) -> DecisionRecord:
    unit = fields.unit_of_analysis
    return DecisionRecord(
        decision_point=UNIT_POINT,
        chosen=unit if unit and unit.strip() else "(not stated)",
        decided_by="model",
        rationale="stated by the model; not checked by code (see G12)",
    )


def draft_plan_records(fields, profile: ModelSafeView) -> tuple[DecisionRecord, ...]:
    """Records for outcome, outcome type, predictors and unit of analysis, in that order."""
    outcome_col = next((c for c in profile.columns if c.name == fields.outcome_name), None)
    if outcome_col is None:
        raise ValueError(f"Outcome column '{fields.outcome_name}' is not in the dataset profile.")
    return (
        _outcome_record(fields, profile),
        _outcome_type_record(fields, outcome_col),
        _predictors_record(fields, profile),
        _unit_record(fields),
    )
