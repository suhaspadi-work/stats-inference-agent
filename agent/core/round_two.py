"""
The exploratory round (M1.3b): after a comparison, a regression of the same outcome on the grouping column
plus up to MAX_OTHER_PREDICTORS other columns, labelled exploratory. Pure: no graph, no model call, no import
of core.nodes, and the inputs are never mutated. Every profile column gets a recorded reason for being
included or excluded.
"""
from __future__ import annotations

from core.capabilities import is_numeric_dtype_name
from core.model_safe_view import ModelSafeView
from core.plan import AnalysisPlan, QuestionType
from core.regression import MAX_CATEGORICAL_LEVELS
from core.routing import COMPARISON_TYPES
from core.stage_plan import plan_steps

MAX_OTHER_PREDICTORS = 5
OFFER_NOTE = "labelled exploratory; its results are not corrected for multiple testing"


def _is_float(dtype: str) -> bool:
    return str(dtype).startswith("float")


def propose_exploratory_predictors(profile: ModelSafeView, group_column: str, outcome_name: str) -> dict:
    """{"predictors": [group_column, *others], "reasons": {every column: reason}, "included", "excluded"}."""
    reasons, included, excluded, others = {}, [group_column], [], []
    reasons[group_column] = "the grouping column of the main analysis"
    for col in profile.columns:
        name = col.name
        if name == group_column:
            continue
        numeric = is_numeric_dtype_name(col.dtype)
        if name == outcome_name:
            reason, ok = "the outcome cannot also be a predictor", False
        elif col.distinct_count < 2:
            reason, ok = "constant or empty", False
        elif not _is_float(col.dtype) and col.distinct_count == profile.row_count - col.missing_count:
            reason, ok = "identifier-like: every value is distinct", False
        elif not numeric and col.distinct_count > MAX_CATEGORICAL_LEVELS:
            reason, ok = f"text column with more than {MAX_CATEGORICAL_LEVELS} distinct values", False
        elif len(others) >= MAX_OTHER_PREDICTORS:
            reason, ok = f"beyond the limit of {MAX_OTHER_PREDICTORS} other predictors", False
        else:
            reason, ok = ("numeric column" if numeric else f"text column with {col.distinct_count} levels"), True
        reasons[name] = reason
        if ok:
            others.append(name)
            included.append(name)
        else:
            excluded.append(name)
    return {"predictors": [group_column, *others], "reasons": reasons, "included": included, "excluded": excluded}


def _modeling_step(stage_plan: dict) -> dict:
    return next(s for s in plan_steps(stage_plan) if s["step"] == "modeling")


def exploratory_offer(stage_plan: dict, plan: AnalysisPlan, proposal: dict) -> bool:
    """True only for a comparison whose stage plan offered exploratory modeling and whose data has an extra predictor."""
    if stage_plan is None or plan.question_type.value not in COMPARISON_TYPES:
        return False
    modeling = _modeling_step(stage_plan)
    return modeling["state"] == "offered" and modeling["exploratory"] and len(proposal["predictors"]) > 1


def stage_gate_payload(plan: AnalysisPlan, result: dict, last_capability: str, round_index: int,
                       stage_plan: dict, proposal: dict) -> dict:
    return {
        "type": "stage_gate",
        "question": "The main analysis is finished. Run an exploratory follow-up?",
        "request_text": plan.request_text,
        "finished": {
            "round": round_index,
            "capability": last_capability,
            "question_type": plan.question_type.value,
            "method": plan.primary_method,
            "outcome": plan.outcome_name,
            "predictors": list(plan.candidate_predictors or ()),
            "p_value": (result or {}).get("p_value"),
        },
        "offer": {
            "step": "modeling",
            "capability": "regression",
            "exploratory": True,
            "reason": _modeling_step(stage_plan)["reason"],
            "outcome": plan.outcome_name,
            "proposed_predictors": list(proposal["predictors"]),
            "excluded": {name: proposal["reasons"][name] for name in proposal["excluded"]},
            "note": OFFER_NOTE,
        },
        "allowed_decisions": ["run", "skip"],
    }


def build_exploratory_plan(plan: AnalysisPlan, dataset_handle: str, predictors) -> AnalysisPlan:
    """A new plan (new id) for the exploratory regression; round one's plan is not changed."""
    new = AnalysisPlan(
        request_text=plan.request_text,
        session_id=plan.session_id,
        question_type=QuestionType.DRIVER_ANALYSIS,
        causal_status=plan.causal_status,
        outcome_name=plan.outcome_name,
        outcome_type=plan.outcome_type,
        unit_of_analysis=plan.unit_of_analysis,
        data_sources=(dataset_handle,),
        candidate_predictors=tuple(predictors),
        classification_decision=plan.classification_decision,
        alpha=plan.alpha,
        allowed_claims=plan.allowed_claims,
    )
    new = new.refine(candidate_predictors=tuple(predictors))
    for issue in plan.unaddressed_issues:
        new = new.record_unaddressed_issue(issue)
    return new
