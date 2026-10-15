"""
The stage plan (M1.1): which of the four steps of the three-stage design a run will take, decided by code
from the classification, the model-safe profile and the detected data-quality issues. Pure: no model call,
no data access, and no import of core.nodes. The result is a plain dict (it is stored in state).

  stage 1  prepare and describe   clean, describe
  stage 2  statistical inference  inference
  stage 3  data modeling          modeling

Each step's state is planned (the question needs it), offered (the question does not need it, so the user
chooses; "run" is the default choice), unavailable (it would apply but is not built yet) or not_applicable.
"""
from __future__ import annotations

from core.capabilities import is_numeric_dtype_name
from core.model_safe_view import ModelSafeView
from core.plan import ClassificationDecision

SCHEMA_VERSION = 1
STEP_STATES = ("planned", "offered", "unavailable", "not_applicable")
STEP_NAMES = ("clean", "describe", "inference", "modeling")

UNSUPPORTED_MESSAGE = ("Association questions are not supported yet. To see which variables relate to a column, "
                       "ask what drives that column.")
_COMPARISON_TYPES = ("two_group_comparison", "ab_test", "multi_group_comparison")
_MODELING_RULE = "3 or more columns with at least 1 numeric"


def _step(name, state, run, reason, evidence, exploratory=False):
    return {"step": name, "state": state, "run": run, "exploratory": exploratory, "reason": reason,
            "evidence": evidence}


def _clean_step(issues) -> dict:
    types = sorted({i.issue_type for i in issues})
    columns = sorted({c for i in issues for c in i.affected_columns})
    evidence = {"issue_count": len(issues), "issue_types": types, "affected_columns": columns}
    if issues:
        return _step("clean", "planned", True,
                     f"{len(issues)} data-quality issue(s) found ({', '.join(types)}); "
                     "cleaning proposals will be shown for approval", evidence)
    return _step("clean", "not_applicable", False, "no data-quality issues found by the detectors", evidence)


def _offered_describe(describe_evidence) -> dict:
    return _step("describe", "offered", True,
                 "describing the finished dataset is part of stage 1; the question does not need it, so you choose",
                 describe_evidence)


def _modeling_for_comparison(profile: ModelSafeView) -> dict:
    numeric = sum(1 for c in profile.columns if is_numeric_dtype_name(c.dtype))
    evidence = {"column_count": profile.column_count, "numeric_column_count": numeric,
                "rule": _MODELING_RULE, "provisional": True}
    if profile.column_count >= 3 and numeric >= 1:
        return _step("modeling", "offered", False,
                     "a regression could adjust the comparison for other columns; exploratory, so you choose",
                     evidence, exploratory=True)
    return _step("modeling", "not_applicable", False,
                 "fewer than 3 columns or no numeric column, so no adjusted model is possible", evidence)


def build_stage_plan(classification: ClassificationDecision, profile: ModelSafeView, issues) -> dict:
    """The stage plan for this run. Raises ValueError for a question type it does not know."""
    qt = classification.question_type.value
    describe_evidence = {"row_count": profile.row_count, "column_count": profile.column_count}
    qt_evidence = {"question_type": qt}
    status, message = "ok", None

    if qt == "descriptive":
        clean = _clean_step(issues)
        describe = _step("describe", "planned", True, "the question asks for a description of the data",
                         describe_evidence)
        inference = _step("inference", "not_applicable", False, "a descriptive question has no hypothesis to test",
                          dict(qt_evidence))
        modeling = _step("modeling", "not_applicable", False, "a descriptive question has no model to fit",
                         dict(qt_evidence))
    elif qt in _COMPARISON_TYPES:
        clean = _clean_step(issues)
        describe = _offered_describe(describe_evidence)
        inference = _step("inference", "planned", True, "the question compares groups", dict(qt_evidence))
        modeling = _modeling_for_comparison(profile)
    elif qt == "driver_analysis":
        clean = _clean_step(issues)
        describe = _offered_describe(describe_evidence)
        inference = _step("inference", "unavailable", False, "per-predictor association tests are not built yet (M3.3)",
                          {"available_from": "M3.3"})
        modeling = _step("modeling", "planned", True, "the question asks which variables drive the outcome",
                         dict(qt_evidence))
    elif qt == "association":
        status, message = "unsupported", UNSUPPORTED_MESSAGE
        reason = "the question is not supported yet (G1)"
        clean_evidence = _clean_step(issues)["evidence"]
        clean = _step("clean", "not_applicable", False, reason, clean_evidence)
        describe = _step("describe", "not_applicable", False, reason, describe_evidence)
        inference = _step("inference", "not_applicable", False, reason, dict(qt_evidence))
        modeling = _step("modeling", "not_applicable", False, reason, dict(qt_evidence))
    else:
        raise ValueError(f"No stage plan rule for question type '{qt}'.")

    return {
        "schema_version": SCHEMA_VERSION,
        "question_type": qt,
        "status": status,
        "message": message,
        "stages": [
            {"stage": 1, "name": "prepare and describe", "steps": [clean, describe]},
            {"stage": 2, "name": "statistical inference", "steps": [inference]},
            {"stage": 3, "name": "data modeling", "steps": [modeling]},
        ],
    }


def plan_steps(plan: dict) -> list[dict]:
    """The four steps in order (clean, describe, inference, modeling)."""
    return [step for stage in plan["stages"] for step in stage["steps"]]
