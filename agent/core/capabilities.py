"""
Capability registry (M0.1, commit 1 of 2). Nothing imports this module yet,
so adding it changes no behavior.

A capability is one technique family inside a stage. Two kinds of entry:
  Analysis   (Stages 2 and 3): applicability, select, execute, report.
             Lifecycle: select -> freeze -> execute -> report.
  Diagnostic (Stage 1): compute and report. Read-only, no freeze.
Stage 1 operations (assemble, clean, transform) get their own kind in M2.

Routing is still by question type, exactly as the graph nodes do it today.
Applicability is a set of pure functions over the model-safe profile (column
types and distinct counts), so it needs no data access.
Everything that crosses a state boundary is a plain dict.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, Optional, Union

from core.dataset import Dataset, DatasetStore
from core.model_safe_view import ModelSafeView
from core.plan import AnalysisPlan, MethodDecision
from core.testing import select_two_group_method, execute_two_group_test
from core.multi_group import select_multi_group_method, execute_multi_group_test
from core.regression import select_regression_method, execute_regression
from core.descriptive import compute_descriptive_stats
from core.report import (
    generate_report, generate_multi_group_report,
    generate_regression_report, generate_descriptive_report,
)


@dataclass(frozen=True)
class Applicability:
    eligible: bool
    reason: str
    evidence: dict


_NUMERIC_PREFIXES = ("int", "uint", "float")


def is_numeric_dtype_name(dtype: str) -> bool:
    """Conservative: booleans count as non-numeric here."""
    return str(dtype).startswith(_NUMERIC_PREFIXES)


def _check_inputs(plan: AnalysisPlan, profile: ModelSafeView):
    """Shared checks. Returns (problem or None, columns by name, predictors)."""
    cols = {c.name: c for c in profile.columns}
    predictors = tuple(plan.candidate_predictors or ())
    missing = [n for n in (plan.outcome_name, *predictors) if n not in cols]
    if missing:
        return Applicability(False, f"Columns not in the dataset: {missing}.", {"missing_columns": missing}), cols, predictors
    outcome = cols[plan.outcome_name]
    if not is_numeric_dtype_name(outcome.dtype):
        return Applicability(
            False,
            f"Outcome '{plan.outcome_name}' has data type '{outcome.dtype}', which a numeric analysis cannot use.",
            {"outcome": plan.outcome_name, "outcome_dtype": outcome.dtype},
        ), cols, predictors
    return None, cols, predictors


def _applicable_two_group(plan: AnalysisPlan, profile: ModelSafeView) -> Applicability:
    problem, cols, predictors = _check_inputs(plan, profile)
    if problem:
        return problem
    if len(predictors) != 1:
        return Applicability(False, f"Needs exactly one grouping column; got {len(predictors)}.", {"predictor_count": len(predictors)})
    group = predictors[0]
    levels = cols[group].distinct_count
    if levels != 2:
        return Applicability(False, f"'{group}' has {levels} levels; a two-group comparison needs exactly 2.", {"group_column": group, "levels": levels})
    return Applicability(True, f"'{group}' has exactly 2 levels and the outcome is numeric.", {"group_column": group, "levels": levels})


def _applicable_multi_group(plan: AnalysisPlan, profile: ModelSafeView) -> Applicability:
    problem, cols, predictors = _check_inputs(plan, profile)
    if problem:
        return problem
    if len(predictors) != 1:
        return Applicability(False, f"Needs exactly one grouping column; got {len(predictors)}.", {"predictor_count": len(predictors)})
    group = predictors[0]
    levels = cols[group].distinct_count
    if levels < 3:
        return Applicability(False, f"'{group}' has {levels} levels; a multi-group comparison needs 3 or more.", {"group_column": group, "levels": levels})
    return Applicability(True, f"'{group}' has {levels} levels and the outcome is numeric.", {"group_column": group, "levels": levels})


def _applicable_regression(plan: AnalysisPlan, profile: ModelSafeView) -> Applicability:
    problem, cols, predictors = _check_inputs(plan, profile)
    if problem:
        return problem
    if not predictors:
        return Applicability(False, "Needs at least one predictor column.", {"predictor_count": 0})
    if plan.outcome_name in predictors:
        return Applicability(False, f"The outcome '{plan.outcome_name}' cannot also be a predictor.", {"outcome": plan.outcome_name})
    return Applicability(True, f"Numeric outcome with {len(predictors)} predictor(s).", {"predictor_count": len(predictors)})


def _require_method(plan: AnalysisPlan) -> str:
    method = plan.primary_method
    if method is None:
        raise ValueError("Cannot execute: no method was selected on the plan before freeze.")
    return method


# ------------------------------------------------------------------ two-group

def _select_two_group(dataset, store, plan):
    return select_two_group_method(
        dataset=dataset, store=store,
        outcome_col=plan.outcome_name, group_col=plan.candidate_predictors[0],
    ).decision


def _execute_two_group(dataset, store, plan):
    r = execute_two_group_test(
        dataset=dataset, store=store, outcome_col=plan.outcome_name,
        group_col=plan.candidate_predictors[0], method=_require_method(plan),
    )
    return {
        "result_type": "two_group",
        "method": r.method,
        "statistic": r.statistic,
        "p_value": r.p_value,
        "effect_size": r.effect_size,
        "confidence_interval": r.confidence_interval,
        "group_a_n": r.group_a_n,
        "group_b_n": r.group_b_n,
        "group_a_mean": r.group_a_mean,
        "group_b_mean": r.group_b_mean,
    }


# ----------------------------------------------------------------- multi-group

def _select_multi_group(dataset, store, plan):
    return select_multi_group_method(
        dataset=dataset, store=store,
        outcome_col=plan.outcome_name, group_col=plan.candidate_predictors[0],
    ).decision


def _execute_multi_group(dataset, store, plan):
    r = execute_multi_group_test(
        dataset=dataset, store=store, outcome_col=plan.outcome_name,
        group_col=plan.candidate_predictors[0], method=_require_method(plan),
    )
    return {
        "result_type": "multi_group",
        "method": r.method,
        "statistic": r.statistic,
        "p_value": r.p_value,
        "post_hoc_method": r.post_hoc_method,
        "pairwise_comparisons": [
            {
                "group_a": c.group_a, "group_b": c.group_b, "mean_diff": c.mean_diff,
                "ci_lower": c.ci_lower, "ci_upper": c.ci_upper,
                "p_value": c.p_value, "significant": c.significant,
            }
            for c in r.pairwise_comparisons
        ],
        "group_ns": r.group_ns,
        "group_means": r.group_means,
    }


# ------------------------------------------------------------------ regression

def _select_regression(dataset, store, plan):
    return select_regression_method(
        dataset=dataset, store=store, outcome_col=plan.outcome_name,
        predictors=list(plan.candidate_predictors or ()),
    ).decision


def _execute_regression(dataset, store, plan):
    method = _require_method(plan)
    # The encodings the approver saw are stored in the MethodDecision evidence.
    decision = plan.latest_decision
    stored = (decision.evidence.get("encodings") if decision else None) or {}
    encodings = {
        col: {k: spec.get(k) for k in ("scheme", "reference", "order", "source")}
        for col, spec in stored.items()
    } or None
    r = execute_regression(
        dataset=dataset, store=store, outcome_col=plan.outcome_name,
        predictors=list(plan.candidate_predictors or ()), method=method, encodings=encodings,
    )
    return {
        "result_type": "regression",
        "method": r.method,
        "n_observations": r.n_observations,
        "n_dropped_missing": r.n_dropped_missing,
        "r_squared": r.r_squared,
        "adj_r_squared": r.adj_r_squared,
        "aic": r.aic,
        "f_statistic": r.f_statistic,
        "f_p_value": r.f_p_value,
        "coefficients": [asdict(c) for c in r.coefficients],
        "encodings": r.encodings,
        "alternative_fits": [
            {
                "label": f.label, "description": f.description,
                "same_fitted_model": f.same_fitted_model,
                "r_squared": f.r_squared, "adj_r_squared": f.adj_r_squared, "aic": f.aic,
                "coefficients": [asdict(c) for c in f.coefficients],
            }
            for f in r.alternative_fits
        ],
    }


# ----------------------------------------------------------------- descriptive

def _compute_descriptive(dataset, store):
    r = compute_descriptive_stats(dataset, store)
    return {
        "row_count": r.row_count,
        "column_stats": [
            {
                "name": c.name, "dtype": c.dtype, "is_numeric": c.is_numeric,
                "mean": c.mean, "median": c.median, "std": c.std,
                "min": c.min, "max": c.max, "q1": c.q1, "q3": c.q3, "skewness": c.skewness,
                "mode": c.mode, "top_value_counts": list(c.top_value_counts),
            }
            for c in r.column_stats
        ],
        "correlation_matrix": r.correlation_matrix,
        "chart_data": list(r.chart_data),
    }


# -------------------------------------------------------------------- entries

@dataclass(frozen=True)
class Analysis:
    name: str
    stage: int
    result_type: str                 # tag stored in the result dict
    predictor_rule: str              # "exactly_one" or "any"
    applicability: Callable[[AnalysisPlan, ModelSafeView], Applicability]
    select: Callable[[Dataset, DatasetStore, AnalysisPlan], MethodDecision]
    execute: Callable[[Dataset, DatasetStore, AnalysisPlan], dict]
    report: Callable[[AnalysisPlan, dict], str]
    kind: str = "analysis"
    guard_causal_language: bool = True

    def validate_predictors(self, plan: AnalysisPlan, phase: str) -> None:
        predictors = plan.candidate_predictors or ()
        if self.predictor_rule == "exactly_one" and len(predictors) != 1:
            raise ValueError(
                f"{phase} requires exactly one grouping predictor; "
                f"got {len(predictors)}: {predictors}."
            )


@dataclass(frozen=True)
class Diagnostic:
    name: str
    stage: int
    result_key: str                  # AgentState key the result is stored under
    compute: Callable[[Dataset, DatasetStore], dict]
    report: Callable[[AnalysisPlan, dict], str]
    kind: str = "diagnostic"
    guard_causal_language: bool = False


TWO_GROUP = Analysis(
    name="two_group", stage=2, result_type="two_group", predictor_rule="exactly_one",
    applicability=_applicable_two_group, select=_select_two_group,
    execute=_execute_two_group, report=generate_report,
)
MULTI_GROUP = Analysis(
    name="multi_group", stage=2, result_type="multi_group", predictor_rule="exactly_one",
    applicability=_applicable_multi_group, select=_select_multi_group,
    execute=_execute_multi_group, report=generate_multi_group_report,
)
REGRESSION = Analysis(
    name="regression", stage=3, result_type="regression", predictor_rule="any",
    applicability=_applicable_regression, select=_select_regression,
    execute=_execute_regression, report=generate_regression_report,
)
DESCRIPTIVE = Diagnostic(
    name="descriptive", stage=1, result_key="descriptive_result",
    compute=_compute_descriptive, report=generate_descriptive_report,
)

ANALYSES = (TWO_GROUP, MULTI_GROUP, REGRESSION)

_BY_QUESTION_TYPE = {
    "descriptive": DESCRIPTIVE,
    "two_group_comparison": TWO_GROUP,
    "ab_test": TWO_GROUP,
    "association": TWO_GROUP,  # no association capability yet (gap G1, planned M3.3)
    "multi_group_comparison": MULTI_GROUP,
    "driver_analysis": REGRESSION,
}
_BY_RESULT_TYPE = {a.result_type: a for a in ANALYSES}


def capability_for_plan(plan: AnalysisPlan) -> Union[Analysis, Diagnostic]:
    try:
        return _BY_QUESTION_TYPE[plan.question_type.value]
    except KeyError:
        raise ValueError(f"No capability is registered for question type '{plan.question_type.value}'.")


def analysis_for_result(result: Optional[dict]) -> Analysis:
    """Dispatch a result to its report by the result's own tag. Untagged results are two-group, as before."""
    return _BY_RESULT_TYPE[(result or {}).get("result_type", "two_group")]


def evaluate_all(plan: AnalysisPlan, profile: ModelSafeView) -> list[tuple[Analysis, Applicability]]:
    """Every analysis with its applicability for this plan. Later milestones use this for candidate lists."""
    return [(a, a.applicability(plan, profile)) for a in ANALYSES]