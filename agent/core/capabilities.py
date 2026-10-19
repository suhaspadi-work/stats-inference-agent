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
from core.plan import AnalysisPlan, MethodDecision, OutcomeType
from core.decision_record import DecisionRecord
from core.execution_records import multi_group_execution_records
from core.selection_records import (
    two_group_selection_record, multi_group_selection_record, regression_selection_records,
)
from core.testing import select_two_group_method, execute_two_group_test
from core.multi_group import select_multi_group_method, execute_multi_group_test
from core.regression import select_regression_method, execute_regression
from core.descriptive import column_stats_to_dicts, compute_descriptive_stats
from core.diagnostics import compute_diagnostics, diagnostics_to_dict
from core import wrangling
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


# A grouping column has 2 to 10 distinct values, whatever its dtype. The 10 mirrors
# regression.MAX_CATEGORICAL_LEVELS (a test asserts the two are equal).
GROUPING_MIN_LEVELS = 2
GROUPING_MAX_LEVELS = 10


def is_grouping_column(column) -> bool:
    """True when the column has between GROUPING_MIN_LEVELS and GROUPING_MAX_LEVELS distinct values."""
    return GROUPING_MIN_LEVELS <= column.distinct_count <= GROUPING_MAX_LEVELS


def _check_inputs(plan: AnalysisPlan, profile: ModelSafeView):
    """Shared checks. Returns (problem or None, columns by name, predictors)."""
    cols = {c.name: c for c in profile.columns}
    predictors = tuple(plan.candidate_predictors or ())
    missing = [n for n in (plan.outcome_name, *predictors) if n not in cols]
    if missing:
        return Applicability(False, f"Columns not in the dataset: {missing}.", {"missing_columns": missing}), cols, predictors
    if plan.outcome_type is OutcomeType.BINARY:
        return Applicability(
            False,
            f"Outcome '{plan.outcome_name}' is binary; binary outcomes are not supported yet (G3).",
            {"outcome": plan.outcome_name, "outcome_type": "binary"},
        ), cols, predictors
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
    if levels > GROUPING_MAX_LEVELS:
        return Applicability(
            False,
            f"'{group}' has {levels} distinct values; a grouping column can have {GROUPING_MIN_LEVELS} to {GROUPING_MAX_LEVELS}.",
            {"group_column": group, "levels": levels, "max_levels": GROUPING_MAX_LEVELS},
        )
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
    if levels > GROUPING_MAX_LEVELS:
        return Applicability(
            False,
            f"'{group}' has {levels} distinct values; a grouping column can have {GROUPING_MIN_LEVELS} to {GROUPING_MAX_LEVELS}.",
            {"group_column": group, "levels": levels, "max_levels": GROUPING_MAX_LEVELS},
        )
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


def _plain_charts(charts) -> list:
    """Run-time charts as plain lists and dicts (tuples become lists; NaN values are kept)."""
    def plain(v):
        if isinstance(v, dict):
            return {k: plain(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [plain(x) for x in v]
        return v
    return [plain(c) for c in (charts or ())]


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
        "group_a_label": r.group_a_label,
        "group_b_label": r.group_b_label,
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
        "chart_data": _plain_charts(r.chart_data),
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
        "chart_data": _plain_charts(r.chart_data),
    }


# ----------------------------------------------------------------- descriptive

def _compute_descriptive(dataset, store):
    r = compute_descriptive_stats(dataset, store)
    return {
        "row_count": r.row_count,
        "column_stats": column_stats_to_dicts(r.column_stats),
        "correlation_matrix": r.correlation_matrix,
        "chart_data": list(r.chart_data),
    }


def _compute_diagnostics(dataset, store):
    return diagnostics_to_dict(compute_diagnostics(dataset, store))


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
    # Builds the selection DecisionRecord(s) from the MethodDecision (M0.2 steps 5 to 7):
    # one record for the test families, a tuple (method, then one per encoding) for regression
    record: Optional[Callable[[MethodDecision], Union[DecisionRecord, tuple[DecisionRecord, ...]]]] = None
    # M0.2 step 11: records for choices made at run time, from the result and the selection record
    execution_records: Optional[Callable[[dict, Optional[dict]], tuple[DecisionRecord, ...]]] = None

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
    report: Optional[Callable[[AnalysisPlan, dict], str]] = None  # None: the result is not reported (diagnose)
    kind: str = "diagnostic"
    guard_causal_language: bool = False


@dataclass(frozen=True)
class OperationCapability:
    """
    M2.3a: the third registry kind (D8), a Stage 1 operation. transform is pure (df, params) -> (new_df, details,
    warnings); execute wraps the same transform, so the dry-run preview and the execution are the same code.
    """
    name: str                        # the op_type
    stage: int
    transform: Callable
    execute: Callable
    describe: Callable
    kind: str = "operation"
    guard_causal_language: bool = False
    # M2.3b: an operation that reads other tables (a join): transform(df, params, frames) and
    # execute(dataset, store, operation, tables)
    multi_input: bool = False


def _operation(name, transform, execute, multi_input=False) -> OperationCapability:
    return OperationCapability(name=name, stage=1, transform=transform, execute=execute,
                               describe=wrangling.describe_operation, multi_input=multi_input)


OPERATIONS = {op.name: op for op in (
    _operation("cast", wrangling.transform_cast, wrangling.execute_cast),
    _operation("rename", wrangling.transform_rename, wrangling.execute_rename),
    _operation("filter", wrangling.transform_filter, wrangling.execute_filter),
    _operation("dedupe", wrangling.transform_dedupe, wrangling.execute_dedupe),
    _operation("drop_column", wrangling.transform_drop_column, wrangling.execute_drop_column),
    _operation("impute", wrangling.transform_impute, wrangling.execute_impute),
    _operation("normalize_text", wrangling.transform_normalize_text, wrangling.execute_normalize_text),
    _operation("join", wrangling.transform_join, wrangling.execute_join, multi_input=True),
)}


def operation_named(op_type: str) -> OperationCapability:
    """The registry entry of an operation type; a ValueError names an unknown one."""
    try:
        return OPERATIONS[op_type]
    except KeyError:
        raise ValueError(f"No operation is registered for type '{op_type}'.")


def run_transform(op_type: str, df, params: dict, frames=None):
    """M2.3b: an operation's pure transform; a multi_input operation also gets frames (table name -> DataFrame)."""
    operation = operation_named(op_type)
    if operation.multi_input:
        return operation.transform(df, params, frames or {})
    return operation.transform(df, params)


TWO_GROUP = Analysis(
    name="two_group", stage=2, result_type="two_group", predictor_rule="exactly_one",
    applicability=_applicable_two_group, select=_select_two_group,
    execute=_execute_two_group, report=generate_report,
    record=two_group_selection_record,
)
MULTI_GROUP = Analysis(
    name="multi_group", stage=2, result_type="multi_group", predictor_rule="exactly_one",
    applicability=_applicable_multi_group, select=_select_multi_group,
    execute=_execute_multi_group, report=generate_multi_group_report,
    record=multi_group_selection_record,
    execution_records=multi_group_execution_records,
)
REGRESSION = Analysis(
    name="regression", stage=3, result_type="regression", predictor_rule="any",
    applicability=_applicable_regression, select=_select_regression,
    execute=_execute_regression, report=generate_regression_report,
    record=regression_selection_records,
)
DESCRIPTIVE = Diagnostic(
    name="descriptive", stage=1, result_key="descriptive_result",
    compute=_compute_descriptive, report=generate_descriptive_report,
)
# M0.6 (G16): the chart-free diagnostics. Nothing routes to it yet; the Stage 1 controller (M1, M2) will.
DIAGNOSE = Diagnostic(
    name="diagnose", stage=1, result_key="diagnostics_result", compute=_compute_diagnostics,
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


def capability_for_question_type(question_type: str) -> Union[Analysis, Diagnostic]:
    """Read-only routing lookup: the capability a question type value dispatches to."""
    try:
        return _BY_QUESTION_TYPE[question_type]
    except KeyError:
        raise ValueError(f"No capability is registered for question type '{question_type}'.")


def capability_for_plan(plan: AnalysisPlan) -> Union[Analysis, Diagnostic]:
    return capability_for_question_type(plan.question_type.value)


def capability_named(name: str) -> Union[Analysis, Diagnostic]:
    """The registry entry (an analysis, the descriptive diagnostic or diagnose) with this name."""
    for entry in (*ANALYSES, DESCRIPTIVE, DIAGNOSE):
        if entry.name == name:
            return entry
    raise ValueError(f"No capability is named '{name}'.")


def analysis_for_result(result: Optional[dict]) -> Analysis:
    """Dispatch a result to its report by the result's own tag. Untagged results are two-group, as before."""
    return _BY_RESULT_TYPE[(result or {}).get("result_type", "two_group")]


def evaluate_all(plan: AnalysisPlan, profile: ModelSafeView) -> list[tuple[Analysis, Applicability]]:
    """Every analysis with its applicability for this plan. Later milestones use this for candidate lists."""
    return [(a, a.applicability(plan, profile)) for a in ANALYSES]