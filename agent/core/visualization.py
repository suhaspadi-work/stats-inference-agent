"""
Computes chart-ready data for the 15 locked-in chart types, per
core.plan.MethodDecision.chart_data's documented shape: every function
here returns a plain, JSON-serializable dict (lists/dicts of numbers and
strings only -- never a numpy array, pandas object, or matplotlib figure),
so the backend stays fully decoupled from whatever renders these later.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def _to_plain_list(arr) -> list:
    """Numpy arrays/pandas Series must never leak into chart_data raw -- same discipline as every other numeric boundary in this project."""
    return [float(x) for x in arr]


def histogram(values: np.ndarray, column_name: str) -> dict:
    return {
        "chart_type": "histogram",
        "title": f"Distribution of {column_name}",
        "data": {"values": _to_plain_list(values)},
    }


def bar_chart(categories: list[str], counts: list[int], column_name: str) -> dict:
    return {
        "chart_type": "bar_chart",
        "title": f"Frequency of {column_name}",
        "data": {"categories": [str(c) for c in categories], "counts": [int(c) for c in counts]},
    }


def boxplot_by_group(groups: dict[str, np.ndarray], outcome_name: str) -> dict:
    return {
        "chart_type": "boxplot_by_group",
        "title": f"{outcome_name} by group",
        "data": {"groups": {str(k): _to_plain_list(v) for k, v in groups.items()}},
    }


def boxplot_single_column(values: np.ndarray, column_name: str) -> dict:
    return {
        "chart_type": "boxplot_single_column",
        "title": f"{column_name} (outlier check)",
        "data": {"values": _to_plain_list(values)},
    }


def qq_plot(values: np.ndarray, label: str) -> dict:
    """Theoretical vs. sample quantiles for a normality check -- the visual companion to a Shapiro-Wilk test."""
    osm, osr = stats.probplot(values, dist="norm", fit=False)
    return {
        "chart_type": "qq_plot",
        "title": f"Q-Q plot: {label}",
        "data": {"theoretical_quantiles": _to_plain_list(osm), "sample_quantiles": _to_plain_list(osr)},
    }


def scatter_with_fit(x: np.ndarray, y: np.ndarray, fit_x: np.ndarray, fit_y: np.ndarray, x_name: str, y_name: str) -> dict:
    return {
        "chart_type": "scatter_with_fit",
        "title": f"{y_name} vs. {x_name}",
        "data": {
            "x": _to_plain_list(x), "y": _to_plain_list(y),
            "fit_x": _to_plain_list(fit_x), "fit_y": _to_plain_list(fit_y),
        },
    }


def residual_plot(fitted_values: np.ndarray, residuals: np.ndarray) -> dict:
    return {
        "chart_type": "residual_plot",
        "title": "Residuals vs. fitted values",
        "data": {"fitted_values": _to_plain_list(fitted_values), "residuals": _to_plain_list(residuals)},
    }


def residual_qq_plot(residuals: np.ndarray) -> dict:
    osm, osr = stats.probplot(residuals, dist="norm", fit=False)
    return {
        "chart_type": "residual_qq_plot",
        "title": "Q-Q plot of residuals",
        "data": {"theoretical_quantiles": _to_plain_list(osm), "sample_quantiles": _to_plain_list(osr)},
    }


def coefficient_plot(predictors: list[str], estimates: list[float], ci_lower: list[float], ci_upper: list[float]) -> dict:
    return {
        "chart_type": "coefficient_plot",
        "title": "Coefficient estimates with 95% CI",
        "data": {
            "predictors": list(predictors), "estimates": [float(e) for e in estimates],
            "ci_lower": [float(c) for c in ci_lower], "ci_upper": [float(c) for c in ci_upper],
        },
    }


def residuals_vs_leverage(leverage: np.ndarray, standardized_residuals: np.ndarray) -> dict:
    return {
        "chart_type": "residuals_vs_leverage",
        "title": "Residuals vs. leverage",
        "data": {"leverage": _to_plain_list(leverage), "standardized_residuals": _to_plain_list(standardized_residuals)},
    }


def correlation_heatmap(columns: list[str], matrix: np.ndarray) -> dict:
    return {
        "chart_type": "correlation_heatmap",
        "title": "Correlation matrix",
        "data": {"columns": list(columns), "matrix": [[float(v) for v in row] for row in matrix]},
    }


def missingness_chart(columns: list[str], missing_pcts: list[float]) -> dict:
    return {
        "chart_type": "missingness_chart",
        "title": "Missing values by column",
        "data": {"categories": list(columns), "counts": [float(p) for p in missing_pcts]},
    }


def scatter_matrix(columns: list[str], pairs: dict[tuple[str, str], tuple[list[float], list[float]]]) -> dict:
    return {
        "chart_type": "scatter_matrix",
        "title": "Pairwise scatter plots",
        "data": {
            "columns": list(columns),
            "pairs": {f"{a}|{b}": {"x": x, "y": y} for (a, b), (x, y) in pairs.items()},
        },
    }


def pairwise_categorical_chart(col_a_name: str, col_b_name: str, categories_a: list[str], categories_b: list[str], counts: list[list[int]]) -> dict:
    return {
        "chart_type": "pairwise_categorical_chart",
        "title": f"{col_a_name} by {col_b_name}",
        "data": {
            "categories_a": list(categories_a), "categories_b": list(categories_b),
            "counts": [[int(c) for c in row] for row in counts],
        },
    }


def post_hoc_comparison_chart(comparisons: list[str], mean_diffs: list[float], ci_lower: list[float], ci_upper: list[float], significant: list[bool]) -> dict:
    return {
        "chart_type": "post_hoc_comparison_chart",
        "title": "Pairwise group comparisons",
        "data": {
            "comparisons": list(comparisons), "mean_diffs": [float(d) for d in mean_diffs],
            "ci_lower": [float(c) for c in ci_lower], "ci_upper": [float(c) for c in ci_upper],
            "significant": [bool(s) for s in significant],
        },
    }