"""
Descriptive/EDA capability: computes real summary statistics and
chart-ready data for a dataset, with no inferential method selection
involved (unlike two-group/multi-group/regression, EDA never chooses
between competing statistical approaches -- it just describes what's
there). This is why DescriptiveResult stands alone, not wrapped in a
MethodDecision the way test-selection results are.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from core.dataset import Dataset, DatasetStore
from core.ingestion import HIGH_CARDINALITY_THRESHOLD
import core.visualization as viz

MAX_SCATTER_MATRIX_COLUMNS = 5


@dataclass(frozen=True)
class ColumnDescriptiveStats:
    name: str
    dtype: str
    is_numeric: bool
    # Numeric-only fields (None for categorical columns)
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    q1: float | None = None
    q3: float | None = None
    skewness: float | None = None
    # Categorical-only fields (None for numeric columns)
    mode: str | None = None
    top_value_counts: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    # Non-null distinct values, the same count as the profile's. Above HIGH_CARDINALITY_THRESHOLD a
    # text column's individual values are withheld: no top_value_counts, mode or bar chart.
    distinct_count: int = 0


@dataclass(frozen=True)
class DescriptiveResult:
    """
    The real, computed output of an EDA pass -- every number here comes
    from pandas/numpy/scipy directly, never from the model. chart_data
    aggregates every chart produced across all columns/combinations.
    """
    row_count: int
    column_stats: tuple[ColumnDescriptiveStats, ...]
    correlation_matrix: dict | None  # {"columns": [...], "matrix": [[...]]} or None if <2 numeric columns
    chart_data: tuple[dict, ...] = field(default_factory=tuple)


def _column_groups(df: pd.DataFrame):
    """Numeric columns, non-numeric columns, non-null distinct counts, and the non-numeric columns at or
    below HIGH_CARDINALITY_THRESHOLD, all in df order."""
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    distinct_counts = {c: int(df[c].nunique(dropna=True)) for c in df.columns}
    low_cardinality_cols = [c for c in categorical_cols if distinct_counts[c] <= HIGH_CARDINALITY_THRESHOLD]
    return numeric_cols, categorical_cols, distinct_counts, low_cardinality_cols


def _top_5(series: pd.Series):
    value_counts = series.value_counts()
    return value_counts, tuple((str(v), int(c)) for v, c in value_counts.head(5).items())


def compute_column_stats(df: pd.DataFrame) -> tuple[ColumnDescriptiveStats, ...]:
    """One ColumnDescriptiveStats per column in df order. A text column above HIGH_CARDINALITY_THRESHOLD
    keeps only its name, dtype and distinct_count (M0.5)."""
    numeric_cols, _, distinct_counts, low_cardinality_cols = _column_groups(df)
    column_stats = []
    for col in df.columns:
        series = df[col].dropna()
        if col in numeric_cols:
            column_stats.append(ColumnDescriptiveStats(
                name=col, dtype=str(df[col].dtype), is_numeric=True,
                mean=float(series.mean()), median=float(series.median()), std=float(series.std()),
                min=float(series.min()), max=float(series.max()),
                q1=float(series.quantile(0.25)), q3=float(series.quantile(0.75)),
                skewness=float(scipy_stats.skew(series)) if len(series) >= 3 else 0.0,
                distinct_count=distinct_counts[col],
            ))
        elif col not in low_cardinality_cols:
            column_stats.append(ColumnDescriptiveStats(
                name=col, dtype=str(df[col].dtype), is_numeric=False, distinct_count=distinct_counts[col],
            ))
        else:
            value_counts, top_5 = _top_5(series)
            column_stats.append(ColumnDescriptiveStats(
                name=col, dtype=str(df[col].dtype), is_numeric=False,
                mode=str(value_counts.index[0]) if len(value_counts) else None,
                top_value_counts=top_5, distinct_count=distinct_counts[col],
            ))
    return tuple(column_stats)


def compute_correlation(df: pd.DataFrame) -> dict | None:
    """{"columns": [...], "matrix": [[...]]} over the numeric columns, or None with fewer than two."""
    numeric_cols = _column_groups(df)[0]
    if len(numeric_cols) < 2:
        return None
    corr_df = df[numeric_cols].corr()
    return {"columns": numeric_cols, "matrix": corr_df.to_numpy().tolist()}


def compute_charts(df: pd.DataFrame) -> tuple[dict, ...]:
    """Every chart, in order: per column a histogram and box plot (numeric) or a bar chart (text at or
    below the threshold); then the correlation heatmap and scatter matrix when applicable; then the
    missingness chart; then the pairwise chart of the first two low-cardinality text columns."""
    numeric_cols, _, _, low_cardinality_cols = _column_groups(df)
    charts = []
    for col in df.columns:
        series = df[col].dropna()
        if col in numeric_cols:
            charts.append(viz.histogram(series.to_numpy(), col))
            charts.append(viz.boxplot_single_column(series.to_numpy(), col))
        elif col in low_cardinality_cols:
            _, top_5 = _top_5(series)
            charts.append(viz.bar_chart(
                [v for v, _ in top_5], [c for _, c in top_5], col,
            ))

    if len(numeric_cols) >= 2:
        corr_df = df[numeric_cols].corr()
        charts.append(viz.correlation_heatmap(numeric_cols, corr_df.to_numpy()))

        if len(numeric_cols) <= MAX_SCATTER_MATRIX_COLUMNS:
            pairs = {}
            for i, a in enumerate(numeric_cols):
                for b in numeric_cols[i + 1:]:
                    pairs[(a, b)] = (df[a].tolist(), df[b].tolist())
            charts.append(viz.scatter_matrix(numeric_cols, pairs))

    missing_pcts = [float(df[c].isna().mean()) for c in df.columns]
    charts.append(viz.missingness_chart(list(df.columns), missing_pcts))

    if len(low_cardinality_cols) >= 2:
        col_a, col_b = low_cardinality_cols[0], low_cardinality_cols[1]
        crosstab = pd.crosstab(df[col_a], df[col_b])
        charts.append(viz.pairwise_categorical_chart(
            col_a, col_b, list(crosstab.index.astype(str)), list(crosstab.columns.astype(str)),
            crosstab.to_numpy().tolist(),
        ))
    return tuple(charts)


def column_stats_to_dicts(column_stats) -> list[dict]:
    """The plain-dict shape stored in state (descriptive_result and diagnostics_result)."""
    return [
        {
            "name": c.name, "dtype": c.dtype, "is_numeric": c.is_numeric,
            "mean": c.mean, "median": c.median, "std": c.std,
            "min": c.min, "max": c.max, "q1": c.q1, "q3": c.q3, "skewness": c.skewness,
            "mode": c.mode, "top_value_counts": list(c.top_value_counts), "distinct_count": c.distinct_count,
        }
        for c in column_stats
    ]


def compute_descriptive_stats(dataset: Dataset, store: DatasetStore) -> DescriptiveResult:
    """
    The descriptive entry point: the diagnostics engine's statistics plus charts. Reads the real dataset
    (this is one of the few functions, alongside profile_dataset and detect_duplicate_rows, permitted to
    touch raw rows directly). core.diagnostics.compute_diagnostics is the chart-free entry point (M0.6).
    """
    df = store.read(dataset.storage_key)
    return DescriptiveResult(
        row_count=len(df), column_stats=compute_column_stats(df),
        correlation_matrix=compute_correlation(df), chart_data=compute_charts(df),
    )
