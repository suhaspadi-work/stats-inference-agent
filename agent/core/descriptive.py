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


def compute_descriptive_stats(dataset: Dataset, store: DatasetStore) -> DescriptiveResult:
    """
    The single entry point for EDA. Reads the real dataset (this is one of
    the few functions, alongside profile_dataset and detect_duplicate_rows,
    permitted to touch raw rows directly) and computes genuine statistics
    and chart data for every column, plus cross-column views where relevant.
    """
    df = store.read(dataset.storage_key)
    column_stats = []
    charts = []

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    distinct_counts = {c: int(df[c].nunique(dropna=True)) for c in df.columns}
    low_cardinality_cols = [c for c in categorical_cols if distinct_counts[c] <= HIGH_CARDINALITY_THRESHOLD]

    for col in df.columns:
        series = df[col].dropna()
        is_numeric = col in numeric_cols

        if is_numeric:
            column_stats.append(ColumnDescriptiveStats(
                name=col, dtype=str(df[col].dtype), is_numeric=True,
                mean=float(series.mean()), median=float(series.median()), std=float(series.std()),
                min=float(series.min()), max=float(series.max()),
                q1=float(series.quantile(0.25)), q3=float(series.quantile(0.75)),
                skewness=float(scipy_stats.skew(series)) if len(series) >= 3 else 0.0,
                distinct_count=distinct_counts[col],
            ))
            charts.append(viz.histogram(series.to_numpy(), col))
            charts.append(viz.boxplot_single_column(series.to_numpy(), col))
        elif col not in low_cardinality_cols:
            column_stats.append(ColumnDescriptiveStats(
                name=col, dtype=str(df[col].dtype), is_numeric=False, distinct_count=distinct_counts[col],
            ))
        else:
            value_counts = series.value_counts()
            top_5 = tuple((str(v), int(c)) for v, c in value_counts.head(5).items())
            column_stats.append(ColumnDescriptiveStats(
                name=col, dtype=str(df[col].dtype), is_numeric=False,
                mode=str(value_counts.index[0]) if len(value_counts) else None,
                top_value_counts=top_5, distinct_count=distinct_counts[col],
            ))
            charts.append(viz.bar_chart(
                [v for v, _ in top_5], [c for _, c in top_5], col,
            ))

    correlation_matrix = None
    if len(numeric_cols) >= 2:
        corr_df = df[numeric_cols].corr()
        correlation_matrix = {"columns": numeric_cols, "matrix": corr_df.to_numpy().tolist()}
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

    return DescriptiveResult(
        row_count=len(df), column_stats=tuple(column_stats),
        correlation_matrix=correlation_matrix, chart_data=tuple(charts),
    )