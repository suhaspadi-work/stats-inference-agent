"""
The data-quality snapshot of the analysis table (M2.2): plain numbers and column names only, never a cell value
and never a raw minimum or maximum. Built on core.diagnostics.compute_diagnostics (counts, missingness, skewness);
the outlier fences are computed from the stored rows. Information only: outliers and skew get proposals in M2.6.
No snapshot number reaches a prompt.
"""
from __future__ import annotations

import pandas as pd

from core.dataset import Dataset, DatasetStore
from core.diagnostics import compute_diagnostics

OUTLIER_IQR_MULTIPLIER = 1.5  # Tukey's fences: below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR
SKEW_THRESHOLD = 1.0          # a column is skewed when |skewness| exceeds this


def _sig6(x) -> float:
    return float(f"{float(x):.6g}")


def _outliers(df: pd.DataFrame) -> list:
    found = []
    for column in df.columns:
        series = df[column]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        values = series.dropna()
        if values.empty:
            continue
        q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
        iqr = q3 - q1
        lower, upper = q1 - OUTLIER_IQR_MULTIPLIER * iqr, q3 + OUTLIER_IQR_MULTIPLIER * iqr
        below, above = int((values < lower).sum()), int((values > upper).sum())
        if below or above:
            found.append({"column": str(column), "lower_fence": _sig6(lower), "upper_fence": _sig6(upper),
                          "below": below, "above": above})
    return found


def quality_snapshot(dataset: Dataset, store: DatasetStore, table_name=None) -> dict:
    """The quality numbers of one table, as a JSON-clean dict."""
    diagnostics = compute_diagnostics(dataset, store)
    df = store.read(dataset.storage_key)
    missing = [{"column": m.name, "missing": m.missing_count, "missing_pct": m.missing_pct}
               for m in diagnostics.missing_by_column if m.missing_count > 0]
    outliers = _outliers(df)
    skewed = [{"column": c.name, "skewness": round(float(c.skewness), 3)} for c in diagnostics.column_stats
              if c.is_numeric and c.skewness is not None and pd.notna(c.skewness) and abs(c.skewness) > SKEW_THRESHOLD]
    return {
        "table": table_name or dataset.name,
        "handle": dataset.handle,
        "rows": diagnostics.row_count,
        "columns": diagnostics.column_count,
        "duplicate_rows": diagnostics.duplicate_row_count,
        "missing": missing,
        "columns_with_missing": len(missing),
        "constant_columns": [c.name for c in diagnostics.column_stats if (c.distinct_count or 0) < 2],
        "outliers": outliers,
        "columns_with_outliers": len(outliers),
        "skewed_columns": skewed,
    }


def profile_numbers(profiles: dict, analysis_table) -> list:
    """The carried tables' numbers, from their profiles only (no data is read)."""
    return [{"name": name, "rows": p.row_count, "columns": p.column_count,
             "columns_with_missing": sum(1 for c in p.columns if c.missing_count > 0)}
            for name, p in (profiles or {}).items() if name != analysis_table]
