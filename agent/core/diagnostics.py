"""
Diagnostics (M0.6, G16): the same statistics engine as the descriptive capability, without charts. It is
the data check a later stage reads before choosing an analysis, not a report for the user. Outlier and
identifier flags and cross-table checks are M2.2.
"""
from __future__ import annotations
from dataclasses import dataclass

from core.dataset import Dataset, DatasetStore
from core.descriptive import ColumnDescriptiveStats, column_stats_to_dicts, compute_column_stats, compute_correlation


@dataclass(frozen=True)
class ColumnMissing:
    name: str
    missing_count: int
    missing_pct: float


@dataclass(frozen=True)
class DiagnosticsResult:
    row_count: int
    column_count: int
    column_stats: tuple[ColumnDescriptiveStats, ...]
    missing_by_column: tuple[ColumnMissing, ...]
    duplicate_row_count: int
    correlation_matrix: dict | None
    schema_version: int = 1


def compute_diagnostics(dataset: Dataset, store: DatasetStore) -> DiagnosticsResult:
    """Reads the real dataset once. The duplicate count is the same expression detect_duplicate_rows uses."""
    df = store.read(dataset.storage_key)
    return DiagnosticsResult(
        row_count=len(df),
        column_count=len(df.columns),
        column_stats=compute_column_stats(df),
        missing_by_column=tuple(
            ColumnMissing(name=c, missing_count=int(df[c].isna().sum()), missing_pct=float(df[c].isna().mean()))
            for c in df.columns
        ),
        duplicate_row_count=int(df.duplicated().sum()),
        correlation_matrix=compute_correlation(df),
    )


def diagnostics_to_dict(result: DiagnosticsResult) -> dict:
    """The plain dict stored in state; it has no chart_data key."""
    return {
        "row_count": result.row_count,
        "column_count": result.column_count,
        "column_stats": column_stats_to_dicts(result.column_stats),
        "missing_by_column": [
            {"name": m.name, "missing_count": m.missing_count, "missing_pct": m.missing_pct}
            for m in result.missing_by_column
        ],
        "duplicate_row_count": result.duplicate_row_count,
        "correlation_matrix": result.correlation_matrix,
        "schema_version": result.schema_version,
    }
