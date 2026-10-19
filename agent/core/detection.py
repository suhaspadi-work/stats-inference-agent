from __future__ import annotations
import re
from dataclasses import dataclass, field

import pandas as pd

from core.dataset import Dataset, DatasetStore
from core.model_safe_view import ModelSafeView
from core.thresholds import NUMERIC_TEXT_PARSE_RATE

MISSINGNESS_THRESHOLD = 0.05  # flag a column if more than 5% of its values are missing


@dataclass(frozen=True)
class CandidateIssue:
    """
    A real, detected data-quality issue -- purely mechanical, no model
    judgment involved. This is the input to 7b's proposal step, which
    decides WHAT TO DO about each issue; this function only decides
    WHETHER an issue exists, using concrete thresholds against real numbers.
    """
    issue_type: str                      # "duplicate_rows" | "high_missingness" | "inconsistent_text" | "numeric_as_text"
    description: str
    affected_columns: tuple[str, ...]
    evidence: dict = field(default_factory=dict)


def detect_duplicate_rows(dataset: Dataset, store: DatasetStore, df=None) -> CandidateIssue | None:
    """Real row-level duplicate count -- not derivable from ModelSafeView alone, so this reads the actual data
    (or uses an already-read frame)."""
    df = store.read(dataset.storage_key) if df is None else df
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count == 0:
        return None
    return CandidateIssue(
        issue_type="duplicate_rows",
        description=f"{duplicate_count} exact duplicate row(s) detected across all columns.",
        affected_columns=tuple(df.columns),
        evidence={"duplicate_row_count": duplicate_count, "total_rows": len(df)},
    )


def detect_high_missingness(profile: ModelSafeView) -> list[CandidateIssue]:
    """Uses ONLY the ModelSafeView -- never touches raw data, consistent with the privacy boundary."""
    issues = []
    for col in profile.columns:
        if col.missing_pct > MISSINGNESS_THRESHOLD:
            issues.append(CandidateIssue(
                issue_type="high_missingness",
                description=f"Column '{col.name}' is {col.missing_pct:.1%} missing.",
                affected_columns=(col.name,),
                evidence={"missing_pct": col.missing_pct, "missing_count": col.missing_count},
            ))
    return issues


_WHITESPACE = re.compile(r"\s+")
_LEADING_ZERO = re.compile(r"^[-+]?0\d")


def _is_text(series: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _trim(value) -> str:
    return _WHITESPACE.sub(" ", str(value).strip())


def detect_inconsistent_text(df: pd.DataFrame) -> list[CandidateIssue]:
    """
    M2.3a step 2: a text column whose levels merge after trimming spaces and ignoring case, or whose values carry
    extra spaces. Only columns at or below HIGH_CARDINALITY_THRESHOLD distinct values are checked (a withheld
    column is never examined level by level). The issue holds counts only, never a level: its evidence reaches
    the proposer's prompt.
    """
    from core.ingestion import HIGH_CARDINALITY_THRESHOLD
    issues = []
    for column in df.columns:
        series = df[column]
        if not _is_text(series):
            continue
        values = series.dropna().astype(str)
        if not 1 <= values.nunique() <= HIGH_CARDINALITY_THRESHOLD:
            continue
        trimmed = values.map(_trim)
        normalized = trimmed.str.lower()
        evidence = {"levels_before": int(values.nunique()), "levels_after_trim": int(trimmed.nunique()),
                    "levels_after": int(normalized.nunique()), "cells_changed_trim": int((trimmed != values).sum()),
                    "cells_changed_full": int((normalized != values).sum())}
        merges = evidence["levels_after"] < evidence["levels_before"]
        if not merges and evidence["cells_changed_trim"] == 0:
            continue
        description = (f"Column '{column}' has {evidence['levels_before']} distinct values that reduce to "
                       f"{evidence['levels_after']} after trimming spaces and ignoring case." if merges else
                       f"Column '{column}' has {evidence['cells_changed_trim']} value(s) with leading, trailing or "
                       "repeated spaces.")
        issues.append(CandidateIssue(issue_type="inconsistent_text", description=description,
                                     affected_columns=(column,), evidence=evidence))
    return issues


def detect_numeric_as_text(df: pd.DataFrame) -> list[CandidateIssue]:
    """
    M2.3a step 2: a text column whose values are numbers (at least NUMERIC_TEXT_PARSE_RATE of the non-missing ones).
    A column with leading zeros looks like a code and is not flagged. Counts and ratios only.
    """
    issues = []
    for column in df.columns:
        series = df[column]
        if not _is_text(series):
            continue
        stripped = series.dropna().astype(str).str.strip()
        if len(stripped) < 10 or stripped.nunique() < 2:
            continue
        if stripped.str.match(_LEADING_ZERO).any():
            continue
        parsed = pd.to_numeric(stripped, errors="coerce")
        rate = float(parsed.notna().mean())
        if rate < NUMERIC_TEXT_PARSE_RATE:
            continue
        issues.append(CandidateIssue(
            issue_type="numeric_as_text",
            description=f"Column '{column}' is stored as text but {rate:.1%} of its values are numbers.",
            affected_columns=(column,),
            evidence={"non_missing": int(len(stripped)), "parse_rate": round(rate, 6),
                      "unparseable_count": int(parsed.isna().sum())},
        ))
    return issues


def detect_all_issues(dataset: Dataset, profile: ModelSafeView, store: DatasetStore) -> list[CandidateIssue]:
    """The single entry point 7b's proposal step will call. The data is read once."""
    df = store.read(dataset.storage_key)
    issues = []
    dup_issue = detect_duplicate_rows(dataset, store, df)
    if dup_issue:
        issues.append(dup_issue)
    issues.extend(detect_high_missingness(profile))
    issues.extend(detect_inconsistent_text(df))
    issues.extend(detect_numeric_as_text(df))
    return issues