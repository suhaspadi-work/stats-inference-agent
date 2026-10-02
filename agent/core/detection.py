from __future__ import annotations
from dataclasses import dataclass, field

from core.dataset import Dataset, DatasetStore
from core.model_safe_view import ModelSafeView

MISSINGNESS_THRESHOLD = 0.05  # flag a column if more than 5% of its values are missing


@dataclass(frozen=True)
class CandidateIssue:
    """
    A real, detected data-quality issue -- purely mechanical, no model
    judgment involved. This is the input to 7b's proposal step, which
    decides WHAT TO DO about each issue; this function only decides
    WHETHER an issue exists, using concrete thresholds against real numbers.
    """
    issue_type: str                      # "duplicate_rows" | "high_missingness"
    description: str
    affected_columns: tuple[str, ...]
    evidence: dict = field(default_factory=dict)


def detect_duplicate_rows(dataset: Dataset, store: DatasetStore) -> CandidateIssue | None:
    """Real row-level duplicate count -- not derivable from ModelSafeView alone, so this reads the actual data."""
    df = store.read(dataset.storage_key)
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


def detect_all_issues(dataset: Dataset, profile: ModelSafeView, store: DatasetStore) -> list[CandidateIssue]:
    """The single entry point 7b's proposal step will call."""
    issues = []
    dup_issue = detect_duplicate_rows(dataset, store)
    if dup_issue:
        issues.append(dup_issue)
    issues.extend(detect_high_missingness(profile))
    return issues