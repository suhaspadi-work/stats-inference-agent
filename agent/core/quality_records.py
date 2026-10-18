"""
The audit records of the diagnostics engine (M2.2). Pure: no graph, no model calls, no import of core.nodes. They
record numbers a deterministic step computed, never a cell value.

  stage1.quality.snapshot   the data-quality numbers of the analysis table, before or after cleaning
  stage1.links.scan         candidate keys between tables, found by shared column names
"""
from __future__ import annotations

import copy

from core.decision_record import Candidate, DecisionRecord

QUALITY_SNAPSHOT_POINT = "stage1.quality.snapshot"
LINKS_POINT = "stage1.links.scan"

_RATIONALES = {"before": "the data-quality numbers of the analysis table before cleaning",
               "after": "the data-quality numbers of the analysis table after cleaning"}


def quality_snapshot_record(label: str, snapshot: dict, carried=None) -> DecisionRecord:
    if label not in _RATIONALES:
        raise ValueError(f"Unknown quality snapshot label '{label}'.")
    other = "after" if label == "before" else "before"
    evidence = {**copy.deepcopy(snapshot), "label": label}
    if label == "before":
        evidence["carried"] = [dict(c) for c in (carried or [])]
    return DecisionRecord(
        decision_point=QUALITY_SNAPSHOT_POINT, chosen=label, decided_by="code_rule",
        candidates=(
            Candidate(option=label, eligible=True, reason=f"chosen: {_RATIONALES[label]}"),
            Candidate(option=other, eligible=False, reason=f"not this step: {_RATIONALES[other]}"),
        ),
        rationale=_RATIONALES[label],
        evidence=evidence,
    )


def links_record(scan: dict) -> DecisionRecord:
    found = scan["candidates_total"] > 0
    return DecisionRecord(
        decision_point=LINKS_POINT, chosen="found" if found else "none", decided_by="code_rule",
        candidates=(
            Candidate(option="found", eligible=found, reason="tables share key-like columns by name"),
            Candidate(option="none", eligible=not found, reason="no shared key columns between the tables"),
        ),
        rationale=("candidate keys between tables, found by shared column names" if found
                   else "no shared key columns between the tables"),
        evidence=copy.deepcopy(scan),
    )
