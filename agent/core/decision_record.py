from __future__ import annotations
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Optional
import uuid


DECIDED_BY_VALUES = ("code_rule", "model", "human")
GENERATED_BY_VALUES = ("code", "model")


def _plain(obj: Any) -> Any:
    """Recursively convert tuples to lists and copy dicts, so the result is JSON-safe plain Python."""
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    return obj


def _check_required(d: dict, required: tuple[str, ...], type_name: str) -> None:
    missing = [k for k in required if k not in d]
    if missing:
        raise ValueError(f"{type_name}.from_dict: missing required key(s): {', '.join(missing)}")


def _known_keys(cls: type, d: dict) -> dict:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in names}


@dataclass(frozen=True)
class Candidate:
    """One option available at a decision point, with whether it was eligible and why."""
    option: str
    eligible: Optional[bool]       # None means eligibility was not determined
    reason: str                    # generated from evidence for statistical techniques, never canned
    evidence: dict[str, Any] = field(default_factory=dict)
    generated_by: str = "code"     # "code" or "model"

    def __post_init__(self):
        if self.generated_by not in GENERATED_BY_VALUES:
            raise ValueError(
                f"Candidate.generated_by must be one of {GENERATED_BY_VALUES}, got {self.generated_by!r}"
            )

    def to_dict(self) -> dict:
        return {
            "option": self.option,
            "eligible": self.eligible,
            "reason": self.reason,
            "evidence": _plain(self.evidence),
            "generated_by": self.generated_by,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Candidate:
        _check_required(d, ("option", "eligible", "reason"), "Candidate")
        return cls(**_known_keys(cls, d))


@dataclass(frozen=True)
class DecisionRecord:
    """
    The standard record for every decision a node makes (ARCHITECTURE section 4).
    A generalization of MethodDecision: which decision point, what was chosen
    (including "skipped"), the full candidate set with eligibility and reasons,
    the code-computed evidence, and who decided -- code, model, or a human,
    including whether a human overrode the original choice.
    """
    decision_point: str                            # stable name, e.g. "stage2.test_selection"
    chosen: str                                    # the option taken; "skipped" is valid
    decided_by: str                                # "code_rule", "model" or "human"
    candidates: tuple[Candidate, ...] = field(default_factory=tuple)
    rationale: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)  # numbers computed by code only
    overridden_by_human: bool = False
    original_choice: Optional[str] = None          # what was chosen before a human override
    chart_data: tuple[dict, ...] = field(default_factory=tuple)
    supersedes: Optional[str] = None               # id of the DecisionRecord this replaces, if any
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: int = 1

    def __post_init__(self):
        if not self.decision_point or not self.decision_point.strip():
            raise ValueError("DecisionRecord.decision_point must be a non-empty string")
        if not self.chosen or not self.chosen.strip():
            raise ValueError("DecisionRecord.chosen must be a non-empty string (use 'skipped' for a skip)")
        if self.decided_by not in DECIDED_BY_VALUES:
            raise ValueError(
                f"DecisionRecord.decided_by must be one of {DECIDED_BY_VALUES}, got {self.decided_by!r}"
            )
        if self.overridden_by_human:
            if self.decided_by != "human":
                raise ValueError(
                    "DecisionRecord.overridden_by_human is True, so decided_by must be 'human', "
                    f"got {self.decided_by!r}"
                )
            if self.original_choice is None:
                raise ValueError(
                    "DecisionRecord.overridden_by_human is True, so original_choice must be set"
                )

    def to_dict(self) -> dict:
        """Plain dicts, lists, strings, numbers, booleans and None only (safe for LangGraph state and the event log)."""
        return {
            "decision_point": self.decision_point,
            "chosen": self.chosen,
            "decided_by": self.decided_by,
            "candidates": [c.to_dict() for c in self.candidates],
            "rationale": self.rationale,
            "evidence": _plain(self.evidence),
            "overridden_by_human": self.overridden_by_human,
            "original_choice": self.original_choice,
            "chart_data": _plain(self.chart_data),
            "supersedes": self.supersedes,
            "id": self.id,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DecisionRecord:
        """Rebuild from to_dict output. Unknown keys are ignored; missing optional keys take their defaults."""
        _check_required(d, ("decision_point", "chosen", "decided_by"), "DecisionRecord")
        kwargs = _known_keys(cls, d)
        if "candidates" in kwargs:
            kwargs["candidates"] = tuple(Candidate.from_dict(c) for c in kwargs["candidates"])
        if "chart_data" in kwargs:
            kwargs["chart_data"] = tuple(kwargs["chart_data"])
        return cls(**kwargs)
