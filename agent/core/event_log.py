from __future__ import annotations
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Type, TypeVar

T = TypeVar("T")


def _serialize(obj: Any) -> Any:
    """Recursively convert dataclasses/Enums/tuples into JSON-safe plain Python."""
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


class EventLog:
    """
    An append-only JSON Lines log for Dataset, Operation, and AnalysisPlan
    records. Each line is one record, tagged with its type so it can be
    correctly reconstructed later. This is what lets an analysis session
    survive beyond a single process, and what makes the audit trail
    (MethodDecision history, WranglingRationale, Deviations) something a
    reviewer can actually load back and inspect after the fact.

    v1 implementation: a single file per session. Not a database — no
    querying beyond "read everything and filter in Python" — but honest
    about that limit, and swappable behind this same interface later.
    """

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record_type: str, record: Any) -> None:
        """record_type is a plain string tag, e.g. 'dataset', 'operation', 'plan'."""
        entry = {"record_type": record_type, "data": _serialize(record)}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def read_all(self) -> list[dict]:
        """Return every logged entry as raw {record_type, data} dicts, in append order."""
        if not self.log_path.exists():
            return []
        entries = []
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def read_by_type(self, record_type: str) -> list[dict]:
        """Return only the 'data' dicts for entries matching record_type."""
        return [e["data"] for e in self.read_all() if e["record_type"] == record_type]