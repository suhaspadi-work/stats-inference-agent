from __future__ import annotations
import json
import math
import re
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Type, TypeVar

T = TypeVar("T")


_NON_FINITE = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


def _tag_float(value: float):
    """Non-finite floats become {"$float": "nan" | "inf" | "-inf"}, so every line is strict JSON."""
    if math.isnan(value):
        return {"$float": "nan"}
    if math.isinf(value):
        return {"$float": "inf" if value > 0 else "-inf"}
    return value


def _serialize(obj: Any) -> Any:
    """Recursively convert dataclasses/Enums/tuples into JSON-safe plain Python."""
    if isinstance(obj, float):
        return _tag_float(obj)
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


def _restore(obj: Any) -> Any:
    """Undo _tag_float: a dict that is exactly {"$float": "nan" | "inf" | "-inf"} becomes that float."""
    if isinstance(obj, dict):
        if len(obj) == 1 and obj.get("$float") in _NON_FINITE:
            return _NON_FINITE[obj["$float"]]
        return {k: _restore(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore(v) for v in obj]
    return obj


_UNSAFE_ID_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def events_path(base_dir: Path, thread_id: str) -> Path:
    """
    base_dir / <safe thread id> / "events.jsonl". Characters outside A-Z a-z 0-9 . _ - become "_";
    ".", ".." and an id that is empty after that become "_". Creates no directories.
    """
    if not isinstance(thread_id, str) or not thread_id:
        raise ValueError(f"thread_id must be a non-empty string, got {thread_id!r}")
    safe = _UNSAFE_ID_CHARS.sub("_", thread_id)
    if safe in ("", ".", ".."):
        safe = "_"
    return Path(base_dir) / safe / "events.jsonl"


def read_decision_log(path: Path) -> list[dict]:
    """The decision records in an event log file, in file order, with non-finite floats restored."""
    path = Path(path)
    if not path.exists():
        return []
    return EventLog(log_path=path).read_by_type("decision")


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

    def append(self, record_type: str, record: Any, meta: dict | None = None) -> None:
        """
        record_type is a plain string tag, e.g. 'dataset', 'operation', 'plan'. meta (who wrote the
        entry, for example) is stored only when given, so entries without it are unchanged.
        Lines are strict JSON: allow_nan=False raises on any non-finite float that was not tagged.
        """
        entry = {"record_type": record_type, "data": _serialize(record)}
        if meta is not None:
            entry["meta"] = _serialize(meta)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, allow_nan=False) + "\n")

    def read_all(self) -> list[dict]:
        """Return every logged entry as raw {record_type, data} dicts, in append order."""
        if not self.log_path.exists():
            return []
        entries = []
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(_restore(json.loads(line)))
        return entries

    def read_by_type(self, record_type: str) -> list[dict]:
        """Return only the 'data' dicts for entries matching record_type."""
        return [e["data"] for e in self.read_all() if e["record_type"] == record_type]