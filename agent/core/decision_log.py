from __future__ import annotations
import json
from typing import Any

import numpy as np

from core.decision_record import DecisionRecord

# Raw data must never enter a decision record. Detected by type name, the same
# way assert_model_safe does, so pandas is not imported here.
_FORBIDDEN_TYPES = ("DataFrame", "Series", "ndarray")
_PLAIN_TYPES = (dict, list, str, int, float, bool, type(None))


def _to_plain(value: Any, decision_point: str) -> Any:
    """Copy value, turning numpy scalars into Python scalars and refusing raw data containers."""
    type_name = type(value).__name__
    if type_name in _FORBIDDEN_TYPES:
        raise ValueError(
            f"Decision record {decision_point!r} contains a {type_name}; raw data must never "
            f"enter a decision record. Store computed summary numbers instead."
        )
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {k: _to_plain(v, decision_point) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v, decision_point) for v in value]
    return value


def _first_unplain_type(value: Any) -> str | None:
    """Name the first type that is not plain JSON-safe Python, searching keys and values."""
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                return type(k).__name__
            found = _first_unplain_type(v)
            if found:
                return found
        return None
    if isinstance(value, list):
        for v in value:
            found = _first_unplain_type(v)
            if found:
                return found
        return None
    if not isinstance(value, _PLAIN_TYPES):
        return type(value).__name__
    return None


def log_update(*records: DecisionRecord) -> dict:
    """
    Build the state update a node returns to append decision records:
    {"decision_log": [plain dict per record, in order]}. AgentState merges
    decision_log with operator.add, so earlier records are never overwritten.

    Each dict is checked here, at creation: numpy scalars become Python
    scalars, raw data (ndarray, DataFrame, Series) is refused, and the result
    must pass json.dumps. The records themselves are not modified.
    """
    entries = []
    for record in records:
        if not isinstance(record, DecisionRecord):
            raise TypeError(
                f"log_update accepts only DecisionRecord objects, got {type(record).__name__}"
            )
        entry = _to_plain(record.to_dict(), record.decision_point)
        try:
            json.dumps(entry)
        except (TypeError, ValueError) as e:
            offending = _first_unplain_type(entry) or type(e).__name__
            raise ValueError(
                f"Decision record {record.decision_point!r} is not JSON-serializable: "
                f"contains a {offending} ({e})"
            ) from e
        entries.append(entry)
    return {"decision_log": entries}
