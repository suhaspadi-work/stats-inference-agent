from __future__ import annotations
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSafeView:
    """
    The ONLY shape of data a tool is permitted to return to the model. This
    is the structural enforcement of the middle privacy tier: schema, types,
    missingness, and value frequencies/format patterns — never raw rows.

    There is deliberately no field here that could hold a raw row or a full
    column of values. If a future tool wants to show the model more, that is
    a conscious decision to add a new, explicitly-named field here — not an
    accident of passing a dataframe through.
    """
    dataset_handle: str
    row_count: int
    column_count: int
    columns: tuple["ColumnSummary", ...]
    schema_version: int = 1


@dataclass(frozen=True)
class ColumnSummary:
    """Per-column summary info — value frequencies and patterns, never raw values."""
    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    distinct_count: int
    # Top value frequencies as (value_as_string, count) — a small, bounded
    # sample of the MOST COMMON values, not the actual column. Capped at a
    # small N so this can never become "basically the whole column" for a
    # low-cardinality field the way returning all distinct values could.
    top_values: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    # e.g. "looks like MM/DD/YYYY", "looks like an email address" — a pattern
    # description, never an actual value from the column
    detected_format: str | None = None
    schema_version: int = 1


_FORBIDDEN_TYPES = ("DataFrame", "Series", "ndarray")
_MAX_DEPTH = 8


def _forbidden_type_error(type_name: str, path: str) -> TypeError:
    where = f" (at {path})" if path else ""
    return TypeError(
        f"Refusing to pass a {type_name} to the model — only ModelSafeView "
        f"is permitted. This is the privacy-tier boundary; build a "
        f"ModelSafeView from this object instead.{where}"
    )


def _walk(obj: Any, path: str, depth: int, seen: set) -> None:
    type_name = type(obj).__name__
    if type_name in _FORBIDDEN_TYPES:
        raise _forbidden_type_error(type_name, path)
    if depth >= _MAX_DEPTH or isinstance(obj, (str, bytes, int, float, bool)) or obj is None:
        return
    if id(obj) in seen:
        return
    if isinstance(obj, dict):
        seen.add(id(obj))
        if "chart_type" in obj and "data" in obj:
            raise ValueError(f"Refusing to pass a chart payload ({obj.get('chart_type')!r}) to the model at {path}.")
        for key, value in obj.items():
            child = f"{path}.{key}" if path else str(key)
            if key == "chart_data" and value:
                raise ValueError(f"Refusing to pass chart data to the model at {child}.")
            _walk(value, child, depth + 1, seen)
    elif isinstance(obj, (list, tuple)):
        seen.add(id(obj))
        for i, value in enumerate(obj):
            _walk(value, f"{path}[{i}]", depth + 1, seen)
    elif is_dataclass(obj) and not isinstance(obj, type):
        seen.add(id(obj))
        for f in fields(obj):
            value = getattr(obj, f.name)
            child = f"{path}.{f.name}" if path else f.name
            if f.name == "chart_data" and value:
                raise ValueError(f"Refusing to pass chart data to the model at {child}.")
            _walk(value, child, depth + 1, seen)


def assert_model_safe(obj: Any, name: str = "") -> None:
    """
    A cheap runtime guard: call this at the boundary right before anything
    is handed to the model, to catch a raw DataFrame/Series/list-of-rows
    being passed by mistake. Not a substitute for the type system, but a
    second line of defense during development.

    Since M0.5 it also walks dicts, lists, tuples and dataclass fields (8 levels deep,
    cycle-safe) and raises ValueError, naming the path, for any chart payload (a dict with
    both "chart_type" and "data") or any non-empty "chart_data". Empty chart_data passes.
    """
    type_name = type(obj).__name__
    if type_name in _FORBIDDEN_TYPES:
        raise _forbidden_type_error(type_name, "")
    _walk(obj, name, 0, set())
