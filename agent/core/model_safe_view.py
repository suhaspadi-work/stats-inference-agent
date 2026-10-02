from __future__ import annotations
from dataclasses import dataclass, field
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


def assert_model_safe(obj: Any) -> None:
    """
    A cheap runtime guard: call this at the boundary right before anything
    is handed to the model, to catch a raw DataFrame/Series/list-of-rows
    being passed by mistake. Not a substitute for the type system, but a
    second line of defense during development.
    """
    forbidden_types = ("DataFrame", "Series", "ndarray")
    type_name = type(obj).__name__
    if type_name in forbidden_types:
        raise TypeError(
            f"Refusing to pass a {type_name} to the model — only ModelSafeView "
            f"is permitted. This is the privacy-tier boundary; build a "
            f"ModelSafeView from this object instead."
        )