from __future__ import annotations
from pathlib import Path
from typing import Optional
import re
import pandas as pd

from core.dataset import Dataset, DatasetStore
from core.model_safe_view import ModelSafeView, ColumnSummary

MAX_TOP_VALUES = 5  # caps top_values so a low-cardinality column's frequency
                     # table stays a schema summary, not a copy of the column
HIGH_CARDINALITY_THRESHOLD = 20  # above this many distinct values, top_values is withheld entirely

def load_dataset(path: Path, session_id: str, name: str, store: DatasetStore) -> Dataset:
    """
    Read a raw uploaded file (CSV or Excel) and register it as version 1 of
    a Dataset, storing a normalized CSV copy in `store`. The original file's
    location and format are irrelevant to everything downstream — only the
    Dataset handle and the store matter from here on.
    """
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Expected .csv, .xlsx, or .xls.")

    storage_key = f"{session_id}/{name}_v1.csv"
    store.write(storage_key, df)

    return Dataset(
        session_id=session_id,
        name=name,
        version=1,
        storage_key=storage_key,
        parent_version=None,
        created_from_operation=None,
        original_filename=path.name,
    )


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _detect_format(series: pd.Series) -> Optional[str]:
    """
    Cheap heuristic pattern detection on a column's values — returns a
    DESCRIPTION of the pattern, never an example value. This helps the
    model reason about a column (e.g. "this needs date parsing") without
    ever seeing an actual value from it.
    """
    non_null = series.dropna()
    if non_null.empty:
        return None
    sample = non_null.astype(str).head(200)

    if sample.str.match(_EMAIL_RE).mean() > 0.8:
        return "looks like an email address"

    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    if parsed.notna().mean() > 0.8:
        return "looks like a date/datetime"

    return None


def profile_dataset(dataset: Dataset, store: DatasetStore) -> ModelSafeView:
    """
    The ONLY function permitted to read a Dataset's real rows. Everything it
    returns is a ModelSafeView: schema, types, missingness, capped top-value
    frequencies, and detected format patterns. Never raw rows, never a full
    column of values — this is the structural privacy boundary from Phase 1,
    now doing real work.
    """
    df = store.read(dataset.storage_key)

    columns = []
    for col in df.columns:
        series = df[col]
        missing_count = int(series.isna().sum())
        distinct_count = int(series.nunique(dropna=True))

           # top_values is only safe to populate for genuinely low-cardinality
        # columns (categories, flags) -- a high-cardinality column (emails,
        # names, free text, IDs) would leak real, near-unique raw values
        # through "top values" even at a small MAX_TOP_VALUES, since with
        # enough distinct values, "most frequent" starts to mean "arbitrary
        # individual record." This threshold is deliberately well above
        # MAX_TOP_VALUES itself, so showing a FEW frequent categories out of
        # a MODERATE number of distinct values is still fine (that's the
        # whole point of top_values), but a column that's mostly-unique
        # never gets its actual values surfaced at all.
        if distinct_count <= HIGH_CARDINALITY_THRESHOLD:
            value_counts = series.dropna().astype(str).value_counts().head(MAX_TOP_VALUES)
            top_values = tuple((str(v), int(c)) for v, c in value_counts.items())
        else:
            top_values = ()

        columns.append(ColumnSummary(
            name=str(col),
            dtype=str(series.dtype),
            missing_count=missing_count,
            missing_pct=round(missing_count / len(df), 4) if len(df) else 0.0,
            distinct_count=distinct_count,
            top_values=top_values,
            detected_format=_detect_format(series),
        ))

    return ModelSafeView(
        dataset_handle=dataset.handle,
        row_count=len(df),
        column_count=len(df.columns),
        columns=tuple(columns),
    )