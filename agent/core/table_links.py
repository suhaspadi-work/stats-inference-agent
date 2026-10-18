"""
Candidate keys between tables (M2.2). Candidates come from the profiles only: columns are bucketed by normalized
name, so the cost is linear in the number of columns plus the pairs inside each bucket, never every pair of tables.
Exact numbers (overlap, orphans, cardinality, the inner-join row count) are computed with pandas only for the top
candidates. No link or number reaches a prompt.
"""
from __future__ import annotations

import re

from pandas.api.types import is_numeric_dtype

from core.capabilities import is_numeric_dtype_name

KEY_LIKE_RATIO = 0.9  # a key-like column has distinct non-missing values on at least 90% of its non-missing rows
TOP_LINKS = 20        # limits the computation and the display, never the number of tables
MAX_STORED_CANDIDATES = 200

_NON_ALNUM = re.compile(r"[^0-9a-z]")


def normalize_name(name) -> str:
    return _NON_ALNUM.sub("", str(name).lower())


def _ratio(distinct, non_missing):
    return distinct / non_missing if non_missing > 0 else 0.0


def candidate_links(profiles: dict) -> list:
    """Candidate key pairs between different tables, from the ModelSafeViews only, best first."""
    order = {name: i for i, name in enumerate(profiles)}
    buckets = {}
    for table, profile in profiles.items():
        for c in profile.columns:
            non_missing = profile.row_count - c.missing_count
            buckets.setdefault(normalize_name(c.name), []).append((table, c.name, str(c.dtype), c.distinct_count,
                                                                   non_missing))
    found = []
    for columns in buckets.values():
        if len(columns) < 2:
            continue
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                a, b = columns[i], columns[j]
                if a[0] == b[0]:
                    continue
                if order[a[0]] > order[b[0]]:
                    a, b = b, a
                if is_numeric_dtype_name(a[2]) != is_numeric_dtype_name(b[2]):
                    continue
                if a[2].startswith("float") or b[2].startswith("float"):
                    continue
                ratio_a, ratio_b = _ratio(a[3], a[4]), _ratio(b[3], b[4])
                key_a, key_b = a[4] > 0 and ratio_a >= KEY_LIKE_RATIO, b[4] > 0 and ratio_b >= KEY_LIKE_RATIO
                if not (key_a or key_b):
                    continue
                found.append({"table_a": a[0], "column_a": a[1], "table_b": b[0], "column_b": b[1],
                              "key_like_a": key_a, "key_like_b": key_b, "score": round(max(ratio_a, ratio_b), 6)})
    found.sort(key=lambda c: (-c["score"], c["table_a"], c["column_a"], c["table_b"], c["column_b"]))
    return found


def _cardinality(unique_a, unique_b) -> str:
    if unique_a and unique_b:
        return "one_to_one"
    if unique_a:
        return "one_to_many"
    if unique_b:
        return "many_to_one"
    return "many_to_many"


def _frame(dataset, store, frames):
    if frames is None:
        return store.read(dataset.storage_key)
    if dataset.storage_key not in frames:
        frames[dataset.storage_key] = store.read(dataset.storage_key)
    return frames[dataset.storage_key]


def link_numbers(dataset_a, column_a, dataset_b, column_b, store, frames=None) -> dict:
    """Plain numbers for one candidate key pair; keys are compared as given (missing keys ignored and counted).
    frames (optional) caches the tables by storage key, so a scan reads each table once."""
    a = _frame(dataset_a, store, frames)[column_a]
    b = _frame(dataset_b, store, frames)[column_b]
    keys_a, keys_b = a.dropna(), b.dropna()
    counts_a, counts_b = keys_a.value_counts(), keys_b.value_counts()
    matched = counts_a.index.intersection(counts_b.index)
    inner = int((counts_a[matched] * counts_b[matched]).sum()) if len(matched) else 0
    unique_a, unique_b = bool((counts_a <= 1).all()), bool((counts_b <= 1).all())
    if not (is_numeric_dtype(a) and is_numeric_dtype(b)):  # text keys (pandas 3 uses the str dtype)
        norm_a = {str(k).strip().lower() for k in counts_a.index}
        norm_b = {str(k).strip().lower() for k in counts_b.index}
        matched_normalized = len(norm_a & norm_b)
    else:
        matched_normalized = len(matched)
    rows_a = len(a)
    return {
        "rows_a": rows_a, "rows_b": len(b),
        "missing_a": int(a.isna().sum()), "missing_b": int(b.isna().sum()),
        "distinct_a": int(len(counts_a)), "distinct_b": int(len(counts_b)),
        "matched_distinct": int(len(matched)),
        "unique_a": unique_a, "unique_b": unique_b,
        "cardinality": _cardinality(unique_a, unique_b),
        "orphan_rows_a": int((~keys_a.isin(counts_b.index)).sum()),
        "orphan_rows_b": int((~keys_b.isin(counts_a.index)).sum()),
        "inner_join_rows": inner,
        "fan_out": round(inner / rows_a, 3) if rows_a else None,
        "matched_distinct_normalized": int(matched_normalized),
        "normalizing_helps": matched_normalized > len(matched),
    }


def scan_links(tables: dict, profiles: dict, store, top_n=None) -> dict:
    """The candidates (at most 200 stored, with the true total) and the exact numbers of the first top_n (TOP_LINKS,
    read at call time, when not given). Each table of a computed pair is read once."""
    top_n = TOP_LINKS if top_n is None else top_n
    candidates = candidate_links(profiles)
    frames = {}
    computed = [{**c, **link_numbers(tables[c["table_a"]], c["column_a"], tables[c["table_b"]], c["column_b"], store,
                                     frames)}
                for c in candidates[:top_n]]
    return {"candidates": candidates[:MAX_STORED_CANDIDATES], "candidates_total": len(candidates),
            "computed": computed, "not_computed": len(candidates) - len(computed)}
