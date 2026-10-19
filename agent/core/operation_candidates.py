"""
Code-enumerated cleaning candidates and their dry run (M2.3a). For each detected issue the code lists every
operation that could fix it, with its eligibility, the reason, and a preview of its impact in plain numbers,
computed by running the registered transform on the in-memory frame (nothing is written). The model only chooses
a candidate. Preview numbers are diagnostics: they never reach a prompt.
"""
from __future__ import annotations

import pandas as pd

from core.capabilities import run_transform
from core.ingestion import unique_table_name
from core.thresholds import (
    DISCRETE_MAX_DISTINCT, GROUP_MAX_LEVELS, GROUP_MIN_OBSERVED, IMPUTE_MAX_MISSING_PCT, IMPUTE_MIN_OBSERVED,
    MAX_GROUP_COLUMNS,
)
from core.wrangling import describe_operation, normalize_key

LEAVE = "leave_as_is"


def _frame(dataset, store_or_frame) -> pd.DataFrame:
    return store_or_frame if isinstance(store_or_frame, pd.DataFrame) else store_or_frame.read(dataset.storage_key)


def _float(x):
    return None if x is None else float(x)


def dry_run(dataset, store_or_frame, op_type: str, params: dict, frames=None) -> dict:
    """The impact preview of one operation: the same transform as the execution, on an in-memory frame. frames
    (table name -> DataFrame) holds the other tables a join reads."""
    return _public(_dry_run(dataset, store_or_frame, op_type, params, frames))


def _dry_run(dataset, store_or_frame, op_type: str, params: dict, frames=None) -> dict:
    """dry_run plus the transform's own details (for the plain-words sentence)."""
    df = _frame(dataset, store_or_frame)
    new_df, details, warnings = run_transform(op_type, df, params, frames)
    rows_before, rows_after = len(df), len(new_df)
    shift, cells = None, 0
    if op_type in ("dedupe", "filter"):
        rows_changed = rows_before - rows_after
        cells = rows_changed * len(df.columns)  # the cells of the removed rows
        affected = list(params.get("subset") or df.columns) if op_type == "dedupe" else [params["column"]]
    elif op_type == "impute":
        rows_changed = cells = int(details["cells_filled"])
        affected = [params["column"]] + ([details["indicator"]] if details.get("indicator") else [])
        if details.get("mean_before") is not None:
            shift = {"column": details["column"], "mean_before": _float(details["mean_before"]),
                     "mean_after": _float(details["mean_after"]), "std_before": _float(details["std_before"]),
                     "std_after": _float(details["std_after"])}
    elif op_type == "drop_column":
        rows_changed, cells, affected = 0, rows_before, [params["column"]]  # every cell of the dropped column
    elif op_type == "normalize_text":
        rows_changed = cells = int(details["cells_changed"])
        affected = [params["column"]]
    elif op_type == "join":  # M2.3b: rows removed by an inner join; the cells left missing by unmatched rows
        rows_changed = rows_before - rows_after
        cells, affected = int(details["new_missing_cells"]), list(details["columns_added"])
    elif op_type == "cast" and params.get("to_type") == "numeric":
        rows_changed, cells, affected = 0, int(details["non_missing_before"]), [params["column"]]
        if details["became_missing"]:
            warnings = (f"{details['became_missing']} value(s) that are not numbers become missing",)
    elif op_type == "cast":
        column = params["column"]
        old, new = df[column], new_df[column]
        became_missing = int((old.notna() & new.isna()).sum())
        changed_type = int(new.notna().sum()) if str(old.dtype) != str(new.dtype) else 0
        rows_changed, cells, affected = 0, became_missing + changed_type, [column]
    else:  # rename
        rows_changed, cells, affected = 0, 0, list(params.get("mapping") or {})
    return {
        "rows_before": rows_before, "rows_after": rows_after,
        "columns_before": len(df.columns), "columns_after": len(new_df.columns),
        "rows_changed": int(rows_changed),
        "share_changed": round(rows_changed / rows_before, 4) if rows_before else 0.0,
        "columns_affected": [str(c) for c in affected], "cells_changed": int(cells),
        "shift": shift, "warnings": list(warnings),
        "details": details,
    }


def _zero_preview(df) -> dict:
    return {"rows_before": len(df), "rows_after": len(df), "columns_before": len(df.columns),
            "columns_after": len(df.columns), "rows_changed": 0, "share_changed": 0.0, "columns_affected": [],
            "cells_changed": 0, "shift": None, "warnings": [], "details": {}}


def _candidate(df, dataset, cid, op_type, params, eligible, reason, frames=None) -> dict:
    if not eligible:
        return {"id": cid, "op_type": op_type, "params": params, "eligible": False, "reason": reason,
                "what_was_run": describe_operation(op_type, params), "preview": None}
    if op_type is None:
        preview = _zero_preview(df)
    else:
        preview = _dry_run(dataset, df, op_type, params, frames)
    words = {**preview["details"]}
    if op_type in ("dedupe", "filter"):
        words["rows_removed"] = preview["rows_changed"]
    return {"id": cid, "op_type": op_type, "params": params, "eligible": True, "reason": reason,
            "what_was_run": describe_operation(op_type, params, words), "preview": _public(preview)}


def _public(preview) -> dict:
    """The preview without the transform's raw details (they are folded into what_was_run and shift)."""
    return {k: v for k, v in preview.items() if k != "details"}


def _is_numeric(series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _imputable(series) -> str | None:
    """The reason an imputation is not offered for this column, or None when it is."""
    if series.isna().mean() > IMPUTE_MAX_MISSING_PCT:
        return f"more than {IMPUTE_MAX_MISSING_PCT:.0%} of the values are missing"
    if int(series.notna().sum()) < IMPUTE_MIN_OBSERVED:
        return f"fewer than {IMPUTE_MIN_OBSERVED} observed values"
    return None


def _group_columns(df, column) -> list:
    pool = [c for c in df.columns if c != column and not _is_numeric(df[c])
            and 2 <= df[c].nunique(dropna=True) <= GROUP_MAX_LEVELS]
    order = {c: i for i, c in enumerate(df.columns)}
    return sorted(pool, key=lambda c: (df[c].nunique(dropna=True), order[c]))[:MAX_GROUP_COLUMNS]


def _missingness_candidates(df, dataset, column) -> list:
    series = df[column]
    numeric = _is_numeric(series)
    pct = float(series.isna().mean())
    blocked = _imputable(series)
    out = [
        _candidate(df, dataset, LEAVE, None, {}, True,
                   "keep the missing values; the analysis leaves those rows out where it needs the column"),
        _candidate(df, dataset, "delete_rows", "filter", {"column": column, "condition": "not_null", "value": None},
                   True, f"removes the rows where {column} is missing ({pct:.1%} of the rows)"),
    ]
    if len(df.columns) > 1:
        out.append(_candidate(df, dataset, "drop_column", "drop_column", {"column": column}, True,
                              f"removes the column {column} and keeps every row"))
    else:
        out.append(_candidate(df, dataset, "drop_column", "drop_column", {"column": column}, False,
                              f"{column} is the only column"))
    for strategy in ("mean", "median"):
        reason = "needs a numeric column" if not numeric else blocked
        out.append(_candidate(df, dataset, f"impute_{strategy}", "impute", {"column": column, "strategy": strategy},
                              reason is None, reason or f"fills the missing values with the column's {strategy}"))
    reason = blocked
    if reason is None and numeric and series.nunique(dropna=True) > DISCRETE_MAX_DISTINCT:
        reason = "the column is continuous; a mode would be arbitrary"
    out.append(_candidate(df, dataset, "impute_mode", "impute", {"column": column, "strategy": "mode"}, reason is None,
                          reason or "fills the missing values with the most frequent value"))
    out.append(_candidate(df, dataset, "fill_missing_category", "impute",
                          {"column": column, "strategy": "constant", "value": "missing"}, not numeric,
                          "needs a text column" if numeric else "marks the missing values with their own label 'missing'"))
    if numeric:
        for group in _group_columns(df, column):
            reason = blocked
            if reason is None and df[group].isna().any():
                reason = f"the group column {group} has missing values"
            if reason is None:
                observed = series.notna().groupby(df[group]).sum()
                if int(observed.min()) < GROUP_MIN_OBSERVED:
                    reason = f"a group of {group} has fewer than {GROUP_MIN_OBSERVED} observed values of {column}"
            out.append(_candidate(df, dataset, f"impute_group_median_{group}", "impute",
                                  {"column": column, "strategy": "median", "group_by": group}, reason is None,
                                  reason or f"fills each missing value with the median of its {group} group"))
    reason = "needs a numeric column" if not numeric else blocked
    if reason is None and f"{column}_was_missing" in df.columns:
        reason = f"the column {column}_was_missing already exists"
    out.append(_candidate(df, dataset, "impute_median_indicator", "impute",
                          {"column": column, "strategy": "median", "indicator": True}, reason is None,
                          reason or "fills with the median and adds a column marking which values were missing"))
    return out


def _changes(df, dataset, op_type, params) -> bool:
    return _dry_run(dataset, df, op_type, params)["cells_changed"] > 0


def _text_candidates(df, dataset, column) -> list:
    trim = {"column": column, "lowercase": False}
    lower = {"column": column, "lowercase": True}
    trims, lowers = _changes(df, dataset, "normalize_text", trim), _changes(df, dataset, "normalize_text", lower)
    return [
        _candidate(df, dataset, LEAVE, None, {}, True, "keep the values as they are"),
        _candidate(df, dataset, "trim_text", "normalize_text", trim, trims,
                   "removes extra spaces" if trims else "no value has extra spaces"),
        _candidate(df, dataset, "trim_and_lowercase", "normalize_text", lower, lowers,
                   "removes extra spaces and treats upper and lower case as the same" if lowers else "no value changes"),
    ]


def _numeric_text_candidates(df, dataset, column) -> list:
    codes = df[column].dropna().astype(str).str.strip().str.match(r"^[-+]?0\d").any()
    return [
        _candidate(df, dataset, LEAVE, None, {}, True, "keep the values as they are"),
        _candidate(df, dataset, "convert_to_number", "cast", {"column": column, "to_type": "numeric"}, not codes,
                   "values with leading zeros look like codes, not numbers" if codes
                   else "the values that are numbers are converted; the others become missing"),
    ]


def _join_candidates(df, dataset, issue, tables, frames) -> list:
    """
    M2.3b: the join candidates of one joinable_table issue. A join that would multiply rows (a right key that
    repeats) is never eligible; join_left_normalized only when comparing after trimming spaces and ignoring case
    matches more distinct keys and the right key stays unique.
    """
    ev = issue.evidence
    right_name, left_key, right_key = ev["right_table"], ev["left_key"], ev["right_key"]
    right = frames[right_name]
    left_keys = set(df[left_key].dropna().tolist())
    right_keys = right[right_key].dropna()
    repeats = bool(right_keys.duplicated().any())
    right_set = set(right_keys.tolist())
    exact = sum(1 for k in left_keys if k in right_set)
    normalized_right = right_keys.map(normalize_key)
    normalized_set = set(normalized_right.tolist())
    normalized = sum(1 for k in left_keys if normalize_key(k) in normalized_set)
    normalized_repeats = bool(normalized_right.duplicated().any())
    base = dataset.name  # a derived analysis table's name is used as is
    params = {"right_table": right_name, "left_key": left_key, "right_key": right_key, "how": "left",
              "normalize_keys": False, "result_name": unique_table_name(f"{base}_{right_name}", set(tables or {}))}
    blocked = (f"the key {right_key} repeats in {right_name}: a join would multiply rows" if repeats
               else "no key value matches" if exact == 0 else None)
    if normalized <= exact:
        normalized_blocked = "comparing after trimming spaces and ignoring case matches no more keys"
    elif normalized_repeats:
        normalized_blocked = f"the key repeats in {right_name} once its spaces and case are ignored"
    else:
        normalized_blocked = None
    return [
        _candidate(df, dataset, LEAVE, None, {}, True, "do not join; the table stays carried along"),
        _candidate(df, dataset, "join_left", "join", params, blocked is None,
                   blocked or (f"keeps every row of the analysis table; rows whose key is not found in {right_name} "
                               "get missing values in the added columns"), frames),
        _candidate(df, dataset, "join_inner", "join", {**params, "how": "inner"}, blocked is None,
                   blocked or f"keeps only the rows whose key is found in {right_name}", frames),
        _candidate(df, dataset, "join_left_normalized", "join", {**params, "normalize_keys": True},
                   normalized_blocked is None,
                   normalized_blocked or "compares the keys after trimming spaces and ignoring case, which matches more "
                                         "rows", frames),
    ]


def enumerate_candidates(issue, dataset, profile, store, df=None, tables=None, frames=None) -> list:
    """Every candidate fix for one issue, JSON-clean; reads the data at most once (pass df to read none). M2.3b: a
    joinable_table issue also reads its right table from tables (once, cached in frames when given)."""
    df = df if df is not None else store.read(dataset.storage_key)
    if issue.issue_type == "joinable_table":
        frames = frames if frames is not None else {}
        right_name = issue.evidence["right_table"]
        if right_name not in frames:
            frames[right_name] = store.read(tables[right_name].storage_key)
        return _join_candidates(df, dataset, issue, tables, frames)
    if issue.issue_type == "duplicate_rows":
        return [
            _candidate(df, dataset, "dedupe_all", "dedupe", {"subset": None}, True, "exact duplicate rows were found"),
            _candidate(df, dataset, LEAVE, None, {}, True, "keep the rows as they are; repeated events can be legitimate"),
        ]
    if issue.issue_type == "high_missingness":
        return _missingness_candidates(df, dataset, issue.affected_columns[0])
    if issue.issue_type == "inconsistent_text":
        return _text_candidates(df, dataset, issue.affected_columns[0])
    if issue.issue_type == "numeric_as_text":
        return _numeric_text_candidates(df, dataset, issue.affected_columns[0])
    return [_candidate(df, dataset, LEAVE, None, {}, True, "no fix is registered for this issue type")]


def recompute_candidates(candidates, dataset, frame, basis: str, frames=None) -> list:
    """
    M2.3a: the eligible candidates' previews (and plain words) recomputed on frame, the data as it will be when
    this operation runs; every preview gains "basis". Eligibility is not re-decided. A dry run that raises keeps
    the earlier preview.
    """
    out = []
    for c in candidates:
        c = dict(c)
        if c["eligible"]:
            try:
                fresh = _candidate(frame, dataset, c["id"], c["op_type"], c["params"], True, c["reason"], frames)
                c["preview"], c["what_was_run"] = {**fresh["preview"], "basis": basis}, fresh["what_was_run"]
            except ValueError:
                if c["preview"] is not None:
                    c["preview"] = {**c["preview"], "basis": c["preview"].get("basis", "uploaded_data")}
        out.append(c)
    return out
