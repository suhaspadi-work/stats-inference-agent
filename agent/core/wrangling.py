"""
Cleaning operations (M2.3a). Every operation is a PURE transform(df, params) -> (new_df, details, warnings) that
never touches the store; each execute_* reads the frame, runs its transform and writes through _execute. The
dry-run preview (core/operation_candidates.py) runs the same transform, so a preview and the execution agree.
details carry numbers and column names only (plus an aggregate fill value or a constant label).
"""
from __future__ import annotations

import math
import re

import pandas as pd

from core.dataset import Dataset, DatasetStore
from core.ingestion import HIGH_CARDINALITY_THRESHOLD
from core.operation import (
    Operation, OperationImpact, ApprovalStatus, WranglingRationale,
)

FILTER_CONDITIONS = {"equals", "not_equals", "greater_than", "less_than", "is_null", "not_null"}
IMPUTE_STRATEGIES = ("mean", "median", "mode", "constant")


def _propose(op_type: str, dataset: Dataset, params: dict, rationale: WranglingRationale) -> Operation:
    """Shared helper: every wrangling function proposes an Operation the same way."""
    return Operation(
        op_type=op_type,
        session_id=dataset.session_id,
        input_handles=(dataset.handle,),
        params=params,
        approval_status=ApprovalStatus.PENDING,
        rationale=rationale,
    )


def _execute(dataset: Dataset, store: DatasetStore, operation: Operation, new_df: pd.DataFrame, old_df: pd.DataFrame,
             details: dict | None = None, warnings: tuple = ()) -> tuple[Dataset, Operation]:
    """
    Shared helper: write the new data, produce the next Dataset version, and
    record the impact on the Operation. Only called after approval, per the
    Operation lifecycle from Phase 1.
    """
    new_storage_key = f"{dataset.session_id}/{dataset.name}_v{dataset.version + 1}.csv"
    store.write(new_storage_key, new_df)
    new_dataset = dataset.next_version(new_storage_key=new_storage_key, operation_id=operation.id)

    impact = OperationImpact(
        rows_before=len(old_df), rows_after=len(new_df),
        columns_before=len(old_df.columns), columns_after=len(new_df.columns),
        warnings=tuple(warnings), details=dict(details or {}),
    )
    finished_operation = operation.with_result(impact=impact, output_handle=new_dataset.handle)
    return new_dataset, finished_operation


def _run(dataset: Dataset, store: DatasetStore, operation: Operation, transform) -> tuple[Dataset, Operation]:
    """Read, transform, write. A transform's ValueError or TypeError marks the operation FAILED."""
    if operation.approval_status != ApprovalStatus.APPROVED:
        raise ValueError("Operation must be APPROVED before execution.")
    df = store.read(dataset.storage_key)
    try:
        new_df, details, warnings = transform(df, operation.params)
    except (ValueError, TypeError) as e:
        return dataset, operation.mark_failed(str(e))
    return _execute(dataset, store, operation, new_df, df, details, warnings)


# --- transforms ---------------------------------------------------------------------------------------------------

def transform_cast(df: pd.DataFrame, params: dict):
    column, to_type = params["column"], params["to_type"]
    if to_type == "numeric":
        return _cast_numeric(df, column)
    new_df = df.copy()
    try:
        new_df[column] = new_df[column].astype(to_type)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cast failed: {e}") from e
    return new_df, {}, ()


def _cast_numeric(df: pd.DataFrame, column):
    """M2.3a step 2: a lenient conversion to numbers; a value that is not a number becomes missing (never raises)."""
    if column not in df.columns:
        raise ValueError(f"Cast failed: column not found: {column}")
    new_df = df.copy()
    series = df[column]
    present = series.notna()
    converted = pd.to_numeric(series[present].astype(str).str.strip(), errors="coerce")
    new_df[column] = pd.to_numeric(converted.reindex(series.index), errors="coerce")
    details = {"column": str(column), "to_type": "numeric", "non_missing_before": int(present.sum()),
               "became_missing": int(converted.isna().sum())}
    return new_df, details, ()


_SPACES = re.compile(r"\s+")


def normalize_key(value) -> str:
    """M2.3b: a key compared after trimming spaces (runs of spaces become one) and ignoring case."""
    return _SPACES.sub(" ", str(value).strip()).lower()


def transform_normalize_text(df: pd.DataFrame, params: dict):
    """M2.3a step 2: trim spaces (and runs of spaces) in a text column, and optionally ignore case."""
    column, lowercase = params.get("column"), bool(params.get("lowercase", False))
    if column not in df.columns:
        raise ValueError(f"Cannot normalize — column not found: {column}")
    series = df[column]
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        raise ValueError(f"normalize_text needs a text column; {column} is numeric.")
    present = series.notna()
    before = series[present].astype(str)
    after = before.map(lambda v: _SPACES.sub(" ", v.strip()))
    if lowercase:
        after = after.str.lower()
    new_df = df.copy()
    new_df[column] = new_df[column].astype(object)
    new_df.loc[present, column] = after
    details = {"column": str(column), "lowercase": lowercase, "levels_before": int(before.nunique()),
               "levels_after": int(after.nunique()), "cells_changed": int((after != before).sum())}
    return new_df, details, ()


def _join_keys(series: pd.Series, normalize: bool) -> pd.Series:
    """The keys as compared: as given, or normalized; a missing key stays missing (it never matches)."""
    if not normalize:
        return series
    return series.map(lambda v: normalize_key(v) if pd.notna(v) else None)


def _added_name(name, right_table, taken) -> str:
    """A right column's name in the result: kept, or f"{name}_{right_table}" (then _2, _3) when it is taken."""
    if name not in taken:
        return name
    base = f"{name}_{right_table}"
    candidate, n = base, 2
    while candidate in taken:
        candidate, n = f"{base}_{n}", n + 1
    return candidate


AGG_STATS = ("mean", "sum")


def _summarize(right: pd.DataFrame, right_key, right_table, aggregate: dict, normalize: bool) -> pd.DataFrame:
    """
    M2.3b step 2: the carried table summarized per key (a unique key by construction): the key, then (count) the rows
    per key under f"rows_in_{right_table}" (every row counts), then f"{column}_{stat}" for each column and statistic,
    over the observed values (a key whose values are all missing gets a missing value, never 0).
    """
    columns = list(aggregate.get("columns") or [])
    stats = list(aggregate.get("stats") or [])
    count = bool(aggregate.get("count"))
    for column in columns:
        if column not in right.columns:
            raise ValueError(f"Cannot summarize — column not found: {column}")
        series = right[column]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            raise ValueError(f"Cannot summarize — {column} is not numeric.")
    for stat in stats:
        if stat not in AGG_STATS:
            raise ValueError(f"Unsupported statistic: {stat}. Must be mean or sum.")
    if not count and not (columns and stats):
        raise ValueError("Cannot summarize — nothing to summarize.")
    keys = _join_keys(right[right_key], normalize)
    present = keys.notna()
    grouped = right.loc[present].groupby(keys[present], sort=False)
    summary = pd.DataFrame(index=grouped.size().index)
    if count:
        summary[f"rows_in_{right_table}"] = grouped.size()
    for column in columns:
        for stat in stats:
            values = grouped[column].mean() if stat == "mean" else grouped[column].sum(min_count=1)
            summary[f"{column}_{stat}"] = values
    summary.index.name = right_key
    return summary.reset_index()


def transform_join(df: pd.DataFrame, params: dict, frames: dict):
    """
    M2.3b: join a carried table (frames[right_table]) to the analysis frame df (the left side) on one key. The right
    key must be unique, so a join never multiplies rows; a left join keeps every left row in order, an inner join
    only the matched ones. The right key is not added again; a right column whose name is taken gets the table name.
    """
    right_table, left_key, right_key = params.get("right_table"), params.get("left_key"), params.get("right_key")
    how, normalize = params.get("how", "left"), bool(params.get("normalize_keys", False))
    if right_table not in (frames or {}):
        raise ValueError(f"Cannot join — table not found: {right_table}")
    right = frames[right_table]
    if left_key not in df.columns:
        raise ValueError(f"Cannot join — column not found: {left_key}")
    if right_key not in right.columns:
        raise ValueError(f"Cannot join — column not found: {right_key}")
    if how not in ("left", "inner"):
        raise ValueError(f"Unsupported join type: {how}. Must be left or inner.")
    aggregate = params.get("aggregate")
    carried = right
    if aggregate:  # M2.3b step 2: summarize the carried table per key; the summary's key is unique
        right = _summarize(carried, right_key, right_table, aggregate, normalize)
    left_keys = _join_keys(df[left_key], normalize)
    right_keys = right[right_key] if aggregate else _join_keys(right[right_key], normalize)
    present = right_keys.notna()
    if right_keys[present].duplicated().any():
        raise ValueError(f"Cannot join: the key {right_key} repeats in {right_table}, so the join would multiply rows.")
    lookup = pd.Index(right_keys[present].tolist())
    positions = lookup.get_indexer(left_keys.tolist())
    positions[left_keys.isna().to_numpy()] = -1  # a missing key never matches
    matched = positions >= 0
    right_rows = right.loc[present].reset_index(drop=True)
    added_columns, renamed, taken = [], {}, set(df.columns)
    for column in right.columns:
        if column == right_key:
            continue
        name = _added_name(column, right_table, taken)
        taken.add(name)
        added_columns.append((column, name))
        if name != column:
            renamed[str(column)] = str(name)
    added = right_rows[[c for c, _ in added_columns]].reindex(positions)  # -1 is no row: every value missing
    added.columns = [name for _, name in added_columns]
    added.index = df.index
    count_column = None
    if aggregate and aggregate.get("count"):
        count_column = dict(added_columns)[f"rows_in_{right_table}"]
        added[count_column] = added[count_column].fillna(0).astype(int)  # no matching row: a real zero
    new_df = pd.concat([df, added], axis=1)
    unmatched = int((~matched).sum())
    if how == "inner":
        new_df = new_df[matched]
    kept_unmatched = 0 if how == "inner" else unmatched
    missing_columns = len(added_columns) - (1 if count_column is not None else 0)
    details = {"right_table": str(right_table), "left_key": str(left_key), "right_key": str(right_key),
               "rows_left": len(df), "rows_right": len(carried), "rows_after": len(new_df),
               "matched_rows": int(matched.sum()), "unmatched_left_rows": unmatched,
               "columns_added": [str(name) for _, name in added_columns], "renamed_columns": renamed,
               # cells missing only because their row found no match (a matched row copies the right table's value)
               "new_missing_cells": kept_unmatched * missing_columns,
               "how": how, "normalize_keys": normalize}
    warnings = ()
    if unmatched and how == "left":
        warnings = (f"{unmatched} row(s) have no match in {right_table}",)
    elif unmatched:
        warnings = (f"{unmatched} row(s) have no match in {right_table} and are removed",)
    if aggregate:  # counts and names only: the carried table is summarized as uploaded (it is never cleaned, G78)
        columns, stats = list(aggregate.get("columns") or []), list(aggregate.get("stats") or [])
        duplicates = int(carried.duplicated().sum())
        missing = {str(c): int(carried[c].isna().sum()) for c in columns}
        details.update({
            "aggregate": {"rows_summarized": len(carried), "keys": len(right), "columns": [str(c) for c in columns],
                          "stats": stats, "count_column": count_column},
            "carried_duplicate_rows": duplicates, "carried_missing": missing})
        if duplicates:
            warnings += (f"{duplicates} exact duplicate row(s) in {right_table} are counted again in the summary",)
        which = " and ".join(s for s in AGG_STATS if s in stats)
        for column in columns:
            if missing[str(column)] and which:
                warnings += (f"{column} is {missing[str(column)] / len(carried):.1%} missing in {right_table}: its "
                             f"{which} use{'s' if ' and ' not in which else ''} only the rows that have a value",)
    return new_df, details, warnings


def transform_rename(df: pd.DataFrame, params: dict):
    mapping = params["mapping"]
    missing = [c for c in mapping if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot rename — columns not found: {missing}")
    return df.rename(columns=mapping), {}, ()


def transform_filter(df: pd.DataFrame, params: dict):
    # M2.4a: an optional "unmatched_of" (a carried table name) only words the operation; the filter ignores it
    column, condition, value = params["column"], params["condition"], params.get("value")
    if column not in df.columns:
        raise ValueError(f"Column not found: {column}")
    if condition not in FILTER_CONDITIONS:
        raise ValueError(f"Unsupported filter condition: {condition}. Must be one of {FILTER_CONDITIONS}.")
    if condition == "equals":
        new_df = df[df[column] == value]
    elif condition == "not_equals":
        new_df = df[df[column] != value]
    elif condition == "greater_than":
        new_df = df[df[column] > value]
    elif condition == "less_than":
        new_df = df[df[column] < value]
    elif condition == "is_null":
        new_df = df[df[column].isna()]
    else:
        new_df = df[df[column].notna()]
    return new_df, {}, ()


def transform_dedupe(df: pd.DataFrame, params: dict):
    subset = params.get("subset")
    if subset:
        missing = [c for c in subset if c not in df.columns]
        if missing:
            raise ValueError(f"Cannot dedupe — columns not found: {missing}")
    return df.drop_duplicates(subset=subset or None), {}, ()


def transform_drop_column(df: pd.DataFrame, params: dict):
    column = params.get("column")
    if column not in df.columns:
        raise ValueError(f"Cannot drop — column not found: {column}")
    if len(df.columns) == 1:
        raise ValueError(f"Cannot drop {column}: it is the only column.")
    return df.drop(columns=[column]), {"column": str(column)}, ()


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _plain(x):
    """A numpy or pandas scalar as a plain Python value; NaN as None."""
    if x is None:
        return None
    if hasattr(x, "item"):
        x = x.item()
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def _moments(series: pd.Series):
    if not _is_numeric(series):
        return None, None
    values = series.dropna()
    return (_plain(float(values.mean())) if len(values) else None,
            _plain(float(values.std())) if len(values) > 1 else None)


def _statistic(series: pd.Series, strategy: str):
    values = series.dropna()
    if values.empty:
        return None
    if strategy == "mean":
        return float(values.mean())
    if strategy == "median":
        return float(values.median())
    return values.mode().iloc[0]


def transform_impute(df: pd.DataFrame, params: dict):
    """Fill the missing values of one column; the row count never changes."""
    column, strategy = params.get("column"), params.get("strategy")
    group_by, indicator = params.get("group_by"), bool(params.get("indicator"))
    if column not in df.columns:
        raise ValueError(f"Cannot impute — column not found: {column}")
    if strategy not in IMPUTE_STRATEGIES:
        raise ValueError(f"Unsupported imputation strategy: {strategy}. Must be one of {list(IMPUTE_STRATEGIES)}.")
    series = df[column]
    numeric = _is_numeric(series)
    if strategy in ("mean", "median") and not numeric:
        raise ValueError(f"The {strategy} needs a numeric column; {column} is not numeric.")
    if strategy == "constant":
        if numeric:
            raise ValueError(f"A constant label needs a text column; {column} is numeric.")
        if not isinstance(params.get("value"), str):
            raise ValueError("A constant fill needs a text value.")
        if group_by is not None:
            raise ValueError("A constant fill cannot be computed per group.")
    if group_by is not None and (group_by not in df.columns or group_by == column):
        raise ValueError(f"Cannot impute by group — group column not usable: {group_by}")
    indicator_name = f"{column}_was_missing"
    if indicator and indicator_name in df.columns:
        raise ValueError(f"Cannot add the indicator: the column {indicator_name} already exists.")

    new_df = df.copy()
    missing = series.isna()
    if indicator:
        new_df[indicator_name] = missing.astype(int)
    mean_before, std_before = _moments(series)
    fallback_cells, fill_value, warnings = 0, None, ()
    if strategy == "constant":
        new_df[column] = series.fillna(params["value"])
    else:
        overall = _statistic(series, strategy)
        if overall is None:
            raise ValueError(f"Cannot impute {column}: it has no observed values.")
        if group_by is None:
            new_df[column] = series.fillna(overall)
            fill_value = _plain(overall)
            # a text mode is a cell value: name it only for a column that is not withheld (M0.5 privacy tier)
            if strategy == "mode" and not numeric and series.nunique(dropna=True) > HIGH_CARDINALITY_THRESHOLD:
                fill_value = None
        else:
            per_group = series.groupby(df[group_by]).agg(lambda s: _statistic(s, strategy))
            group_fill = df[group_by].map(per_group)
            no_group_value = missing & group_fill.isna()
            fallback_cells = int(no_group_value.sum())
            new_df[column] = series.fillna(group_fill).fillna(overall)
            if fallback_cells:
                warnings = (f"{fallback_cells} cell(s) used the overall {strategy} because their group had no "
                            "observed value",)
    mean_after, std_after = _moments(new_df[column])
    details = {"column": str(column), "strategy": strategy, "cells_filled": int(missing.sum()), "fill_value": fill_value,
               "mean_before": mean_before, "mean_after": mean_after, "std_before": std_before, "std_after": std_after,
               "fallback_cells": fallback_cells}
    if group_by is not None:
        details["group_by"] = str(group_by)
    if indicator:
        details["indicator"] = indicator_name
    return new_df, details, warnings


def transform_fill_unmatched(df: pd.DataFrame, params: dict):
    """
    M2.4a: fill the given (sum) columns with params["value"] for the rows of an aggregate join whose key has no rows
    in the carried table (the count column is 0): an empty sum is 0. A row with a count above 0 is never touched, so a
    sum that is missing because all its values were missing stays missing. The row count never changes.
    """
    count_column, columns, value = params.get("count_column"), list(params.get("columns") or []), params.get("value", 0)
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Cannot fill — column not found: {column}")
        if not _is_numeric(df[column]):
            raise ValueError(f"Cannot fill — {column} is not numeric.")
    if count_column not in df.columns:
        raise ValueError(f"Cannot fill — column not found: {count_column}")
    if not _is_numeric(df[count_column]):
        raise ValueError(f"Cannot fill — {count_column} is not numeric.")
    new_df = df.copy()
    unmatched = df[count_column] == 0
    filled_rows = pd.Series(False, index=df.index)
    cells = 0
    for column in columns:
        fill = unmatched & df[column].isna()
        new_df.loc[fill, column] = value
        filled_rows |= fill
        cells += int(fill.sum())
    details = {"count_column": str(count_column), "columns": [str(c) for c in columns], "value": value,
               "rows_filled": int(filled_rows.sum()), "cells_filled": cells}
    return new_df, details, ()


# --- plain words ------------------------------------------------------------------------------------------------------

_FILTER_WORDS = {"equals": "equals {value}", "not_equals": "does not equal {value}",
                 "greater_than": "is greater than {value}", "less_than": "is less than {value}",
                 "is_null": "has no value", "not_null": "has a value"}


def _number(x) -> str:
    return f"{x:.4g}" if isinstance(x, float) else str(x)


def describe_operation(op_type, params, details=None) -> str:
    """One plain-words sentence for an operation; with details (a preview or the executed impact) it can name
    counts and the aggregate fill value."""
    params, details = params or {}, details or {}
    removed = details.get("rows_removed")
    if op_type is None:
        return "Leave the data as it is."
    if op_type == "dedupe":
        subset = params.get("subset")
        scope = "all columns" if not subset else "comparing " + ", ".join(str(c) for c in subset)
        count = f"{removed} " if removed is not None else ""
        return f"Remove {count}exact duplicate rows ({scope})."
    if op_type == "filter" and params.get("unmatched_of"):  # M2.4a: the unmatched rows of an aggregate join
        text = f"Remove the rows that have no rows in {params['unmatched_of']}"
        return text + (f" ({removed} removed)." if removed is not None else ".")
    if op_type == "filter":
        words = _FILTER_WORDS.get(params.get("condition"), str(params.get("condition")))
        text = f"Keep only rows where {params.get('column')} " + words.format(value=params.get("value"))
        return text + (f" (removes {removed} rows)." if removed is not None else ".")
    if op_type == "drop_column":
        return f"Drop the column {params.get('column')}."
    if op_type == "impute":
        column, strategy = params.get("column"), params.get("strategy")
        if strategy == "constant":
            text = f"Fill the missing values of {column} with the label '{params.get('value')}'"
        elif params.get("group_by"):
            text = f"Fill the missing values of {column} with the {strategy} of each {params['group_by']}"
        else:
            fill = details.get("fill_value")
            text = f"Fill the missing values of {column} with the {strategy}" + (f" ({_number(fill)})" if fill is not None else "")
        if params.get("indicator"):
            text += f" and add a {column}_was_missing indicator column"
        return text + "."
    if op_type == "cast":
        if params.get("to_type") == "numeric":
            lost = details.get("became_missing")
            return (f"Convert {params.get('column')} to numbers ({lost} value(s) that are not numbers become missing)."
                    if lost else f"Convert {params.get('column')} to numbers.")
        return f"Change the type of {params.get('column')} to {params.get('to_type')}."
    if op_type == "normalize_text":
        text = (f"Trim spaces and ignore case in {params.get('column')}" if params.get("lowercase")
                else f"Trim spaces in {params.get('column')}")
        before, after = details.get("levels_before"), details.get("levels_after")
        if before is not None and after is not None and before != after:
            text += f" ({before} distinct values become {after})"
        return text + "."
    if op_type == "join" and params.get("aggregate"):  # M2.3b step 2: summarize per key, then join
        aggregate = params["aggregate"]
        keep = "keeping only the rows that have a match" if params.get("how") == "inner" else "keeping every row"
        which = " and ".join(s for s in AGG_STATS if s in (aggregate.get("stats") or []))
        columns = list(aggregate.get("columns") or [])
        count = bool(aggregate.get("count"))
        if not (columns and which):
            what = "a row count"  # also the impossible case with nothing to summarize (the transform refuses it)
        else:
            of = (f"the {which} of {' and '.join(str(c) for c in columns)}" if "columns_added" in details
                  else f"the {which} of its numeric columns")
            what = f"a row count, and {of}" if count else of  # M2.3b step 2: no count, no "a row count"
        text = (f"Summarize {params.get('right_table')} per {params.get('left_key')} ({what}), then join the summary "
                f"to the analysis table, {keep}.")
        if params.get("normalize_keys"):
            text += " Keys are compared after trimming spaces and ignoring case."
        added = details.get("columns_added")
        if added is not None:
            names = [str(n) for n in added]
            shown = ", ".join(names[:5]) + (f" and {len(names) - 5} more" if len(names) > 5 else "")
            text += f" Adds {len(names)} column(s): {shown}." if names else " Adds no column."
        return text
    if op_type == "join":
        keep = "keeping only the rows that have a match" if params.get("how") == "inner" else "keeping every row"
        text = f"Join {params.get('right_table')} to the analysis table on {params.get('left_key')}, {keep}."
        if params.get("normalize_keys"):
            text += " Keys are compared after trimming spaces and ignoring case."
        added = details.get("columns_added")
        if added is not None:
            names = [str(n) for n in added]
            shown = ", ".join(names[:5]) + (f" and {len(names) - 5} more" if len(names) > 5 else "")
            text += f" Adds {len(names)} column(s): {shown}." if names else " Adds no column."
        return text
    if op_type == "fill_unmatched":  # M2.4a: without details no column name (the proposer sees these words)
        right = params.get("right_table")
        if not details:
            return f"Fill the empty sums with zero for the rows that have no rows in {right}."
        columns = details.get("columns", params.get("columns") or [])
        text = f"Fill {', '.join(str(c) for c in columns)} with zero for the rows that have no rows in {right}"
        cells = details.get("cells_filled")
        return text + (f" ({cells} value(s) filled)." if cells is not None else ".")
    if op_type == "rename":
        mapping = params.get("mapping") or {}
        return "Rename the columns: " + ", ".join(f"{a} to {b}" for a, b in mapping.items()) + "."
    return f"{op_type} {params}"


# --- propose and execute (the executors wrap the transforms) ------------------------------------------------------------

def propose_cast(dataset: Dataset, store: DatasetStore, column: str, to_type: str, rationale: WranglingRationale) -> Operation:
    """Propose casting one column to a new dtype. No execution yet — approval happens first."""
    return _propose("cast", dataset, {"column": column, "to_type": to_type}, rationale)


def execute_cast(dataset: Dataset, store: DatasetStore, operation: Operation) -> tuple[Dataset, Operation]:
    return _run(dataset, store, operation, transform_cast)


def propose_rename(dataset: Dataset, store: DatasetStore, mapping: dict[str, str], rationale: WranglingRationale) -> Operation:
    """Propose renaming one or more columns. Low-risk, but still logged and reviewable."""
    return _propose("rename", dataset, {"mapping": mapping}, rationale)


def execute_rename(dataset: Dataset, store: DatasetStore, operation: Operation) -> tuple[Dataset, Operation]:
    return _run(dataset, store, operation, transform_rename)


def propose_filter(dataset: Dataset, store: DatasetStore, column: str, condition: str, value, rationale: WranglingRationale) -> Operation:
    """
    Propose filtering rows. `condition` is one of a small, safe whitelist —
    never an arbitrary expression string — per the declarative-operations
    design decision (no free-form code from the model).
    """
    allowed = {"equals", "not_equals", "greater_than", "less_than", "is_null", "not_null"}
    if condition not in allowed:
        raise ValueError(f"Unsupported filter condition: {condition}. Must be one of {allowed}.")
    return _propose("filter", dataset, {"column": column, "condition": condition, "value": value}, rationale)


def execute_filter(dataset: Dataset, store: DatasetStore, operation: Operation) -> tuple[Dataset, Operation]:
    return _run(dataset, store, operation, transform_filter)


def propose_dedupe(dataset: Dataset, store: DatasetStore, subset: list[str] | None, rationale: WranglingRationale) -> Operation:
    """Propose dropping exact duplicate rows, optionally scoped to a subset of columns."""
    return _propose("dedupe", dataset, {"subset": subset}, rationale)


def execute_dedupe(dataset: Dataset, store: DatasetStore, operation: Operation) -> tuple[Dataset, Operation]:
    return _run(dataset, store, operation, transform_dedupe)


def execute_drop_column(dataset: Dataset, store: DatasetStore, operation: Operation) -> tuple[Dataset, Operation]:
    return _run(dataset, store, operation, transform_drop_column)


def execute_impute(dataset: Dataset, store: DatasetStore, operation: Operation) -> tuple[Dataset, Operation]:
    return _run(dataset, store, operation, transform_impute)


def execute_normalize_text(dataset: Dataset, store: DatasetStore, operation: Operation) -> tuple[Dataset, Operation]:
    return _run(dataset, store, operation, transform_normalize_text)


def execute_fill_unmatched(dataset: Dataset, store: DatasetStore, operation: Operation) -> tuple[Dataset, Operation]:
    return _run(dataset, store, operation, transform_fill_unmatched)


def execute_join(dataset: Dataset, store: DatasetStore, operation: Operation, tables: dict) -> tuple[Dataset, Operation]:
    """
    M2.3b: run a join on the analysis table (dataset, the left side) with tables[params["right_table"]]; the result
    is a NEW derived table (its own version chain, parents both inputs). A ValueError marks the operation FAILED.
    """
    if operation.approval_status != ApprovalStatus.APPROVED:
        raise ValueError("Operation must be APPROVED before execution.")
    params = operation.params
    right_table, result_name = params.get("right_table"), params.get("result_name")
    right = (tables or {}).get(right_table)
    if right is None:
        return dataset, operation.mark_failed(f"Cannot join — table not found: {right_table}")
    if not result_name:
        return dataset, operation.mark_failed("Cannot join — no result_name was given.")
    left_df = store.read(dataset.storage_key)
    try:
        new_df, details, warnings = transform_join(left_df, params, {right_table: store.read(right.storage_key)})
    except (ValueError, TypeError) as e:
        return dataset, operation.mark_failed(str(e))
    storage_key = f"{dataset.session_id}/{result_name}_v1.csv"
    store.write(storage_key, new_df)
    derived = Dataset.derived(dataset.session_id, result_name, storage_key, parents=(dataset.handle, right.handle),
                              operation_id=operation.id)
    impact = OperationImpact(rows_before=len(left_df), rows_after=len(new_df), columns_before=len(left_df.columns),
                             columns_after=len(new_df.columns), warnings=tuple(warnings), details=dict(details))
    return derived, operation.with_result(impact=impact, output_handle=derived.handle)
