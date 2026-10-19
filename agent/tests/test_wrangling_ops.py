"""
M2.3a: the pure transforms of core/wrangling.py (drop_column, impute and the four earlier operations), their
plain-words descriptions, and the executors that wrap them. Expected numbers are computed independently with pandas.
"""
import numpy as np
import pandas as pd
import pytest

from core.dataset import LocalDiskStore
from core.ingestion import load_dataset
from core.operation import ApprovalStatus, Operation
from core.wrangling import (
    describe_operation, execute_cast, execute_dedupe, execute_drop_column, execute_filter, execute_impute,
    execute_rename, transform_cast, transform_dedupe, transform_drop_column, transform_filter, transform_impute,
    transform_rename,
)


class CountingStore(LocalDiskStore):
    def __init__(self, base_dir):
        super().__init__(base_dir)
        self.writes = []

    def write(self, storage_key, data):
        self.writes.append(storage_key)
        super().write(storage_key, data)


@pytest.fixture
def store(tmp_path):
    return CountingStore(tmp_path / "store")


def _frame():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8],
        "region": ["East", "West", "East", "West", "East", "West", "North", None],
        "tenure": [10.0, np.nan, 30.0, 40.0, np.nan, 60.0, np.nan, 80.0],
        "visits": [1, 2, 2, 3, np.nan, 3, 3, 4],
        "flag": [True, False, True, False, True, False, True, False],
    })


def _dataset(df, tmp_path, store, name="ops"):
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    return load_dataset(path, session_id="s", name=name, store=store)


def _op(op_type, params):
    return Operation(op_type=op_type, session_id="s", input_handles=("ops@v1",), params=params,
                     approval_status=ApprovalStatus.APPROVED)


# --- purity ------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("transform, params", [
    (transform_cast, {"column": "id", "to_type": "float64"}),
    (transform_rename, {"mapping": {"id": "key"}}),
    (transform_filter, {"column": "tenure", "condition": "not_null", "value": None}),
    (transform_dedupe, {"subset": None}),
    (transform_drop_column, {"column": "visits"}),
    (transform_impute, {"column": "tenure", "strategy": "median", "indicator": True}),
], ids=["cast", "rename", "filter", "dedupe", "drop_column", "impute"])
def test_every_transform_is_pure(transform, params, store):
    df = _frame()
    before = df.copy()
    transform(df, params)
    pd.testing.assert_frame_equal(df, before)
    assert store.writes == []


# --- drop_column ---------------------------------------------------------------------------------------------------------

def test_drop_column():
    new_df, details, warnings = transform_drop_column(_frame(), {"column": "tenure"})
    assert list(new_df.columns) == ["id", "region", "visits", "flag"] and details == {"column": "tenure"} and warnings == ()
    with pytest.raises(ValueError) as error:
        transform_drop_column(_frame(), {"column": "nope"})
    assert str(error.value) == "Cannot drop — column not found: nope"
    with pytest.raises(ValueError) as error:
        transform_drop_column(pd.DataFrame({"only": [1]}), {"column": "only"})
    assert str(error.value) == "Cannot drop only: it is the only column."


# --- impute ----------------------------------------------------------------------------------------------------------------

def _expected(df, column, filled):
    observed = df[column].dropna()
    return {"mean_before": float(observed.mean()), "std_before": float(observed.std()),
            "mean_after": float(filled.mean()), "std_after": float(filled.std())}


@pytest.mark.parametrize("strategy", ["mean", "median", "mode"])
def test_impute_numeric(strategy):
    df = _frame()
    observed = df["tenure"].dropna()
    value = {"mean": observed.mean(), "median": observed.median(), "mode": observed.mode().iloc[0]}[strategy]
    new_df, details, warnings = transform_impute(df, {"column": "tenure", "strategy": strategy})
    filled = df["tenure"].fillna(value)
    pd.testing.assert_series_equal(new_df["tenure"], filled)
    assert len(new_df) == len(df) and list(new_df.columns) == list(df.columns) and warnings == ()
    assert details == {"column": "tenure", "strategy": strategy, "cells_filled": 3, "fill_value": float(value),
                       "fallback_cells": 0, **_expected(df, "tenure", filled)}


def test_impute_mode_of_a_text_column():
    new_df, details, _ = transform_impute(_frame(), {"column": "region", "strategy": "mode"})
    mode = _frame()["region"].dropna().mode().iloc[0]
    assert new_df["region"].isna().sum() == 0 and details["fill_value"] == mode and details["cells_filled"] == 1
    assert (details["mean_before"], details["mean_after"], details["std_before"], details["std_after"]) == (None,) * 4


def test_impute_a_constant_label():
    new_df, details, _ = transform_impute(_frame(), {"column": "region", "strategy": "constant", "value": "missing"})
    assert new_df["region"].tolist()[-1] == "missing" and details["fill_value"] is None and details["cells_filled"] == 1
    with pytest.raises(ValueError, match="A constant label needs a text column; tenure is numeric."):
        transform_impute(_frame(), {"column": "tenure", "strategy": "constant", "value": "missing"})
    with pytest.raises(ValueError, match="A constant fill needs a text value."):
        transform_impute(_frame(), {"column": "region", "strategy": "constant", "value": 3})


@pytest.mark.parametrize("column", ["region", "flag"])
@pytest.mark.parametrize("strategy", ["mean", "median"])
def test_mean_and_median_need_a_numeric_column(column, strategy):
    with pytest.raises(ValueError, match=f"The {strategy} needs a numeric column; {column} is not numeric."):
        transform_impute(_frame(), {"column": column, "strategy": strategy})


def test_invalid_impute_params():
    with pytest.raises(ValueError, match="Unsupported imputation strategy: wild"):
        transform_impute(_frame(), {"column": "tenure", "strategy": "wild"})
    with pytest.raises(ValueError, match="Cannot impute — column not found: nope"):
        transform_impute(_frame(), {"column": "nope", "strategy": "mean"})
    with pytest.raises(ValueError, match="no observed values"):
        transform_impute(pd.DataFrame({"a": [np.nan, np.nan], "b": [1, 2]}), {"column": "a", "strategy": "mean"})


def test_impute_by_group_with_a_fallback():
    df = _frame()
    new_df, details, warnings = transform_impute(df, {"column": "tenure", "strategy": "median", "group_by": "region"})
    medians = df.groupby("region")["tenure"].median()
    overall = df["tenure"].median()
    expected = df["tenure"].fillna(df["region"].map(medians)).fillna(overall)
    pd.testing.assert_series_equal(new_df["tenure"], expected)
    # North (one row, its tenure missing) and the missing region have no observed value: both use the overall median
    assert details["fallback_cells"] == 1 + 0 and details["fill_value"] is None and details["group_by"] == "region"
    assert warnings == ("1 cell(s) used the overall median because their group had no observed value",)


def test_the_missing_indicator():
    df = _frame()
    new_df, details, _ = transform_impute(df, {"column": "tenure", "strategy": "median", "indicator": True})
    assert new_df["tenure_was_missing"].tolist() == df["tenure"].isna().astype(int).tolist()
    assert details["indicator"] == "tenure_was_missing" and len(new_df) == len(df)
    clash = df.assign(tenure_was_missing=0)
    with pytest.raises(ValueError, match="the column tenure_was_missing already exists"):
        transform_impute(clash, {"column": "tenure", "strategy": "median", "indicator": True})


# --- describe_operation ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("op_type, params, details, sentence", [
    ("impute", {"column": "tenure", "strategy": "median"}, {"fill_value": 24.8},
     "Fill the missing values of tenure with the median (24.8)."),
    ("impute", {"column": "tenure", "strategy": "median"}, None, "Fill the missing values of tenure with the median."),
    ("impute", {"column": "name", "strategy": "mode"}, {"fill_value": None}, "Fill the missing values of name with the mode."),
    ("impute", {"column": "tenure", "strategy": "median", "group_by": "region"}, None,
     "Fill the missing values of tenure with the median of each region."),
    ("impute", {"column": "region", "strategy": "constant", "value": "missing"}, None,
     "Fill the missing values of region with the label 'missing'."),
    ("impute", {"column": "tenure", "strategy": "median", "indicator": True}, {"fill_value": 24.8},
     "Fill the missing values of tenure with the median (24.8) and add a tenure_was_missing indicator column."),
    ("filter", {"column": "tenure", "condition": "not_null", "value": None}, None,
     "Keep only rows where tenure has a value."),
    ("filter", {"column": "tenure", "condition": "not_null", "value": None}, {"rows_removed": 36},
     "Keep only rows where tenure has a value (removes 36 rows)."),
    ("filter", {"column": "tenure", "condition": "greater_than", "value": 10}, None,
     "Keep only rows where tenure is greater than 10."),
    ("drop_column", {"column": "tenure"}, None, "Drop the column tenure."),
    ("dedupe", {"subset": None}, None, "Remove exact duplicate rows (all columns)."),
    ("dedupe", {"subset": None}, {"rows_removed": 4}, "Remove 4 exact duplicate rows (all columns)."),
    ("dedupe", {"subset": ["region", "tenure"]}, None, "Remove exact duplicate rows (comparing region, tenure)."),
    ("cast", {"column": "id", "to_type": "float64"}, None, "Change the type of id to float64."),
    ("rename", {"mapping": {"a": "b", "c": "d"}}, None, "Rename the columns: a to b, c to d."),
    (None, {}, None, "Leave the data as it is."),
], ids=["median_value", "median_plain", "withheld_mode", "group", "constant", "indicator", "filter", "filter_count", "filter_gt",
        "drop", "dedupe", "dedupe_count", "dedupe_subset", "cast", "rename", "leave"])
def test_describe_operation(op_type, params, details, sentence):
    assert describe_operation(op_type, params, details) == sentence


# --- the executors wrap the transforms ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("execute, transform, op_type, params", [
    (execute_drop_column, transform_drop_column, "drop_column", {"column": "visits"}),
    (execute_impute, transform_impute, "impute", {"column": "tenure", "strategy": "mean", "indicator": True}),
    (execute_dedupe, transform_dedupe, "dedupe", {"subset": None}),
    (execute_filter, transform_filter, "filter", {"column": "tenure", "condition": "not_null", "value": None}),
    (execute_rename, transform_rename, "rename", {"mapping": {"id": "key"}}),
    (execute_cast, transform_cast, "cast", {"column": "id", "to_type": "float64"}),
], ids=["drop_column", "impute", "dedupe", "filter", "rename", "cast"])
def test_the_executor_runs_the_transform(execute, transform, op_type, params, tmp_path, store):
    ds = _dataset(_frame(), tmp_path, store)
    df = store.read(ds.storage_key)
    expected, details, warnings = transform(df, params)
    new_ds, finished = execute(ds, store, _op(op_type, params))
    pd.testing.assert_frame_equal(store.read(new_ds.storage_key), pd.read_csv(_csv(expected, tmp_path)))
    impact = finished.impact
    assert (impact.rows_before, impact.rows_after, impact.columns_before, impact.columns_after) == (
        len(df), len(expected), len(df.columns), len(expected.columns))
    assert impact.details == details and impact.warnings == warnings
    if op_type in ("dedupe", "filter", "rename", "cast"):
        assert impact.details == {} and impact.warnings == ()  # the four earlier operations record as before


def _csv(df, tmp_path):
    path = tmp_path / "expected.csv"
    df.to_csv(path, index=False)
    return path


def test_an_unknown_filter_condition_now_fails_cleanly(tmp_path, store):
    ds = _dataset(_frame(), tmp_path, store)
    new_ds, finished = execute_filter(ds, store, _op("filter", {"column": "tenure", "condition": "between", "value": 3}))
    assert new_ds is ds and finished.approval_status == ApprovalStatus.FAILED
    assert finished.failure_reason.startswith("Unsupported filter condition: between.")


def test_the_mode_of_a_withheld_text_column_is_not_named():
    names = [f"NAME-{i}" for i in range(30)] + [None, None]
    new_df, details, _ = transform_impute(pd.DataFrame({"name": names, "x": range(32)}), {"column": "name", "strategy": "mode"})
    assert details["fill_value"] is None and new_df["name"].isna().sum() == 0  # filled, but the value is not reported
    assert describe_operation("impute", {"column": "name", "strategy": "mode"}, details) == (
        "Fill the missing values of name with the mode.")
