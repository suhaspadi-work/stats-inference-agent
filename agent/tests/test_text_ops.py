"""
M2.3a step 2: the normalize_text transform, the lenient numeric cast, their plain-words descriptions and the
normalize_text executor. Expected numbers are computed independently with pandas.
"""
import re

import numpy as np
import pandas as pd
import pytest

from core.capabilities import OPERATIONS
from core.dataset import LocalDiskStore
from core.ingestion import load_dataset
from core.operation import ApprovalStatus, Operation
from core.wrangling import describe_operation, execute_normalize_text, transform_cast, transform_normalize_text


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _frame():
    return pd.DataFrame({
        "region": [" East", "east ", "EAST", "West", "west  ", None, "North", "a  b", "a b"],
        "tenure": ["12", " 34.5 ", "unknown", "  ", "1,234", None, "7", "8", "9"],
        "x": np.arange(9, dtype=float),
    })


def _normalized(series, lowercase):
    present = series.dropna().astype(str)
    out = present.str.replace(r"\s+", " ", regex=True).str.strip()
    return out.str.lower() if lowercase else out


# --- normalize_text --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("transform, params", [
    (transform_normalize_text, {"column": "region", "lowercase": True}),
    (transform_cast, {"column": "tenure", "to_type": "numeric"}),
], ids=["normalize_text", "numeric_cast"])
def test_the_transforms_are_pure(transform, params):
    df = _frame()
    before = df.copy()
    transform(df, params)
    pd.testing.assert_frame_equal(df, before)


@pytest.mark.parametrize("lowercase", [False, True], ids=["trim", "trim_and_lowercase"])
def test_normalize_text(lowercase):
    df = _frame()
    new_df, details, warnings = transform_normalize_text(df, {"column": "region", "lowercase": lowercase})
    expected = _normalized(df["region"], lowercase)
    assert new_df["region"].dropna().tolist() == expected.tolist()
    assert new_df["region"].isna().tolist() == df["region"].isna().tolist()  # missing stays missing
    assert new_df.drop(columns="region").equals(df.drop(columns="region")) and warnings == ()
    before = df["region"].dropna().astype(str)
    assert details == {"column": "region", "lowercase": lowercase, "levels_before": before.nunique(),
                       "levels_after": expected.nunique(), "cells_changed": int((expected != before).sum())}


def test_strip_collapse_and_case():
    df = pd.DataFrame({"c": ["  a   b ", "a b", "A B"]})
    assert transform_normalize_text(df, {"column": "c", "lowercase": False})[0]["c"].tolist() == ["a b", "a b", "A B"]
    assert transform_normalize_text(df, {"column": "c", "lowercase": True})[0]["c"].tolist() == ["a b", "a b", "a b"]


def test_normalize_text_errors():
    with pytest.raises(ValueError) as error:
        transform_normalize_text(_frame(), {"column": "x", "lowercase": False})
    assert str(error.value) == "normalize_text needs a text column; x is numeric."
    with pytest.raises(ValueError) as error:
        transform_normalize_text(_frame(), {"column": "nope", "lowercase": False})
    assert str(error.value) == "Cannot normalize — column not found: nope"


# --- the numeric cast ------------------------------------------------------------------------------------------------

def test_the_lenient_numeric_cast():
    df = _frame()
    new_df, details, warnings = transform_cast(df, {"column": "tenure", "to_type": "numeric"})
    assert pd.api.types.is_numeric_dtype(new_df["tenure"]) and warnings == ()
    expected = [12.0, 34.5, np.nan, np.nan, np.nan, np.nan, 7.0, 8.0, 9.0]  # "unknown", "  " and "1,234" are not numbers
    pd.testing.assert_series_equal(new_df["tenure"], pd.Series(expected, name="tenure"), check_dtype=False)
    present = df["tenure"].notna()
    became = int(pd.to_numeric(df["tenure"][present].str.strip(), errors="coerce").isna().sum())
    assert became == 3
    assert details == {"column": "tenure", "to_type": "numeric", "non_missing_before": int(present.sum()),
                       "became_missing": became}
    assert df["tenure"].tolist() == _frame()["tenure"].tolist()  # the input is unchanged


def test_the_numeric_cast_of_a_missing_column():
    with pytest.raises(ValueError) as error:
        transform_cast(_frame(), {"column": "nope", "to_type": "numeric"})
    assert str(error.value) == "Cast failed: column not found: nope"


def test_another_cast_has_empty_details():
    new_df, details, warnings = transform_cast(_frame(), {"column": "x", "to_type": "int64"})
    assert new_df["x"].dtype == "int64" and details == {} and warnings == ()
    with pytest.raises(ValueError, match="Cast failed"):
        transform_cast(_frame(), {"column": "region", "to_type": "float64"})


# --- describe_operation ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("op_type, params, details, sentence", [
    ("normalize_text", {"column": "region", "lowercase": False}, None, "Trim spaces in region."),
    ("normalize_text", {"column": "region", "lowercase": True}, None, "Trim spaces and ignore case in region."),
    ("normalize_text", {"column": "region", "lowercase": True}, {"levels_before": 8, "levels_after": 3},
     "Trim spaces and ignore case in region (8 distinct values become 3)."),
    ("normalize_text", {"column": "region", "lowercase": False}, {"levels_before": 3, "levels_after": 3},
     "Trim spaces in region."),
    ("cast", {"column": "tenure", "to_type": "numeric"}, None, "Convert tenure to numbers."),
    ("cast", {"column": "tenure", "to_type": "numeric"}, {"became_missing": 0}, "Convert tenure to numbers."),
    ("cast", {"column": "tenure", "to_type": "numeric"}, {"became_missing": 3},
     "Convert tenure to numbers (3 value(s) that are not numbers become missing)."),
], ids=["trim", "lowercase", "levels", "equal_levels", "convert", "convert_none_lost", "convert_lost"])
def test_describe_operation(op_type, params, details, sentence):
    assert describe_operation(op_type, params, details) == sentence


def test_a_description_without_details_holds_no_digit():
    for params in ({"column": "region", "lowercase": False}, {"column": "region", "lowercase": True}):
        assert not re.search(r"\d", describe_operation("normalize_text", params))
    assert not re.search(r"\d", describe_operation("cast", {"column": "tenure", "to_type": "numeric"}))


# --- the executor ----------------------------------------------------------------------------------------------------

def test_the_executor_runs_the_transform(tmp_path, store):
    path = tmp_path / "ops.csv"
    _frame().to_csv(path, index=False)
    ds = load_dataset(path, session_id="s", name="ops", store=store)
    df = store.read(ds.storage_key)
    params = {"column": "region", "lowercase": True}
    expected, details, warnings = transform_normalize_text(df, params)
    op = Operation(op_type="normalize_text", session_id="s", input_handles=(ds.handle,), params=params,
                   approval_status=ApprovalStatus.APPROVED)
    new_ds, finished = execute_normalize_text(ds, store, op)
    expected_path = tmp_path / "expected.csv"
    expected.to_csv(expected_path, index=False)
    pd.testing.assert_frame_equal(store.read(new_ds.storage_key), pd.read_csv(expected_path))
    impact = finished.impact
    assert (impact.rows_before, impact.rows_after, impact.columns_before, impact.columns_after) == (
        len(df), len(expected), len(df.columns), len(expected.columns))
    assert impact.details == details and impact.warnings == warnings and details["cells_changed"] > 0


def test_normalize_text_is_registered():
    entry = OPERATIONS["normalize_text"]
    assert entry.transform is transform_normalize_text and entry.execute is execute_normalize_text
