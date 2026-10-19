"""
M2.3b: the join operation (core/wrangling.py transform_join and execute_join), its plain words, and the registry's
multi_input entry. Expected frames and numbers are computed independently with pandas merge.
"""
import numpy as np
import pandas as pd
import pytest

from core.capabilities import OPERATIONS, run_transform
from core.dataset import LocalDiskStore
from core.ingestion import load_dataset
from core.operation import ApprovalStatus, Operation
from core.wrangling import describe_operation, execute_join, transform_dedupe, transform_join


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _orders():
    return pd.DataFrame({"order_id": [1, 2, 3, 4, 5, 6], "customer_id": [3, 1, 3, 9, 2, np.nan],
                         "total_spend": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})


def _customers():
    return pd.DataFrame({"customer_id": [1, 2, 3], "region": ["East", "West", None], "tier": ["basic", "plus", "pro"]})


def _params(**extra):
    return {"right_table": "customers", "left_key": "customer_id", "right_key": "customer_id", "how": "left",
            "normalize_keys": False, "result_name": "orders_customers", **extra}


def _join(left, right, **extra):
    return transform_join(left, _params(**extra), {"customers": right})


# --- the transform -----------------------------------------------------------------------------------------------

def test_the_transform_is_pure():
    left, right = _orders(), _customers()
    before_left, before_right = left.copy(), right.copy()
    _join(left, right)
    _join(left, right, how="inner", normalize_keys=True)
    pd.testing.assert_frame_equal(left, before_left)
    pd.testing.assert_frame_equal(right, before_right)


def test_a_left_join():
    left, right = _orders(), _customers()
    new_df, details, warnings = _join(left, right)
    expected = left.merge(right, on="customer_id", how="left")  # the key is not repeated
    pd.testing.assert_frame_equal(new_df.reset_index(drop=True), expected, check_dtype=False)
    assert list(new_df.columns) == ["order_id", "customer_id", "total_spend", "region", "tier"]
    assert new_df["order_id"].tolist() == left["order_id"].tolist() and len(new_df) == len(left)  # the left order
    assert new_df.loc[new_df["order_id"] == 4, ["region", "tier"]].isna().all(axis=None)  # key 9: no match
    assert new_df.loc[new_df["order_id"] == 6, ["region", "tier"]].isna().all(axis=None)  # a missing key never matches
    matched = int(left["customer_id"].isin(right["customer_id"]).sum())
    assert matched == 4
    assert details == {"right_table": "customers", "left_key": "customer_id", "right_key": "customer_id",
                       "rows_left": 6, "rows_right": 3, "rows_after": 6, "matched_rows": matched,
                       "unmatched_left_rows": 6 - matched, "columns_added": ["region", "tier"], "renamed_columns": {},
                       # 2 unmatched rows x 2 added columns; customer 3's missing region was missing in customers
                       "new_missing_cells": (6 - matched) * 2, "how": "left", "normalize_keys": False}
    assert int(new_df["region"].isna().sum()) == details["new_missing_cells"] // 2 + 2  # plus customer 3, twice
    assert warnings == ("2 row(s) have no match in customers",)


def test_an_inner_join():
    left, right = _orders(), _customers()
    new_df, details, warnings = _join(left, right, how="inner")
    expected = left.merge(right, on="customer_id", how="inner")
    keep = left["customer_id"].isin(right["customer_id"])
    assert new_df["order_id"].tolist() == left.loc[keep, "order_id"].tolist()  # the left order, matched rows only
    pd.testing.assert_frame_equal(new_df.sort_values("order_id").reset_index(drop=True),
                                  expected.sort_values("order_id").reset_index(drop=True), check_dtype=False)
    assert (details["rows_after"], details["matched_rows"], details["unmatched_left_rows"]) == (4, 4, 2)
    assert details["new_missing_cells"] == 0 and details["how"] == "inner"
    assert warnings == ("2 row(s) have no match in customers and are removed",)


def test_no_warning_when_every_row_matches():
    left = _orders().iloc[[0, 1, 2, 4]]
    new_df, details, warnings = _join(left, _customers())
    assert warnings == () and details["unmatched_left_rows"] == 0 and details["new_missing_cells"] == 0
    assert len(new_df) == len(left)  # a unique right key never multiplies rows


def test_name_clashes():
    left = _orders().assign(region="L", region_customers="taken")
    right = _customers().assign(order_id=[7, 8, 9])
    new_df, details, _ = _join(left, right)
    assert details["columns_added"] == ["region_customers_2", "tier", "order_id_customers"]
    assert details["renamed_columns"] == {"region": "region_customers_2", "order_id": "order_id_customers"}
    assert new_df["region"].tolist() == ["L"] * 6 and new_df["order_id"].tolist() == _orders()["order_id"].tolist()


def test_a_normalized_join_keeps_the_left_keys():
    left = pd.DataFrame({"region": ["East", "West", "North", None], "x": [1, 2, 3, 4]})
    right = pd.DataFrame({"zone": ["east ", " West"], "manager": ["Ana", "Ben"]})
    params = {"right_table": "zones", "left_key": "region", "right_key": "zone", "how": "left", "normalize_keys": True}
    new_df, details, _ = transform_join(left, params, {"zones": right})
    assert new_df["manager"].tolist()[:2] == ["Ana", "Ben"] and new_df["manager"].isna().tolist()[2:] == [True, True]
    assert new_df["region"].tolist()[:3] == ["East", "West", "North"]  # the left key values as they were
    assert "zone" not in new_df.columns and details["matched_rows"] == 2
    exact, _, _ = transform_join(left, {**params, "normalize_keys": False}, {"zones": right})
    assert exact["manager"].isna().all()  # as given, nothing matches


def test_the_safety_rule():
    repeats = pd.DataFrame({"customer_id": [1, 1, 2], "region": ["East", "West", "East"]})
    with pytest.raises(ValueError) as error:
        _join(_orders(), repeats)
    assert str(error.value) == "Cannot join: the key customer_id repeats in customers, so the join would multiply rows."
    left = pd.DataFrame({"k": ["a", "b"]})
    after_normalizing = pd.DataFrame({"k": ["A", "a "], "v": [1, 2]})
    params = {"right_table": "customers", "left_key": "k", "right_key": "k", "how": "left"}
    transform_join(left, params, {"customers": after_normalizing})  # unique as given
    with pytest.raises(ValueError) as error:
        transform_join(left, {**params, "normalize_keys": True}, {"customers": after_normalizing})
    assert str(error.value) == "Cannot join: the key k repeats in customers, so the join would multiply rows."


@pytest.mark.parametrize("params, frames, message", [
    (_params(left_key="nope"), {"customers": _customers()}, "Cannot join — column not found: nope"),
    (_params(right_key="nope"), {"customers": _customers()}, "Cannot join — column not found: nope"),
    (_params(right_table="other"), {"customers": _customers()}, "Cannot join — table not found: other"),
    (_params(how="outer"), {"customers": _customers()}, "Unsupported join type: outer. Must be left or inner."),
], ids=["left_key", "right_key", "table", "how"])
def test_the_errors(params, frames, message):
    with pytest.raises(ValueError) as error:
        transform_join(_orders(), params, frames)
    assert str(error.value) == message


# --- describe_operation ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("params, details, sentence", [
    (_params(), None, "Join customers to the analysis table on customer_id, keeping every row."),
    (_params(how="inner"), None, "Join customers to the analysis table on customer_id, keeping only the rows that have "
                                 "a match."),
    (_params(normalize_keys=True), None, "Join customers to the analysis table on customer_id, keeping every row. Keys "
                                         "are compared after trimming spaces and ignoring case."),
    (_params(), {"columns_added": ["region", "tier"]},
     "Join customers to the analysis table on customer_id, keeping every row. Adds 2 column(s): region, tier."),
    (_params(how="inner", normalize_keys=True), {"columns_added": [f"c{i}" for i in range(7)]},
     "Join customers to the analysis table on customer_id, keeping only the rows that have a match. Keys are compared "
     "after trimming spaces and ignoring case. Adds 7 column(s): c0, c1, c2, c3, c4 and 2 more."),
    (_params(), {"columns_added": []}, "Join customers to the analysis table on customer_id, keeping every row. Adds no "
                                       "column."),
], ids=["left", "inner", "normalized", "details", "many_columns", "no_column"])
def test_describe_operation(params, details, sentence):
    assert describe_operation("join", params, details) == sentence


# --- execute_join ------------------------------------------------------------------------------------------------

def _tables(tmp_path, store):
    tables = {}
    for name, frame in (("orders", _orders()), ("customers", _customers())):
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        tables[name] = load_dataset(path, session_id="s", name=name, store=store)
    return tables


def _op(params, status=ApprovalStatus.APPROVED, handles=("orders@v1", "customers@v1")):
    return Operation(op_type="join", session_id="s", input_handles=handles, params=params, approval_status=status)


def test_execute_join_makes_a_derived_table(tmp_path, store):
    tables = _tables(tmp_path, store)
    left, right = store.read(tables["orders"].storage_key), store.read(tables["customers"].storage_key)
    expected, details, warnings = transform_join(left, _params(), {"customers": right})
    op = _op(_params())
    derived, finished = execute_join(tables["orders"], store, op, tables)
    assert (derived.name, derived.version, derived.parent_version) == ("orders_customers", 1, None)
    assert derived.parents == (tables["orders"].handle, tables["customers"].handle)
    assert derived.created_from_operation == op.id and derived.storage_key == "s/orders_customers_v1.csv"
    assert derived.handle == "orders_customers@v1" and finished.output_handle == derived.handle
    path = tmp_path / "expected.csv"
    expected.to_csv(path, index=False)
    pd.testing.assert_frame_equal(store.read(derived.storage_key), pd.read_csv(path))
    impact = finished.impact
    assert (impact.rows_before, impact.rows_after, impact.columns_before, impact.columns_after) == (
        len(left), len(expected), len(left.columns), len(expected.columns))
    assert impact.details == details and impact.warnings == warnings
    assert store.read(tables["customers"].storage_key).equals(right)  # the carried table is untouched


def test_execute_join_needs_approval_and_fails_cleanly(tmp_path, store):
    tables = _tables(tmp_path, store)
    with pytest.raises(ValueError, match="Operation must be APPROVED before execution."):
        execute_join(tables["orders"], store, _op(_params(), ApprovalStatus.PENDING), tables)
    same, failed = execute_join(tables["orders"], store, _op(_params(left_key="nope")), tables)
    assert same is tables["orders"] and failed.approval_status == ApprovalStatus.FAILED
    assert failed.failure_reason == "Cannot join — column not found: nope"
    same, failed = execute_join(tables["orders"], store, _op(_params(right_table="other")), tables)
    assert same is tables["orders"] and failed.failure_reason == "Cannot join — table not found: other"


# --- the registry ------------------------------------------------------------------------------------------------

def test_the_registry():
    entry = OPERATIONS["join"]
    assert entry.multi_input is True and entry.transform is transform_join and entry.execute is execute_join
    assert all(not op.multi_input for name, op in OPERATIONS.items() if name != "join")
    frames = {"customers": _customers()}
    pd.testing.assert_frame_equal(run_transform("join", _orders(), _params(), frames)[0],
                                  transform_join(_orders(), _params(), frames)[0])
    df = pd.concat([_orders(), _orders().iloc[[0]]], ignore_index=True)
    assert run_transform("dedupe", df, {"subset": None}, frames)[1:] == transform_dedupe(df, {"subset": None})[1:]
    pd.testing.assert_frame_equal(run_transform("dedupe", df, {"subset": None})[0],
                                  transform_dedupe(df, {"subset": None})[0])
