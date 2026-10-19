"""
M2.3b step 2: the join with an "aggregate" parameter (core/wrangling.py): the carried table is summarized per key (a
row count, and the mean and sum of chosen numeric columns), then joined like a plain join. Expected frames and numbers
are computed independently with pandas groupby and merge.
"""
import re

import numpy as np
import pandas as pd
import pytest

from core.dataset import LocalDiskStore
from core.ingestion import load_dataset
from core.operation import ApprovalStatus, Operation
from core.wrangling import describe_operation, execute_join, transform_join
from tests.test_join_ops import _customers as _step1_customers, _orders as _step1_orders, _params as _step1_params

SUMMARY = {"count": True, "columns": ["spend", "qty"], "stats": ["mean", "sum"]}


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _left():
    return pd.DataFrame({"customer_id": [1, 2, 3, 4], "region": ["E", "W", "E", "W"]})


def _right():
    # key 2 has an exact duplicate row; key 3's spend is all missing; the last row has a missing key; key 4 has no rows
    return pd.DataFrame({"customer_id": [1, 1, 2, 2, 2, 3, np.nan], "spend": [10.0, np.nan, 5.0, 5.0, np.nan, np.nan, 99.0],
                         "qty": [1, 2, 3, 3, 5, 6, 7]})


def _params(aggregate=SUMMARY, **extra):
    return {"right_table": "orders", "left_key": "customer_id", "right_key": "customer_id", "how": "left",
            "normalize_keys": False, "result_name": "customers_orders", "aggregate": aggregate, **extra}


def _run(left=None, right=None, **params):
    return transform_join(_left() if left is None else left, _params(**params),
                          {"orders": _right() if right is None else right})


def _expected_summary(right, columns, stats):
    keyed = right.dropna(subset=["customer_id"])
    grouped = keyed.groupby("customer_id")
    out = pd.DataFrame({"rows_in_orders": grouped.size()})
    for column in columns:
        for stat in stats:
            out[f"{column}_{stat}"] = grouped[column].mean() if stat == "mean" else grouped[column].sum(min_count=1)
    return out.reset_index()


# --- the summary and the join ------------------------------------------------------------------------------------

def test_the_transform_is_pure():
    left, right = _left(), _right()
    before_left, before_right = left.copy(), right.copy()
    _run(left, right)
    _run(left, right, normalize_keys=True, how="inner")
    pd.testing.assert_frame_equal(left, before_left)
    pd.testing.assert_frame_equal(right, before_right)


def test_the_summary_and_the_join():
    new_df, details, warnings = _run()
    summary = _expected_summary(_right(), ["spend", "qty"], ["mean", "sum"])
    assert summary["rows_in_orders"].tolist() == [2, 3, 1]  # the missing-key row is not counted
    expected = _left().merge(summary, on="customer_id", how="left")
    expected["rows_in_orders"] = expected["rows_in_orders"].fillna(0).astype(int)  # no rows: a real zero
    pd.testing.assert_frame_equal(new_df.reset_index(drop=True), expected, check_dtype=False)
    assert list(new_df.columns) == ["customer_id", "region", "rows_in_orders", "spend_mean", "spend_sum", "qty_mean",
                                    "qty_sum"]
    assert new_df["customer_id"].tolist() == [1, 2, 3, 4]  # the left rows and their order
    row3 = new_df[new_df["customer_id"] == 3].iloc[0]
    assert np.isnan(row3["spend_sum"]) and np.isnan(row3["spend_mean"])  # all missing: missing, not 0
    row4 = new_df[new_df["customer_id"] == 4].iloc[0]
    assert row4["rows_in_orders"] == 0 and row4[["spend_mean", "spend_sum", "qty_mean", "qty_sum"]].isna().all()
    assert details == {
        "right_table": "orders", "left_key": "customer_id", "right_key": "customer_id", "rows_left": 4, "rows_right": 7,
        "rows_after": 4, "matched_rows": 3, "unmatched_left_rows": 1,
        "columns_added": ["rows_in_orders", "spend_mean", "spend_sum", "qty_mean", "qty_sum"], "renamed_columns": {},
        "new_missing_cells": 1 * 4,  # one unmatched row x the four non-count columns; key 3's missing spend is not new
        "how": "left", "normalize_keys": False,
        "aggregate": {"rows_summarized": 7, "keys": 3, "columns": ["spend", "qty"], "stats": ["mean", "sum"],
                      "count_column": "rows_in_orders"},
        "carried_duplicate_rows": int(_right().duplicated().sum()), "carried_missing": {"spend": 3, "qty": 0}}
    assert details["carried_duplicate_rows"] == 1
    assert warnings == ("1 row(s) have no match in orders",
                        "1 exact duplicate row(s) in orders are counted again in the summary",
                        f"spend is {3 / 7:.1%} missing in orders: its mean and sum use only the rows that have a value")
    assert warnings[2] == "spend is 42.9% missing in orders: its mean and sum use only the rows that have a value"


def test_the_order_of_the_statistics_follows_the_request():
    new_df, details, _ = _run(aggregate={"count": True, "columns": ["qty"], "stats": ["sum", "mean"]})
    assert details["columns_added"] == ["rows_in_orders", "qty_sum", "qty_mean"]


@pytest.mark.parametrize("stats, verb", [(["mean"], "its mean uses"), (["sum"], "its sum uses"),
                                         (["mean", "sum"], "its mean and sum use")], ids=["mean", "sum", "both"])
def test_the_missing_share_warning(stats, verb):
    _, details, warnings = _run(aggregate={"count": True, "columns": ["spend"], "stats": stats})
    assert warnings[-1] == f"spend is 42.9% missing in orders: {verb} only the rows that have a value"
    assert details["columns_added"] == ["rows_in_orders", *[f"spend_{s}" for s in stats]]


def test_a_clean_carried_table_has_no_carried_warning():
    clean = pd.DataFrame({"customer_id": [1, 1, 2, 3, 4], "spend": [1.0, 2.0, 3.0, 4.0, 5.0]})
    _, details, warnings = _run(right=clean, aggregate={"count": True, "columns": ["spend"], "stats": ["mean"]})
    assert warnings == () and details["carried_duplicate_rows"] == 0 and details["carried_missing"] == {"spend": 0}


def test_a_count_only_summary():
    new_df, details, warnings = _run(aggregate={"count": True, "columns": [], "stats": []})
    assert details["columns_added"] == ["rows_in_orders"] and new_df["rows_in_orders"].tolist() == [2, 3, 1, 0]
    assert details["new_missing_cells"] == 0  # only the count column, and it holds 0 for no rows
    assert details["carried_missing"] == {} and warnings == (
        "1 row(s) have no match in orders", "1 exact duplicate row(s) in orders are counted again in the summary")


def test_normalized_keys_fall_in_one_group():
    left = pd.DataFrame({"region": ["East", "West", "North"], "x": [1, 2, 3]})
    right = pd.DataFrame({"region": ["East ", "east", "WEST", "west ", "west"], "amount": [1.0, 2.0, 3.0, 4.0, 5.0]})
    params = {"right_table": "orders", "left_key": "region", "right_key": "region", "how": "left", "normalize_keys": True,
              "aggregate": {"count": True, "columns": ["amount"], "stats": ["sum"]}}
    new_df, details, _ = transform_join(left, params, {"orders": right})
    assert new_df["rows_in_orders"].tolist() == [2, 3, 0] and new_df["amount_sum"].tolist()[:2] == [3.0, 12.0]
    assert new_df["region"].tolist() == ["East", "West", "North"]  # the left keys as they were
    assert details["aggregate"]["keys"] == 2


def test_a_repeating_left_key_does_not_multiply_rows():
    left = pd.DataFrame({"customer_id": [1, 1, 2, 2, 2, 5], "x": range(6)})
    new_df, details, _ = _run(left=left)
    assert len(new_df) == len(left) == details["rows_after"]
    assert new_df["rows_in_orders"].tolist() == [2, 2, 3, 3, 3, 0]


def test_the_inner_form():
    new_df, details, warnings = _run(how="inner")
    assert new_df["customer_id"].tolist() == [1, 2, 3] and details["new_missing_cells"] == 0
    assert warnings[0] == "1 row(s) have no match in orders and are removed"


def test_a_column_name_clash():
    left = _left().assign(rows_in_orders="taken", spend_mean=0.0)
    new_df, details, _ = _run(left=left)
    assert details["columns_added"] == ["rows_in_orders_orders", "spend_mean_orders", "spend_sum", "qty_mean", "qty_sum"]
    assert details["aggregate"]["count_column"] == "rows_in_orders_orders"
    assert new_df["rows_in_orders_orders"].tolist() == [2, 3, 1, 0] and new_df["rows_in_orders"].tolist() == ["taken"] * 4


@pytest.mark.parametrize("aggregate", [None, "absent"], ids=["none", "absent"])
def test_a_plain_join_is_unchanged(aggregate):
    params = _step1_params() if aggregate == "absent" else {**_step1_params(), "aggregate": None}
    frames = {"customers": _step1_customers()}
    new_df, details, warnings = transform_join(_step1_orders(), params, frames)
    expected = _step1_orders().merge(_step1_customers(), on="customer_id", how="left")
    pd.testing.assert_frame_equal(new_df.reset_index(drop=True), expected, check_dtype=False)
    assert "aggregate" not in details and "carried_missing" not in details
    assert details["new_missing_cells"] == 4 and warnings == ("2 row(s) have no match in customers",)


@pytest.mark.parametrize("aggregate, message", [
    ({"count": True, "columns": ["nope"], "stats": ["mean"]}, "Cannot summarize — column not found: nope"),
    ({"count": True, "columns": ["label"], "stats": ["mean"]}, "Cannot summarize — label is not numeric."),
    ({"count": True, "columns": ["flag"], "stats": ["sum"]}, "Cannot summarize — flag is not numeric."),
    ({"count": True, "columns": ["spend"], "stats": ["median"]}, "Unsupported statistic: median. Must be mean or sum."),
    ({"count": False, "columns": [], "stats": []}, "Cannot summarize — nothing to summarize."),
], ids=["missing", "text", "bool", "statistic", "nothing"])
def test_the_errors(aggregate, message):
    right = _right().assign(label="x", flag=True)
    with pytest.raises(ValueError) as error:
        _run(right=right, aggregate=aggregate)
    assert str(error.value) == message


# --- describe_operation ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("aggregate, extra, sentence", [
    ({"count": True, "columns": [], "stats": []}, {},
     "Summarize orders per customer_id (a row count), then join the summary to the analysis table, keeping every row."),
    ({"count": True, "columns": ["spend"], "stats": ["mean"]}, {},
     "Summarize orders per customer_id (a row count, and the mean of its numeric columns), then join the summary to "
     "the analysis table, keeping every row."),
    ({"count": True, "columns": ["spend"], "stats": ["sum"]}, {"how": "inner"},
     "Summarize orders per customer_id (a row count, and the sum of its numeric columns), then join the summary to the "
     "analysis table, keeping only the rows that have a match."),
    (SUMMARY, {"normalize_keys": True},
     "Summarize orders per customer_id (a row count, and the mean and sum of its numeric columns), then join the "
     "summary to the analysis table, keeping every row. Keys are compared after trimming spaces and ignoring case."),
], ids=["count", "mean", "sum_inner", "both_normalized"])
def test_the_words_without_details(aggregate, extra, sentence):
    text = describe_operation("join", _params(aggregate, **extra))
    assert text == sentence
    assert not re.search(r"\d", text) and "spend" not in text and "qty" not in text  # it reaches the prompt


def test_the_words_with_details():
    _, details, _ = _run()
    assert describe_operation("join", _params(), details) == (
        "Summarize orders per customer_id (a row count, and the mean and sum of spend and qty), then join the summary to "
        "the analysis table, keeping every row. Adds 5 column(s): rows_in_orders, spend_mean, spend_sum, qty_mean, "
        "qty_sum.")
    assert describe_operation("join", _params({"count": True, "columns": [], "stats": []}), {"columns_added": [
        "rows_in_orders"]}) == ("Summarize orders per customer_id (a row count), then join the summary to the analysis "
                                "table, keeping every row. Adds 1 column(s): rows_in_orders.")
    assert describe_operation("join", _step1_params()) == (
        "Join customers to the analysis table on customer_id, keeping every row.")  # a plain join as before


# --- execute_join ------------------------------------------------------------------------------------------------

def test_execute_an_aggregate_join(tmp_path, store):
    tables = {}
    for name, frame in (("customers", _left()), ("orders", _right())):
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        tables[name] = load_dataset(path, session_id="s", name=name, store=store)
    left, right = store.read(tables["customers"].storage_key), store.read(tables["orders"].storage_key)
    expected, details, warnings = transform_join(left, _params(), {"orders": right})
    op = Operation(op_type="join", session_id="s", input_handles=("customers@v1", "orders@v1"), params=_params(),
                   approval_status=ApprovalStatus.APPROVED)
    derived, finished = execute_join(tables["customers"], store, op, tables)
    assert (derived.name, derived.version, derived.parents, derived.created_from_operation, derived.storage_key) == (
        "customers_orders", 1, ("customers@v1", "orders@v1"), op.id, "s/customers_orders_v1.csv")
    path = tmp_path / "expected.csv"
    expected.to_csv(path, index=False)
    pd.testing.assert_frame_equal(store.read(derived.storage_key), pd.read_csv(path))
    impact = finished.impact
    assert (impact.rows_before, impact.rows_after, impact.columns_before, impact.columns_after) == (
        4, 4, 2, 7)
    assert impact.details == details and impact.warnings == warnings
    assert store.read(tables["orders"].storage_key).equals(right)  # the carried table is untouched


# --- the wording honours aggregate["count"] -------------------------------------------------------------------------

@pytest.mark.parametrize("stats, which", [(["mean"], "mean"), (["sum"], "sum"), (["mean", "sum"], "mean and sum")],
                         ids=["mean", "sum", "both"])
def test_the_words_without_a_count(stats, which):
    aggregate = {"count": False, "columns": ["spend"], "stats": stats}
    assert describe_operation("join", _params(aggregate)) == (
        f"Summarize orders per customer_id (the {which} of its numeric columns), then join the summary to the analysis "
        "table, keeping every row.")
    _, details, _ = _run(aggregate=aggregate)
    names = ", ".join(f"spend_{s}" for s in stats)
    assert details["columns_added"] == [f"spend_{s}" for s in stats] and details["aggregate"]["count_column"] is None
    assert describe_operation("join", _params(aggregate), details) == (
        f"Summarize orders per customer_id (the {which} of spend), then join the summary to the analysis table, "
        f"keeping every row. Adds {len(stats)} column(s): {names}.")


def test_the_words_with_a_count_are_unchanged():
    aggregate = {"count": True, "columns": ["spend"], "stats": ["mean"]}
    assert describe_operation("join", _params(aggregate)) == (
        "Summarize orders per customer_id (a row count, and the mean of its numeric columns), then join the summary to "
        "the analysis table, keeping every row.")
    _, details, _ = _run(aggregate=aggregate)
    assert describe_operation("join", _params(aggregate), details) == (
        "Summarize orders per customer_id (a row count, and the mean of spend), then join the summary to the analysis "
        "table, keeping every row. Adds 2 column(s): rows_in_orders, spend_mean.")
    assert describe_operation("join", _params({"count": False, "columns": [], "stats": []})) == (  # cannot run
        "Summarize orders per customer_id (a row count), then join the summary to the analysis table, keeping every row.")
