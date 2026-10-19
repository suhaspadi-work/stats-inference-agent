"""
M2.3b step 2: the candidates of an aggregatable_table issue (core/operation_candidates.py): the ids, the summarized
columns chosen by code, the eligibility rules with their exact reasons, a dry-run preview that writes nothing and equals
the executed aggregate join, the existing preview keys only, and plain words that hold no column name, digit or value.
"""
import json
import re

import numpy as np
import pandas as pd
import pytest

from core.detection import CandidateIssue
from core.operation import ApprovalStatus, Operation
from core.operation_candidates import _summarized_columns, dry_run, enumerate_candidates, recompute_candidates
from core.thresholds import MAX_AGG_COLUMNS
from core.wrangling import describe_operation, execute_join
from tests.test_operation_candidates import CountingStore, _dataset

PREVIEW_KEYS = {"rows_before", "rows_after", "columns_before", "columns_after", "rows_changed", "share_changed",
                "columns_affected", "cells_changed", "shift", "warnings"}
IDS = ["leave_as_is", "aggregate_left_summary", "aggregate_left_count", "aggregate_left_normalized"]


@pytest.fixture
def store(tmp_path):
    return CountingStore(tmp_path / "store")


def _issue(right="orders", left_key="customer_id", right_key="customer_id"):
    return CandidateIssue(issue_type="aggregatable_table", description="d", affected_columns=(left_key,),
                          evidence={"right_table": right, "left_key": left_key, "right_key": right_key})


def _customers(n=20):
    return pd.DataFrame({"customer_id": range(1, n + 1), "region": ["East", "West"] * (n // 2)})


def _orders(n=60):
    rng = np.random.default_rng(9)
    df = pd.DataFrame({"customer_id": rng.integers(1, 16, n),
                       "amount": rng.normal(20, 3, n), "quantity": rng.integers(1, 5, n),
                       "channel": rng.choice(["web", "store"], n), "gift": rng.choice([True, False], n)})
    df.loc[df.index % 7 == 0, "amount"] = np.nan
    return pd.concat([df, df.iloc[[3, 4]]], ignore_index=True)  # 2 exact duplicate rows


def _setup(tmp_path, store, left, right, extra=None):
    tables = {"customers": _dataset(left, tmp_path, store, "customers"), "orders": _dataset(right, tmp_path, store, "orders")}
    tables.update(extra or {})
    return tables


def _by_id(tmp_path, store, left=None, right=None, extra=None):
    tables = _setup(tmp_path, store, _customers() if left is None else left, _orders() if right is None else right, extra)
    candidates = enumerate_candidates(_issue(), tables["customers"], None, store, tables=tables, frames={})
    return tables, {c["id"]: c for c in candidates}


# --- ids, columns, eligibility -----------------------------------------------------------------------------------

def test_the_ids_columns_and_reasons(tmp_path, store):
    tables, c = _by_id(tmp_path, store)
    assert list(c) == IDS and all(cid.replace("_", "").isalnum() for cid in c)
    summary = {"count": True, "columns": ["amount", "quantity"], "stats": ["mean", "sum"]}
    base = {"right_table": "orders", "left_key": "customer_id", "right_key": "customer_id", "how": "left",
            "normalize_keys": False, "result_name": "customers_orders"}
    assert c["aggregate_left_summary"]["params"] == {**base, "aggregate": summary}
    assert c["aggregate_left_count"]["params"] == {**base, "aggregate": {"count": True, "columns": [], "stats": []}}
    assert c["aggregate_left_normalized"]["params"] == {**base, "normalize_keys": True, "aggregate": summary}
    assert (c["leave_as_is"]["eligible"], c["leave_as_is"]["reason"]) == (True, "do not join; the table stays carried along")
    assert (c["aggregate_left_summary"]["eligible"], c["aggregate_left_summary"]["reason"]) == (
        True, "summarizes orders per key (a row count, and the mean and sum of its numeric columns) and keeps every row "
              "of the analysis table")
    assert (c["aggregate_left_count"]["eligible"], c["aggregate_left_count"]["reason"]) == (
        True, "summarizes orders per key as a row count and keeps every row of the analysis table")
    assert (c["aggregate_left_normalized"]["eligible"], c["aggregate_left_normalized"]["reason"]) == (
        False, "comparing after trimming spaces and ignoring case matches no more keys")


def test_the_summarized_columns():
    assert MAX_AGG_COLUMNS == 5
    eight = pd.DataFrame({"k": [1, 1, 2], **{f"n{chr(97 + i)}": [1.5, 2.5, 3.5] for i in range(8)}})
    assert _summarized_columns(eight, "k") == ["na", "nb", "nc", "nd", "ne"]  # column order, at most 5
    mixed = pd.DataFrame({"k": [1, 1, 2], "row_id": [7, 8, 9], "count": [1, 1, 2], "flag": [True, False, True],
                          "label": ["a", "b", "c"], "price": [1.0, 2.0, 2.0]})
    assert _summarized_columns(mixed, "k") == ["count", "price"]  # the identifier, bool and text columns are skipped


def test_no_numeric_column(tmp_path, store):
    orders = _orders()[["customer_id", "channel"]]
    _, c = _by_id(tmp_path, store, right=orders)
    assert (c["aggregate_left_summary"]["eligible"], c["aggregate_left_summary"]["reason"]) == (
        False, "orders has no numeric column to summarize")
    assert c["aggregate_left_count"]["eligible"]
    assert c["aggregate_left_normalized"]["params"]["aggregate"] == {"count": True, "columns": [], "stats": []}


def test_no_key_value_matches(tmp_path, store):
    orders = _orders().assign(customer_id=lambda d: d["customer_id"] + 100)
    _, c = _by_id(tmp_path, store, right=orders)
    for cid in ("aggregate_left_summary", "aggregate_left_count"):
        assert (c[cid]["eligible"], c[cid]["reason"], c[cid]["preview"]) == (False, "no key value matches", None)


def test_the_normalized_candidate(tmp_path, store):
    left = pd.DataFrame({"code": ["Ka", "Kb", "Kc"] * 4, "y": range(12)})
    right = pd.DataFrame({"code": ["ka ", "ka ", "KB", "KB", "Kc"], "amount": [1.0, 2.0, 3.0, 4.0, 5.0]})
    tables = {"customers": _dataset(left, tmp_path, store, "customers"), "orders": _dataset(right, tmp_path, store, "orders")}
    c = {x["id"]: x for x in enumerate_candidates(_issue(left_key="code", right_key="code"), tables["customers"], None,
                                                   store, tables=tables)}
    assert (c["aggregate_left_normalized"]["eligible"], c["aggregate_left_normalized"]["reason"]) == (
        True, "compares the keys after trimming spaces and ignoring case, which matches more rows")


def test_the_result_name_is_unique(tmp_path, store):
    taken = _dataset(_customers(), tmp_path, store, "customers_orders")
    _, c = _by_id(tmp_path, store, extra={"customers_orders": taken})
    assert {c[cid]["params"]["result_name"] for cid in IDS[1:]} == {"customers_orders_2"}


# --- previews ----------------------------------------------------------------------------------------------------

def test_every_preview_equals_the_executed_join(tmp_path, store):
    tables, c = _by_id(tmp_path, store)
    right = store.read(tables["orders"].storage_key)
    duplicates, missing = int(right.duplicated().sum()), int(right["amount"].isna().sum())
    assert (duplicates, missing) == (2, 9)
    checked = 0
    for cand in c.values():
        if not cand["eligible"] or cand["op_type"] is None:
            continue
        preview = cand["preview"]
        assert set(preview) == PREVIEW_KEYS, cand["id"]
        writes = len(store.writes)
        assert dry_run(tables["customers"], store, "join", cand["params"], {"orders": right}) == preview
        assert len(store.writes) == writes  # a dry run writes nothing
        op = Operation(op_type="join", session_id="s", input_handles=("customers@v1", "orders@v1"), params=cand["params"],
                       approval_status=ApprovalStatus.APPROVED)
        _, finished = execute_join(tables["customers"], store, op, tables)
        impact = finished.impact
        assert (preview["rows_before"], preview["rows_after"], preview["columns_before"], preview["columns_after"]) == (
            impact.rows_before, impact.rows_after, impact.columns_before, impact.columns_after), cand["id"]
        assert preview["rows_changed"] == impact.rows_before - impact.rows_after == 0
        assert preview["cells_changed"] == impact.details["new_missing_cells"]
        assert preview["columns_affected"] == impact.details["columns_added"] and preview["warnings"] == list(impact.warnings)
        assert f"{duplicates} exact duplicate row(s) in orders are counted again in the summary" in preview["warnings"]
        checked += 1
    assert checked == 2
    assert (f"amount is {missing / len(right):.1%} missing in orders: its mean and sum use only the rows that have a value"
            in c["aggregate_left_summary"]["preview"]["warnings"])


def test_reads_and_recompute(tmp_path, store):
    tables = _setup(tmp_path, store, _customers(), _orders())
    store.reads.clear()
    frames = {}
    df = store.read(tables["customers"].storage_key)
    candidates = enumerate_candidates(_issue(), tables["customers"], None, store, df, tables=tables, frames=frames)
    enumerate_candidates(_issue(), tables["customers"], None, store, df, tables=tables, frames=frames)
    assert store.reads == [tables["customers"].storage_key, tables["orders"].storage_key]  # the right table once
    other = df.iloc[:8]
    fresh = recompute_candidates(candidates, tables["customers"], other, "after_earlier_operations", frames)
    for old, new in zip(candidates, fresh):
        if old["eligible"] and old["op_type"] is not None:
            assert new["preview"]["basis"] == "after_earlier_operations" and new["preview"]["rows_before"] == 8
            assert {k: v for k, v in new["preview"].items() if k != "basis"} == dry_run(
                tables["customers"], other, "join", old["params"], frames)


def test_plain_words_hold_no_name_digit_or_value(tmp_path, store):
    orders = pd.DataFrame({"customer_id": [1, 1, 2, 2, 3], "zz_secret_amount": [31415.926, 2.0, 3.0, 4.0, 5.0],
                           "zz_note": ["CANARY-ONE", "CANARY-TWO", "x", "y", "z"]})
    _, c = _by_id(tmp_path, store, right=orders)
    for cand in c.values():
        json.dumps(cand, allow_nan=False)
        assert "CANARY" not in json.dumps(cand) and "31415" not in json.dumps(cand), cand["id"]
        if cand["op_type"] is not None:
            words = describe_operation(cand["op_type"], cand["params"])  # what the proposer sees
            assert not re.search(r"\d", words) and "zz_" not in words, cand["id"]
