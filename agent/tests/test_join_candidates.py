"""
M2.3b: the candidates of a joinable_table issue (core/operation_candidates.py): the ids, the eligibility rules with
their exact reasons, a unique result name, a dry-run preview that writes nothing and equals the executed join, the
existing preview keys only, the sequential recomputation, and no cell value anywhere.
"""
import json

import numpy as np
import pandas as pd
import pytest

from core.capabilities import run_transform
from core.detection import CandidateIssue
from core.operation import ApprovalStatus, Operation
from core.operation_candidates import dry_run, enumerate_candidates, recompute_candidates
from core.wrangling import execute_join
from tests.test_operation_candidates import CountingStore, _dataset

PREVIEW_KEYS = {"rows_before", "rows_after", "columns_before", "columns_after", "rows_changed", "share_changed",
                "columns_affected", "cells_changed", "shift", "warnings"}
IDS = ["leave_as_is", "join_left", "join_inner", "join_left_normalized"]


@pytest.fixture
def store(tmp_path):
    return CountingStore(tmp_path / "store")


def _issue(right="customers", left_key="customer_id", right_key="customer_id"):
    return CandidateIssue(issue_type="joinable_table", description="d", affected_columns=(left_key,),
                          evidence={"right_table": right, "left_key": left_key, "right_key": right_key})


def _orders(n=40):
    rng = np.random.default_rng(8)
    return pd.DataFrame({"customer_id": rng.integers(1, 15, n), "total_spend": rng.normal(50, 5, n)})


def _customers(ids=range(1, 13)):
    ids = list(ids)
    return pd.DataFrame({"customer_id": ids, "region": ["East", "West"] * (len(ids) // 2) + ["East"] * (len(ids) % 2)})


def _setup(tmp_path, store, left, right, right_name="customers", extra=None):
    tables = {"orders": _dataset(left, tmp_path, store, "orders"), right_name: _dataset(right, tmp_path, store, right_name)}
    tables.update(extra or {})
    return tables


def _by_id(tmp_path, store, left, right, issue=None, extra=None, right_name="customers"):
    tables = _setup(tmp_path, store, left, right, right_name, extra)
    cands = enumerate_candidates(issue or _issue(right_name), tables["orders"], None, store, tables=tables, frames={})
    return tables, {c["id"]: c for c in cands}


# --- ids and eligibility -----------------------------------------------------------------------------------------

def test_the_ids_and_the_eligible_reasons(tmp_path, store):
    _, c = _by_id(tmp_path, store, _orders(), _customers())
    assert list(c) == IDS
    for cid in c:
        assert cid.replace("_", "").isalnum(), cid
    assert (c["leave_as_is"]["eligible"], c["leave_as_is"]["reason"]) == (True, "do not join; the table stays carried along")
    assert (c["join_left"]["eligible"], c["join_left"]["reason"]) == (
        True, "keeps every row of the analysis table; rows whose key is not found in customers get missing values in "
              "the added columns")
    assert (c["join_inner"]["eligible"], c["join_inner"]["reason"]) == (
        True, "keeps only the rows whose key is found in customers")
    assert (c["join_left_normalized"]["eligible"], c["join_left_normalized"]["reason"]) == (
        False, "comparing after trimming spaces and ignoring case matches no more keys")
    assert c["join_left"]["params"] == {"right_table": "customers", "left_key": "customer_id", "right_key": "customer_id",
                                        "how": "left", "normalize_keys": False, "result_name": "orders_customers"}
    assert c["join_inner"]["params"]["how"] == "inner" and c["join_left_normalized"]["params"]["normalize_keys"] is True


def test_a_repeating_right_key(tmp_path, store):
    _, c = _by_id(tmp_path, store, _orders(), pd.DataFrame({"customer_id": [1, 1, 2], "region": ["East", "West", "East"]}))
    for cid in ("join_left", "join_inner"):
        assert (c[cid]["eligible"], c[cid]["reason"], c[cid]["preview"]) == (
            False, "the key customer_id repeats in customers: a join would multiply rows", None)


def test_no_key_value_matches(tmp_path, store):
    _, c = _by_id(tmp_path, store, _orders(), _customers(range(100, 110)))
    for cid in ("join_left", "join_inner"):
        assert (c[cid]["eligible"], c[cid]["reason"]) == (False, "no key value matches")


def test_the_normalized_candidate(tmp_path, store):
    left = pd.DataFrame({"region": ["East", "West", "West", "North"] * 5, "x": range(20)})
    loose = pd.DataFrame({"region": ["east ", "WEST", "North"], "manager": ["Ana", "Ben", "Cy"]})
    _, c = _by_id(tmp_path, store, left, loose, _issue("zones", "region", "region"), right_name="zones")
    assert (c["join_left"]["eligible"], c["join_inner"]["eligible"]) == (True, True)  # North matches as given
    assert (c["join_left_normalized"]["eligible"], c["join_left_normalized"]["reason"]) == (
        True, "compares the keys after trimming spaces and ignoring case, which matches more rows")
    collides = pd.DataFrame({"region": ["east", "EAST ", "West"], "manager": ["Ana", "Ben", "Cy"]})
    _, c = _by_id(tmp_path, store, left, collides, _issue("zones", "region", "region"), right_name="zones")
    assert (c["join_left_normalized"]["eligible"], c["join_left_normalized"]["reason"]) == (
        False, "the key repeats in zones once its spaces and case are ignored")


def test_the_result_name_is_unique(tmp_path, store):
    taken = _dataset(_orders(), tmp_path, store, "orders_customers")
    _, c = _by_id(tmp_path, store, _orders(), _customers(), extra={"orders_customers": taken})
    assert {c[cid]["params"]["result_name"] for cid in IDS[1:]} == {"orders_customers_2"}


# --- previews ----------------------------------------------------------------------------------------------------

def test_every_preview_equals_the_executed_join(tmp_path, store):
    tables, c = _by_id(tmp_path, store, _orders(), _customers())
    checked = 0
    for cand in c.values():
        if not cand["eligible"] or cand["op_type"] is None:
            continue
        assert set(cand["preview"]) == PREVIEW_KEYS, cand["id"]
        writes = len(store.writes)
        frames = {"customers": store.read(tables["customers"].storage_key)}
        assert dry_run(tables["orders"], store, "join", cand["params"], frames) == cand["preview"]
        assert len(store.writes) == writes  # a dry run writes nothing
        op = Operation(op_type="join", session_id="s", input_handles=(tables["orders"].handle, tables["customers"].handle),
                       params=cand["params"], approval_status=ApprovalStatus.APPROVED)
        _, finished = execute_join(tables["orders"], store, op, tables)
        impact, preview = finished.impact, cand["preview"]
        assert (preview["rows_before"], preview["rows_after"], preview["columns_before"], preview["columns_after"]) == (
            impact.rows_before, impact.rows_after, impact.columns_before, impact.columns_after), cand["id"]
        assert preview["rows_changed"] == impact.rows_before - impact.rows_after, cand["id"]
        assert preview["cells_changed"] == impact.details["new_missing_cells"], cand["id"]
        assert preview["columns_affected"] == impact.details["columns_added"] == ["region"]
        assert preview["warnings"] == list(impact.warnings) and preview["shift"] is None
        checked += 1
    assert checked == 2  # join_left and join_inner


def test_the_right_table_is_read_once_per_call(tmp_path, store):
    tables = _setup(tmp_path, store, _orders(), _customers())
    store.reads.clear()
    enumerate_candidates(_issue(), tables["orders"], None, store, tables=tables)
    assert sorted(store.reads) == sorted([tables["orders"].storage_key, tables["customers"].storage_key])
    store.reads.clear()
    frames = {}
    df = store.read(tables["orders"].storage_key)
    enumerate_candidates(_issue(), tables["orders"], None, store, df, tables=tables, frames=frames)
    enumerate_candidates(_issue(), tables["orders"], None, store, df, tables=tables, frames=frames)
    assert store.reads == [tables["orders"].storage_key, tables["customers"].storage_key]  # cached in frames
    assert list(frames) == ["customers"]


def test_recompute_on_another_left_frame(tmp_path, store):
    tables, c = _by_id(tmp_path, store, _orders(), _customers())
    frames = {"customers": store.read(tables["customers"].storage_key)}
    other = store.read(tables["orders"].storage_key).iloc[:15]
    fresh = recompute_candidates(list(c.values()), tables["orders"], other, "after_earlier_operations", frames)
    for old, new in zip(c.values(), fresh):
        if not old["eligible"] or old["op_type"] is None:
            continue
        _, details, warnings = run_transform("join", other, old["params"], frames)
        assert new["preview"]["basis"] == "after_earlier_operations"
        assert {k: v for k, v in new["preview"].items() if k != "basis"} == dry_run(
            tables["orders"], other, "join", old["params"], frames)
        assert new["preview"]["rows_before"] == 15 and new["preview"]["cells_changed"] == details["new_missing_cells"]


# --- JSON and cell values ----------------------------------------------------------------------------------------

def test_no_cell_value_in_a_candidate(tmp_path, store):
    left = pd.DataFrame({"code": ["CANARY-KEY-A", "CANARY-KEY-B", "CANARY-KEY-Z"] * 4, "x": range(12)})
    right = pd.DataFrame({"code": ["CANARY-KEY-A", "CANARY-KEY-B"], "label": ["CANARY-VALUE-1", "CANARY-VALUE-2"]})
    _, c = _by_id(tmp_path, store, left, right, _issue("lookup", "code", "code"), right_name="lookup")
    for cand in c.values():
        text = json.dumps(cand, allow_nan=False)
        assert "CANARY" not in text, cand["id"]  # id, reason, what_was_run, params, preview and its warnings
