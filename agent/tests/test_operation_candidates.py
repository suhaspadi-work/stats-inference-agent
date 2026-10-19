"""
M2.3a: the code-enumerated cleaning candidates (core/operation_candidates.py): the eligibility rules with their exact
reasons and boundaries, the dry-run preview (nothing written; its numbers equal the executed impact), JSON-clean
candidates that never name a cell value, and at most one read of the data per call.
"""
import json

import numpy as np
import pandas as pd
import pytest

from core.capabilities import operation_named
from core.dataset import LocalDiskStore
from core.detection import CandidateIssue
from core.ingestion import load_dataset
from core.operation import ApprovalStatus, Operation
from core.operation_candidates import dry_run, enumerate_candidates
from core.thresholds import DISCRETE_MAX_DISTINCT, GROUP_MIN_OBSERVED, IMPUTE_MAX_MISSING_PCT, IMPUTE_MIN_OBSERVED

CANARY = "CANDIDATE-CANARY"


class CountingStore(LocalDiskStore):
    def __init__(self, base_dir):
        super().__init__(base_dir)
        self.reads, self.writes = [], []

    def read(self, storage_key):
        self.reads.append(storage_key)
        return super().read(storage_key)

    def write(self, storage_key, data):
        self.writes.append(storage_key)
        super().write(storage_key, data)


@pytest.fixture
def store(tmp_path):
    return CountingStore(tmp_path / "store")


def _dataset(df, tmp_path, store, name):
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    return load_dataset(path, session_id="s", name=name, store=store)


def _issue(column):
    return CandidateIssue(issue_type="high_missingness", description=f"{column} is missing", affected_columns=(column,),
                          evidence={})


def _candidates(df, tmp_path, store, column, name="c"):
    ds = _dataset(df, tmp_path, store, name)
    return {c["id"]: c for c in enumerate_candidates(_issue(column), ds, None, store)}


def _with_missing(values, missing):
    values = list(values)
    for i in missing:
        values[i] = np.nan
    return values


# --- the eligibility matrix ----------------------------------------------------------------------------------------------

def test_a_numeric_column_with_twenty_percent_missing(tmp_path, store):
    n = 100
    df = pd.DataFrame({"x": _with_missing(np.linspace(0, 99, n), range(20)), "y": range(n)})
    c = _candidates(df, tmp_path, store, "x")
    assert list(c) == ["leave_as_is", "delete_rows", "drop_column", "impute_mean", "impute_median", "impute_mode",
                       "fill_missing_category", "impute_median_indicator"]
    assert all(c[k]["eligible"] for k in ("leave_as_is", "delete_rows", "drop_column", "impute_mean", "impute_median",
                                          "impute_median_indicator"))
    assert (c["impute_mode"]["eligible"], c["impute_mode"]["reason"]) == (False, "the column is continuous; a mode would be arbitrary")
    assert (c["fill_missing_category"]["eligible"], c["fill_missing_category"]["reason"]) == (False, "needs a text column")
    assert c["delete_rows"]["reason"] == "removes the rows where x is missing (20.0% of the rows)"


def test_the_missing_share_boundary(tmp_path, store):
    at = pd.DataFrame({"x": _with_missing(np.linspace(0, 99, 100), range(40)), "y": range(100)})
    over = pd.DataFrame({"x": _with_missing(np.linspace(0, 99, 100), range(41)), "y": range(100)})
    assert IMPUTE_MAX_MISSING_PCT == 0.40
    assert _candidates(at, tmp_path, store, "x", "at")["impute_mean"]["eligible"]
    c = _candidates(over, tmp_path, store, "x", "over")
    for k in ("impute_mean", "impute_median", "impute_mode", "impute_median_indicator"):
        assert (c[k]["eligible"], c[k]["reason"]) == (False, "more than 40% of the values are missing"), k
        assert c[k]["preview"] is None
    assert c["delete_rows"]["eligible"] and c["drop_column"]["eligible"]


def test_the_observed_values_boundary(tmp_path, store):
    assert IMPUTE_MIN_OBSERVED == 10
    enough = pd.DataFrame({"x": _with_missing(np.linspace(0, 1, 15), range(5)), "y": range(15)})   # 10 observed
    short = pd.DataFrame({"x": _with_missing(np.linspace(0, 1, 15), range(6)), "y": range(15)})    # 9 observed
    assert _candidates(enough, tmp_path, store, "x", "enough")["impute_median"]["eligible"]
    c = _candidates(short, tmp_path, store, "x", "short")
    assert (c["impute_median"]["eligible"], c["impute_median"]["reason"]) == (False, "fewer than 10 observed values")


def test_the_mode_boundary(tmp_path, store):
    assert DISCRETE_MAX_DISTINCT == 20
    discrete = pd.DataFrame({"x": _with_missing([i % 20 for i in range(100)], range(10)), "y": range(100)})
    continuous = pd.DataFrame({"x": _with_missing([i % 21 for i in range(105)], range(10)), "y": range(105)})
    assert discrete["x"].nunique() == 20 and continuous["x"].nunique() == 21
    assert _candidates(discrete, tmp_path, store, "x", "discrete")["impute_mode"]["eligible"]
    small = pd.DataFrame({"x": _with_missing([i % 5 for i in range(50)], range(5)), "y": range(50)})
    assert _candidates(small, tmp_path, store, "x", "five")["impute_mode"]["eligible"]
    c = _candidates(continuous, tmp_path, store, "x", "continuous")
    assert c["impute_mode"]["reason"] == "the column is continuous; a mode would be arbitrary"


def test_a_text_column_with_missing_values(tmp_path, store):
    labels = [f"{CANARY}-{i % 30}" for i in range(100)]
    df = pd.DataFrame({"t": _with_missing(labels, range(8)), "y": range(100)})
    c = _candidates(df, tmp_path, store, "t")
    assert c["fill_missing_category"]["eligible"]
    assert c["fill_missing_category"]["what_was_run"] == "Fill the missing values of t with the label 'missing'."
    for k in ("impute_mean", "impute_median", "impute_median_indicator"):
        assert (c[k]["eligible"], c[k]["reason"]) == (False, "needs a numeric column"), k
    assert c["impute_mode"]["eligible"]
    assert not any(k.startswith("impute_group_median") for k in c)


def _grouped(levels_by_column, n=120, target_missing=range(0, 120, 7)):  # 7: every level keeps observed values
    data = {"target": _with_missing(np.linspace(0, 1, n), target_missing)}
    for name, levels in levels_by_column.items():
        data[name] = [f"{name}{i % levels}" for i in range(n)]
    return pd.DataFrame(data)


def test_only_three_group_columns_are_offered(tmp_path, store):
    df = _grouped({"g4": 4, "g2a": 2, "g3": 3, "g2b": 2, "g12": 12, "g13": 13})
    c = _candidates(df, tmp_path, store, "target")
    groups = [k for k in c if k.startswith("impute_group_median_")]
    assert groups == ["impute_group_median_g2a", "impute_group_median_g2b", "impute_group_median_g3"]
    assert all(c[k]["eligible"] for k in groups)
    assert c["impute_group_median_g2a"]["params"] == {"column": "target", "strategy": "median", "group_by": "g2a"}


def test_the_level_bounds_of_a_group_column(tmp_path, store):
    c = _candidates(_grouped({"g12": 12, "g13": 13, "one": 1}), tmp_path, store, "target")
    assert [k for k in c if k.startswith("impute_group_median_")] == ["impute_group_median_g12"]
    assert c["impute_group_median_g12"]["eligible"]


def test_group_columns_that_are_not_eligible(tmp_path, store):
    df = _grouped({"gmiss": 2, "gsmall": 2})
    df.loc[3, "gmiss"] = None
    df["gsmall"] = ["rare" if i < 4 else "common" for i in range(len(df))]  # rows 0 to 3: row 0 missing, 3 observed
    c = _candidates(df, tmp_path, store, "target")
    assert (c["impute_group_median_gmiss"]["eligible"], c["impute_group_median_gmiss"]["reason"]) == (
        False, "the group column gmiss has missing values")
    assert GROUP_MIN_OBSERVED == 5
    assert (c["impute_group_median_gsmall"]["eligible"], c["impute_group_median_gsmall"]["reason"]) == (
        False, "a group of gsmall has fewer than 5 observed values of target")


def test_the_indicator_name_clash(tmp_path, store):
    df = pd.DataFrame({"x": _with_missing(np.linspace(0, 1, 50), range(10)), "x_was_missing": 0})
    c = _candidates(df, tmp_path, store, "x")
    assert (c["impute_median_indicator"]["eligible"], c["impute_median_indicator"]["reason"]) == (
        False, "the column x_was_missing already exists")


def test_duplicate_rows_candidates(tmp_path, store):
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    ds = _dataset(df, tmp_path, store, "dupes")
    issue = CandidateIssue(issue_type="duplicate_rows", description="d", affected_columns=("a", "b"), evidence={})
    dedupe, leave = enumerate_candidates(issue, ds, None, store)
    assert (dedupe["id"], dedupe["op_type"], dedupe["params"], dedupe["eligible"]) == ("dedupe_all", "dedupe", {"subset": None}, True)
    assert dedupe["what_was_run"] == "Remove 1 exact duplicate rows (all columns)."
    assert leave["id"] == "leave_as_is" and leave["op_type"] is None
    assert {k: leave["preview"][k] for k in ("rows_changed", "cells_changed", "share_changed")} == {
        "rows_changed": 0, "cells_changed": 0, "share_changed": 0.0}
    assert leave["preview"]["rows_before"] == leave["preview"]["rows_after"] == 3


# --- JSON and cell values ---------------------------------------------------------------------------------------------------

def test_candidates_are_json_clean_and_hold_no_cell_value(tmp_path, store):
    n = 100
    df = pd.DataFrame({
        "t": _with_missing([f"{CANARY}-{i % 30}" for i in range(n)], range(10)),  # 30 levels: withheld
        "x": _with_missing(np.linspace(1000.123, 1999.987, n), range(15)),
        "g": [f"{CANARY}-group{i % 2}" for i in range(n)],
    })
    ds = _dataset(df, tmp_path, store, "canary")
    for column in ("t", "x"):
        for c in enumerate_candidates(_issue(column), ds, None, store):
            assert CANARY not in json.dumps(c), (column, c["id"])  # id, reason, what_was_run, params, preview
            if c["id"] == "impute_mode" and column == "t":
                assert c["what_was_run"] == "Fill the missing values of t with the mode."
                op = operation_named("impute")
                assert op.transform(df, c["params"])[1]["fill_value"] is None


@pytest.mark.parametrize("levels, named", [(20, True), (21, False)], ids=["twenty_levels", "twenty_one_levels"])
def test_the_text_mode_boundary(levels, named, tmp_path, store):
    values = [f"{CANARY}-{i % levels}" for i in range(120)]
    values += [f"{CANARY}-0"] * 10  # one clear mode
    df = pd.DataFrame({"t": _with_missing(values, range(1, 12)), "y": range(130)})
    assert df["t"].nunique() == levels
    c = _candidates(df, tmp_path, store, "t", f"levels_{levels}")["impute_mode"]
    assert c["eligible"]
    if named:
        assert c["what_was_run"] == f"Fill the missing values of t with the mode ({CANARY}-0)."
    else:
        assert c["what_was_run"] == "Fill the missing values of t with the mode." and CANARY not in json.dumps(c)


def test_a_low_cardinality_text_mode_is_named(tmp_path, store):
    df = pd.DataFrame({"region": _with_missing(["East"] * 50 + ["West"] * 30 + ["North"] * 20, range(80, 90)),
                       "y": range(100)})
    c = _candidates(df, tmp_path, store, "region")["impute_mode"]
    assert c["what_was_run"] == "Fill the missing values of region with the mode (East)."
    assert operation_named("impute").transform(df, c["params"])[1]["fill_value"] == "East"


# --- reads and writes -----------------------------------------------------------------------------------------------------------

def test_one_read_per_call_and_none_with_a_frame(tmp_path, store):
    df = pd.DataFrame({"x": _with_missing(np.linspace(0, 1, 50), range(10)), "g": ["a", "b"] * 25})
    ds = _dataset(df, tmp_path, store, "reads")
    store.reads.clear()
    enumerate_candidates(_issue("x"), ds, None, store)
    assert store.reads == [ds.storage_key]
    store.reads.clear()
    enumerate_candidates(_issue("x"), ds, None, store, store.read(ds.storage_key))
    assert store.reads == [ds.storage_key]  # only the read made by the test itself
    assert store.writes == [ds.storage_key]  # the upload; no candidate wrote anything


# --- the preview equals the executed impact ----------------------------------------------------------------------------------

def _frame_for_previews():
    n = 60
    return pd.DataFrame({
        "x": _with_missing(np.linspace(10, 70, n), range(0, n, 5)),
        "k": _with_missing([i % 4 for i in range(n)], range(1, n, 7)),
        "t": _with_missing(["a", "b", "c"] * (n // 3), range(2, n, 9)),
        "g": ["east", "west"] * (n // 2),
    })


@pytest.mark.parametrize("column", ["x", "k", "t"])
def test_dry_run_equals_the_executed_impact(column, tmp_path, store):
    df = _frame_for_previews()
    ds = _dataset(df, tmp_path, store, f"exec_{column}")
    for c in enumerate_candidates(_issue(column), ds, None, store):
        if not c["eligible"] or c["op_type"] is None:
            continue
        writes = len(store.writes)
        preview = dry_run(ds, store, c["op_type"], c["params"])
        assert len(store.writes) == writes  # a dry run writes nothing
        op = Operation(op_type=c["op_type"], session_id="s", input_handles=(ds.handle,), params=c["params"],
                       approval_status=ApprovalStatus.APPROVED)
        new_ds, finished = operation_named(c["op_type"]).execute(ds, store, op)
        impact = finished.impact
        assert (preview["rows_before"], preview["rows_after"], preview["columns_before"], preview["columns_after"]) == (
            impact.rows_before, impact.rows_after, impact.columns_before, impact.columns_after), c["id"]
        if c["op_type"] == "impute":
            assert preview["cells_changed"] == impact.details["cells_filled"], c["id"]
            if preview["shift"] is not None:
                assert preview["shift"] == {"column": column, "mean_before": impact.details["mean_before"],
                                            "mean_after": impact.details["mean_after"],
                                            "std_before": impact.details["std_before"],
                                            "std_after": impact.details["std_after"]}, c["id"]
        else:
            assert preview["rows_changed"] == impact.rows_before - impact.rows_after, c["id"]
        assert preview == c["preview"], c["id"]


def test_no_candidate_id_has_a_colon_or_whitespace(tmp_path, store):
    # M2.3a: a live model shortened "impute_group_median:region" to "impute_group_median"; ids are whole names
    cases = [
        ("x", pd.DataFrame({"x": _with_missing(np.linspace(0, 99, 100), range(20)), "y": range(100)})),
        ("t", pd.DataFrame({"t": _with_missing([f"v{i % 30}" for i in range(100)], range(8)), "y": range(100)})),
        ("target", _grouped({"g4": 4, "g2a": 2, "g3": 3, "g2b": 2, "g12": 12, "g13": 13})),
        ("x", pd.DataFrame({"x": _with_missing(np.linspace(0, 1, 50), range(10)), "x_was_missing": 0})),
    ]
    ids = []
    for i, (column, df) in enumerate(cases):
        ids += list(_candidates(df, tmp_path, store, column, f"ids_{i}"))
    dupes = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    ds = _dataset(dupes, tmp_path, store, "ids_dupes")
    issue = CandidateIssue(issue_type="duplicate_rows", description="d", affected_columns=("a", "b"), evidence={})
    ids += [c["id"] for c in enumerate_candidates(issue, ds, None, store)]
    assert "impute_group_median_g2a" in ids and "dedupe_all" in ids
    for cid in ids:
        assert ":" not in cid and not any(ch.isspace() for ch in cid), cid
