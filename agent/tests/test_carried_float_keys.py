"""
M2.4b (G80): a numeric key column with a missing value is read as float; a float column whose non-missing values are all
whole numbers now counts as an integer-like key, so its table is linked, joined and aggregated. Expected values are
computed independently with pandas.
"""
import time

import numpy as np
import pandas as pd
import pytest

from core.detection import detect_join_issues
from core.graph import build_agent_graph
from core.ingestion import profile_dataset
from core.plan import QuestionType
from core.table_links import candidate_links, scan_links
from core.wrangling import transform_join
from tests.test_decision_point_guard import _dataset, choose, confident, fake_draft
from tests.test_join_run import _bundle, _state, _store_frame
from tests.test_operation_candidates import CountingStore
from tests.test_prompt_canaries import fake_model  # noqa: F401 (fixture)
from tests.test_stage1_loop import _run


@pytest.fixture
def store(tmp_path):
    return CountingStore(tmp_path / "store")


LEFT_FLOAT = pd.DataFrame({"customer_id": [1, 2, 3, None, 5, 6], "region": list("EWEWEW")})
RIGHT_INT = pd.DataFrame({"customer_id": [1, 2, 3, 4, 5], "tier": list("abcde")})
MANY_INT = pd.DataFrame({"customer_id": [1, 1, 2, 5, 5, 5], "amount": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
LEFT_INT = pd.DataFrame({"customer_id": [1, 2, 3, 4, 5, 6], "region": list("EWEWEW")})
RIGHT_FLOAT = pd.DataFrame({"customer_id": [1, 2, 3, None, 5], "tier": list("abcde")})
MANY_FLOAT = pd.DataFrame({"customer_id": [1, 1, 2, None, 5, 5], "amount": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})


def _tables(tmp_path, store, frames):
    tables = {name: _dataset(df, tmp_path, store, name) for name, df in frames.items()}
    return tables, {name: profile_dataset(ds, store=store) for name, ds in tables.items()}


# --- candidate_links and scan_links ------------------------------------------------------------------------------

def test_candidate_links_skips_a_float_key_without_whole_floats(tmp_path, store):
    tables, profiles = _tables(tmp_path, store, {"left": LEFT_FLOAT, "right": RIGHT_INT})
    assert str(store.read(tables["left"].storage_key)["customer_id"].dtype) == "float64"
    assert candidate_links(profiles) == []  # exactly as before M2.4b
    (link,) = candidate_links(profiles, whole_floats={("left", "customer_id")})
    assert (link["table_a"], link["column_a"], link["table_b"], link["column_b"]) == (
        "left", "customer_id", "right", "customer_id")


def test_scan_links_finds_a_float_key_reading_each_table_once(tmp_path, store):
    tables, profiles = _tables(tmp_path, store, {"left": LEFT_INT, "right": RIGHT_FLOAT, "many": MANY_FLOAT})
    store.reads.clear()
    scan = scan_links(tables, profiles, store)
    pairs = {(c["table_a"], c["table_b"]) for c in scan["computed"]}
    assert pairs == {("left", "right"), ("left", "many"), ("right", "many")}
    assert sorted(store.reads) == sorted(ds.storage_key for ds in tables.values())  # each table once


def test_several_float_columns_and_fifty_tables(tmp_path, store):
    frames = {"left": LEFT_INT.assign(visits=[1.0, 2.0, None, 4.0, 5.0, 6.0])}
    for i in range(48):
        frames[f"t{i:02d}"] = pd.DataFrame({f"only_{i:02d}": [1.5, 2.5]})
    frames["right"] = RIGHT_FLOAT.assign(visits=[1.0, None, 3.0, 4.0, 5.0])
    tables, profiles = _tables(tmp_path, store, frames)
    store.reads.clear()
    started = time.perf_counter()
    scan = scan_links(tables, profiles, store)
    assert time.perf_counter() - started < 10
    assert {(c["column_a"], c["column_b"]) for c in scan["computed"]} == {("customer_id", "customer_id"),
                                                                         ("visits", "visits")}
    assert sorted(store.reads) == sorted([tables["left"].storage_key, tables["right"].storage_key])  # once each


def test_a_fractional_float_is_not_a_key(tmp_path, store):
    fractional = pd.DataFrame({"customer_id": [1.5, 2.0, 3.0, None], "x": [1, 2, 3, 4]})
    tables, profiles = _tables(tmp_path, store, {"left": LEFT_INT, "fractional": fractional})
    assert scan_links(tables, profiles, store)["computed"] == []


def test_a_whole_float_in_one_table_is_not_read(tmp_path, store):
    # lonely_id is a whole-number float, but no other table has a column of that name: it is not even read
    alone = pd.DataFrame({"lonely_id": [1.0, 2.0, None], "note": ["a", "b", "c"]})
    tables, profiles = _tables(tmp_path, store, {"left": LEFT_INT, "alone": alone})
    store.reads.clear()
    scan_links(tables, profiles, store)
    assert store.reads.count(tables["alone"].storage_key) == 0


# --- the four join shapes -------------------------------------------------------------------------------------------

BASE = {"left_key": "customer_id", "right_key": "customer_id", "how": "left", "normalize_keys": False}


@pytest.mark.parametrize("left, right", [(LEFT_FLOAT, RIGHT_INT), (LEFT_INT, RIGHT_FLOAT)], ids=["left_float", "right_float"])
def test_a_plain_join(left, right):
    joined, details, _ = transform_join(left, {**BASE, "right_table": "right", "result_name": "x"}, {"right": right})
    present = right.dropna(subset=["customer_id"])
    expected = left["customer_id"].map(dict(zip(present["customer_id"], present["tier"])))
    expected[left["customer_id"].isna()] = np.nan  # a missing key never matches
    assert joined["tier"].fillna("<na>").tolist() == expected.fillna("<na>").tolist()
    assert details["matched_rows"] == int(expected.notna().sum())


@pytest.mark.parametrize("left, many", [(LEFT_FLOAT, MANY_INT), (LEFT_INT, MANY_FLOAT)], ids=["left_float", "right_float"])
def test_an_aggregate_join(left, many):
    params = {**BASE, "right_table": "many", "result_name": "y",
              "aggregate": {"count": True, "columns": ["amount"], "stats": ["sum"]}}
    joined, details, _ = transform_join(left, params, {"many": many})
    keyed = many.dropna(subset=["customer_id"])  # a right row with a missing key is not counted
    counts = keyed.groupby("customer_id").size()
    sums = keyed.groupby("customer_id")["amount"].sum()
    expected_counts = [int(counts.get(k, 0)) if pd.notna(k) else 0 for k in left["customer_id"]]
    assert joined["rows_in_many"].tolist() == expected_counts
    expected_sums = [float(sums[k]) if pd.notna(k) and k in sums.index else None for k in left["customer_id"]]
    assert [None if pd.isna(v) else float(v) for v in joined["amount_sum"]] == expected_sums


def test_the_issue_type_follows_whether_the_key_repeats(tmp_path, store):
    tables, profiles = _tables(tmp_path, store, {"left": LEFT_INT, "unique": RIGHT_FLOAT})
    assert [i.issue_type for i in detect_join_issues("left", tables, profiles, store)] == ["joinable_table"]
    tables, profiles = _tables(tmp_path, store, {"left": LEFT_INT, "repeats": MANY_FLOAT})
    assert [i.issue_type for i in detect_join_issues("left", tables, profiles, store)] == ["aggregatable_table"]


# --- end to end ------------------------------------------------------------------------------------------------------

def test_the_graph_joins_a_float_keyed_table(fake_model, tmp_path, store):
    rng = np.random.default_rng(91)
    main = pd.DataFrame({"customer_id": range(1, 41), "region": ["East", "West"] * 20,
                         "total_spend": rng.normal(50, 5, 40)})
    keys = [float(k) for k in range(1, 41)]
    keys[5] = None  # one missing key: the column is read as float
    tiers = pd.DataFrame({"customer_id": keys, "tier": ["basic", "plus"] * 20})
    tables = _bundle(tmp_path, store, {"main": main, "tiers": tiers})
    assert str(_store_frame(store, tables["tiers"])["customer_id"].dtype) == "float64"
    app = build_agent_graph(
        store, classify_fn=confident(QuestionType.TWO_GROUP_COMPARISON), draft_fn=fake_draft(["region"]),
        propose_fn=choose({}))
    reject_missing = lambda p: {"decisions": [{"index": o["index"], "action": "reject" if o["op_type"] != "join"
                                               else "approve"} for o in p["proposed_operations"]]}
    values, _, _ = _run(app, _state(tables), "graph", wrangling_approval=reject_missing)
    joined = _store_frame(store, values["dataset"])
    expected = main.merge(tiers.dropna(subset=["customer_id"]).astype({"customer_id": int}), on="customer_id",
                          how="left")
    assert joined["tier"].fillna("<na>").tolist() == expected["tier"].fillna("<na>").tolist()
    assert int(joined["tier"].isna().sum()) == 1  # customer 6: its key was missing in tiers
