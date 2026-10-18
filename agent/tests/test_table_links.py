"""
M2.2: candidate keys between tables (core/table_links.py). Candidates come from the profiles only, bucketed by
normalized column name, so the cost never grows with every pair of tables; exact numbers are computed with pandas
only for the top candidates and checked here against pandas merges.
"""
import json
import time

import numpy as np
import pandas as pd
import pytest

from core.dataset import Dataset, LocalDiskStore
from core.ingestion import load_dataset
from core.model_safe_view import ColumnSummary, ModelSafeView
from core.table_links import KEY_LIKE_RATIO, TOP_LINKS, candidate_links, link_numbers, normalize_name, scan_links
from tests.test_capabilities import make_profile


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _view(rows, columns, handle="t@v1"):
    """columns: [(name, dtype, distinct, missing)]"""
    return ModelSafeView(dataset_handle=handle, row_count=rows, column_count=len(columns),
                         columns=tuple(ColumnSummary(name=n, dtype=d, missing_count=m, missing_pct=m / rows if rows else 0,
                                                     distinct_count=k) for n, d, k, m in columns))


def _dataset(df, tmp_path, store, name):
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    return load_dataset(path, session_id="s", name=name, store=store)


# --- normalize_name -------------------------------------------------------------------------------------------------

def test_normalize_name():
    assert normalize_name("Customer_ID") == normalize_name("customer id") == normalize_name("CUSTOMERID") == "customerid"
    assert normalize_name("") == "" and normalize_name("a-b.c") == "abc"


# --- candidate_links ---------------------------------------------------------------------------------------------------

def test_one_key_like_side():
    profiles = {"orders": make_profile({"Customer_ID": ("int64", 40)}),        # 40 of 100: not key-like
                "customers": make_profile({"customer id": ("int64", 100)})}    # 100 of 100: key-like
    (c,) = candidate_links(profiles)
    assert (c["table_a"], c["column_a"], c["table_b"], c["column_b"]) == ("orders", "Customer_ID", "customers", "customer id")
    assert (c["key_like_a"], c["key_like_b"], c["score"]) == (False, True, 1.0)


def test_pairs_that_are_not_candidates():
    assert candidate_links({"a": make_profile({"k": ("int64", 10)}), "b": make_profile({"k": ("int64", 20)})}) == []
    assert candidate_links({"a": make_profile({"k": ("float64", 100)}), "b": make_profile({"k": ("float64", 100)})}) == []
    assert candidate_links({"a": make_profile({"k": ("int64", 100)}), "b": make_profile({"k": ("str", 100)})}) == []
    same_table = {"a": make_profile({"Key": ("int64", 100), "key": ("int64", 100)}), "b": make_profile({"x": ("int64", 1)})}
    assert candidate_links(same_table) == []
    empty = {"a": _view(5, [("k", "int64", 0, 5)]), "b": _view(5, [("k", "int64", 1, 0)])}
    assert candidate_links(empty) == []  # no non-missing rows on one side, too few distinct on the other


def test_key_like_threshold():
    exactly = {"a": _view(10, [("k", "int64", 9, 0)]), "b": _view(10, [("k", "int64", 2, 0)])}
    (c,) = candidate_links(exactly)
    assert c["key_like_a"] and c["score"] == 0.9 == KEY_LIKE_RATIO
    missing = {"a": _view(10, [("k", "int64", 8, 2)]), "b": _view(10, [("k", "int64", 2, 0)])}
    assert candidate_links(missing)[0]["key_like_a"]  # 8 distinct of 8 non-missing rows


def test_three_tables_and_the_order():
    profiles = {"z": make_profile({"id": ("int64", 100)}), "a": make_profile({"id": ("int64", 100)}),
                "m": make_profile({"id": ("int64", 100)})}
    links = candidate_links(profiles)
    assert [(c["table_a"], c["table_b"]) for c in links] == [("a", "m"), ("z", "a"), ("z", "m")]
    assert all(c["score"] == 1.0 for c in links)
    scored = {"x": make_profile({"k1": ("int64", 95), "k2": ("int64", 100)}),
              "y": make_profile({"k1": ("int64", 95), "k2": ("int64", 100)})}
    assert [c["column_a"] for c in candidate_links(scored)] == ["k2", "k1"]  # the higher score first
    assert candidate_links(scored) == candidate_links(scored)


def test_two_thousand_tables_without_shared_names():
    profiles = {f"t{i}": make_profile({f"c{i}_{j}": ("int64", 100) for j in range(5)}) for i in range(2000)}
    start = time.perf_counter()
    assert candidate_links(profiles) == []
    assert time.perf_counter() - start < 1.0


def test_three_hundred_tables_sharing_one_key():
    profiles = {f"t{i:03d}": make_profile({"id": ("int64", 100)}) for i in range(300)}
    assert len(candidate_links(profiles)) == 300 * 299 // 2


# --- link_numbers --------------------------------------------------------------------------------------------------------

def _merge_rows(a, b, col_a, col_b):
    return len(a.dropna(subset=[col_a]).merge(b.dropna(subset=[col_b]), left_on=col_a, right_on=col_b, how="inner"))


@pytest.mark.parametrize("keys_a, keys_b, cardinality", [
    ([1, 2, 3, 4], [1, 2, 3, 9], "one_to_one"),
    ([1, 2, 3], [1, 1, 2, 2, 2, 7], "one_to_many"),
    ([1, 1, 2, 2, 3], [1, 2, 5], "many_to_one"),
    ([1, 1, 2, 3, 3], [1, 1, 1, 3, 4], "many_to_many"),
], ids=["one_to_one", "one_to_many", "many_to_one", "many_to_many"])
def test_link_numbers(keys_a, keys_b, cardinality, tmp_path, store):
    a = pd.DataFrame({"k": keys_a, "x": range(len(keys_a))})
    b = pd.DataFrame({"key": keys_b, "y": range(len(keys_b))})
    n = link_numbers(_dataset(a, tmp_path, store, "a"), "k", _dataset(b, tmp_path, store, "b"), "key", store)
    assert n["cardinality"] == cardinality
    sa, sb = set(keys_a), set(keys_b)
    assert (n["rows_a"], n["rows_b"], n["distinct_a"], n["distinct_b"]) == (len(keys_a), len(keys_b), len(sa), len(sb))
    assert n["matched_distinct"] == len(sa & sb)
    assert n["orphan_rows_a"] == sum(k not in sb for k in keys_a) and n["orphan_rows_b"] == sum(k not in sa for k in keys_b)
    assert n["inner_join_rows"] == _merge_rows(a, b, "k", "key")
    assert n["fan_out"] == round(n["inner_join_rows"] / len(keys_a), 3)
    assert n["matched_distinct_normalized"] == n["matched_distinct"] and n["normalizing_helps"] is False
    json.dumps(n)


def test_missing_keys(tmp_path, store):
    a = pd.DataFrame({"k": [1, 2, None, None]})
    b = pd.DataFrame({"k": [1, None, 3]})
    n = link_numbers(_dataset(a, tmp_path, store, "a"), "k", _dataset(b, tmp_path, store, "b"), "k", store)
    assert (n["missing_a"], n["missing_b"], n["distinct_a"], n["distinct_b"]) == (2, 1, 2, 2)
    assert (n["matched_distinct"], n["orphan_rows_a"], n["orphan_rows_b"]) == (1, 1, 1)
    assert n["inner_join_rows"] == _merge_rows(a, b, "k", "k") == 1


def test_text_keys_that_differ_by_case_and_spaces(tmp_path, store):
    a = pd.DataFrame({"region": ["East ", "West", "North"]})
    b = pd.DataFrame({"region": ["east", "West", "south"]})
    ds_a, ds_b = _dataset(a, tmp_path, store, "a"), _dataset(b, tmp_path, store, "b")
    assert store.read(ds_a.storage_key)["region"].tolist() == ["East ", "West", "North"]  # read back as stored
    n = link_numbers(ds_a, "region", ds_b, "region", store)
    assert (n["matched_distinct"], n["matched_distinct_normalized"], n["normalizing_helps"]) == (1, 2, True)


def test_no_overlap(tmp_path, store):
    a, b = pd.DataFrame({"k": [1, 2, 3]}), pd.DataFrame({"k": [7, 8]})
    n = link_numbers(_dataset(a, tmp_path, store, "a"), "k", _dataset(b, tmp_path, store, "b"), "k", store)
    assert (n["matched_distinct"], n["orphan_rows_a"], n["orphan_rows_b"], n["inner_join_rows"]) == (0, 3, 2, 0)


# --- scan_links ------------------------------------------------------------------------------------------------------------

class CountingStore(LocalDiskStore):
    def __init__(self, base_dir):
        super().__init__(base_dir)
        self.reads = []

    def read(self, storage_key):
        self.reads.append(storage_key)
        return super().read(storage_key)


def test_scan_links(tmp_path):
    store = CountingStore(tmp_path / "counting")
    n = 100
    a_df = pd.DataFrame({**{f"k{i:02d}": range(n) for i in range(25)}, **{f"m{i:02d}": [0, 1] * (n // 2) for i in range(25)}})
    b_df = pd.DataFrame({f"k{i:02d}": range(n) for i in range(25)})
    ds_a, ds_b = _dataset(a_df, tmp_path, store, "a"), _dataset(b_df, tmp_path, store, "b")
    ds_c = Dataset(session_id="s", name="c", version=1, storage_key="s/never_written.csv", parent_version=None,
                   created_from_operation=None)
    profiles = {"a": make_profile({**{f"k{i:02d}": ("int64", 100) for i in range(25)},
                                   **{f"m{i:02d}": ("int64", 2) for i in range(25)}}),
                "b": make_profile({f"k{i:02d}": ("int64", 100) for i in range(25)}),
                "c": make_profile({f"m{i:02d}": ("int64", 95) for i in range(25)})}  # score 0.95: after the k pairs (make_profile has 100 rows)
    store.reads.clear()
    scan = scan_links({"a": ds_a, "b": ds_b, "c": ds_c}, profiles, store)
    assert (scan["candidates_total"], len(scan["computed"]), scan["not_computed"]) == (50, TOP_LINKS, 30)
    assert all((c["table_a"], c["table_b"]) == ("a", "b") for c in scan["computed"])
    assert sorted(store.reads) == sorted([ds_a.storage_key, ds_b.storage_key])  # each once (20 pairs); c never read
    assert scan_links({"a": ds_a, "b": ds_b, "c": ds_c}, profiles, store, top_n=3)["not_computed"] == 47
    json.dumps(scan)


def test_top_links_is_read_at_call_time(tmp_path, monkeypatch):
    import core.table_links as table_links
    store = CountingStore(tmp_path / "counting")
    df = pd.DataFrame({f"k{i}": range(100) for i in range(5)})
    ds_a, ds_b = _dataset(df, tmp_path, store, "a"), _dataset(df, tmp_path, store, "b")
    profiles = {"a": make_profile({f"k{i}": ("int64", 100) for i in range(5)}),
                "b": make_profile({f"k{i}": ("int64", 100) for i in range(5)})}
    assert len(scan_links({"a": ds_a, "b": ds_b}, profiles, store)["computed"]) == 5
    monkeypatch.setattr(table_links, "TOP_LINKS", 2)
    scan = scan_links({"a": ds_a, "b": ds_b}, profiles, store)
    assert (len(scan["computed"]), scan["not_computed"]) == (2, 3)


def test_link_numbers_without_frames_reads_each_call(tmp_path):
    store = CountingStore(tmp_path / "counting")
    df = pd.DataFrame({"k": [1, 2, 3]})
    ds_a, ds_b = _dataset(df, tmp_path, store, "a"), _dataset(df, tmp_path, store, "b")
    store.reads.clear()
    link_numbers(ds_a, "k", ds_b, "k", store)
    link_numbers(ds_a, "k", ds_b, "k", store)
    assert store.reads == [ds_a.storage_key, ds_b.storage_key] * 2
    frames = {}
    link_numbers(ds_a, "k", ds_b, "k", store, frames)
    link_numbers(ds_a, "k", ds_b, "k", store, frames)
    assert len(store.reads) == 6 and set(frames) == {ds_a.storage_key, ds_b.storage_key}


def test_the_stored_candidates_are_capped():
    profiles = {"a": make_profile({f"k{i:03d}": ("int64", 100) for i in range(250)}),
                "b": make_profile({f"k{i:03d}": ("int64", 100) for i in range(250)})}
    scan = scan_links({}, profiles, store=None, top_n=0)
    assert (scan["candidates_total"], len(scan["candidates"]), scan["computed"], scan["not_computed"]) == (250, 200, [], 250)


def test_no_candidates(tmp_path, store):
    profiles = {"a": make_profile({"x": ("int64", 100)}), "b": make_profile({"y": ("int64", 100)})}
    scan = scan_links({}, profiles, store)
    assert scan == {"candidates": [], "candidates_total": 0, "computed": [], "not_computed": 0}
    json.dumps(scan)
