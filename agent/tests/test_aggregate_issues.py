"""
M2.3b step 2: aggregatable_table issues (core/detection.py detect_join_issues). A carried table whose best usable link
has a REPEATING carried key and at least one match raises one aggregatable_table issue (names only in its evidence); a
unique-key link wins a tie and raises the joinable_table issue of step 1; a table already joined in raises neither.
"""
import dataclasses
import json
import re
import time

import numpy as np
import pandas as pd
import pytest

from core.dataset import Dataset, LocalDiskStore
from core.detection import detect_all_issues, detect_join_issues
from core.ingestion import profile_dataset
from core.table_links import scan_links
from tests.test_join_issues import _bundle


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _customers(n=20):
    rng = np.random.default_rng(6)
    return pd.DataFrame({"customer_id": range(1, n + 1), "region": rng.choice(["East", "West"], n)})


def _orders(n=60, key="customer_id"):
    rng = np.random.default_rng(7)
    return pd.DataFrame({key: rng.integers(1, 16, n), "amount": rng.normal(20, 3, n), "quantity": rng.integers(1, 5, n)})


def _issues(tmp_path, store, frames, analysis="customers"):
    tables, profiles = _bundle(tmp_path, store, frames)
    return detect_join_issues(analysis, tables, profiles, store)


# --- the issue ---------------------------------------------------------------------------------------------------

def test_a_repeating_key_raises_an_aggregatable_issue(tmp_path, store):
    (issue,) = _issues(tmp_path, store, {"customers": _customers(), "orders": _orders()})
    assert (issue.issue_type, issue.affected_columns) == ("aggregatable_table", ("customer_id",))
    assert issue.evidence == {"right_table": "orders", "left_key": "customer_id", "right_key": "customer_id"}
    assert issue.description == ("The table 'orders' (columns: customer_id, amount, quantity) has several rows for each "
                                 "value of the key 'customer_id' it shares with the analysis table.")


def test_the_two_name_key_and_the_column_cap(tmp_path, store):
    wide = _orders(key="CustomerID").assign(**{f"col_{chr(97 + i)}": 1.0 for i in range(14)})
    (issue,) = _issues(tmp_path, store, {"customers": _customers(), "orders": wide})
    names = list(wide.columns)
    assert issue.evidence == {"right_table": "orders", "left_key": "customer_id", "right_key": "CustomerID"}
    assert issue.description == (f"The table 'orders' (columns: {', '.join(names[:15])} and {len(names) - 15} more) has "
                                 "several rows for each value of the key 'customer_id' (called 'CustomerID' there) it "
                                 "shares with the analysis table.")


def test_a_many_to_many_link_raises_one(tmp_path, store):
    main = pd.DataFrame({"code": ["a", "b"] * 20, "y": range(40)})
    carried = pd.DataFrame({"code": [f"c{chr(97 + i)}" for i in range(19)] + ["a"], "z": range(20)})
    carried.loc[0, "code"] = "a"
    tables, profiles = _bundle(tmp_path, store, {"main": main, "carried": carried})
    assert scan_links(tables, profiles, store)["computed"][0]["cardinality"] == "many_to_many"
    assert [i.issue_type for i in detect_join_issues("main", tables, profiles, store)] == ["aggregatable_table"]


def test_no_match_raises_none_and_a_normalized_match_is_usable(tmp_path, store):
    far = pd.DataFrame({"customer_id": [101, 101, 102], "amount": [1.0, 2.0, 3.0]})
    assert _issues(tmp_path, store, {"customers": _customers(), "far": far}) == []
    main = pd.DataFrame({"code": [f"K{chr(97 + i)}" for i in range(20)], "y": range(20)})
    loose = pd.DataFrame({"code": ["ka ", "ka ", "kb", "kb"], "amount": [1.0, 2.0, 3.0, 4.0]})  # repeats as given
    tables, profiles = _bundle(tmp_path, store, {"main": main, "loose": loose})
    link = scan_links(tables, profiles, store)["computed"][0]
    assert (link["matched_distinct"], link["matched_distinct_normalized"]) == (0, 2)
    (issue,) = detect_join_issues("main", tables, profiles, store)
    assert issue.issue_type == "aggregatable_table"


def test_a_unique_key_link_wins_a_tie(tmp_path, store):
    main = _customers().assign(account=lambda d: [f"acc{chr(97 + i)}" for i in range(len(d))])
    both = pd.DataFrame({"customer_id": range(1, 21), "account": ["acca", "accb"] * 10, "tier": ["x", "y"] * 10})
    tables, profiles = _bundle(tmp_path, store, {"customers": main, "both": both})
    scores = {c["column_a"]: (c["score"], c["unique_b"]) for c in scan_links(tables, profiles, store)["computed"]}
    # a unique carried key always has a ratio of 1.0, so a repeating-key link can at most tie it (here both are 1.0)
    assert scores == {"account": (1.0, False), "customer_id": (1.0, True)}
    (issue,) = detect_join_issues("customers", tables, profiles, store)
    assert (issue.issue_type, issue.evidence["left_key"]) == ("joinable_table", "customer_id")  # unique before repeating


def test_two_carried_tables_in_profile_order_and_none_between_carried(tmp_path, store):
    regions = pd.DataFrame({"region": ["East", "West"], "manager": ["Ana", "Ben"]})
    issues = _issues(tmp_path, store, {"customers": _customers(), "orders": _orders(), "regions": regions})
    assert [(i.issue_type, i.evidence["right_table"]) for i in issues] == [
        ("aggregatable_table", "orders"), ("joinable_table", "regions")]
    one = pd.DataFrame({"code": ["a", "a", "b"], "x": [1, 2, 3]})
    two = pd.DataFrame({"code": ["a", "b", "c"], "y": [4, 5, 6]})
    assert _issues(tmp_path, store, {"customers": _customers(), "one": one, "two": two}) == []


def test_no_number_in_an_issue(tmp_path, store):
    orders = _orders().rename(columns={"quantity": "units"})
    (issue,) = _issues(tmp_path, store, {"customers": _customers(), "orders": orders})
    assert not re.search(r"\d", json.dumps(dataclasses.asdict(issue)))  # the names hold no digit


# --- the lineage skip --------------------------------------------------------------------------------------------

def test_an_aggregate_joined_table_raises_none(tmp_path, store):
    tables, plain = _bundle(tmp_path, store, {"customers": _customers(), "orders": _orders()})
    joined = Dataset.derived("s", "customers_orders", tables["customers"].storage_key,
                             parents=("customers@v1", "orders@v1"), operation_id="op")
    for analysis in (joined, joined.next_version(tables["customers"].storage_key, "op2")):  # v1, then a v2
        all_tables = {**tables, "customers_orders": analysis}
        profiles = {**plain, "customers_orders": profile_dataset(analysis, store=store)}
        assert detect_join_issues("customers_orders", all_tables, profiles, store) == []
    assert [i.issue_type for i in detect_join_issues("customers", tables, plain, store)] == ["aggregatable_table"]


# --- detect_all_issues and the cost --------------------------------------------------------------------------------

def test_detect_all_issues(tmp_path, store):
    main = _customers().assign(note=lambda d: np.where(np.arange(len(d)) % 3 == 0, "x ", "y"))  # stray spaces
    tables, profiles = _bundle(tmp_path, store, {"customers": main, "orders": _orders()})
    before = detect_all_issues(tables["customers"], profiles["customers"], store)
    assert [i.issue_type for i in before] == ["inconsistent_text"]
    issues = detect_all_issues(tables["customers"], profiles["customers"], store, tables=tables, profiles=profiles,
                               analysis="customers")
    assert issues[:1] == before and [i.issue_type for i in issues[1:]] == ["aggregatable_table"]


def test_five_hundred_tables(tmp_path, store):
    frames = {"customers": _customers()}
    for i in range(497):
        frames[f"t{i:03d}"] = pd.DataFrame({f"only_{i:03d}": [i]})
    for name in ("orders", "visits", "returns"):
        frames[name] = _orders()
    tables, profiles = _bundle(tmp_path, store, frames)
    started = time.perf_counter()
    issues = detect_join_issues("customers", tables, profiles, store)
    assert time.perf_counter() - started < 10
    assert [(i.issue_type, i.evidence["right_table"]) for i in issues] == [
        ("aggregatable_table", "orders"), ("aggregatable_table", "visits"), ("aggregatable_table", "returns")]
