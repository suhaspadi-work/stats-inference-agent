"""
M2.3b: the join issues (core/detection.py detect_join_issues and detect_all_issues). One joinable_table issue per
carried table whose best link with the analysis table has a unique carried key and at least one match (as given or
after normalizing); the analysis table is always the left side; the evidence holds names only and no link number.
"""
import dataclasses
import json
import re
import time

import numpy as np
import pandas as pd
import pytest

from core.dataset import LocalDiskStore
from core.detection import detect_all_issues, detect_join_issues
from core.ingestion import load_bundle, profile_dataset


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _bundle(tmp_path, store, frames):
    folder = tmp_path / "files"
    folder.mkdir(exist_ok=True)
    paths = []
    for name, frame in frames.items():
        paths.append(folder / f"{name}.csv")
        frame.to_csv(paths[-1], index=False)
    tables = load_bundle(paths, session_id="s", store=store)
    return tables, {name: profile_dataset(ds, store=store) for name, ds in tables.items()}


def _main(n=60):
    rng = np.random.default_rng(5)
    return pd.DataFrame({"customer_id": rng.integers(1, 21, n), "region": rng.choice(["East", "West"], n),
                         "total_spend": rng.normal(50, 5, n)})


def _issues(tmp_path, store, frames, analysis="main"):
    tables, profiles = _bundle(tmp_path, store, frames)
    return detect_join_issues(analysis, tables, profiles, store)


REGIONS = pd.DataFrame({"region": ["East", "West"], "manager": ["Ana", "Ben"]})


# --- a usable link -----------------------------------------------------------------------------------------------

def test_a_many_to_one_link(tmp_path, store):
    (issue,) = _issues(tmp_path, store, {"main": _main(), "regions": REGIONS})
    assert (issue.issue_type, issue.affected_columns) == ("joinable_table", ("region",))
    assert issue.evidence == {"right_table": "regions", "left_key": "region", "right_key": "region"}
    assert issue.description == ("The table 'regions' (columns: region, manager) shares the key 'region' with the "
                                 "analysis table.")


def test_keys_with_different_names_and_the_analysis_table_second(tmp_path, store):
    customers = pd.DataFrame({"CustomerID": range(1, 21), "tier": ["basic", "plus"] * 10})
    (issue,) = _issues(tmp_path, store, {"customers": customers, "main": _main()})  # the analysis table is second
    assert issue.affected_columns == ("customer_id",)
    assert issue.evidence == {"right_table": "customers", "left_key": "customer_id", "right_key": "CustomerID"}
    assert issue.description == ("The table 'customers' (columns: CustomerID, tier) shares the key 'customer_id' "
                                 "(called 'CustomerID' there) with the analysis table.")


def test_the_column_list_is_capped(tmp_path, store):
    wide = pd.DataFrame({"region": ["East", "West"], **{f"col_{chr(97 + i)}": ["x", "y"] for i in range(17)}})
    (issue,) = _issues(tmp_path, store, {"main": _main(), "wide": wide})
    names = list(wide.columns)
    assert issue.description == (f"The table 'wide' (columns: {', '.join(names[:15])} and 3 more) shares the key "
                                 "'region' with the analysis table.")


def test_no_value_or_link_number_in_an_issue(tmp_path, store):
    lookup = pd.DataFrame({"region": ["East", "West"], "note": ["CANARY-NOTE-ONE", "CANARY-NOTE-TWO"]})
    (issue,) = _issues(tmp_path, store, {"main": _main(), "lookup": lookup})
    text = json.dumps(dataclasses.asdict(issue))
    assert "CANARY" not in text and "East" not in text and "Ana" not in text
    assert not re.search(r"\d", text)  # the names hold no digit, so any digit would be a number from the scan


# --- no issue ----------------------------------------------------------------------------------------------------

def _cardinality(tmp_path, store, frames):
    from core.table_links import scan_links
    tables, profiles = _bundle(tmp_path, store, frames)
    (link,) = scan_links(tables, profiles, store)["computed"]
    return link["cardinality"], detect_join_issues("main", tables, profiles, store)


def test_a_repeating_carried_key_raises_an_aggregatable_issue_not_a_join(tmp_path, store):
    main = _main().drop(columns="region").assign(customer_id=range(1, 61))  # unique on the analysis side
    carried = pd.DataFrame({"customer_id": [1, 1, 2, 3] * 5, "score": range(20)})
    cardinality, issues = _cardinality(tmp_path, store, {"main": main, "carried": carried})
    # M2.3b step 2: no joinable_table issue; the repeating key raises the aggregatable_table issue instead
    assert cardinality == "one_to_many" and [i.issue_type for i in issues] == ["aggregatable_table"]
    (issue,) = issues
    assert issue.affected_columns == ("customer_id",)
    assert issue.evidence == {"right_table": "carried", "left_key": "customer_id", "right_key": "customer_id"}
    assert issue.description == ("The table 'carried' (columns: customer_id, score) has several rows for each value "
                                 "of the key 'customer_id' it shares with the analysis table.")


def test_a_many_to_many_link_raises_an_aggregatable_issue_not_a_join(tmp_path, store):
    main = pd.DataFrame({"code": ["a", "b"] * 20, "y": range(40)})
    carried = pd.DataFrame({"code": [f"c{chr(97 + i)}" for i in range(19)] + ["a"], "z": range(20)})
    carried.loc[0, "code"] = "a"  # 19 distinct of 20 values: key-like, but "a" repeats
    cardinality, issues = _cardinality(tmp_path, store, {"main": main, "carried": carried})
    # M2.3b step 2: no joinable_table issue; the repeating key raises the aggregatable_table issue instead
    assert cardinality == "many_to_many" and [i.issue_type for i in issues] == ["aggregatable_table"]
    (issue,) = issues
    assert (issue.affected_columns, issue.evidence) == (
        ("code",), {"right_table": "carried", "left_key": "code", "right_key": "code"})
    assert issue.description == ("The table 'carried' (columns: code, z) has several rows for each value of the key "
                                 "'code' it shares with the analysis table.")


def test_no_match_at_all_raises_no_issue(tmp_path, store):
    assert _issues(tmp_path, store, {"main": _main(), "far": pd.DataFrame({"region": ["North", "South"], "m": [1, 2]})}) == []


def test_a_match_after_normalizing_is_usable(tmp_path, store):
    loose = pd.DataFrame({"region": ["east ", "WEST"], "manager": ["Ana", "Ben"]})
    (issue,) = _issues(tmp_path, store, {"main": _main(), "loose": loose})
    assert issue.evidence["right_table"] == "loose"


# --- several tables ----------------------------------------------------------------------------------------------

def test_two_carried_tables_in_profile_order(tmp_path, store):
    customers = pd.DataFrame({"customer_id": range(1, 21), "tier": ["basic", "plus"] * 10})
    issues = _issues(tmp_path, store, {"main": _main(), "regions": REGIONS, "customers": customers})
    assert [(i.evidence["right_table"], i.evidence["left_key"]) for i in issues] == [
        ("regions", "region"), ("customers", "customer_id")]


def test_one_carried_table_with_two_shared_columns(tmp_path, store):
    main = _main().assign(tier=lambda d: np.where(d["region"] == "East", "basic", "plus"))
    both = pd.DataFrame({"region": ["East", "West"], "tier": ["basic", "plus"], "manager": ["Ana", "Ben"]})
    (issue,) = _issues(tmp_path, store, {"main": main, "both": both})
    assert issue.evidence == {"right_table": "both", "left_key": "region", "right_key": "region"}  # same score: names


def test_a_link_between_two_carried_tables_raises_no_issue(tmp_path, store):
    one = pd.DataFrame({"code": ["a", "b", "c"], "x": [1, 2, 3]})
    two = pd.DataFrame({"code": ["a", "b", "c"], "y": [4, 5, 6]})
    assert _issues(tmp_path, store, {"main": _main(), "one": one, "two": two}) == []


# --- detect_all_issues -------------------------------------------------------------------------------------------

def _all_kinds():
    df = _main(100)
    df["region"] = np.where(np.arange(100) % 4 == 0, "east ", df["region"])  # mergeable text
    df["tenure"] = [f"{v:.1f}" for v in np.linspace(1, 50, 100)]
    df.loc[3, "tenure"] = "unknown"  # numbers stored as text
    df.loc[:9, "total_spend"] = np.nan  # 10% missing
    return pd.concat([df, df.iloc[[50, 51]]], ignore_index=True)


def test_without_the_new_arguments_nothing_changes(tmp_path, store):
    tables, profiles = _bundle(tmp_path, store, {"main": _all_kinds(), "regions": REGIONS})
    ds, profile = tables["main"], profiles["main"]
    before = detect_all_issues(ds, profile, store)
    assert [i.issue_type for i in before] == ["duplicate_rows", "high_missingness", "inconsistent_text",
                                              "numeric_as_text"]
    assert detect_all_issues(ds, profile, store, tables={"main": ds}, profiles={"main": profile}, analysis="main") == before
    assert detect_all_issues(ds, profile, store, tables=tables, profiles=None, analysis="main") == before
    assert detect_all_issues(ds, profile, store, tables=tables, profiles=profiles, analysis=None) == before
    issues = detect_all_issues(ds, profile, store, tables=tables, profiles=profiles, analysis="main")
    assert issues[:4] == before and [i.issue_type for i in issues[4:]] == ["joinable_table"]  # the join issue last


def test_five_hundred_tables(tmp_path, store):
    frames = {"main": _main()}
    for i in range(497):
        frames[f"t{i:03d}"] = pd.DataFrame({f"only_{i:03d}": [i]})
    frames["regions"] = REGIONS
    frames["customers"] = pd.DataFrame({"customer_id": range(1, 21), "tier": ["basic", "plus"] * 10})
    frames["managers"] = pd.DataFrame({"region": ["East", "West"], "boss": ["Cy", "Di"]})
    tables, profiles = _bundle(tmp_path, store, frames)
    assert len(tables) == 501
    started = time.perf_counter()
    issues = detect_join_issues("main", tables, profiles, store)
    assert time.perf_counter() - started < 10
    assert [i.evidence["right_table"] for i in issues] == ["regions", "customers", "managers"]
