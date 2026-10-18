"""
M2.1: intake of several files (core/ingestion.py unique_table_name and load_bundle) and the starting tables of a
question (core/nodes.py make_profile_node and _starting_tables). There is no limit on the number of tables. A
"datasets" input wins and is consumed; existing tables are reused only while they mirror state["dataset"].
"""
import dataclasses
import pickle
import time

import pandas as pd
import pytest

import core.ingestion as ingestion
from core.dataset import LocalDiskStore
from core.ingestion import load_bundle, unique_table_name
from core.model_safe_view import ModelSafeView
from core.nodes import _starting_tables, make_profile_node


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _csv(tmp_path, name, rows=3, folder="files"):
    directory = tmp_path / folder
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    pd.DataFrame({"id": range(rows), "value": [i * 1.5 for i in range(rows)]}).to_csv(path, index=False)
    return path


def _bundle(tmp_path, store, n, rows=3, session_id="s"):
    paths = [_csv(tmp_path, f"t{i:03d}.csv", rows) for i in range(n)]
    return load_bundle(paths, session_id=session_id, store=store)


def _check_mirror(update):
    tables, profiles, analysis = update["tables"], update["profiles"], update["analysis_table"]
    assert update["dataset"] is tables[analysis] and update["profile"] is profiles[analysis]
    assert set(profiles) == set(tables)
    for name, profile in profiles.items():
        assert isinstance(profile, ModelSafeView) and profile.dataset_handle == tables[name].handle


# --- unique_table_name ------------------------------------------------------------------------------------------------

def test_unique_table_name():
    assert unique_table_name("customers", set()) == "customers"
    assert unique_table_name("my file (2).v1", set()) == "my_file__2__v1"
    assert unique_table_name("sales-2026_q1", set()) == "sales-2026_q1"
    assert unique_table_name("", set()) == "table" and unique_table_name(None, set()) == "table"
    assert unique_table_name("a", {"a"}) == "a_2"
    assert unique_table_name("a", {"a", "a_2"}) == "a_3"
    taken = {"x"}
    assert unique_table_name("x", taken) == "x_2" and taken == {"x"}  # the set is read, not changed


def test_thousands_of_names():
    taken = set()
    start = time.perf_counter()
    for i in range(5000):
        taken.add(unique_table_name("x!" if i % 2 else "x?", taken))
    assert len(taken) == 5000 and "x_" in taken
    assert time.perf_counter() - start < 30.0


# --- load_bundle --------------------------------------------------------------------------------------------------------

def test_load_bundle(tmp_path, store):
    paths = [_csv(tmp_path, "customers.csv", 4), _csv(tmp_path, "regions.csv", 2),
             _csv(tmp_path, "customers.csv", 3, folder="other")]
    tables = load_bundle(paths, session_id="s", store=store)
    assert list(tables) == ["customers", "regions", "customers_2"]
    for name, path in zip(tables, paths):
        ds = tables[name]
        assert (ds.name, ds.version, ds.original_filename, ds.parent_version) == (name, 1, path.name, None)
        assert store.read(ds.storage_key).equals(pd.read_csv(path))
    assert [len(store.read(ds.storage_key)) for ds in tables.values()] == [4, 2, 3]


def test_load_bundle_with_names(tmp_path, store):
    paths = [_csv(tmp_path, "a.csv"), _csv(tmp_path, "b.csv")]
    tables = load_bundle(paths, session_id="s", store=store, names=["main table", "main table"])
    assert list(tables) == ["main_table", "main_table_2"]


def test_load_bundle_errors(tmp_path, store):
    with pytest.raises(ValueError) as error:
        load_bundle([], session_id="s", store=store)
    assert str(error.value) == "No files were given."
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    with pytest.raises(ValueError, match=r"^notes\.txt: Unsupported file type: \.txt"):
        load_bundle([_csv(tmp_path, "ok.csv"), bad], session_id="s", store=store)


def test_no_limit_on_the_number_of_tables(tmp_path, store):
    assert not hasattr(ingestion, "MAX_TABLES")
    tables = _bundle(tmp_path, store, 60, rows=2)
    assert len(tables) == 60 and len(set(tables)) == 60
    assert all(ds.version == 1 and len(store.read(ds.storage_key)) == 2 for ds in tables.values())


# --- the starting tables ---------------------------------------------------------------------------------------------------

def test_a_datasets_input_wins(tmp_path, store):
    tables = _bundle(tmp_path, store, 3)
    other = _bundle(tmp_path / "x", store, 1, session_id="other")
    node = make_profile_node(store)
    stale = {"tables": other, "analysis_table": "t000", "dataset": next(iter(other.values()))}
    for given in (list(tables.values()), dict(tables)):
        update = node({**stale, "datasets": given})  # the persisted analysis_table is not read in branch (a)
        assert list(update["tables"]) == list(tables) and update["analysis_table"] == "t000"
        assert update["datasets"] is None and update["main_table"] is None
        _check_mirror(update)
    twin = dataclasses.replace(tables["t001"], name="t000")
    with pytest.raises(ValueError, match="Two tables share the name 't000'"):
        node({"datasets": [tables["t000"], twin]})


def test_existing_tables_are_reused_while_they_mirror_the_dataset(tmp_path, store):
    tables = _bundle(tmp_path, store, 3)
    copy_of_main = pickle.loads(pickle.dumps(tables["t000"]))  # equal, not the same object (a checkpoint copy)
    assert copy_of_main == tables["t000"] and copy_of_main is not tables["t000"]
    state = {"tables": dict(tables), "analysis_table": "t000", "dataset": copy_of_main}
    assert list(_starting_tables(state)[0]) == ["t000", "t001", "t002"]
    update = make_profile_node(store)(state)
    assert list(update["tables"]) == ["t000", "t001", "t002"] and update["analysis_table"] == "t000"
    _check_mirror(update)
    replaced = dataclasses.replace(tables["t000"])  # also equal: a value copy
    assert list(_starting_tables({**state, "dataset": replaced})[0]) == ["t000", "t001", "t002"]


def test_a_new_upload_on_a_thread_with_tables(tmp_path, store):
    tables = _bundle(tmp_path, store, 3)
    fresh = _bundle(tmp_path / "new", store, 1, session_id="new")["t000"]
    fresh = dataclasses.replace(fresh, name="fresh")
    update = make_profile_node(store)({"tables": dict(tables), "analysis_table": "t000", "dataset": fresh})
    assert update["tables"] == {"fresh": fresh} and update["analysis_table"] == "fresh"
    _check_mirror(update)


def test_a_single_dataset(tmp_path, store):
    ds = _bundle(tmp_path, store, 1)["t000"]
    update = make_profile_node(store)({"dataset": ds})
    assert update["tables"] == {"t000": ds} and update["analysis_table"] == "t000" and update["dataset"] == ds
    _check_mirror(update)


def test_the_analysis_table(tmp_path, store):
    tables = _bundle(tmp_path, store, 3)
    node = make_profile_node(store)
    assert node({"datasets": list(tables.values())})["analysis_table"] == "t000"
    update = node({"datasets": list(tables.values()), "main_table": "t002"})  # the user's choice is main_table
    assert update["analysis_table"] == "t002" and update["dataset"] is tables["t002"]
    _check_mirror(update)
    with pytest.raises(ValueError) as error:
        node({"datasets": list(tables.values()), "main_table": "nope"})  # the user's choice is main_table
    assert str(error.value) == "Unknown analysis table 'nope'. Tables: t000, t001, t002."


def test_an_unknown_table_among_many(tmp_path, store):
    tables = _bundle(tmp_path, store, 30)
    with pytest.raises(ValueError) as error:
        _starting_tables({"datasets": list(tables.values()), "main_table": "nope"})  # the choice is main_table
    names = ", ".join(sorted(tables)[:20])
    assert str(error.value) == f"Unknown analysis table 'nope'. Tables: {names} and 10 more."


def test_sixty_tables_are_profiled(tmp_path, store):
    tables = _bundle(tmp_path, store, 60)
    update = make_profile_node(store)({"datasets": list(tables.values())})
    assert len(update["profiles"]) == 60
    _check_mirror(update)


# --- main_table: the input; analysis_table: the result (M2.1) ---------------------------------------------------------------

def test_main_table_is_consumed(tmp_path, store):
    tables = _bundle(tmp_path, store, 3)
    node = make_profile_node(store)
    first = node({"datasets": list(tables.values()), "main_table": "t001"})
    assert first["analysis_table"] == "t001" and first["main_table"] is None and first["datasets"] is None
    later = {**first, "dataset": first["dataset"]}  # a later question: no main_table, the mirror holds (branch b)
    assert node(later)["analysis_table"] == "t001"
    assert node({**later, "main_table": "t002"})["analysis_table"] == "t002"  # an explicit choice in branch (b)
    fresh = node({"datasets": list(tables.values())})  # a new bundle without a choice: its first table (branch a)
    assert fresh["analysis_table"] == "t000"


def test_a_main_table_that_does_not_exist(tmp_path, store):
    tables = _bundle(tmp_path, store, 2)
    node = make_profile_node(store)
    with pytest.raises(ValueError) as error:
        node({"datasets": list(tables.values()), "main_table": "missing"})
    assert str(error.value) == "Unknown analysis table 'missing'. Tables: t000, t001."
    state = {"tables": dict(tables), "analysis_table": "t000", "dataset": tables["t000"], "main_table": "missing"}
    with pytest.raises(ValueError, match="Unknown analysis table 'missing'. Tables: t000, t001."):
        node(state)  # branch (b)
    with pytest.raises(ValueError, match=r"Unknown analysis table 'missing'\. Tables: solo\."):
        node({"dataset": dataclasses.replace(tables["t000"], name="solo"), "main_table": "missing"})  # branch (c)


def test_the_persisted_analysis_table_is_read_only_in_branch_b(tmp_path, store):
    tables = _bundle(tmp_path, store, 3)
    reused = {"tables": dict(tables), "analysis_table": "t002", "dataset": tables["t002"]}
    assert _starting_tables(reused)[1] == "t002"  # (b): the tables mirror the dataset
    assert _starting_tables({**reused, "datasets": list(tables.values())})[1] == "t000"  # (a): ignored
    fresh = dataclasses.replace(tables["t000"], name="solo", id="another-id")
    assert _starting_tables({**reused, "dataset": fresh}) == ({"solo": fresh}, "solo")  # (c): ignored


def test_a_hand_built_state_without_main_table(tmp_path, store):
    tables = _bundle(tmp_path, store, 2)
    update = make_profile_node(store)({"tables": dict(tables), "analysis_table": "t001", "dataset": tables["t001"]})
    assert update["analysis_table"] == "t001" and update["dataset"] is update["tables"]["t001"]
    _check_mirror(update)


# --- unreadable files, no dataset, and many names sharing a base (M2.1 review) -------------------------------------------

def test_no_dataset_was_given(store):
    with pytest.raises(ValueError) as error:
        _starting_tables({"request_text": "q"})
    assert str(error.value) == "No dataset was given."
    with pytest.raises(ValueError, match=r"^No dataset was given\.$"):
        make_profile_node(store)({"datasets": None, "tables": {}})


@pytest.mark.parametrize("name, content, kind", [
    ("empty.csv", b"", "EmptyDataError"),
    ("binary.csv", bytes(range(256)) * 8, None),
    ("quote.csv", b'a,b\n1,"unclosed\n2,3\n', None),
], ids=["empty", "binary", "unbalanced_quote"])
def test_an_unreadable_file_in_a_bundle(name, content, kind, tmp_path, store):
    good = _csv(tmp_path, "good.csv")
    bad = tmp_path / name
    bad.write_bytes(content)
    with pytest.raises(ValueError) as error:
        load_bundle([good, bad], session_id="s", store=store)
    assert str(error.value).startswith(f"{name}: ")
    assert kind is None or kind in str(error.value)
    assert error.value.__cause__ is not None


def test_counters_give_the_same_names():
    raw = [f"n{i % 7}" if i % 3 else f"other {i % 4}" for i in range(50)]
    plain, fast, taken_a, taken_b, counters = [], [], set(), set(), {}
    for name in raw:
        plain.append(unique_table_name(name, taken_a))
        taken_a.add(plain[-1])
        fast.append(unique_table_name(name, taken_b, counters))
        taken_b.add(fast[-1])
    assert plain == fast and len(set(fast)) == 50


def test_twenty_thousand_names_sharing_a_base():
    taken, counters = set(), {}
    start = time.perf_counter()
    for _ in range(20000):
        taken.add(unique_table_name("x", taken, counters))
    assert time.perf_counter() - start < 2.0 and len(taken) == 20000


def test_three_hundred_files_with_the_same_stem(tmp_path, store):
    paths = [_csv(tmp_path, "x.csv", 1, folder=f"d{i:03d}") for i in range(300)]
    tables = load_bundle(paths, session_id="s", store=store)
    assert list(tables) == ["x", *(f"x_{i}" for i in range(2, 301))]
