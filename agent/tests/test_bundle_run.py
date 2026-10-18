"""
M2.1: runs of the real graph (guard fakes, fake chat model) with several tables. The analysis table is analysed
exactly as a single upload is; the other tables are profiled and carried, never cleaned and never sent to a model;
dataset and profile always mirror the analysis table; a second question on the same thread keeps the carried tables
when it passes the cleaned dataset back, as the product's "Ask another question" does.
"""
import copy

import numpy as np
import pandas as pd
import pytest
from langgraph.types import Command

from core.dataset import LocalDiskStore
from core.graph import build_agent_graph
from core.ingestion import load_bundle
from core.nodes import with_dataset
from core.plan import QuestionType
from tests.test_decision_point_guard import confident, fake_draft, fake_propose, must_not_propose
from tests.test_exploratory_round import _frame
from tests.test_prompt_canaries import _base_df, _with_duplicates, canary_hits, fake_model  # noqa: F401 (fixture)
from tests.test_stage1_report import _answer, _drive, _messy_frame

QUESTION = "Did West spend more?"
CARRIED_COLUMN = "zz_carried_only_column"
CARRIED_NAME = "carried_lookup"
CARRIED_STRINGS = ("CARRIED-CANARY-ALPHA", "CARRIED-CANARY-BRAVO")
CARRIED_NUMBER = 31415.926535


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _carried_df(rows=2):
    return pd.DataFrame({"region": (["East", "West"] * rows)[:rows], CARRIED_COLUMN: list(CARRIED_STRINGS)[:rows],
                         "budget": [CARRIED_NUMBER, 2.5][:rows]})


def _bundle(tmp_path, store, frames, session="s-bundle"):
    """frames: an ordered dict name -> DataFrame, written as CSV files and loaded with load_bundle."""
    folder = tmp_path / f"files-{session}"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, frame in frames.items():
        paths.append(folder / f"{name}.csv")
        frame.to_csv(paths[-1], index=False)
    return load_bundle(paths, session_id=session, store=store)


def _app(store, question_type=QuestionType.TWO_GROUP_COMPARISON, predictors=("region",), propose_fn=fake_propose):
    return build_agent_graph(store, classify_fn=confident(question_type), draft_fn=fake_draft(list(predictors)),
                             propose_fn=propose_fn)


def _bundle_state(tables, **extra):
    return {"request_text": QUESTION, "session_id": "s-bundle", "datasets": list(tables.values()), **extra}


def _single_state(tables, name):
    return {"request_text": QUESTION, "session_id": "s-bundle", "dataset": tables[name]}


def _points(values):
    return [r["decision_point"] for r in values["decision_log"]]


def _record(values, point):
    return next(r for r in reversed(values["decision_log"]) if r["decision_point"] == point)


def _stage1(values):
    return next(e for e in values["stage_reports"] if e["stage"] == "stage1")


def _sections(entry):
    return {s["heading"]: s["text"] for s in entry["sections"]}


def _assert_mirror(values):
    analysis = values["analysis_table"]
    assert values["tables"][analysis] == values["dataset"]
    assert values["profiles"][analysis] == values["profile"]


# --- (a) a single upload ------------------------------------------------------------------------------------------------

def test_a_single_file_run(fake_model, tmp_path, store):
    tables = _bundle(tmp_path, store, {"main": _frame()})
    values, _, _ = _drive(_app(store, propose_fn=must_not_propose), _single_state(tables, "main"), "single")
    assert list(values["tables"]) == ["main"] and values["analysis_table"] == "main"
    assert _record(values, "planner.stage_plan")["evidence"]["tables"] == [
        {"name": "main", "rows": len(_frame()), "columns": _frame().shape[1], "role": "analysis"}]
    assert "Tables" not in _sections(_stage1(values))
    assert values["plan"].status.value == "executed" and values["report"]
    _assert_mirror(values)


# --- (b) a two-table bundle -------------------------------------------------------------------------------------------------

def test_a_two_table_bundle_analyses_the_main_table_only(fake_model, tmp_path, store):
    tables = _bundle(tmp_path, store, {"main": _frame(), CARRIED_NAME: _carried_df()})
    carried_before = tables[CARRIED_NAME]
    carried_bytes = (store.base_dir / carried_before.storage_key).read_bytes()
    alone, _, _ = _drive(_app(store, propose_fn=must_not_propose), _single_state(tables, "main"), "alone")
    values, _, _ = _drive(_app(store, propose_fn=must_not_propose), _bundle_state(tables), "bundle")
    assert values["test_result"]["method"] == alone["test_result"]["method"]
    assert values["test_result"]["p_value"] == alone["test_result"]["p_value"]
    assert _points(values) == _points(alone)
    assert [t["name"] for t in _record(values, "planner.stage_plan")["evidence"]["tables"]] == ["main", CARRIED_NAME]
    assert values["tables"][CARRIED_NAME] == carried_before
    assert (store.base_dir / carried_before.storage_key).read_bytes() == carried_bytes
    _assert_mirror(values)


# --- (c) a messy bundle: cleaning applies to the main table only ---------------------------------------------------------------

def test_cleaning_applies_to_the_main_table_only(fake_model, tmp_path, store):
    tables = _bundle(tmp_path, store, {"main": _messy_frame(), CARRIED_NAME: _carried_df()})
    values, pauses, _ = _drive(_app(store), _bundle_state(tables), "messy-bundle")
    assert "wrangling_approval" in [p["type"] for p in pauses]
    assert values["tables"]["main"].version > 1 and values["tables"]["main"] == values["dataset"]
    assert values["tables"][CARRIED_NAME] == tables[CARRIED_NAME] and values["tables"][CARRIED_NAME].version == 1
    applied = [r for r in values["decision_log"] if r["decision_point"] == "stage1.clean.execution"
               and r["chosen"] == "applied"]
    assert applied and all(h.startswith("main@") for r in applied for h in r["evidence"]["input_handles"])
    _assert_mirror(values)


# --- (d) the Stage 1 report lists the tables -----------------------------------------------------------------------------------

def _line(name, frame, role):
    tail = "main table" if role == "analysis" else "carried along; not used until joins arrive"
    return f"{name}: {len(frame)} rows, {frame.shape[1]} columns ({tail})"


def test_the_tables_section(fake_model, tmp_path, store):
    main, carried = _frame(), _carried_df()
    tables = _bundle(tmp_path, store, {"main": main, CARRIED_NAME: carried})
    values, _, _ = _drive(_app(store, propose_fn=must_not_propose), _bundle_state(tables), "tables")
    entry = _stage1(values)
    headings = [s["heading"] for s in entry["sections"]]
    assert headings[:3] == ["Checked", "Tables", "Changed"]
    assert _sections(entry)["Tables"].split("\n") == [_line("main", main, "analysis"),
                                                      _line(CARRIED_NAME, carried, "carried")]
    assert _record(values, "planner.stage_plan")["id"] in entry["source_records"]

    single, _, _ = _drive(_app(store, propose_fn=must_not_propose), _single_state(tables, "main"), "no-tables")
    assert "Tables" not in _sections(_stage1(single))
    assert _record(single, "planner.stage_plan")["id"] not in _stage1(single)["source_records"]


def test_thirty_tables(fake_model, tmp_path, store):
    frames = {"main": _frame(), **{f"extra_{i:02d}": pd.DataFrame({"k": [i]}) for i in range(29)}}
    tables = _bundle(tmp_path, store, frames)
    values, _, _ = _drive(_app(store, propose_fn=must_not_propose), _bundle_state(tables), "thirty")
    lines = _sections(_stage1(values))["Tables"].split("\n")
    assert lines[0] == _line("main", frames["main"], "analysis")
    assert lines[1:10] == [_line(f"extra_{i:02d}", frames[f"extra_{i:02d}"], "carried") for i in range(9)]
    assert lines[10:] == ["and 20 more carried tables"]
    assert len(_record(values, "planner.stage_plan")["evidence"]["tables"]) == 30
    _assert_mirror(values)


# --- (e) privacy: nothing of a carried table reaches a prompt ----------------------------------------------------------------------

def test_no_carried_table_reaches_a_prompt(fake_model, tmp_path, store):
    prompts, config = fake_model
    config.update(question_type=QuestionType.TWO_GROUP_COMPARISON, predictors=["region"])
    tables = _bundle(tmp_path, store, {"main": _with_duplicates(_base_df()), CARRIED_NAME: _carried_df()})
    values, pauses, _ = _drive(build_agent_graph(store), _bundle_state(tables), "privacy")
    assert "wrangling_approval" in [p["type"] for p in pauses] and values["report"]
    stages = {p["stage"] for p in prompts}
    assert {"report", "_ClassificationOut", "_DraftPlanOut", "_WranglingProposalsOut"} <= stages
    assert canary_hits(prompts) == []  # the main table's canaries, as before
    assert canary_hits(prompts, canaries=(*CARRIED_STRINGS, CARRIED_NUMBER)) == []
    for p in prompts:
        assert CARRIED_COLUMN not in p["text"] and "budget" not in p["text"], p["stage"]
        assert CARRIED_NAME not in p["text"], p["stage"]


# --- (f) another analysis table ------------------------------------------------------------------------------------------------------

def test_an_analysis_table_other_than_the_first(fake_model, tmp_path, store):
    tables = _bundle(tmp_path, store, {CARRIED_NAME: _carried_df(), "main": _frame()})
    # the user's choice of the analysis table is the main_table input
    values, _, _ = _drive(_app(store, propose_fn=must_not_propose), _bundle_state(tables, main_table="main"),
                          "second-table")
    assert values["analysis_table"] == "main" and values["dataset"] == tables["main"]
    # a checkpoint turns the plan's tuple fields into lists (M0.4); the value is what matters
    assert list(values["plan"].data_sources) == [tables["main"].handle] and values["plan"].outcome_name == "total_spend"
    assert values["tables"][CARRIED_NAME] == tables[CARRIED_NAME] and list(values["tables"]) == [CARRIED_NAME, "main"]
    _assert_mirror(values)


# --- (g) the mirror after every node ----------------------------------------------------------------------------------------------------

def test_every_node_keeps_the_mirror(fake_model, tmp_path, store):
    tables = _bundle(tmp_path, store, {"main": _messy_frame(), CARRIED_NAME: _carried_df()})
    app = _app(store)
    config = {"configurable": {"thread_id": "mirror"}}
    state, checked, graph_input = {}, [], _bundle_state(tables)
    while True:
        for event in app.stream(graph_input, config, stream_mode="updates"):
            for node, update in event.items():
                if node == "__interrupt__" or not isinstance(update, dict):
                    continue
                state.update(update)
                if "tables" in state and ("dataset" in update or "profile" in update or "tables" in update):
                    analysis = state["analysis_table"]
                    assert state["tables"][analysis] == state["dataset"], node
                    if "profile" in state:
                        assert state["profiles"][analysis] == state["profile"], node
                    checked.append(node)
        snapshot = app.get_state(config)
        if not snapshot.interrupts:
            break
        graph_input = Command(resume=_answer(snapshot.interrupts[0].value, {}))
    assert {"profile", "wrangle_review"} <= set(checked)


# --- (h) with_dataset on a hand-built state ---------------------------------------------------------------------------------------------

def test_with_dataset_without_tables(tmp_path, store):
    ds = _bundle(tmp_path, store, {"main": _frame()})["main"]
    assert with_dataset({}, ds) == {"dataset": ds, "tables": {"main": ds}}
    assert with_dataset({}, ds, "PROFILE") == {"dataset": ds, "tables": {"main": ds}, "profile": "PROFILE",
                                               "profiles": {"main": "PROFILE"}}
    state = {"tables": {"main": "old", "other": "x"}, "analysis_table": "main", "profiles": {"other": "p"}}
    before = copy.deepcopy(state)
    update = with_dataset(state, ds, "P")
    assert update["tables"] == {"main": ds, "other": "x"} and update["profiles"] == {"other": "p", "main": "P"}
    assert state == before


# --- (i) the second question on the same thread ----------------------------------------------------------------------------------------

def _invoke_to_end(app, graph_input, thread):
    """As the product does it: invoke, answer each pause, and return the values invoke returned."""
    config = {"configurable": {"thread_id": thread}}
    result = app.invoke(graph_input, config)
    while "__interrupt__" in result:
        result = app.invoke(Command(resume=_answer(result["__interrupt__"][0].value, {})), config)
    return result


def test_a_second_question_keeps_the_carried_tables(fake_model, tmp_path, store):
    app = _app(store)
    tables = _bundle(tmp_path, store, {"main": _messy_frame(), CARRIED_NAME: _carried_df()})
    first = _invoke_to_end(app, _bundle_state(tables), "thread")
    cleaned = first["dataset"]
    assert cleaned.version > 1 and first["plan"].status.value == "executed"

    second = _invoke_to_end(app, {"request_text": QUESTION, "session_id": "s-bundle", "dataset": cleaned}, "thread")
    assert list(second["tables"]) == ["main", CARRIED_NAME] and second["analysis_table"] == "main"
    assert second["tables"][CARRIED_NAME] == tables[CARRIED_NAME]
    # a checkpoint turns the plan's tuple fields into lists (M0.4); the value is what matters
    assert list(second["plan"].data_sources) == [cleaned.handle] and second["plan"].status.value == "executed"
    _assert_mirror(second)

    fresh = _bundle(tmp_path, store, {"fresh": _frame(seed=63)}, session="s-fresh")["fresh"]
    third = _invoke_to_end(app, {"request_text": QUESTION, "session_id": "s-bundle", "dataset": fresh}, "thread")
    assert third["tables"] == {"fresh": fresh} and third["analysis_table"] == "fresh"
    _assert_mirror(third)

    bundle2 = _bundle(tmp_path, store, {"again": _frame(seed=64), "lookup2": _carried_df()}, session="s-two")
    fourth = _invoke_to_end(app, _bundle_state(bundle2), "thread")
    assert list(fourth["tables"]) == ["again", "lookup2"] and CARRIED_NAME not in fourth["tables"]
    assert fourth["analysis_table"] == "again"
    _assert_mirror(fourth)


# --- the checkpoint keeps a Dataset equal (Dataset.__post_init__) -------------------------------------------------------------

def test_checkpointed_datasets_equal_their_originals(fake_model, tmp_path, store):
    from core.checkpoint_types import make_serde
    from core.dataset import Dataset
    tables = _bundle(tmp_path, store, {"main": _frame(), CARRIED_NAME: _carried_df()})
    app = _app(store, propose_fn=must_not_propose)
    config = {"configurable": {"thread_id": "checkpoint"}}
    assert "__interrupt__" in app.invoke(_bundle_state(tables), config)  # the first pause: the stage-plan gate
    checkpointed = app.get_state(config).values["tables"]
    assert list(checkpointed) == list(tables)
    for name, ds in checkpointed.items():
        assert ds == tables[name] and isinstance(ds.parents, tuple)
        assert hash(ds) == hash(tables[name])

    listed = Dataset(session_id="s", name="j", version=1, storage_key="k", parent_version=None,
                     created_from_operation="op", parents=["a@v1"])
    assert listed.parents == ("a@v1",)
    serde = make_serde()
    restored = serde.loads_typed(serde.dumps_typed(listed))
    assert restored == listed and restored.parents == ("a@v1",) and type(restored) is Dataset


# --- a new bundle on a thread that already has tables (the stale analysis table) ------------------------------------------------

@pytest.mark.parametrize("second_names", [("first_new", "second_new"), ("first_new", "main")],
                         ids=["new_names", "main_at_second_position"])
def test_a_new_bundle_uses_its_own_first_table(second_names, fake_model, tmp_path, store):
    app = _app(store)
    first = _bundle(tmp_path, store, {"main": _messy_frame(), "carried": _carried_df()})
    q1 = _invoke_to_end(app, _bundle_state(first), "stale")
    assert q1["analysis_table"] == "main" and q1["plan"].status.value == "executed"
    a, b = second_names
    frames = {a: _frame(seed=71), b: _frame(seed=72) if b == "main" else _carried_df()}
    second = _bundle(tmp_path, store, frames, session="s-second")
    q2 = _invoke_to_end(app, _bundle_state(second), "stale")
    assert q2["analysis_table"] == a
    assert list(q2["tables"]) == [a, b] and q2["dataset"] == second[a]
    # a checkpoint turns the plan's tuple fields into lists (M0.4); the value is what matters
    assert list(q2["plan"].data_sources) == [second[a].handle]
    _assert_mirror(q2)
