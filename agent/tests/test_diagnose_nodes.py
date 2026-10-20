"""
M2.2: the diagnose and rediagnose nodes in the real graph (guard fakes, fake chat model). diagnose records the
"before" quality snapshot of the analysis table (and, with several tables, the carried tables' numbers and the
links between tables); rediagnose re-profiles after cleaning changed the data (G68) and records the "after"
snapshot. The Stage 1 report shows a Data quality section and a Links between tables section; nothing reaches a
prompt.
"""
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
from langgraph.types import Command
from scipy import stats as scipy_stats

import core.table_links as table_links_module
from core.dataset import LocalDiskStore
from core.decision_points import DECISION_NODES, NODE_STAGES, registered_points
from core.graph import build_agent_graph
from core.ingestion import profile_dataset
from core.plan import QuestionType
from core.quality import OUTLIER_IQR_MULTIPLIER, SKEW_THRESHOLD
from core.quality_records import LINKS_POINT, QUALITY_SNAPSHOT_POINT, quality_snapshot_record
from core.stage_reports import stage1_report
from tests.test_bundle_run import (
    CARRIED_NAME, _app, _bundle, _bundle_state, _carried_df, _record, _sections, _single_state, _stage1,
)
from tests.test_decision_point_guard import must_not_propose, reject_joins
from tests.test_exploratory_round import _frame
from tests.test_prompt_canaries import _base_df, _with_duplicates, canary_hits, fake_model  # noqa: F401 (fixture)
from tests.test_stage1_report import _answer, _drive, _messy_frame
from tests.test_table_links import CountingStore

BEFORE = "the data-quality numbers of the analysis table before cleaning"
AFTER = "the data-quality numbers of the analysis table after cleaning"
NOT_CHANGED = "Cleaning did not change the data."


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _plain(view):
    """Every field of a dataclass as plain JSON values (tuples and lists compare alike)."""
    return json.loads(json.dumps(dataclasses.asdict(view)))


def _snapshots(values, label):
    return [r for r in values["decision_log"] if r["decision_point"] == QUALITY_SNAPSHOT_POINT and r["chosen"] == label]


def _points(values):
    return [r["decision_point"] for r in values["decision_log"]]


def _reject_all(payload):
    return {"decisions": [{"index": op["index"], "action": "reject", "reason": "keep everything"}
                          for op in payload["proposed_operations"]]}


def _cleaned(df):
    deduped = df.drop_duplicates()
    return deduped[deduped["tenure"].notna()]


def _stream_checking_the_mirror(app, graph_input, thread, **overrides):
    """Run to the end, applying every node update to a copy of the state and checking the mirror after each."""
    config = {"configurable": {"thread_id": thread}}
    state, checked = {}, []
    while True:
        for event in app.stream(graph_input, config, stream_mode="updates"):
            for node, update in event.items():
                if node == "__interrupt__" or not isinstance(update, dict):
                    continue
                state.update(update)
                if "tables" in state and {"dataset", "profile", "tables"} & set(update):
                    analysis = state["analysis_table"]
                    assert state["tables"][analysis] == state["dataset"], node
                    assert state["profiles"][analysis] == state["profile"], node
                    checked.append(node)
        snapshot = app.get_state(config)
        if not snapshot.interrupts:
            return snapshot.values, checked
        graph_input = Command(resume=_answer(snapshot.interrupts[0].value, overrides))


# --- (a) a single-table messy run ----------------------------------------------------------------------------------------

def test_a_messy_run_gets_before_and_after(fake_model, tmp_path, store):
    df = _messy_frame()
    tables = _bundle(tmp_path, store, {"main": df})
    values, checked = _stream_checking_the_mirror(_app(store), _single_state(tables, "main"), "messy")
    (before,), (after,) = _snapshots(values, "before"), _snapshots(values, "after")
    assert (before["rationale"], after["rationale"]) == (BEFORE, AFTER)
    assert (before["evidence"]["rows"], before["evidence"]["duplicate_rows"]) == (len(df), int(df.duplicated().sum()))
    assert before["evidence"]["missing"] == [{"column": "tenure", "missing": int(df["tenure"].isna().sum()),
                                             "missing_pct": float(df["tenure"].isna().mean())}]
    cleaned = _cleaned(df)
    assert (after["evidence"]["rows"], after["evidence"]["duplicate_rows"]) == (len(cleaned), 0)
    assert after["evidence"]["missing"] == [] and after["evidence"]["handle"] == values["dataset"].handle
    assert LINKS_POINT not in _points(values)
    points = _points(values)
    i_plan, i_before = points.index("planner.stage_plan"), points.index(QUALITY_SNAPSHOT_POINT)
    i_scan, i_after = points.index("stage1.clean.scan"), len(points) - 1 - points[::-1].index(QUALITY_SNAPSHOT_POINT)
    assert i_plan < i_before < i_scan < i_after and i_after > points.index("stage1.clean.execution")
    assert values["profile"].row_count == len(cleaned)  # G68: profile is refreshed after cleaning
    assert values["profiles"]["main"] == values["profile"]
    assert {"profile", "wrangle_review", "rediagnose"} <= set(checked)
    json.dumps(before["evidence"]), json.dumps(after["evidence"])


# --- (b) nothing to clean, (c) every operation rejected -------------------------------------------------------------------

def test_nothing_to_clean(fake_model, tmp_path, store):
    tables = _bundle(tmp_path, store, {"main": _frame()})
    values, _, nodes = _drive(_app(store, propose_fn=must_not_propose), _single_state(tables, "main"), "clean")
    assert len(_snapshots(values, "before")) == 1 and _snapshots(values, "after") == []
    assert "rediagnose" not in nodes
    assert values["dataset"] == tables["main"]
    # a checkpoint turns the profile's tuples into lists (M0.4, G67); every value must be the original one
    assert _plain(values["profile"]) == _plain(profile_dataset(tables["main"], store=store))
    assert NOT_CHANGED not in _sections(_stage1(values))["Data quality"]


def test_every_operation_rejected(fake_model, tmp_path, store):
    tables = _bundle(tmp_path, store, {"main": _messy_frame()})
    values, pauses, nodes = _drive(_app(store), _single_state(tables, "main"), "rejected",
                                   wrangling_approval=_reject_all)
    assert "wrangling_approval" in [p["type"] for p in pauses] and "rediagnose" in nodes
    assert _snapshots(values, "after") == []
    assert values["dataset"] == tables["main"]
    # a checkpoint turns the profile's tuples into lists (M0.4, G67); every value must be the original one
    assert _plain(values["profile"]) == _plain(profile_dataset(tables["main"], store=store))
    assert NOT_CHANGED in _sections(_stage1(values))["Data quality"].split("\n")


def _scan(chosen):
    return {"decision_point": "stage1.clean.scan", "chosen": chosen, "id": f"scan-{chosen}",
            "evidence": {"rows": 10, "columns_scanned": 2, "duplicate_row_count": 0,
                         "columns_above_missing_threshold": [], "missing_threshold": 0.05}}


def _snapshot_record(label, handle, rows=10):
    snapshot = {"table": "t", "handle": handle, "rows": rows, "columns": 2, "duplicate_rows": 0, "missing": [],
                "columns_with_missing": 0, "constant_columns": [], "outliers": [], "columns_with_outliers": 0,
                "skewed_columns": []}
    record = quality_snapshot_record(label, snapshot).__dict__
    return {**record, "id": f"{label}-{handle}", "candidates": [], "evidence": dict(record["evidence"])}


def test_the_cleaning_line_composer():
    stage_plan = {"question_type": "two_group_comparison"}

    def quality(records):
        sections = {s["heading"]: s["text"] for s in stage1_report(records, stage_plan)["sections"]}
        return sections["Data quality"].split("\n")
    assert NOT_CHANGED in quality([_scan("propose_fixes"), _snapshot_record("before", "t@v1")])
    assert NOT_CHANGED not in quality([_scan("skipped"), _snapshot_record("before", "t@v1")])
    with_after = quality([_scan("propose_fixes"), _snapshot_record("before", "t@v1"), _snapshot_record("after", "t@v2", 8)])
    assert NOT_CHANGED not in with_after and "After cleaning: 8 rows, 0 exact duplicate rows." in with_after


# --- (d) a two-table bundle ------------------------------------------------------------------------------------------------

def test_a_bundle_records_the_carried_tables_and_the_link(fake_model, tmp_path):
    store = CountingStore(tmp_path / "counting")
    main, carried = _frame(), _carried_df()
    tables = _bundle(tmp_path, store, {"main": main, CARRIED_NAME: carried})
    store.reads.clear()
    # M2.3b: the shared region key raises a join proposal, rejected here
    values, _, _ = _drive(_app(store), _bundle_state(tables), "bundle", wrangling_approval=reject_joins)
    (before,) = _snapshots(values, "before")
    assert before["evidence"]["carried"] == [{"name": CARRIED_NAME, "rows": len(carried), "columns": carried.shape[1],
                                             "columns_with_missing": 0}]
    links = _record(values, LINKS_POINT)
    assert links["chosen"] == "found" and links["rationale"] == "candidate keys between tables, found by shared column names"
    (c,) = links["evidence"]["computed"]
    assert (c["table_a"], c["column_a"], c["table_b"], c["column_b"]) == ("main", "region", CARRIED_NAME, "region")
    counts = main["region"].value_counts()
    assert c["cardinality"] == "many_to_one" and c["matched_distinct"] == len(set(main["region"]) & set(carried["region"]))
    assert (c["orphan_rows_a"], c["orphan_rows_b"]) == (int((~main["region"].isin(carried["region"])).sum()), 0)
    assert c["inner_join_rows"] == len(main.merge(carried, on="region")) == int(counts.sum())
    carried_reads = store.reads.count(tables[CARRIED_NAME].storage_key)
    # its profile and the link numbers (as before); M2.3b: the link scans of the join detection at plan_stages and
    # wrangle_propose, and the join candidates' read; never cleaned or described
    assert carried_reads == 2 + 3


# --- (e) the Stage 1 report text ----------------------------------------------------------------------------------------------

def _outliers(df):
    found = []
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        values = df[column].dropna().to_numpy()
        q1, q3 = np.percentile(values, [25, 75])
        lower, upper = q1 - OUTLIER_IQR_MULTIPLIER * (q3 - q1), q3 + OUTLIER_IQR_MULTIPLIER * (q3 - q1)
        below, above = int((values < lower).sum()), int((values > upper).sum())
        if below or above:
            found.append(f"{column}: {below} below, {above} above")
    return found


def _skewed(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and len(df[c].dropna()) >= 3
            and abs(float(scipy_stats.skew(df[c].dropna()))) > SKEW_THRESHOLD]


def test_the_data_quality_lines(fake_model, tmp_path, store):
    df = _messy_frame()
    tables = _bundle(tmp_path, store, {"main": df})
    values, _, _ = _drive(_app(store), _single_state(tables, "main"), "lines")
    entry = _stage1(values)
    cleaned = _cleaned(df)
    expected = [f"Before: {len(df)} rows, {df.shape[1]} columns, {int(df.duplicated().sum())} exact duplicate rows.",
                f"Missing values: tenure {df['tenure'].isna().mean():.1%}."]
    out_before, out_after = _outliers(df), _outliers(cleaned)
    expected.append(f"Outliers (beyond 1.5 times the interquartile range): {'; '.join(out_before)}." if out_before
                    else "Outliers: none.")
    if _skewed(df):
        expected.append(f"Skewed columns (skewness beyond 1 in either direction): {'; '.join(_skewed(df))}.")
    expected += [f"After cleaning: {len(cleaned)} rows, 0 exact duplicate rows.", "Missing values after cleaning: none.",
                 f"Outliers after cleaning: {'; '.join(out_after) + '.' if out_after else 'none.'}"]
    lines = _sections(entry)["Data quality"].split("\n")
    assert lines == expected and all(line.endswith(".") for line in lines)
    assert [s["heading"] for s in entry["sections"]] == ["Checked", "Data quality", "Changed", "Described"]
    ids = {_snapshots(values, "before")[0]["id"], _snapshots(values, "after")[0]["id"]}
    assert ids <= set(entry["source_records"]) and LINKS_POINT not in _points(values)


def _link_line(c, extra=""):
    words = {"one_to_one": "one to one", "one_to_many": "one to many", "many_to_one": "many to one",
             "many_to_many": "many to many"}
    return (f"{c['table_a']}.{c['column_a']} with {c['table_b']}.{c['column_b']}: {words[c['cardinality']]}; "
            f"{c['matched_distinct']} of {c['distinct_a']} keys in {c['table_a']} are found in {c['table_b']}; "
            f"{c['orphan_rows_a']} rows of {c['table_a']} and {c['orphan_rows_b']} rows of {c['table_b']} have no match; "
            f"an inner join would give {c['inner_join_rows']} rows.{extra}")


def test_the_links_lines(fake_model, tmp_path, store):
    main = _frame()
    carried = pd.DataFrame({"region": ["east", "West "], "manager": ["Ana", "Ben"]})  # differ by case and a space
    tables = _bundle(tmp_path, store, {"main": main, CARRIED_NAME: carried})
    # M2.3b: the keys match after normalizing, so a join is proposed (join_left_normalized), rejected here
    values, _, _ = _drive(_app(store), _bundle_state(tables), "case", wrangling_approval=reject_joins)
    entry = _stage1(values)
    assert [s["heading"] for s in entry["sections"]] == ["Checked", "Tables", "Data quality", "Links between tables",
                                                         "Changed", "Left as is", "Described"]  # M2.3b: the rejected join
    (c,) = _record(values, LINKS_POINT)["evidence"]["computed"]
    assert (c["matched_distinct"], c["matched_distinct_normalized"], c["inner_join_rows"]) == (0, 2, 0)
    assert c["orphan_rows_a"] == len(main) and c["orphan_rows_b"] == 2
    assert _sections(entry)["Links between tables"] == _link_line(
        c, " Matching would improve by 2 keys after trimming spaces and ignoring case.")
    assert _record(values, LINKS_POINT)["id"] in entry["source_records"]


def test_more_candidates_than_computed(fake_model, tmp_path, store, monkeypatch):
    # the node calls scan_links without top_n, which reads TOP_LINKS at call time: the real path
    monkeypatch.setattr(table_links_module, "TOP_LINKS", 1)
    carried = pd.DataFrame({"region": ["East", "West"], "tier": ["basic", "plus"]})
    tables = _bundle(tmp_path, store, {"main": _frame(), CARRIED_NAME: carried})
    values, _, _ = _drive(_app(store), _bundle_state(tables), "more",
                          wrangling_approval=reject_joins)  # M2.3b: the proposed join is rejected
    ev = _record(values, LINKS_POINT)["evidence"]
    assert (ev["candidates_total"], len(ev["computed"]), ev["not_computed"]) == (2, 1, 1)
    lines = _sections(_stage1(values))["Links between tables"].split("\n")
    assert lines == [_link_line(ev["computed"][0]), "and 1 more candidate(s) not computed."]


def test_no_shared_key_columns(fake_model, tmp_path, store):
    tables = _bundle(tmp_path, store, {"main": _frame(), CARRIED_NAME: pd.DataFrame({"zz_code": [1, 2]})})
    values, _, _ = _drive(_app(store, propose_fn=must_not_propose), _bundle_state(tables), "none")
    links = _record(values, LINKS_POINT)
    assert links["chosen"] == "none" and links["rationale"] == "no shared key columns between the tables"
    entry = _stage1(values)
    assert _sections(entry)["Links between tables"] == "No shared key columns were found between the tables."
    assert links["id"] in entry["source_records"]


# --- (f) privacy ------------------------------------------------------------------------------------------------------------

def test_no_quality_or_link_number_reaches_a_prompt(fake_model, tmp_path, store):
    prompts, config = fake_model
    config.update(question_type=QuestionType.TWO_GROUP_COMPARISON, predictors=["region"])
    tables = _bundle(tmp_path, store, {"main": _with_duplicates(_base_df()), CARRIED_NAME: _carried_df()})
    values, _, _ = _drive(build_agent_graph(store), _bundle_state(tables), "privacy")
    assert canary_hits(prompts) == []
    before = _snapshots(values, "before")[0]["evidence"]
    numbers = [f"{o['lower_fence']}" for o in before["outliers"]] + [f"{o['upper_fence']}" for o in before["outliers"]]
    numbers += [f"{s['skewness']}" for s in before["skewed_columns"]]
    for p in prompts:
        text = p["text"].lower()
        for word in ("outlier", "fence", "skewness", "interquartile", "inner join", "orphan", "cardinality"):
            assert word not in text, (p["stage"], word)
        for number in numbers:
            assert number not in p["text"], (p["stage"], number)
    # the cleaning proposer's prompt names the duplicate_rows issue as it always has; that is not a snapshot number


# --- (g) the descriptive and the unsupported question -----------------------------------------------------------------------------

def test_a_descriptive_question(fake_model, tmp_path, store):
    tables = _bundle(tmp_path, store, {"main": _frame()})
    app = _app(store, QuestionType.DESCRIPTIVE, (), must_not_propose)
    values, _, _ = _drive(app, _single_state(tables, "main"), "descriptive")
    assert len(_snapshots(values, "before")) == 1 and _snapshots(values, "after") == []
    assert NOT_CHANGED not in _sections(_stage1(values))["Data quality"]


def test_an_unsupported_question(fake_model, tmp_path, store):
    tables = _bundle(tmp_path, store, {"main": _frame()})
    app = _app(store, QuestionType.ASSOCIATION, ("region", "tier"), must_not_propose)
    values, _, nodes = _drive(app, _single_state(tables, "main"), "unsupported")
    assert "diagnose" not in nodes and QUALITY_SNAPSHOT_POINT not in _points(values)
    assert LINKS_POINT not in _points(values)


# --- (h) the edges, (i) the registry --------------------------------------------------------------------------------------------

def test_graph_edges(tmp_path):
    edges = {(e.source, e.target): e.conditional
             for e in build_agent_graph(LocalDiskStore(base_dir=tmp_path)).get_graph().edges}
    assert edges[("plan_stages", "diagnose")] is True and edges[("stage_plan_gate", "diagnose")] is True
    assert edges[("diagnose", "wrangle_propose")] is False
    assert edges[("wrangle_review", "rediagnose")] is False
    assert edges[("rediagnose", "wrangle_propose")] is True and edges[("rediagnose", "replan")] is True  # M2.4a
    assert edges[("replan", "describe")] is True and edges[("replan", "stage1_report")] is True  # M2.4a: was rediagnose
    assert ("rediagnose", "describe") not in edges and ("rediagnose", "stage1_report") not in edges  # M2.4a
    assert ("wrangle_review", "describe") not in edges and ("wrangle_review", "stage1_report") not in edges
    for target in ("wrangle_review", "describe", "stage1_report"):
        assert edges[("wrangle_propose", target)] is True, target


def test_the_registry(tmp_path):
    assert (len(registered_points()), len(DECISION_NODES)) == (32, 18)  # M2.4a: replan (planner.stage_replan)
    assert DECISION_NODES["diagnose"] == (QUALITY_SNAPSHOT_POINT, LINKS_POINT)
    assert DECISION_NODES["rediagnose"] == (QUALITY_SNAPSHOT_POINT,)
    assert NODE_STAGES["diagnose"] == NODE_STAGES["rediagnose"] == "stage1"
    nodes = build_agent_graph(LocalDiskStore(base_dir=tmp_path)).get_graph().nodes
    assert len([n for n in nodes if not n.startswith("__")]) == 22  # M2.4a: replan
