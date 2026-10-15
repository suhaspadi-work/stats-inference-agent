"""
M0.2 step 13: the real graph writes every decision record to the event log as
each node returns. Model functions are faked; runs stop before the report node.
"""
import json
import math

import numpy as np
import pandas as pd
import pytest
from langgraph.types import Command

from core.dataset import LocalDiskStore
from core.decision_points import DECISION_NODES
from core.event_log import EventLog, events_path, read_decision_log
from core.graph import build_agent_graph
from core.plan import QuestionType
from tests.test_decision_point_guard import (
    _dataset, _two_group_df, _with_duplicate, confident, fake_classify, fake_draft, fake_propose, must_not_propose,
)

STOP = ["execute", "eda"]
S1_RESUMES = [
    "I mean the two regions, East and West.",
    {"action": "approve"},  # M1.3a: the stage-plan gate
    {"decisions": [{"index": 0, "action": "approve"}, {"index": 1, "action": "approve"}]},
    {"action": "approve"},
]


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _s1_dataset(tmp_path, store):
    df = _two_group_df(np.random.default_rng(7))
    df["notes"] = [None if i % 6 == 0 else "ok" for i in range(len(df))]
    return _dataset(_with_duplicate(df), tmp_path, store, "s1")


def _s1_app(store, event_log_dir):
    return build_agent_graph(store, classify_fn=fake_classify(QuestionType.TWO_GROUP_COMPARISON),
                             draft_fn=fake_draft(["region"]), propose_fn=fake_propose, event_log_dir=event_log_dir)


def _start(ds, **extra):
    return {"request_text": "Did West spend more?", "session_id": "session-abc", "dataset": ds, **extra}


def _run(app, state, thread, resumes):
    config = {"configurable": {"thread_id": thread}}
    queue = list(resumes)
    result = app.invoke(state, config, interrupt_after=STOP)
    while "__interrupt__" in result:
        result = app.invoke(Command(resume=queue.pop(0)), config, interrupt_after=STOP)
    assert queue == []
    return app.get_state(config).values


def _strict_lines(path):
    def reject(constant):
        raise ValueError(f"non-strict JSON constant {constant}")
    return [json.loads(line, parse_constant=reject) for line in path.read_text().splitlines() if line.strip()]


def _same(a, b) -> bool:
    """Deep equality where NaN equals NaN."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


# --- a full run --------------------------------------------------------------------

@pytest.fixture
def s1_run(tmp_path, store):
    events = tmp_path / "events"
    state = _run(_s1_app(store, events), _start(_s1_dataset(tmp_path, store)), "s1-thread", S1_RESUMES)
    return state, events_path(events, "s1-thread")


def test_file_log_equals_the_state_log(s1_run):
    state, path = s1_run
    assert len(state["decision_log"]) > 10
    assert read_decision_log(path) == state["decision_log"]


def test_every_entry_names_a_node_declared_for_its_point(s1_run):
    _, path = s1_run
    entries = EventLog(log_path=path).read_by_type("decision")
    raw = EventLog(log_path=path).read_all()
    assert len(raw) == len(entries)
    for entry in raw:
        node = entry["meta"]["node"]
        assert node in DECISION_NODES, node
        assert entry["data"]["decision_point"] in DECISION_NODES[node], (node, entry["data"]["decision_point"])
        assert entry["meta"]["thread_id"] == "s1-thread"
        assert entry["meta"]["session_id"] == "session-abc"


def test_a_fresh_event_log_reads_the_same_records(s1_run):
    state, path = s1_run
    assert [e["data"] for e in EventLog(log_path=path).read_all()] == state["decision_log"]


def test_file_grows_with_each_pause_and_holds_no_duplicates(tmp_path, store):
    events = tmp_path / "events"
    app = _s1_app(store, events)
    config = {"configurable": {"thread_id": "steps"}}
    path = events_path(events, "steps")

    result = app.invoke(_start(_s1_dataset(tmp_path, store)), config, interrupt_after=STOP)
    assert result["__interrupt__"][0].value["type"] == "clarification_needed"
    assert read_decision_log(path) == app.get_state(config).values["decision_log"]
    assert [e["decision_point"] for e in read_decision_log(path)] == [
        "intake.question_type", "intake.causal_status", "intake.clarification_gate"]

    for resume in S1_RESUMES:
        app.invoke(Command(resume=resume), config, interrupt_after=STOP)
        assert read_decision_log(path) == app.get_state(config).values["decision_log"]

    ids = [e["id"] for e in read_decision_log(path)]
    assert len(ids) == len(set(ids))


def test_without_event_log_dir_nothing_is_written(tmp_path, store):
    ds = _s1_dataset(tmp_path, store)
    before = {p for p in tmp_path.rglob("*")}
    state = _run(_s1_app(store, None), _start(ds), "no-log", S1_RESUMES)
    assert state["plan"].status.value == "executed"
    created = {p for p in tmp_path.rglob("*")} - before
    assert not any(p.name == "events.jsonl" or p.name == "no-log" for p in created)
    assert all(p.is_relative_to(tmp_path / "store") for p in created)  # only new dataset versions


# --- NaN through the file -------------------------------------------------------------

def test_dunn_nan_intervals_round_trip_as_strict_json(tmp_path, store):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"region": [g for g in ("East", "North", "West") for _ in range(60)],
                       "total_spend": np.concatenate([rng.exponential(s, 60) for s in (1, 3, 6)])})
    events = tmp_path / "events"
    app = build_agent_graph(store, classify_fn=confident(QuestionType.MULTI_GROUP_COMPARISON),
                            draft_fn=fake_draft(["region"]), propose_fn=must_not_propose, event_log_dir=events)
    state = _run(app, _start(_dataset(df, tmp_path, store, "kw")), "kw", [{"action": "approve"}, {"action": "approve"}])  # M1.3a: gate first
    path = events_path(events, "kw")

    in_state = next(e for e in state["decision_log"] if e["decision_point"] == "stage2.post_hoc")
    in_file = next(e for e in read_decision_log(path) if e["decision_point"] == "stage2.post_hoc")
    assert in_state["chosen"] == "dunn"
    ci = in_file["chart_data"][0]["data"]["ci_lower"]
    assert any(isinstance(v, float) and math.isnan(v) for v in ci)
    assert _same(in_file, in_state)
    _strict_lines(path)  # raises on any NaN or Infinity token


# --- threads and the default graph -----------------------------------------------------

def test_two_threads_write_two_files(tmp_path, store):
    events = tmp_path / "events"
    app = build_agent_graph(store, classify_fn=confident(QuestionType.DESCRIPTIVE), draft_fn=fake_draft([]),
                            propose_fn=must_not_propose, event_log_dir=events)
    ds = _dataset(_two_group_df(np.random.default_rng(3)), tmp_path, store, "desc")
    first = _run(app, _start(ds), "thread-a", [])
    second = _run(app, _start(ds), "thread-b", [])
    assert sorted(p.name for p in events.iterdir()) == ["thread-a", "thread-b"]
    assert read_decision_log(events_path(events, "thread-a")) == first["decision_log"]
    assert read_decision_log(events_path(events, "thread-b")) == second["decision_log"]
    a_ids = {e["id"] for e in first["decision_log"]}
    assert a_ids.isdisjoint(e["id"] for e in second["decision_log"])


def test_default_graph_has_the_same_nodes(tmp_path):
    nodes = set(build_agent_graph(LocalDiskStore(base_dir=tmp_path)).get_graph().nodes) - {"__start__", "__end__"}
    # M1.1: plan_stages; M1.2: describe; M1.3a: stage_plan_gate; M1.3b: stage_gate and start_round after report
    assert nodes == {"classify", "clarify", "reclassify", "profile", "plan_stages", "stage_plan_gate", "wrangle_propose",
                     "wrangle_review", "describe", "draft_plan", "eda", "select_method", "freeze", "execute", "report",
                     "stage_gate", "start_round"}
