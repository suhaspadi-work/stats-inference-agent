import pandas as pd
import pytest
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from core.agent_state import AgentState
from core.dataset import LocalDiskStore
from core.ingestion import load_dataset, profile_dataset
from core.nodes import make_wrangle_node
from core.nodes import _normalize_params


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def make_dataset(df, tmp_path, store, name="data"):
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    return load_dataset(path, session_id="session-abc", name=name, store=store)


def build_wrangle_graph(store):
    graph = StateGraph(AgentState)
    graph.add_node("wrangle", make_wrangle_node(store))
    graph.add_edge(START, "wrangle")
    graph.add_edge("wrangle", END)
    return graph.compile(checkpointer=InMemorySaver())


def test_no_issues_skips_interrupt_entirely(tmp_path, store):
    df = pd.DataFrame({"id": [1, 2, 3], "region": ["East", "West", "East"]})
    ds = make_dataset(df, tmp_path, store)
    profile = profile_dataset(ds, store=store)

    app = build_wrangle_graph(store)
    result = app.invoke(
        {"dataset": ds, "profile": profile, "request_text": "x", "session_id": "session-abc"},
        {"configurable": {"thread_id": "t1"}},
    )

    assert "__interrupt__" not in result
    assert result["dataset"].handle == ds.handle  # unchanged, no new version created


@pytest.mark.live
def test_approving_dedupe_produces_a_new_clean_dataset_version(tmp_path, store):
    df = pd.DataFrame({"id": [1, 2, 2, 3], "region": ["East", "West", "West", "East"]})
    ds = make_dataset(df, tmp_path, store)
    profile = profile_dataset(ds, store=store)

    app = build_wrangle_graph(store)
    config = {"configurable": {"thread_id": "t2"}}
    result = app.invoke(
        {"dataset": ds, "profile": profile, "request_text": "x", "session_id": "session-abc"}, config,
    )

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["type"] == "wrangling_approval"
    assert len(payload["proposed_operations"]) == 1
    assert payload["proposed_operations"][0]["op_type"] == "dedupe"

    final = app.invoke(Command(resume={"decisions": [{"index": 0, "action": "approve"}]}), config)

    assert "__interrupt__" not in final
    assert final["dataset"].version == 2  # new version produced
    cleaned = store.read(final["dataset"].storage_key)
    assert len(cleaned) == 3  # duplicate removed


@pytest.mark.live
def test_rejecting_dedupe_records_unaddressed_issue_via_pending_mechanism(tmp_path, store):
    df = pd.DataFrame({"id": [1, 2, 2, 3], "region": ["East", "West", "West", "East"]})
    ds = make_dataset(df, tmp_path, store)
    profile = profile_dataset(ds, store=store)

    app = build_wrangle_graph(store)
    config = {"configurable": {"thread_id": "t3"}}
    app.invoke({"dataset": ds, "profile": profile, "request_text": "x", "session_id": "session-abc"}, config)

    final = app.invoke(
        Command(resume={"decisions": [{"index": 0, "action": "reject", "reason": "Investigate first"}]}), config,
    )

    assert "__interrupt__" not in final
    assert final["dataset"].version == 1  # unchanged -- nothing executed
    assert "pending_unaddressed_issues" in final
    assert len(final["pending_unaddressed_issues"]) == 1
    assert final["pending_unaddressed_issues"][0].rejection_reason == "Investigate first"

def test_normalize_params_converts_empty_dedupe_subset_to_none():
    """
    Regression test for a real bug found via the live wrangle_node test:
    the model proposes subset=[] meaning 'all columns', but pandas'
    drop_duplicates(subset=[]) is nonsensical -- it must become subset=None.
    """
    assert _normalize_params("dedupe", {"subset": []}) == {"subset": None}
    assert _normalize_params("dedupe", {"subset": None}) == {"subset": None}
    assert _normalize_params("dedupe", {"subset": ["id"]}) == {"subset": ["id"]}
    assert _normalize_params("filter", {"column": "x", "condition": "not_null", "value": None}) == {
        "column": "x", "condition": "not_null", "value": None,
    }