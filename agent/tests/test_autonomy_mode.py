import pandas as pd
import pytest
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from core.agent_state import AgentState
from core.dataset import LocalDiskStore
from core.ingestion import load_dataset, profile_dataset
from core.nodes import make_wrangle_node


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def make_dataset_with_duplicates(tmp_path, store):
    df = pd.DataFrame({"id": [1, 2, 2, 3], "region": ["East", "West", "West", "East"]})
    path = tmp_path / "customers.csv"
    df.to_csv(path, index=False)
    return load_dataset(path, session_id="session-abc", name="customers", store=store)


def build_wrangle_graph(store):
    graph = StateGraph(AgentState)
    graph.add_node("wrangle", make_wrangle_node(store))
    graph.add_edge(START, "wrangle")
    graph.add_edge("wrangle", END)
    return graph.compile(checkpointer=InMemorySaver())


def test_default_autonomy_mode_still_interrupts(tmp_path, store):
    """No autonomy_mode key at all -- must behave exactly like before this change (defaults to 'analyst')."""
    ds = make_dataset_with_duplicates(tmp_path, store)
    profile = profile_dataset(ds, store=store)
    app = build_wrangle_graph(store)

    result = app.invoke(
        {"dataset": ds, "profile": profile, "request_text": "x", "session_id": "session-abc"},
        {"configurable": {"thread_id": "t1"}},
    )
    assert "__interrupt__" in result


def test_analyst_mode_explicit_still_interrupts(tmp_path, store):
    ds = make_dataset_with_duplicates(tmp_path, store)
    profile = profile_dataset(ds, store=store)
    app = build_wrangle_graph(store)

    result = app.invoke(
        {"dataset": ds, "profile": profile, "request_text": "x", "session_id": "session-abc", "autonomy_mode": "analyst"},
        {"configurable": {"thread_id": "t2"}},
    )
    assert "__interrupt__" in result


@pytest.mark.live
def test_stakeholder_mode_auto_approves_duplicate_removal_without_interrupt(tmp_path, store):
    """
    The core autonomy behavior: in stakeholder mode, a duplicate_rows issue
    is auto-approved and executed with NO interrupt at all.
    """
    ds = make_dataset_with_duplicates(tmp_path, store)
    profile = profile_dataset(ds, store=store)
    app = build_wrangle_graph(store)

    result = app.invoke(
        {"dataset": ds, "profile": profile, "request_text": "x", "session_id": "session-abc", "autonomy_mode": "stakeholder"},
        {"configurable": {"thread_id": "t3"}},
    )

    assert "__interrupt__" not in result
    assert result["dataset"].version == 2  # the dedupe actually ran
    cleaned = store.read(result["dataset"].storage_key)
    assert len(cleaned) == 3  # duplicate removed