import pandas as pd
import pytest
from langgraph.graph import StateGraph, START, END

from core.agent_state import AgentState
from core.dataset import LocalDiskStore
from core.ingestion import load_dataset
from core.nodes import make_profile_node


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


@pytest.fixture
def dataset(tmp_path, store):
    df = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "region": ["East", "West", "East"],
        "spend": [100.0, 150.0, 90.0],
    })
    path = tmp_path / "customers.csv"
    df.to_csv(path, index=False)
    return load_dataset(path, session_id="session-abc", name="customers", store=store)


def test_profile_node_populates_state_with_model_safe_view(dataset, store):
    graph = StateGraph(AgentState)
    graph.add_node("profile", make_profile_node(store))
    graph.add_edge(START, "profile")
    graph.add_edge("profile", END)
    app = graph.compile()

    result = app.invoke({"request_text": "x", "session_id": "session-abc", "dataset": dataset})

    assert "profile" in result
    assert result["profile"].row_count == 3
    assert result["profile"].column_count == 3
    assert result["profile"].dataset_handle == dataset.handle


def test_profile_node_output_passes_the_privacy_guard(dataset, store):
    from core.model_safe_view import assert_model_safe

    graph = StateGraph(AgentState)
    graph.add_node("profile", make_profile_node(store))
    graph.add_edge(START, "profile")
    graph.add_edge("profile", END)
    app = graph.compile()

    result = app.invoke({"request_text": "x", "session_id": "session-abc", "dataset": dataset})
    assert_model_safe(result["profile"])