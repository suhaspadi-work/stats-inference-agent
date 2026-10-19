import pandas as pd
import pytest
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from core.agent_state import AgentState
from core.classifier import _ProposedOperationOut
from core.dataset import LocalDiskStore
from core.ingestion import load_dataset, profile_dataset
from core.nodes import make_wrangle_propose_node, make_wrangle_review_node, route_after_wrangle_propose


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def make_dataset_with_duplicates(tmp_path, store):
    df = pd.DataFrame({"id": [1, 2, 2, 3], "region": ["East", "West", "West", "East"]})
    path = tmp_path / "customers.csv"
    df.to_csv(path, index=False)
    return load_dataset(path, session_id="session-abc", name="customers", store=store)


class FakeProposer:
    """Stands in for the model's cleaning proposals (G23) and counts its calls."""
    def __init__(self):
        self.calls = 0

    def __call__(self, issues, candidates, request_text=None):  # M2.3b: and the question
        self.calls += 1
        proposals = []
        for i, issue in enumerate(issues):
            if issue.issue_type == "duplicate_rows":
                choice = "dedupe_all"
            elif issue.issue_type == "high_missingness":
                choice = "delete_rows"
            else:
                raise AssertionError(f"no fake proposal for issue type {issue.issue_type!r}")
            proposals.append(_ProposedOperationOut(
                issue_index=i, candidate_id=choice, rationale="fake",
                alternative_candidate_id="leave_as_is", alternative_rejected_because="fake",
            ))
        return proposals


@pytest.fixture
def fake_proposer():
    return FakeProposer()


def build_wrangle_graph(store, propose_fn=None):
    """wrangle_propose -> (wrangle_review) -> a stub draft_plan, wired as in the real graph.
    Without propose_fn the real (live) proposer is used."""
    graph = StateGraph(AgentState)
    propose = make_wrangle_propose_node(store) if propose_fn is None else make_wrangle_propose_node(store, propose_fn=propose_fn)
    graph.add_node("wrangle_propose", propose)
    graph.add_node("wrangle_review", make_wrangle_review_node(store))
    graph.add_node("draft_plan", lambda state: {})
    graph.add_edge(START, "wrangle_propose")
    graph.add_conditional_edges("wrangle_propose", route_after_wrangle_propose,
                                {"wrangle_review": "wrangle_review", "draft_plan": "draft_plan"})
    graph.add_edge("wrangle_review", "draft_plan")
    graph.add_edge("draft_plan", END)
    return graph.compile(checkpointer=InMemorySaver())


def _assert_one_dedupe_pause(result, fake_proposer):
    assert fake_proposer.calls == 1
    (pause,) = result["__interrupt__"]
    assert pause.value["type"] == "wrangling_approval"
    assert [op["op_type"] for op in pause.value["proposed_operations"]] == ["dedupe"]
    assert "auto_approved_note" not in pause.value


def test_default_autonomy_mode_still_interrupts(tmp_path, store, fake_proposer):
    """No autonomy_mode key at all -- must behave exactly like before this change (defaults to 'analyst')."""
    ds = make_dataset_with_duplicates(tmp_path, store)
    profile = profile_dataset(ds, store=store)
    app = build_wrangle_graph(store, propose_fn=fake_proposer)

    result = app.invoke(
        {"dataset": ds, "profile": profile, "request_text": "x", "session_id": "session-abc"},
        {"configurable": {"thread_id": "t1"}},
    )
    assert "__interrupt__" in result
    _assert_one_dedupe_pause(result, fake_proposer)


def test_analyst_mode_explicit_still_interrupts(tmp_path, store, fake_proposer):
    ds = make_dataset_with_duplicates(tmp_path, store)
    profile = profile_dataset(ds, store=store)
    app = build_wrangle_graph(store, propose_fn=fake_proposer)

    result = app.invoke(
        {"dataset": ds, "profile": profile, "request_text": "x", "session_id": "session-abc", "autonomy_mode": "analyst"},
        {"configurable": {"thread_id": "t2"}},
    )
    assert "__interrupt__" in result
    _assert_one_dedupe_pause(result, fake_proposer)


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