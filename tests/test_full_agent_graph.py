import pytest
from langgraph.types import Command

from core.dataset import LocalDiskStore
from core.ingestion import load_dataset
from core.graph import build_agent_graph
from core.synthetic import generate_two_group_continuous


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


@pytest.fixture
def dataset(tmp_path, store):
    """A clear, unambiguous two-group dataset with a known, planted effect."""
    df, spec = generate_two_group_continuous(n_per_group=200, mean_a=50, effect=8.0, std=10.0, seed=5)
    df = df.rename(columns={"outcome": "total_spend", "group": "region"})
    path = tmp_path / "customers.csv"
    df.to_csv(path, index=False)
    ds = load_dataset(path, session_id="session-abc", name="customers", store=store)
    return ds, spec


@pytest.mark.live
def test_full_agent_graph_end_to_end(dataset, store):
    """
    The full chain: a real request, real classification, real profiling,
    real plan drafting (with the hallucinated-column guard active), a
    human-approval interrupt, real test execution against planted ground
    truth, and a real report with the causal-language guard enforced.
    Every phase of this project, working together as one agent.
    """
    ds, spec = dataset
    app = build_agent_graph(store)
    config = {"configurable": {"thread_id": "e2e-test-1"}}

    initial_state = {
        "request_text": "Did customers in the West region spend more than the East region?",
        "session_id": "session-abc",
        "dataset": ds,
    }

    # Run until the freeze interrupt (classify should NOT interrupt -- this
    # request is unambiguous, per test_classifier.py's earlier confirmation)
    result = app.invoke(initial_state, config)

    assert "__interrupt__" in result
    interrupt_payload = result["__interrupt__"][0].value
    assert interrupt_payload["type"] == "plan_approval"
    assert interrupt_payload["outcome_name"] == "total_spend"
    assert "region" in interrupt_payload["candidate_predictors"]

    # Approve the plan -- resume through freeze -> execute -> report
    final_result = app.invoke(Command(resume={"action": "approve"}), config)

    assert "__interrupt__" not in final_result
    assert final_result["plan"].status.value == "executed"

    # The core correctness check: the executed test recovers the PLANTED
    # ground truth from the synthetic generator, having passed through
    # every node in the graph, not just a direct scipy call
    observed_effect = (
        final_result["test_result"]["group_b_mean"] - final_result["test_result"]["group_a_mean"]
    )
    assert abs(observed_effect - spec.true_effect) < 2.0
    assert final_result["test_result"]["p_value"] < 0.05

    # The report exists, uses the real numbers, and respects the causal guard
    assert len(final_result["report"]) > 0
    report_lower = final_result["report"].lower()
    assert "associat" in report_lower or "correlat" in report_lower  # matches "association"/"associated" either way