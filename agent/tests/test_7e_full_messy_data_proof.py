import pytest
from langgraph.types import Command

from core.dataset import LocalDiskStore
from core.ingestion import load_dataset
from core.graph import build_agent_graph
from core.synthetic import generate_two_group_continuous_with_quality_issues


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


@pytest.fixture
def messy_dataset(tmp_path, store):
    """
    A genuinely messy dataset: a known, planted effect PLUS real duplicate
    rows -- the scenario this entire phase was built to handle. No
    missingness here, to keep this one proof focused on the duplicate-reject
    path specifically (missingness is already covered by 7a/7b/7c's own
    isolated tests).
    """
    df, spec = generate_two_group_continuous_with_quality_issues(
        n_per_group=200, mean_a=50.0, effect=8.0, std=10.0, seed=5,
        n_duplicate_rows=5, missingness_pct=0.0,
    )
    df = df.rename(columns={"outcome": "total_spend", "group": "region"})
    path = tmp_path / "customers.csv"
    df.to_csv(path, index=False)
    ds = load_dataset(path, session_id="session-abc", name="customers", store=store)
    return ds, spec


@pytest.mark.live
def test_full_graph_with_messy_data_and_deliberate_rejection(messy_dataset, store):
    """
    THE capstone proof for Phase 7. A genuinely messy dataset (known effect
    + real duplicates) runs through the ENTIRE graph: classify -> profile
    -> wrangle (interrupts, we REJECT the cleanup) -> draft_plan -> select_method
    -> freeze (interrupts, we approve) -> execute -> report.

    Verifies, in one continuous run, every link in the chain this phase was
    built for: the rejection is never silently dropped, it's recorded as an
    UnaddressedIssue, folded into the plan, surfaced in the method-selection
    evidence, and explicitly stated in the final report -- while the actual
    statistical result still correctly recovers the planted ground truth
    despite the un-cleaned duplicates still sitting in the data.
    """
    ds, spec = messy_dataset
    app = build_agent_graph(store)
    config = {"configurable": {"thread_id": "7e-proof-1"}}

    initial_state = {
        "request_text": "Did customers in the East region spend more than the West region?",
        "session_id": "session-abc",
        "dataset": ds,
    }

    # --- Run to the first interrupt ---
    result = app.invoke(initial_state, config)
    assert "__interrupt__" in result, "Expected the wrangle node to interrupt on detected duplicates"
    wrangle_payload = result["__interrupt__"][0].value
    assert wrangle_payload["type"] == "wrangling_approval"
    assert len(wrangle_payload["proposed_operations"]) >= 1
    dedupe_ops = [op for op in wrangle_payload["proposed_operations"] if op["op_type"] == "dedupe"]
    assert len(dedupe_ops) == 1, "Expected exactly one dedupe proposal for the injected duplicate rows"
    dedupe_index = dedupe_ops[0]["index"]

    # --- Reject the cleanup, deliberately ---
    result = app.invoke(
        Command(resume={"decisions": [
            {"index": dedupe_index, "action": "reject", "reason": "Want to investigate these duplicates manually first"}
        ]}),
        config,
    )

    # --- Should now hit the freeze interrupt (classify was never ambiguous, wrangle is resolved) ---
    assert "__interrupt__" in result, "Expected the freeze node to interrupt next, for plan approval"
    freeze_payload = result["__interrupt__"][0].value
    assert freeze_payload["type"] == "plan_approval"
    assert freeze_payload["outcome_name"] == "total_spend"
    assert "region" in freeze_payload["candidate_predictors"]

    # --- Approve the plan ---
    final_result = app.invoke(Command(resume={"action": "approve"}), config)

    assert "__interrupt__" not in final_result
    assert final_result["plan"].status.value == "executed"

    # --- Link 1: the rejection produced a real, durable UnaddressedIssue ---
    plan = final_result["plan"]
    assert len(plan.unaddressed_issues) == 1
    issue = plan.unaddressed_issues[0]
    assert issue.issue_type == "duplicate_rows"
    assert "investigate" in issue.rejection_reason.lower()

    # --- Link 2: it's surfaced in the method-selection evidence ---
    decision = plan.latest_decision
    assert "unaddressed_issues" in decision.evidence
    assert decision.evidence["unaddressed_issues"][0]["issue_type"] == "duplicate_rows"

    # --- Link 3: the statistics still correctly recover the PLANTED ground truth,
    #     despite the duplicates never being cleaned ---
    test_result = final_result["test_result"]
    observed_effect = test_result["group_b_mean"] - test_result["group_a_mean"]
    assert abs(observed_effect - spec.true_effect) < 2.5  # slightly wider tolerance given the uncleaned duplicates
    assert test_result["p_value"] < 0.05

    # --- Link 4: the final report explicitly states the caveat ---
    report_lower = final_result["report"].lower()
    assert "duplicate" in report_lower
    assert any(phrase in report_lower for phrase in (
        "caveat", "not removed", "left in the data", "not addressed", "unaddressed", "investigate",
    ))
    # And the causal-language guard still holds, since this is observational data
    assert any(phrase in report_lower for phrase in ("associat", "correlat"))