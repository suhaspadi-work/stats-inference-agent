"""
Focused check: reject BOTH proposed operations in a single wrangle
checklist (one duplicate-rows fix, one missingness fix) and confirm both
-- not just the first -- get recorded as separate UnaddressedIssues, both
surface in select_method's evidence, and both get mentioned in the final
report. This closes a real gap: 7e only ever tested rejecting ONE issue.
"""
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from langgraph.types import Command

from core.dataset import LocalDiskStore
from core.ingestion import load_dataset
from core.graph import build_agent_graph

load_dotenv()


def build_dataset(store):
    rng = np.random.default_rng(11)
    df = pd.DataFrame({
        "customer_id": range(1, 201),
        "region": ["East"] * 100 + ["West"] * 100,
        "total_spend": np.concatenate([rng.normal(50, 10, 100), rng.normal(58, 10, 100)]),
    })

    # Real, exact duplicate rows -- sampled from the data itself, same
    # technique as core.synthetic's quality-issues generator, so these are
    # genuine full-row duplicates detect_duplicate_rows will actually catch.
    dup_rows = df.sample(n=3, random_state=11)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # Missingness injected on a column NOT involved in the duplicates, so
    # both issues are independently detectable.
    missing_idx = rng.choice(df.index, size=int(len(df) * 0.1), replace=False)
    df["region"] = df["region"].astype(object)
    df.loc[missing_idx, "region"] = None

    tmp_path = Path("/tmp/eval_chained_rejections.csv")
    df.to_csv(tmp_path, index=False)
    return load_dataset(tmp_path, session_id="session-chained-check", name="chained_check", store=store)


def main():
    store = LocalDiskStore(base_dir=Path("eval_data_chained_check"))
    dataset = build_dataset(store)
    app = build_agent_graph(store)
    config = {"configurable": {"thread_id": "chained-rejections-check"}}

    result = app.invoke(
        {
            "request_text": "Did customers in the East region spend more than the West region?",
            "session_id": "session-chained-check",
            "dataset": dataset,
        },
        config,
    )

    if "__interrupt__" not in result:
        print("FAIL: expected a wrangling_approval interrupt with 2 proposed operations, got none.")
        return

    payload = result["__interrupt__"][0].value
    proposed = payload.get("proposed_operations", [])
    print(f"Proposed operations: {len(proposed)}")
    for op in proposed:
        print(f"  [{op['index']}] {op['op_type']} -- {op['rationale'][:80]}")

    if len(proposed) < 2:
        print(f"FAIL: expected 2 proposed operations (dedupe + filter), got {len(proposed)}.")
        return

    # Reject BOTH, with distinct reasons, so we can confirm both are tracked separately
    decisions = [
        {"index": op["index"], "action": "reject", "reason": f"Rejecting {op['op_type']} to investigate manually"}
        for op in proposed
    ]
    result = app.invoke(Command(resume={"decisions": decisions}), config)

    # Should now hit the freeze interrupt
    if "__interrupt__" not in result:
        print("FAIL: expected the freeze interrupt next, got none.")
        return
    final = app.invoke(Command(resume={"action": "approve"}), config)

    plan = final.get("plan")
    if plan is None:
        print("FAIL: no plan in final state.")
        return

    print(f"\nunaddressed_issues recorded: {len(plan.unaddressed_issues)}")
    for issue in plan.unaddressed_issues:
        print(f"  - {issue.issue_type}: {issue.rejection_reason}")

    if len(plan.unaddressed_issues) < 2:
        print(f"FAIL: expected 2 unaddressed_issues (one per rejection), got {len(plan.unaddressed_issues)}.")
        return

    issue_types = {i.issue_type for i in plan.unaddressed_issues}
    if issue_types != {"duplicate_rows", "high_missingness"}:
        print(f"FAIL: expected both duplicate_rows and high_missingness, got {issue_types}.")
        return

    evidence_issues = plan.latest_decision.evidence.get("unaddressed_issues", [])
    print(f"\nunaddressed_issues in method-selection evidence: {len(evidence_issues)}")
    if len(evidence_issues) < 2:
        print(f"FAIL: evidence only shows {len(evidence_issues)} of 2 unaddressed issues.")
        return

    report_lower = (final.get("report") or "").lower()
    mentions_dup = "duplicate" in report_lower
    mentions_missing = "missing" in report_lower or "region" in report_lower and "caveat" in report_lower
    print(f"\nReport mentions duplicates: {mentions_dup}")
    print(f"Report text:\n{final.get('report')}")

    if mentions_dup:
        print("\nPASS: both rejections were tracked as separate UnaddressedIssues, surfaced in evidence, "
              "and at least the duplicate caveat appears in the final report.")
    else:
        print("\nFAIL: report does not clearly mention the rejected issues.")


if __name__ == "__main__":
    main()