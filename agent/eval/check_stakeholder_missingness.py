"""
Focused, standalone check for a specific 7d design claim: stakeholder mode
should auto-approve duplicate_rows but ALWAYS interrupt on missingness.
Written standalone (not as a battery EvalScenario) because this needs to
inspect the interrupt itself, not auto-resolve past it -- the opposite of
what eval/runner.py's run_scenario is built to do.
"""
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import pandas as pd

from core.dataset import LocalDiskStore
from core.ingestion import load_dataset
from core.graph import build_agent_graph

load_dotenv()


def build_dataset(store):
    rng = np.random.default_rng(9)
    df = pd.DataFrame({
        "customer_id": range(1, 201),
        "region": ["East"] * 100 + ["West"] * 100,
        "total_spend": rng.normal(50, 10, 200),
    })
    missing_idx = rng.choice(df.index, size=20, replace=False)  # 10% missing
    df["region"] = df["region"].astype(object)
    df.loc[missing_idx, "region"] = None

    tmp_path = Path("/tmp/eval_stakeholder_missingness.csv")
    df.to_csv(tmp_path, index=False)
    return load_dataset(tmp_path, session_id="session-stakeholder-check", name="stakeholder_check", store=store)


def main():
    store = LocalDiskStore(base_dir=Path("eval_data_stakeholder_check"))
    dataset = build_dataset(store)
    app = build_agent_graph(store)
    config = {"configurable": {"thread_id": "stakeholder-missingness-check"}}

    result = app.invoke(
        {
            "request_text": "Did customers in the East region spend more than the West region?",
            "session_id": "session-stakeholder-check",
            "dataset": dataset,
            "autonomy_mode": "stakeholder",
        },
        config,
    )

    if "__interrupt__" not in result:
        print("FAIL: No interrupt occurred at all under stakeholder mode -- missingness was silently auto-approved.")
        print(f"Final dataset version: {result.get('dataset').version if result.get('dataset') else None}")
        return

    payload = result["__interrupt__"][0].value
    print(f"Interrupt type: {payload.get('type')}")

    if payload.get("type") != "wrangling_approval":
        print(f"FAIL: Expected a wrangling_approval interrupt, got {payload.get('type')}")
        return

    proposed = payload.get("proposed_operations", [])
    missingness_ops = [op for op in proposed if op["op_type"] == "filter"]
    auto_note = payload.get("auto_approved_note")

    print(f"Proposed operations shown for review: {len(proposed)}")
    print(f"Auto-approved note: {auto_note}")

    if missingness_ops:
        print(f"PASS: The missingness fix ({missingness_ops[0]['op_type']}) correctly required human review, "
              f"even in stakeholder mode.")
    else:
        print("FAIL: No missingness-related operation appeared in the review list -- "
              "it may have been silently auto-approved, violating 7d's design.")


if __name__ == "__main__":
    main()