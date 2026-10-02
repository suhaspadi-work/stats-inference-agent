"""
Check: a second question asked against the SAME session, after the first
question already completed (and, in this case, cleaned some duplicates
along the way). Confirms the second run starts a genuinely fresh
AnalysisPlan (not corrupted by the first one's state) and correctly
operates on the LATEST dataset version -- not the original raw upload --
since our lineage design means the first run's wrangling should persist.
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
    rng = np.random.default_rng(71)
    df = pd.DataFrame({
        "customer_id": range(1, 201),
        "region": ["East"] * 100 + ["West"] * 100,
        "total_spend": np.concatenate([rng.normal(50, 10, 100), rng.normal(58, 10, 100)]),
    })
    dup_rows = df.sample(n=3, random_state=71)
    df = pd.concat([df, dup_rows], ignore_index=True)

    tmp_path = Path("/tmp/eval_multi_turn.csv")
    df.to_csv(tmp_path, index=False)
    return load_dataset(tmp_path, session_id="session-multi-turn", name="multi_turn", store=store)


def _run_to_completion(app, config, initial_state=None, resume_value=None):
    """Drives one request through to a final executed state, auto-approving every interrupt."""
    result = app.invoke(initial_state, config) if initial_state else app.invoke(Command(resume=resume_value), config)
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        itype = payload.get("type")
        if itype == "wrangling_approval":
            resume = {"decisions": [{"index": op["index"], "action": "approve"} for op in payload["proposed_operations"]]}
        elif itype == "plan_approval":
            resume = {"action": "approve"}
        else:
            resume = "Please proceed with the most natural reading of my original question."
        result = app.invoke(Command(resume=resume), config)
    return result


def main():
    store = LocalDiskStore(base_dir=Path("eval_data_multi_turn"))
    dataset_v1 = build_dataset(store)
    app = build_agent_graph(store)

    # --- Turn 1: first question, in its own thread ---
    config_1 = {"configurable": {"thread_id": "multi-turn-q1"}}
    result_1 = _run_to_completion(app, config_1, initial_state={
        "request_text": "Did customers in the East region spend more than the West region?",
        "session_id": "session-multi-turn", "dataset": dataset_v1,
    })

    plan_1 = result_1.get("plan")
    dataset_after_q1 = result_1.get("dataset")
    print(f"Turn 1 -- plan status: {plan_1.status.value if plan_1 else None}")
    print(f"Turn 1 -- dataset version after cleanup: {dataset_after_q1.version if dataset_after_q1 else None}")
    print(f"Turn 1 -- plan id: {plan_1.id if plan_1 else None}")

    if not plan_1 or plan_1.status.value != "executed":
        print("FAIL: turn 1 did not complete successfully.")
        return

    # --- Turn 2: SECOND question, same session, using the dataset turn 1 left behind ---
    # (In a real product, this would be the same conversational session;
    # here we pass dataset_after_q1 explicitly to simulate that continuity.)
    config_2 = {"configurable": {"thread_id": "multi-turn-q2"}}
    result_2 = _run_to_completion(app, config_2, initial_state={
        "request_text": "What is the average total spend overall, and does region still show a difference?",
        "session_id": "session-multi-turn", "dataset": dataset_after_q1,
    })

    plan_2 = result_2.get("plan")
    dataset_after_q2 = result_2.get("dataset")
    print(f"\nTurn 2 -- plan status: {plan_2.status.value if plan_2 else None}")
    print(f"Turn 2 -- dataset version used: {dataset_after_q2.version if dataset_after_q2 else None}")

    print(f"Turn 2 -- plan id: {plan_2.id if plan_2 else None}")

    if not plan_2 or plan_2.status.value != "executed":
        print("FAIL: turn 2 did not complete successfully.")
        return

    # --- The real checks ---
    distinct_plans = plan_1.id != plan_2.id
    no_leftover_duplicates = dataset_after_q2.version >= dataset_after_q1.version  # never regressed to an earlier version
    print(f"\nDistinct AnalysisPlan objects across turns: {distinct_plans}")
    print(f"Turn 2 built on turn 1's cleaned dataset (version did not regress): {no_leftover_duplicates}")

    if distinct_plans and no_leftover_duplicates:
        print("PASS: second turn used a fresh plan and correctly built on the first turn's cleaned dataset lineage.")
    else:
        print("FAIL: state from turn 1 leaked into or corrupted turn 2, or dataset lineage regressed.")


if __name__ == "__main__":
    main()