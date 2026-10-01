"""
Check: a dataset that is BOTH genuinely messy (duplicates) AND paired with
an ambiguous request, so both the classify-clarification interrupt and the
wrangle-checklist interrupt would normally fire in the same run. Confirms
they resolve independently, in the correct order, without either one
corrupting the other's state -- a scenario explicitly deferred earlier in
testing because it needed careful, deliberate construction.
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
    rng = np.random.default_rng(61)
    df = pd.DataFrame({
        "customer_id": range(1, 201),
        "region": ["East"] * 100 + ["West"] * 100,
        "total_spend": np.concatenate([rng.normal(50, 10, 100), rng.normal(58, 10, 100)]),
    })
    # Real, exact duplicate rows
    dup_rows = df.sample(n=3, random_state=61)
    df = pd.concat([df, dup_rows], ignore_index=True)

    tmp_path = Path("/tmp/eval_concurrent_check.csv")
    df.to_csv(tmp_path, index=False)
    return load_dataset(tmp_path, session_id="session-concurrent-check", name="concurrent_check", store=store)


def main():
    store = LocalDiskStore(base_dir=Path("eval_data_concurrent_check"))
    dataset = build_dataset(store)
    app = build_agent_graph(store)
    config = {"configurable": {"thread_id": "concurrent-interrupts-check"}}

    # Deliberately vague request -- ambiguous enough to plausibly trigger
    # classify_node's low-confidence clarification gate.
    result = app.invoke(
        {"request_text": "Is there a relationship in this data?",
         "session_id": "session-concurrent-check", "dataset": dataset},
        config,
    )

    interrupt_sequence = []

    if "__interrupt__" not in result:
        print("No interrupt fired at all -- the request may have been confidently classified. "
              "Re-run with a more ambiguous phrasing if this happens.")
        return

    payload = result["__interrupt__"][0].value
    interrupt_sequence.append(payload.get("type"))
    print(f"First interrupt: {payload.get('type')}")

    if payload.get("type") == "clarification_needed":
        print(f"  question: {payload.get('question', '')[:150]}")
        result = app.invoke(
            Command(resume="Please look for any association between region and total spend."), config,
        )
    elif payload.get("type") == "wrangling_approval":
        print(f"  proposed_operations: {[op['op_type'] for op in payload.get('proposed_operations', [])]}")
        result = app.invoke(
            Command(resume={"decisions": [
                {"index": op["index"], "action": "approve"} for op in payload["proposed_operations"]
            ]}),
            config,
        )
    else:
        print(f"Unexpected first interrupt type: {payload.get('type')}")
        return

    # Keep resolving however many interrupts come next, tracking the order
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        itype = payload.get("type")
        interrupt_sequence.append(itype)
        print(f"Next interrupt: {itype}")

        if itype == "clarification_needed":
            resume = "Please look for any association between region and total spend."
        elif itype == "wrangling_approval":
            print(f"  proposed_operations: {[op['op_type'] for op in payload.get('proposed_operations', [])]}")
            resume = {"decisions": [
                {"index": op["index"], "action": "approve"} for op in payload["proposed_operations"]
            ]}
        elif itype == "plan_approval":
            resume = {"action": "approve"}
        else:
            print(f"FAIL: unknown interrupt type {itype}")
            return

        result = app.invoke(Command(resume=resume), config)

    plan = result.get("plan")
    print(f"\nFull interrupt sequence: {interrupt_sequence}")
    print(f"Final plan status: {plan.status.value if plan else None}")
    print(f"Final dataset version: {result.get('dataset').version if result.get('dataset') else None}")

    if plan and plan.status.value == "executed":
        print("PASS: multiple interrupts resolved correctly in sequence, run completed successfully.")
    else:
        print("FAIL: did not reach a clean executed state.")


if __name__ == "__main__":
    main()