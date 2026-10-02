"""
Final backend testing pass before moving into the product layer. Five
checks covering data-shape extremes and interrupt-path completeness not
yet exercised anywhere else:

1. Tiny dataset (n=3 per group)
2. Single-group column (zero variance in the grouping column itself)
3. Zero-variance outcome column (all identical values)
4. Wrangle-checklist EDIT action (only approve/reject tested so far)
5. Plan rejection actually terminates the graph cleanly (not just that
   clarification_needed is returned, but that the graph run genuinely ends)
"""
import sys
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from langgraph.types import Command

from core.dataset import LocalDiskStore
from core.ingestion import load_dataset
from core.graph import build_agent_graph

load_dotenv()


def _make_dataset(store, name, df):
    tmp_path = Path(f"/tmp/eval_{name}.csv")
    df.to_csv(tmp_path, index=False)
    return load_dataset(tmp_path, session_id=f"session-{name}", name=name, store=store)


def _drain_interrupts(app, config, result, auto_resolve=True):
    """Resolves every interrupt with sensible defaults, returning the final result."""
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


# ---------------------------------------------------------------------------
# Check 1: tiny dataset (n=3 per group).
# ---------------------------------------------------------------------------

def check_tiny_dataset():
    print("\n=== Check 1: tiny dataset (n=3 per group) ===")
    store = LocalDiskStore(base_dir=Path("eval_data_final_gaps_2"))
    df = pd.DataFrame({
        "customer_id": range(1, 7),
        "region": ["East", "East", "East", "West", "West", "West"],
        "total_spend": [48.0, 52.0, 50.0, 60.0, 58.0, 61.0],
    })
    dataset = _make_dataset(store, "tiny_dataset", df)
    app = build_agent_graph(store)
    config = {"configurable": {"thread_id": "tiny-dataset-check"}}

    try:
        result = app.invoke(
            {"request_text": "Did customers in the East region spend more than the West region?",
             "session_id": "session-tiny_dataset", "dataset": dataset},
            config,
        )
        result = _drain_interrupts(app, config, result)
    except Exception as e:
        print(f"Failed: {type(e).__name__}: {e}")
        return

    plan = result.get("plan")
    test_result = result.get("test_result")
    print(f"Plan status: {plan.status.value if plan else None}")
    print(f"Test result: {test_result}")
    if plan and plan.status.value == "executed" and test_result:
        print("PASS: completed on a tiny dataset without crashing.")
    else:
        print("FAIL: did not complete successfully.")


# ---------------------------------------------------------------------------
# Check 2: single-group column (zero variance in the GROUPING column).
# ---------------------------------------------------------------------------

def check_single_group_column():
    print("\n=== Check 2: single-group column (only one region value present) ===")
    store = LocalDiskStore(base_dir=Path("eval_data_final_gaps_2"))
    rng = np.random.default_rng(101)
    df = pd.DataFrame({
        "customer_id": range(1, 201),
        "region": ["East"] * 200,  # only ONE group present at all
        "total_spend": rng.normal(50, 10, 200),
    })
    dataset = _make_dataset(store, "single_group", df)
    app = build_agent_graph(store)
    config = {"configurable": {"thread_id": "single-group-check"}}

    try:
        result = app.invoke(
            {"request_text": "Did customers in the East region spend more than the West region?",
             "session_id": "session-single_group", "dataset": dataset},
            config,
        )
        result = _drain_interrupts(app, config, result)
    except Exception as e:
        print(f"PASS: failed cleanly rather than crashing confusingly: {type(e).__name__}: {e}")
        return

    plan = result.get("plan")
    if plan and plan.status.value == "executed":
        print("FAIL: DANGER -- executed successfully despite only one group being present.")
    else:
        print(f"UNCLEAR: no exception, not executed. plan_status={plan.status.value if plan else None}")


# ---------------------------------------------------------------------------
# Check 3: zero-variance OUTCOME column (all identical values).
# ---------------------------------------------------------------------------

def check_zero_variance_outcome():
    print("\n=== Check 3: zero-variance outcome column (all identical values) ===")
    store = LocalDiskStore(base_dir=Path("eval_data_final_gaps_2"))
    df = pd.DataFrame({
        "customer_id": range(1, 201),
        "region": ["East"] * 100 + ["West"] * 100,
        "total_spend": [50.0] * 200,  # zero variance
    })
    dataset = _make_dataset(store, "zero_variance", df)
    app = build_agent_graph(store)
    config = {"configurable": {"thread_id": "zero-variance-check"}}

    try:
        result = app.invoke(
            {"request_text": "Did customers in the East region spend more than the West region?",
             "session_id": "session-zero_variance", "dataset": dataset},
            config,
        )
        result = _drain_interrupts(app, config, result)
    except Exception as e:
        print(f"PASS: failed cleanly rather than crashing confusingly: {type(e).__name__}: {e}")
        return

    plan = result.get("plan")
    test_result = result.get("test_result")
    print(f"Plan status: {plan.status.value if plan else None}")
    print(f"Test result: {test_result}")
    if plan and plan.status.value == "executed" and test_result:
        p_val = test_result.get("p_value")
        is_nan = p_val != p_val  # NaN check without importing math
        print(f"p-value is NaN: {is_nan}")
        if is_nan:
            print("FAIL: DANGER -- executed 'successfully' but produced a NaN p-value with no guard against it.")
        else:
            print(f"PASS: executed and produced a well-defined result (p={p_val}) despite zero variance.")
    else:
        print(f"UNCLEAR: plan_status={plan.status.value if plan else None}")


# ---------------------------------------------------------------------------
# Check 4: wrangle-checklist EDIT action (never tested -- only approve/reject so far).
# ---------------------------------------------------------------------------

def check_wrangle_edit_action():
    print("\n=== Check 4: wrangle-checklist EDIT action ===")
    store = LocalDiskStore(base_dir=Path("eval_data_final_gaps_2"))
    rng = np.random.default_rng(111)
    df = pd.DataFrame({
        "customer_id": range(1, 201),
        "region": ["East"] * 100 + ["West"] * 100,
        "total_spend": np.concatenate([rng.normal(50, 10, 100), rng.normal(58, 10, 100)]),
    })
    dup_rows = df.sample(n=3, random_state=111)
    df = pd.concat([df, dup_rows], ignore_index=True)
    dataset = _make_dataset(store, "wrangle_edit", df)
    app = build_agent_graph(store)
    config = {"configurable": {"thread_id": "wrangle-edit-check"}}

    result = app.invoke(
        {"request_text": "Did customers in the East region spend more than the West region?",
         "session_id": "session-wrangle_edit", "dataset": dataset},
        config,
    )

    if "__interrupt__" not in result:
        print("INCONCLUSIVE: expected a wrangling_approval interrupt, got none.")
        return

    payload = result["__interrupt__"][0].value
    proposed = payload.get("proposed_operations", [])
    print(f"Proposed: {[(op['index'], op['op_type'], op['params']) for op in proposed]}")
    dedupe_op = next((op for op in proposed if op["op_type"] == "dedupe"), None)
    if dedupe_op is None:
        print("INCONCLUSIVE: no dedupe operation proposed.")
        return

    # Edit the dedupe to scope it to a SPECIFIC subset of columns, rather
    # than accepting the model's original (likely subset=None/all-columns) proposal.
    edited_decisions = [{
        "index": dedupe_op["index"], "action": "edit",
        "params": {"subset": ["customer_id", "region", "total_spend"]},
    }]
    result = app.invoke(Command(resume={"decisions": edited_decisions}), config)
    result = _drain_interrupts(app, config, result)

    plan = result.get("plan")
    dataset_after = result.get("dataset")
    print(f"Plan status: {plan.status.value if plan else None}")
    print(f"Dataset version after edited dedupe: {dataset_after.version if dataset_after else None}")

    if plan and plan.status.value == "executed" and dataset_after and dataset_after.version >= 2:
        print("PASS: edit action on the wrangle checklist was correctly applied and executed.")
    else:
        print("FAIL: edit action did not complete as expected.")


# ---------------------------------------------------------------------------
# Check 5: plan rejection terminates the graph cleanly.
# ---------------------------------------------------------------------------

def check_plan_rejection_terminates_cleanly():
    print("\n=== Check 5: plan rejection terminates the graph cleanly ===")
    store = LocalDiskStore(base_dir=Path("eval_data_final_gaps_2"))
    rng = np.random.default_rng(121)
    df = pd.DataFrame({
        "customer_id": range(1, 201),
        "region": ["East"] * 100 + ["West"] * 100,
        "total_spend": np.concatenate([rng.normal(50, 10, 100), rng.normal(58, 10, 100)]),
    })
    dataset = _make_dataset(store, "plan_rejection", df)
    app = build_agent_graph(store)
    config = {"configurable": {"thread_id": "plan-rejection-check"}}

    result = app.invoke(
        {"request_text": "Did customers in the East region spend more than the West region?",
         "session_id": "session-plan_rejection", "dataset": dataset},
        config,
    )
    if "__interrupt__" not in result:
        print("INCONCLUSIVE: expected a plan_approval interrupt, got none.")
        return

    payload = result["__interrupt__"][0].value
    if payload.get("type") != "plan_approval":
        print(f"INCONCLUSIVE: expected plan_approval, got {payload.get('type')}.")
        return

    result = app.invoke(Command(resume={"action": "reject", "reason": "Testing graph termination"}), config)

    print(f"'__interrupt__' present after rejection: {'__interrupt__' in result}")
    print(f"'clarification_needed' in result: {result.get('clarification_needed')}")
    plan = result.get("plan")
    print(f"Plan status after rejection: {plan.status.value if plan else None}")

    # The real check: does the graph consider this run DONE? Inspect
    # get_state's next tasks -- an empty tuple means the graph has
    # genuinely reached a terminal point, not paused waiting on anything.
    state = app.get_state(config)
    print(f"Graph's next pending tasks: {state.next}")

    if "__interrupt__" not in result and plan and plan.status.value != "frozen" and not state.next:
        print("PASS: plan rejection terminates the graph cleanly -- no further interrupt, no pending tasks, "
              "plan correctly never frozen.")
    else:
        print("FAIL: graph did not terminate cleanly after a plan rejection.")


if __name__ == "__main__":
    check_name = sys.argv[1] if len(sys.argv) > 1 else None
    checks = {
        "tiny": check_tiny_dataset,
        "single_group": check_single_group_column,
        "zero_variance": check_zero_variance_outcome,
        "wrangle_edit": check_wrangle_edit_action,
        "reject_terminates": check_plan_rejection_terminates_cleanly,
    }
    if check_name:
        checks[check_name]()
    else:
        for fn in checks.values():
            fn()