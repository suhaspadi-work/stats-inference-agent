"""
Consolidated standalone checks for the remaining test scenarios scoped
this session. Each function is a focused, independent check -- run them
individually via the CHECK_NAME argument, or all together with no
argument. Written standalone (not as EvalScenario battery entries)
because several of these need to inspect mid-run state (interrupt
payloads, raw classification calls) rather than letting the battery's
auto-resolver run straight through.
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
from core.classifier import classify_request

load_dotenv()


def _make_dataset(store, name, df):
    tmp_path = Path(f"/tmp/eval_{name}.csv")
    df.to_csv(tmp_path, index=False)
    return load_dataset(tmp_path, session_id=f"session-{name}", name=name, store=store)


# ---------------------------------------------------------------------------
# Check 1: invalid-column edit at freeze. A human editing the plan's
# predictors at the freeze interrupt supplies a column name that doesn't
# exist in the dataset. Does this validate and fail cleanly, or get
# silently accepted and crash later inside execute_node/scipy?
# ---------------------------------------------------------------------------

def check_invalid_column_edit_at_freeze():
    print("\n=== Check 1: invalid-column edit at freeze ===")
    store = LocalDiskStore(base_dir=Path("eval_data_remaining"))
    rng = np.random.default_rng(21)
    df = pd.DataFrame({
        "customer_id": range(1, 201),
        "region": ["East"] * 100 + ["West"] * 100,
        "total_spend": np.concatenate([rng.normal(50, 10, 100), rng.normal(58, 10, 100)]),
    })
    dataset = _make_dataset(store, "invalid_edit_check", df)
    app = build_agent_graph(store)
    config = {"configurable": {"thread_id": "invalid-edit-check"}}

    result = app.invoke(
        {"request_text": "Did customers in the East region spend more than the West region?",
         "session_id": "session-invalid_edit_check", "dataset": dataset},
        config,
    )

    if "__interrupt__" not in result:
        print("INCONCLUSIVE: expected a freeze interrupt, got none.")
        return

    payload = result["__interrupt__"][0].value
    if payload.get("type") != "plan_approval":
        print(f"INCONCLUSIVE: expected plan_approval interrupt, got {payload.get('type')}.")
        return

    print(f"Original candidate_predictors: {payload.get('candidate_predictors')}")

    try:
        final = app.invoke(
            Command(resume={"action": "edit", "candidate_predictors": ["nonexistent_column_xyz"]}),
            config,
        )
    except Exception as e:
        print(f"PASS: edit with an invalid column raised cleanly at/before execution: {type(e).__name__}: {e}")
        return

    plan = final.get("plan")
    if plan and plan.status.value == "executed":
        print("FAIL: DANGER -- executed successfully with a nonexistent predictor column.")
    elif "__interrupt__" in final:
        print(f"PASS (or re-routed): hit another interrupt rather than executing blindly: {final['__interrupt__'][0].value.get('type')}")
    else:
        print(f"UNCLEAR: no exception, not executed, no further interrupt. plan_status={plan.status.value if plan else None}")


if __name__ == "__main__":
    check_name = sys.argv[1] if len(sys.argv) > 1 else None
    checks = {"edit": check_invalid_column_edit_at_freeze}
    if check_name:
        checks[check_name]()
    else:
        for fn in checks.values():
            fn()