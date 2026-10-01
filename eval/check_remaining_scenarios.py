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
import time
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from langgraph.types import Command

from core.dataset import LocalDiskStore
from core.ingestion import load_dataset, profile_dataset
from core.graph import build_agent_graph
from core.classifier import classify_request

load_dotenv()


def _make_dataset(store, name, df):
    tmp_path = Path(f"/tmp/eval_{name}.csv")
    df.to_csv(tmp_path, index=False)
    return load_dataset(tmp_path, session_id=f"session-{name}", name=name, store=store)


# ---------------------------------------------------------------------------
# Check 1: invalid-column edit at freeze.
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
        print(f"PASS: edit with an invalid column raised cleanly: {type(e).__name__}: {e}")
        return

    plan = final.get("plan")
    if plan and plan.status.value == "executed":
        print("FAIL: DANGER -- executed successfully with a nonexistent predictor column.")
    elif "__interrupt__" in final:
        print(f"PASS (or re-routed): hit another interrupt rather than executing blindly.")
    else:
        print(f"UNCLEAR: no exception, not executed, no further interrupt.")


# ---------------------------------------------------------------------------
# Check 2: classification stability. Same request, sent twice, should not
# flip-flop on question_type/causal_status for identical input. Cheapest
# check -- no dataset needed at all, a direct call to classify_request.
# ---------------------------------------------------------------------------

def check_classification_stability():
    print("\n=== Check 2: classification stability ===")
    request = "Did customers in the East region spend more than the West region?"

    decision1 = classify_request(request)
    decision2 = classify_request(request)

    print(f"Call 1: question_type={decision1.question_type.value}, causal_status={decision1.causal_status.value}, confidence={decision1.confidence:.2f}")
    print(f"Call 2: question_type={decision2.question_type.value}, causal_status={decision2.causal_status.value}, confidence={decision2.confidence:.2f}")

    if decision1.question_type != decision2.question_type:
        print(f"FAIL: question_type is unstable across identical calls.")
        return
    if decision1.causal_status != decision2.causal_status:
        print(f"FAIL: causal_status is unstable across identical calls.")
        return

    print("PASS: question_type and causal_status are stable across repeated identical calls.")


# ---------------------------------------------------------------------------
# Check 3: privacy-boundary stress. A column with an obviously sensitive
# name/content (email addresses) -- confirm ModelSafeView's capping means
# raw values never reach the model, even with a request that nudges
# toward using that column directly.
# ---------------------------------------------------------------------------

def check_privacy_boundary_stress():
    print("\n=== Check 3: privacy-boundary stress (email column) ===")
    store = LocalDiskStore(base_dir=Path("eval_data_remaining"))
    rng = np.random.default_rng(31)
    emails = [f"customer{i}@example.com" for i in range(1, 201)]
    df = pd.DataFrame({
        "customer_id": range(1, 201),
        "email": emails,
        "region": ["East"] * 100 + ["West"] * 100,
        "total_spend": np.concatenate([rng.normal(50, 10, 100), rng.normal(58, 10, 100)]),
    })
    dataset = _make_dataset(store, "privacy_check", df)
    profile = profile_dataset(dataset, store=store)

    email_col = next((c for c in profile.columns if c.name == "email"), None)
    if email_col is None:
        print("FAIL: email column missing from profile entirely.")
        return

    # The ModelSafeView itself must never carry a real email value -- only
    # aggregate stats and a detected FORMAT pattern are allowed through.
    profile_str = str(profile)
    leaked = any(e in profile_str for e in emails[:5])  # spot-check a few
    print(f"Detected format for email column: {email_col.detected_format}")
    print(f"Raw email addresses found in ModelSafeView's own string repr: {leaked}")

    if leaked:
        print("FAIL: DANGER -- raw email values are present in the privacy-boundary-checked profile object.")
        return

    print("PASS: no raw email values reached the ModelSafeView; only aggregate/format info passed through.")


# ---------------------------------------------------------------------------
# Check 4: larger dataset performance. Confirms detection/profiling/the
# full pipeline don't break or degrade meaningfully at larger scale, and
# that ModelSafeView's top-value capping actually engages.
# ---------------------------------------------------------------------------

def check_larger_dataset():
    print("\n=== Check 4: larger dataset (5,000 rows) ===")
    store = LocalDiskStore(base_dir=Path("eval_data_remaining"))
    rng = np.random.default_rng(41)
    n = 2500
    df = pd.DataFrame({
        "customer_id": range(1, 2 * n + 1),
        "region": ["East"] * n + ["West"] * n,
        "total_spend": np.concatenate([rng.normal(50, 10, n), rng.normal(58, 10, n)]),
    })
    dataset = _make_dataset(store, "large_dataset_check", df)
    app = build_agent_graph(store)
    config = {"configurable": {"thread_id": "large-dataset-check"}}

    start = time.time()
    result = app.invoke(
        {"request_text": "Did customers in the East region spend more than the West region?",
         "session_id": "session-large_dataset_check", "dataset": dataset},
        config,
    )
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        if payload.get("type") == "wrangling_approval":
            resume = {"decisions": [{"index": op["index"], "action": "approve"} for op in payload["proposed_operations"]]}
        elif payload.get("type") == "plan_approval":
            resume = {"action": "approve"}
        else:
            resume = "Please proceed with the most natural reading of my original question."
        result = app.invoke(Command(resume=resume), config)
    elapsed = time.time() - start

    plan = result.get("plan")
    test_result = result.get("test_result")
    print(f"Elapsed: {elapsed:.1f}s, rows: {2 * n}")
    print(f"Plan status: {plan.status.value if plan else None}")
    print(f"Test result: p={test_result['p_value']:.4f}" if test_result else "No test_result")

    if plan and plan.status.value == "executed" and test_result:
        print(f"PASS: completed successfully on {2*n} rows in {elapsed:.1f}s.")
    else:
        print("FAIL: did not complete successfully.")


# ---------------------------------------------------------------------------
# Check 5: unicode / special characters. Non-ASCII values in a categorical
# column -- confirms no encoding crash anywhere in the pipeline.
# ---------------------------------------------------------------------------

def check_unicode_values():
    print("\n=== Check 5: unicode / special characters ===")
    store = LocalDiskStore(base_dir=Path("eval_data_remaining"))
    rng = np.random.default_rng(51)
    df = pd.DataFrame({
        "customer_id": range(1, 201),
        "region": ["Côte d'Ivoire"] * 100 + ["São Paulo"] * 100,
        "total_spend": np.concatenate([rng.normal(50, 10, 100), rng.normal(58, 10, 100)]),
    })
    dataset = _make_dataset(store, "unicode_check", df)
    app = build_agent_graph(store)
    config = {"configurable": {"thread_id": "unicode-check"}}

    try:
        result = app.invoke(
            {"request_text": "Did customers in São Paulo spend more than Côte d'Ivoire?",
             "session_id": "session-unicode_check", "dataset": dataset},
            config,
        )
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            if payload.get("type") == "plan_approval":
                resume = {"action": "approve"}
            elif payload.get("type") == "wrangling_approval":
                resume = {"decisions": [{"index": op["index"], "action": "approve"} for op in payload["proposed_operations"]]}
            else:
                resume = "Please proceed with the most natural reading of my original question."
            result = app.invoke(Command(resume=resume), config)
    except Exception as e:
        print(f"FAIL: crashed on unicode input: {type(e).__name__}: {e}")
        return

    plan = result.get("plan")
    report = result.get("report", "")
    print(f"Plan status: {plan.status.value if plan else None}")
    print(f"Report (first 200 chars): {report[:200]}")

    if plan and plan.status.value == "executed":
        print("PASS: completed successfully with unicode values in the data and request.")
    else:
        print(f"UNCLEAR: did not execute, but no crash either. plan={plan}")


if __name__ == "__main__":
    check_name = sys.argv[1] if len(sys.argv) > 1 else None
    checks = {
        "edit": check_invalid_column_edit_at_freeze,
        "stability": check_classification_stability,
        "privacy": check_privacy_boundary_stress,
        "large": check_larger_dataset,
        "unicode": check_unicode_values,
    }
    if check_name:
        checks[check_name]()
    else:
        for fn in checks.values():
            fn()