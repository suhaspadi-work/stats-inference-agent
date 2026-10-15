"""
Live end-to-end check: one request per capability through the real graph and
real model, auto-approving every interrupt. Run from the agent folder with:
    python -m eval.check_live_capabilities
"""
import json
import math
import tempfile
import time
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from langgraph.types import Command

from core.dataset import LocalDiskStore
from core.ingestion import load_dataset
from core.graph import build_agent_graph
from core.event_log import events_path, read_decision_log
from core.decision_points import registered_points

load_dotenv()


def first_difference(a, b, where="log"):
    """None if a and b are deeply equal (NaN equals NaN), otherwise a description of the first difference."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return None
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return f"{where}: keys differ ({sorted(set(a) ^ set(b))})"
        for k in a:
            diff = first_difference(a[k], b[k], f"{where}.{k}")
            if diff:
                return diff
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{where}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            diff = first_difference(x, y, f"{where}[{i}]")
            if diff:
                return diff
        return None
    return None if a == b else f"{where}: {a!r} vs {b!r}"


def strict_json_problem(path):
    """None if every line of the file is strict JSON (no NaN or Infinity tokens), otherwise the first problem."""
    def reject(constant):
        raise ValueError(f"non-strict JSON constant {constant}")
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                json.loads(line, parse_constant=reject)
            except ValueError as e:
                return f"line {n}: {e}"
    return None


def print_cleaning_lines(log):
    """One line per cleaning or clarification record, with its key evidence."""
    for e in log:
        point = e["decision_point"]
        if not (point.startswith("stage1.clean") or point == "intake.clarification"):
            continue
        ev = e.get("evidence") or {}
        detail = ""
        if point == "stage1.clean.fix":
            detail = f" params={ev.get('params')}"
        elif point == "stage1.clean.approval":
            detail = f" received_action={ev.get('received_action')!r}"
        elif point == "stage1.clean.execution":
            detail = (f" failure={ev.get('failure_reason')!r}" if e["chosen"] == "failed"
                      else f" rows {ev.get('rows_before')} -> {ev.get('rows_after')}")
        print(f"     {point}: chosen={e['chosen']} decided_by={e['decided_by']}{detail}")


def print_lineage(state, log):
    """PASS when the last applied cleaning operation produced the dataset the run ended with."""
    applied = [e for e in log if e["decision_point"] == "stage1.clean.execution" and e["chosen"] == "applied"]
    if not applied:
        print("   lineage: no applied operations")
        return
    last_op = applied[-1]["evidence"].get("operation_id")
    dataset = state.get("dataset")
    made_by = getattr(dataset, "created_from_operation", None)
    print("   lineage: " + ("PASS" if last_op == made_by and last_op is not None
                            else f"FAIL (last applied operation {last_op}, dataset created_from_operation {made_by})"))


def print_route(state, label):
    """For the routing runs: the planner.route record and the plan's final question type, labelled by status."""
    route = next((e for e in reversed(state.get("decision_log") or []) if e["decision_point"] == "planner.route"), None)
    if route is None:
        print(f"   route [{label}]: no planner.route record")
        return
    ev = route.get("evidence") or {}
    plan = state.get("plan")
    print(f"   route [{label}]: outcome={ev.get('status')} chosen={route['chosen']} "
          f"classifier_question_type={ev.get('classifier_question_type')} question_type={ev.get('question_type')} "
          f"plan question_type={plan.question_type.value if plan else None}")
    if ev.get("status") == "unsupported":
        print(f"   route [{label}]: message={ev.get('message')!r}")


def print_stage_plan(log, label):
    """M1.1: the planner.stage_plan record of the run, one line."""
    record = next((e for e in log if e["decision_point"] == "planner.stage_plan"), None)
    if record is None:
        print(f"   stage plan [{label}]: no planner.stage_plan record")
        return
    print(f"   stage plan [{label}]: chosen={record['chosen']} status={(record.get('evidence') or {}).get('status')}")


def print_describe(state_log, label):
    """M1.2: the stage1.describe.execution record of the run, one line."""
    record = next((e for e in state_log if e["decision_point"] == "stage1.describe.execution"), None)
    if record is None:
        print(f"   describe [{label}]: not run")
        return
    print(f"   describe [{label}]: {record['chosen']} ({(record.get('evidence') or {}).get('chart_count')} charts)")


def report_event_log(app, cfg, label, events, all_points, failed):
    """
    Compare the run's event log file with the state's decision_log, check strict JSON, print the
    decision points and (for the messy run) the cleaning lines and lineage. Used after successful and
    failed runs alike; a failed run's points still count toward the final union.
    """
    thread_id = cfg["configurable"]["thread_id"]
    try:
        state = app.get_state(cfg).values or {}
    except Exception as e:
        print(f"   state not available: {type(e).__name__}: {e}")
        state = {}
    state_log = state.get("decision_log") or []
    path = events_path(events, thread_id)
    file_log = read_decision_log(path)
    diff = first_difference(state_log, file_log)
    print(f"   event log [{label}]: {len(state_log)} records in state, {len(file_log)} in file: "
          + ("PASS" if diff is None else f"FAIL ({diff})"))
    if path.exists():
        problem = strict_json_problem(path)
        print("   strict JSON: " + ("PASS" if problem is None else f"FAIL ({problem})"))
    else:
        print("   strict JSON: FAIL (no event log file was written)")
    log = file_log if failed else state_log
    points = [e["decision_point"] for e in log]
    print("   decision points:", points)
    print_stage_plan(log, label)
    print_describe(log, label)
    all_points.update(points)
    if label == "messy":
        print_cleaning_lines(log)
        print_lineage(state, state_log)
    if label in ("mismatch", "unsupported"):
        print_route(state, label)


def main():
    rng = np.random.default_rng(3)
    n = 360
    region = rng.choice(["East", "West", "North"], n, p=[0.5, 0.3, 0.2])
    tenure = rng.normal(24, 8, n)
    visits = rng.normal(10, 3, n)
    shift = {"East": 0.0, "West": 8.0, "North": 15.0}
    spend = 20 + 2.0 * tenure + 1.5 * visits + np.array([shift[r] for r in region]) + rng.normal(0, 5, n)
    df3 = pd.DataFrame({"customer_id": range(1, n + 1), "region": region, "tenure": tenure, "visits": visits, "total_spend": spend})
    df2 = df3[df3.region != "North"].reset_index(drop=True)

    # Messy data for the cleaning path: about 12% of tenure missing (above the 5% threshold)
    # and 4 rows that exactly repeat earlier rows (296 + 4 = 300 rows).
    mrng = np.random.default_rng(11)
    m = 296
    m_region = mrng.choice(["East", "West"], m)
    m_tenure = mrng.normal(24, 8, m)
    m_spend = 20 + 2.0 * m_tenure + np.where(m_region == "West", 6.0, 0.0) + mrng.normal(0, 5, m)
    m_tenure[mrng.choice(m, size=36, replace=False)] = np.nan
    df_messy = pd.DataFrame({"customer_id": range(1, m + 1), "region": m_region,
                             "tenure": m_tenure, "total_spend": m_spend})
    df_messy = pd.concat([df_messy, df_messy.iloc[[5, 40, 120, 250]]], ignore_index=True)

    store = LocalDiskStore(base_dir=Path("/tmp/live_capabilities_store"))
    events = Path(tempfile.mkdtemp(prefix="live_events_"))
    app = build_agent_graph(store, event_log_dir=events)
    all_points = set()

    runs = [
        ("two_group", df2, "Did customers in the West region spend more than the East region?"),
        ("multi_group", df3, "Does total spend differ across East, West, and North regions?"),
        ("regression", df3, "What factors drive total spend? Consider region, tenure and visits."),
        ("descriptive", df3, "Can you describe this dataset for me?"),
        ("messy", df_messy, "Did West customers spend more than East customers?"),
        # M0.3: a two-group question on the three-region data (expected to be reconciled to multi-group)
        ("mismatch", df3, "Did West customers spend more than East customers?"),
        # M0.3: the request that crashed select_method before M0.3 (expected to stop before method selection)
        ("unsupported", df_messy, "Is there a relationship between region, tenure and spend?"),
    ]
    for i, (label, df, question) in enumerate(runs):
        cfg = {"configurable": {"thread_id": f"live-{label}"}}
        try:
            path = Path(f"/tmp/live_{label}.csv")
            df.to_csv(path, index=False)
            ds = load_dataset(path, session_id=f"s-live-{label}", name=f"live_{label}", store=store)
            result = app.invoke({"request_text": question, "session_id": f"s-live-{label}", "dataset": ds}, cfg)
            seen = []
            while "__interrupt__" in result:
                p = result["__interrupt__"][0].value
                seen.append(p.get("type"))
                if p.get("type") in ("plan_approval", "stage_plan_approval"):  # M1.3a: the stage-plan gate
                    resume = {"action": "approve"}
                elif p.get("type") == "wrangling_approval":
                    resume = {"decisions": [{"index": o["index"], "action": "approve"} for o in p["proposed_operations"]]}
                else:
                    resume = question
                result = app.invoke(Command(resume=resume), cfg)
            if result.get("clarification_needed") and not result.get("report"):
                print(f"[{label}] ended before method selection: {result['clarification_needed']}")
                print(f"   interrupts={seen}")
                report_event_log(app, cfg, label, events, all_points, failed=False)
                if i < len(runs) - 1:
                    time.sleep(20)  # stay under the per-minute token limit
                continue
            plan = result["plan"]
            tr = result.get("test_result") or {}
            kind = tr.get("result_type", "descriptive_result" if "descriptive_result" in result else None)
            print(f"[{label}] interrupts={seen} question_type={plan.question_type.value} "
                  f"status={plan.status.value} result_type={kind} method={tr.get('method')}")
            print("   report:", (result.get("report") or "")[:160].replace("\n", " "))

            report_event_log(app, cfg, label, events, all_points, failed=False)
        except Exception as e:
            print(f"[{label}] FAILED: {type(e).__name__}: {str(e)[:200]}")
            report_event_log(app, cfg, label, events, all_points, failed=True)
        if i < len(runs) - 1:
            time.sleep(20)  # stay under the per-minute token limit

    print("\nDecision points across the runs:", sorted(all_points))
    print("Registered points not reached in these runs (information only):",
          sorted(registered_points() - all_points))
    print("Event log folder:", events)


if __name__ == "__main__":
    main()
