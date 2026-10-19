"""
Live end-to-end check: one request per capability through the real graph and
real model, auto-approving every interrupt. Run from the agent folder with:
    python -m eval.check_live_capabilities
Set LIVE_ONLY to a comma-separated list of run labels (for example LIVE_ONLY=exploratory) to run only those;
the final decision-point lines then cover the executed runs only.

The exit code is 0 when every run passed, 3 when the only failures are an exhausted Groq quota (not a code
failure; retry after the reset), and 1 otherwise (G54). A summary is printed at the end.
"""
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from langgraph.types import Command

from core.dataset import LocalDiskStore
from core.ingestion import load_bundle, load_dataset
from core.lineage import check_lineage, edges_from_operations, lineage_graph
from core.graph import build_agent_graph
from core.event_log import events_path, read_decision_log
from core.decision_points import registered_points
from core.explain import explain_method

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
            detail = f" params={ev.get('params')} candidate={ev.get('candidate_id')}"  # M2.3a
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
        return None
    last_op = applied[-1]["evidence"].get("operation_id")
    dataset = state.get("dataset")
    made_by = getattr(dataset, "created_from_operation", None)
    if last_op == made_by and last_op is not None:
        print("   lineage: PASS")
        return None
    problem = f"lineage: FAIL (last applied operation {last_op}, dataset created_from_operation {made_by})"
    print("   " + problem)
    return problem


def print_lineage_graph(state, log):
    """M2.1: with more than one table, the lineage graph of the applied operations must lead from the analysis
    table back to uploaded tables only. Returns the problem line, or None (also when there is one table)."""
    tables = state.get("tables") or {}
    if len(tables) <= 1:
        return None
    applied = [e["evidence"] for e in log if e["decision_point"] == "stage1.clean.execution" and e["chosen"] == "applied"]
    graph = lineage_graph(edges_from_operations(applied))
    # M2.3b: a table a join made is derived, never an uploaded root
    derived = {e["params"].get("result_name") for e in applied if e.get("op_type") == "join"}
    uploaded = {f"{name}@v1" for name in tables if name not in derived}
    problems = check_lineage(state["dataset"].handle, graph, uploaded)
    if not problems:
        print("   lineage graph: PASS")
        return None
    problem = f"lineage graph: FAIL ({'; '.join(problems)})"
    print("   " + problem)
    return problem


def print_quality(log, label):
    """M2.2: the before and after data-quality snapshots and the computed links between tables."""
    def latest(point, chosen=None):
        return next((e for e in reversed(log) if e["decision_point"] == point
                     and (chosen is None or e["chosen"] == chosen)), None)

    def numbers(ev):
        return (f"rows={ev['rows']} duplicates={ev['duplicate_rows']} missing={ev['columns_with_missing']} "
                f"outliers={ev['columns_with_outliers']}")
    before = latest("stage1.quality.snapshot", "before")
    if before is None:
        return
    print(f"   quality [{label}]: before {numbers(before['evidence'])}")
    after = latest("stage1.quality.snapshot", "after")
    if after is not None:
        print(f"   quality [{label}]: after {numbers(after['evidence'])}")
    links = latest("stage1.links.scan")
    for c in ((links or {}).get("evidence") or {}).get("computed") or []:
        print(f"   links [{label}]: {c['table_a']}.{c['column_a']} with {c['table_b']}.{c['column_b']} "
              f"{c['cardinality']} matched={c['matched_distinct']} orphans={c['orphan_rows_a']}/{c['orphan_rows_b']} "
              f"inner_join_rows={c['inner_join_rows']}")


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


def print_stage1_report(state, label):
    """M1.4b: the Stage 1 report of the run, its sections and record count; for the messy run, also its text."""
    entries = [e for e in state.get("stage_reports") or [] if e.get("stage") == "stage1"]
    if not entries:
        print(f"   stage 1 report [{label}]: none")
    for e in entries:
        headings = ", ".join(s["heading"] for s in e["sections"])
        print(f"   stage 1 report [{label}]: {headings} ({len(e['source_records'])} records)")
        if label in ("messy", "bundle", "normalize", "join"):  # M2.1: the bundle run's report also lists its tables
            for line in e["text"].split("\n"):
                print(f"      | {line}")


def print_analysis_reports(state, label):
    """M1.4b step 2b: each analysis round's "how this was chosen" entry (collapsed sections marked *) and the
    summary; for the two_group run, also their full text."""
    entries = state.get("stage_reports") or []

    def headings(e):
        return ", ".join(s["heading"] + ("*" if s.get("collapsed") else "") for s in e["sections"])

    def text(e):
        for line in e["text"].split("\n"):
            print(f"      | {line}")

    for e in (e for e in entries if e.get("stage") == "analysis"):
        print(f"   analysis report [{label}]: round {e['round']}, {headings(e)} ({len(e['source_records'])} records)")
        if label == "two_group":
            text(e)
    for e in (e for e in entries if e.get("stage") == "summary"):
        print(f"   summary [{label}]: {headings(e)}")
        if label == "two_group":
            text(e)


def print_explanation(state, label, chat=None):
    """
    M1.4b step 3: one on-request explanation of the chosen method of round 1, for the two_group run only (the real
    model unless chat is given). Returns None when it passed or did not run, else (error_kind, problem line) with
    error_kind from classify_error.
    """
    if label != "two_group":
        return None
    try:
        explained = explain_method(state["plan"].primary_method, state, 1, chat=chat)
    except Exception as e:
        line = f"explanation [{label}]: FAILED: {type(e).__name__}: {str(e)[:500]}"
        print(f"   {line}")
        return classify_error(e), line
    print(f"   explanation [{label}]: source={explained['source']}")
    for text in explained["answer"].split("\n"):
        print(f"      | {text}")
    return None


def print_report_narrative(state_log, label):
    """M1.4a: whose text became each analysis report, and whether each model attempt passed the guard."""
    records = [e for e in state_log if e["decision_point"] == "report.narrative"]
    if not records:
        print(f"   report narrative [{label}]: none")
    for e in records:
        passed = [a["passed"] for a in (e.get("evidence") or {}).get("attempts", [])]
        print(f"   report narrative [{label}]: {e['chosen']} (attempts passed: {passed})")


def print_rounds(state, label):
    """M1.3b: the finished rounds of the run, one line."""
    rounds = state.get("rounds") or []
    if not rounds:
        print(f"   rounds [{label}]: none")
        return
    print(f"   rounds [{label}]: " + "; ".join(
        f"{r['round']} ({r['capability']}, exploratory={r['exploratory']})" for r in rounds))


def print_exploratory(state_log, label):
    """M1.3b: every boundary-gate answer in order, the round start, and the refusals (step 2b)."""
    gates = [e for e in state_log if e["decision_point"] == "planner.stage_gate"]
    print(f"   stage gate [{label}]: " + (f"chosen={', '.join(e['chosen'] for e in gates)}" if gates
                                          else "no planner.stage_gate record"))
    record = next((e for e in state_log if e["decision_point"] == "planner.round_start"), None)
    print(f"   round start [{label}]: " + (f"chosen={record['chosen']}" if record else "no planner.round_start record"))
    refused = [e for e in gates if e["chosen"] == "refused"]
    rules = ", ".join(v["rule"] for v in (refused[0]["evidence"].get("refusals") or [])) if refused else "none"
    print(f"   refusals [{label}]: {len(refused)} ({rules})")


def print_edit(state_log, label):
    """G54: the answers of the "edit" run at the stage-plan gate and at plan approval, and the re-selection."""
    stage = [e["chosen"] for e in state_log if e["decision_point"] == "planner.stage_plan_approval"]
    plan = [e["chosen"] for e in state_log if e["decision_point"] == "planner.plan_approval"]
    selections = [e for e in state_log if e["decision_point"] == "stage3.regression_method"]
    supersedes = len(selections) >= 2 and selections[1].get("supersedes") == selections[0]["id"]
    print(f"   edit [{label}]: stage plan answers={', '.join(stage)}; plan approval answers={', '.join(plan)}; "
          f"selections={len(selections)} (second supersedes first: {supersedes})")


def report_event_log(app, cfg, label, events, all_points, failed, outcome=None):
    """
    Compare the run's event log file with the state's decision_log, check strict JSON, print the
    decision points and (for the messy run) the cleaning lines and lineage. Used after successful and
    failed runs alike; a failed run's points still count toward the final union. Returns the problems
    found (every FAIL line printed), for the summary (G54). An explanation blocked by the quota marks the outcome
    (when given) as quota-blocked instead of adding a problem.
    """
    problems = []
    thread_id = cfg["configurable"]["thread_id"]
    try:
        state = app.get_state(cfg).values or {}
    except Exception as e:
        print(f"   state not available: {type(e).__name__}: {e}")
        state = {}
    state_log = state.get("decision_log") or []
    path = events_path(events, thread_id)
    file_log = read_decision_log(path)
    if failed and not state_log and not file_log:
        # The failure itself is already reported; there is nothing to compare.
        print(f"   event log [{label}]: not checked (the run failed before any node ran)")
        print("   strict JSON: not checked")
    else:
        diff = first_difference(state_log, file_log)
        print(f"   event log [{label}]: {len(state_log)} records in state, {len(file_log)} in file: "
              + ("PASS" if diff is None else f"FAIL ({diff})"))
        if diff is not None:
            problems.append(f"event log: FAIL ({diff})")
        if path.exists():
            problem = strict_json_problem(path)
            print("   strict JSON: " + ("PASS" if problem is None else f"FAIL ({problem})"))
            if problem is not None:
                problems.append(f"strict JSON: FAIL ({problem})")
        else:
            print("   strict JSON: FAIL (no event log file was written)")
            problems.append("strict JSON: FAIL (no event log file was written)")
    log = file_log if failed else state_log
    points = [e["decision_point"] for e in log]
    print("   decision points:", points)
    print_stage_plan(log, label)
    print_describe(log, label)
    print_quality(log, label)
    print_stage1_report(state, label)
    print_analysis_reports(state, label)
    if not failed:
        explanation = print_explanation(state, label)
        if explanation is not None and explanation[0] == "quota":
            print("   quota exhausted: not a code failure")
            if outcome is not None and outcome.get("error_kind") is None:
                outcome["error_kind"] = "quota"
                outcome["hint"] = quota_reset_hint(explanation[1])
        elif explanation is not None:
            problems.append(explanation[1])
    print_report_narrative(log, label)
    print_rounds(state, label)
    all_points.update(points)
    if label in ("messy", "bundle", "normalize", "join"):  # M2.1: the bundle run is the messy run with a second table
        print_cleaning_lines(log)
        lineage = print_lineage(state, state_log)
        if lineage is not None:
            problems.append(lineage)
        graph = print_lineage_graph(state, state_log)
        if graph is not None:
            problems.append(graph)
    if label in ("mismatch", "unsupported"):
        print_route(state, label)
    if label == "exploratory":
        print_exploratory(log, label)
    if label == "edit":
        print_edit(log, label)
    return problems


def filter_runs(runs, only):
    """M1.3b: the runs named in only (all of them when only is empty); an unknown label raises ValueError."""
    if not only:
        return list(runs)
    unknown = sorted(set(only) - {label for label, _, _ in runs})
    if unknown:
        raise ValueError(f"LIVE_ONLY names unknown run(s) {unknown}; known: {[label for label, _, _ in runs]}")
    return [r for r in runs if r[0] in only]


def classify_error(exc) -> str:
    """G54: "quota" for an exhausted Groq rate limit (not a code failure), "code" for anything else."""
    if "RateLimit" in type(exc).__name__:
        return "quota"
    message = str(exc)
    if "429" in message and "rate limit" in message.lower() and ("tokens per day" in message or "TPD" in message):
        return "quota"
    return "code"


def quota_reset_hint(message: str):
    """The wait Groq names after "Please try again in" (for example "1h2m3.5s"), or None."""
    match = re.search(r"Please try again in (.+?)(?:\.(?:\s|$)|['\"]|$)", message)
    return match.group(1).strip() if match else None


def summarize(outcomes) -> dict:
    """
    G54: outcomes is a list of {"label", "error_kind": None, "quota" or "code", "problems": [...], "hint"}.
    exit_code 0 when every run passed, 3 when every failure is an exhausted quota with no problem, else 1.
    """
    passed, code, quota, lines = 0, 0, 0, []
    for o in outcomes:
        if o.get("error_kind") is None and not o.get("problems"):
            passed += 1
        elif o.get("error_kind") == "quota" and not o.get("problems"):
            quota += 1
            hint = o.get("hint")
            lines.append(f"  {o['label']}: BLOCKED (quota): not a code failure; retry after the reset"
                         + (f" {hint}" if hint else ""))
        else:
            code += 1
            first = (o.get("problems") or [o.get("error") or "unknown error"])[0]
            lines.append(f"  {o['label']}: FAILED (code): {first}")
    exit_code = 0 if code == 0 and quota == 0 else (3 if code == 0 else 1)
    header = f"SUMMARY: {len(outcomes)} runs, {passed} passed, {code} code failures, {quota} quota-blocked"
    return {"exit_code": exit_code, "lines": [header, *lines]}


def _resume_for(label, p, counts, question):
    """The answer to one pause. Every run answers as before; the "edit" run tries refusals first (G54)."""
    kind = p.get("type")
    counts[kind] = counts.get(kind, 0) + 1
    if label == "edit" and kind == "stage_plan_approval" and counts[kind] == 1:
        return {"action": "approve", "run": {"clean": False}}  # clean is not applicable here: refused
    if label == "edit" and kind == "plan_approval":
        if counts[kind] == 1:
            return {"action": "edit", "candidate_predictors": ["customer_id"]}  # an identifier: refused
        if counts[kind] == 2:
            predictors = list(p.get("candidate_predictors") or [])
            return {"action": "edit", "candidate_predictors": predictors[1:] or predictors}
        return {"action": "approve"}
    if kind in ("plan_approval", "stage_plan_approval"):  # M1.3a: the stage-plan gate
        return {"action": "approve"}
    if kind == "stage_gate":  # M1.3b: run the exploratory round only in its own run
        if label != "exploratory":
            return {"action": "skip"}
        if p.get("refused") is None:  # step 2b: first try an identifier-like column, which is refused
            return {"action": "run", "predictors": ["region", "customer_id"]}
        return {"action": "run"}
    if kind == "wrangling_approval":
        if label == "bundle":  # M2.3b: the regions table is proposed as a join; this question does not need it
            return {"decisions": [{"index": o["index"], "action": "reject",
                                   "reason": "the question does not need the regions table"} if o["op_type"] == "join"
                                  else {"index": o["index"], "action": "approve"} for o in p["proposed_operations"]]}
        return {"decisions": [{"index": o["index"], "action": "approve"} for o in p["proposed_operations"]]}
    return question


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

    # M2.3a step 2: inconsistent text (a trailing space and mixed case) and numbers stored as text
    nrng = np.random.default_rng(23)
    k = 200
    n_region = np.array(["East", "east ", "West", "WEST"] * (k // 4))
    n_spend = 50 + np.where(np.isin(n_region, ["West", "WEST"]), 10.0, 0.0) + nrng.normal(0, 5, k)
    n_tenure = [f"{max(1.0, v):.1f}" for v in nrng.normal(24, 8, k)]
    for i in (7, 51, 99, 150):
        n_tenure[i] = "unknown"  # keeps the column text after a CSV round trip
    df_normalize = pd.DataFrame({"customer_id": range(1, k + 1), "region": n_region, "tenure": n_tenure,
                                 "total_spend": n_spend})

    # M2.1: a second, small table carried along with the messy data (unused until joins arrive in M2.3)
    df_regions = pd.DataFrame({"region": ["East", "West"], "manager": ["Ana", "Ben"]})

    # M2.3b: 300 orders (customers repeat) and a customers table with one row per customer; region is on customers
    jrng = np.random.default_rng(29)
    df_customers = pd.DataFrame({"customer_id": range(1, 151), "region": jrng.choice(["East", "West"], 150),
                                 "tier": jrng.choice(["basic", "plus"], 150)})
    order_customer = jrng.integers(1, 151, 300)
    west = df_customers.set_index("customer_id").loc[order_customer, "region"].to_numpy() == "West"
    df_orders = pd.DataFrame({"customer_id": order_customer,
                              "total_spend": 50 + np.where(west, 10.0, 0.0) + jrng.normal(0, 5, 300)})

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
        # M2.1: the messy run as a two-file bundle; the messy table is the analysis table, regions is carried
        ("bundle", {"live_bundle": df_messy, "regions": df_regions}, "Did West customers spend more than East customers?"),
        # M2.3a step 2: text normalization and a numeric conversion, then the same question and answers as messy
        ("normalize", df_normalize, "Did West customers spend more than East customers?"),
        # M2.3b: region is only on the customers table, so the orders need the join; every decision is approve
        ("join", {"orders": df_orders, "customers": df_customers},
         "Did customers in the West region spend more than customers in the East region?"),
        # M0.3: a two-group question on the three-region data (expected to be reconciled to multi-group)
        ("mismatch", df3, "Did West customers spend more than East customers?"),
        # M1.3b: the boundary gate is answered "run", so an exploratory regression follows the comparison
        ("exploratory", df2, "Did customers in the West region spend more than the East region?"),
        # G54: a refused stage-plan choice, a refused edit, then an edit that goes back through method selection
        ("edit", df3, "What factors drive total spend? Consider region, tenure and visits."),
        # M0.3: the request that crashed select_method before M0.3 (expected to stop before method selection)
        ("unsupported", df_messy, "Is there a relationship between region, tenure and spend?"),
    ]
    runs = filter_runs(runs, [s.strip() for s in os.environ.get("LIVE_ONLY", "").split(",") if s.strip()])
    outcomes = []
    for i, (label, df, question) in enumerate(runs):
        cfg = {"configurable": {"thread_id": f"live-{label}"}}
        outcome = {"label": label, "error_kind": None, "problems": [], "hint": None}
        outcomes.append(outcome)
        try:
            start = {"request_text": question, "session_id": f"s-live-{label}"}
            if isinstance(df, dict):  # M2.1: a bundle of several files; the first is the analysis table
                paths = []
                for name, frame in df.items():
                    paths.append(Path(f"/tmp/live_{label}_{name}.csv"))
                    frame.to_csv(paths[-1], index=False)
                tables = load_bundle(paths, session_id=f"s-live-{label}", store=store, names=list(df))
                start["datasets"] = list(tables.values())
            else:
                path = Path(f"/tmp/live_{label}.csv")
                df.to_csv(path, index=False)
                start["dataset"] = load_dataset(path, session_id=f"s-live-{label}", name=f"live_{label}", store=store)
            result = app.invoke(start, cfg)
            seen, counts = [], {}
            while "__interrupt__" in result:
                p = result["__interrupt__"][0].value
                seen.append(p.get("type"))
                result = app.invoke(Command(resume=_resume_for(label, p, counts, question)), cfg)
            if result.get("clarification_needed") and not result.get("report"):
                print(f"[{label}] ended before method selection: {result['clarification_needed']}")
                print(f"   interrupts={seen}")
                outcome["problems"] += report_event_log(app, cfg, label, events, all_points, failed=False)
                if i < len(runs) - 1:
                    time.sleep(20)  # stay under the per-minute token limit
                continue
            plan = result["plan"]
            tr = result.get("test_result") or {}
            kind = tr.get("result_type", "descriptive_result" if "descriptive_result" in result else None)
            print(f"[{label}] interrupts={seen} question_type={plan.question_type.value} "
                  f"status={plan.status.value} result_type={kind} method={tr.get('method')}")
            print("   report:", (result.get("report") or "")[:160].replace("\n", " "))

            outcome["problems"] += report_event_log(app, cfg, label, events, all_points, failed=False,
                                                    outcome=outcome)
            if label == "normalize":  # M2.3a step 2: what the cleaning did to region and tenure
                final = store.read(result["dataset"].storage_key)
                print(f"   normalize [{label}]: region levels after cleaning={final['region'].nunique()} "
                      f"tenure dtype after cleaning={final['tenure'].dtype}")
            if label == "join":  # M2.3b: the analysis table after cleaning is the joined table
                final = store.read(result["dataset"].storage_key)
                print(f"   join [{label}]: analysis table={result.get('analysis_table')} "
                      f"columns={', '.join(str(c) for c in final.columns)} rows={len(final)}")
        except Exception as e:
            message = str(e)
            print(f"[{label}] FAILED: {type(e).__name__}: {message[:500]}")
            outcome["error_kind"] = classify_error(e)
            if outcome["error_kind"] == "quota":
                print("   quota exhausted: not a code failure")
                outcome["hint"] = quota_reset_hint(message)
                if outcome["hint"]:
                    print(f"   retry after the reset: {outcome['hint']}")
            else:
                outcome["problems"].append(f"{type(e).__name__}: {message[:500]}")
            outcome["problems"] += report_event_log(app, cfg, label, events, all_points, failed=True)
        if i < len(runs) - 1:
            time.sleep(20)  # stay under the per-minute token limit

    print("\nDecision points across the runs:", sorted(all_points))
    print("Registered points not reached in these runs (information only):",
          sorted(registered_points() - all_points))
    print("Event log folder:", events)

    summary = summarize(outcomes)
    print()
    for line in summary["lines"]:
        print(line)
    sys.exit(summary["exit_code"])


if __name__ == "__main__":
    main()
