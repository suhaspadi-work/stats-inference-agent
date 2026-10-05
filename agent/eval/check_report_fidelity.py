"""
Report fidelity check. Calls the real report model on results built by the real
analysis code, across varied group names, outcome names and scales, and flags
sentences where the order in which group names appear disagrees with the order
in which their means appear.

HEURISTIC: it reads free text, so it can miss a swap or flag a false one. Read
any flagged sentence yourself. Run from the agent folder:
    python -m eval.check_report_fidelity
"""
import re
import time
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import pandas as pd

from core.capabilities import TWO_GROUP, MULTI_GROUP
from core.dataset import LocalDiskStore
from core.ingestion import load_dataset
from core.plan import AnalysisPlan, QuestionType, CausalStatus, OutcomeType
from core.report import generate_report, generate_multi_group_report

load_dotenv()

REPEATS = 2
PAUSE_SECONDS = 5

TWO_GROUP_SCENARIOS = [
    {"labels": ("East", "West"), "outcome": "total_spend", "base": 50.0, "gap": 10.0},
    {"labels": ("Control", "Variant"), "outcome": "conversion_rate", "base": 0.050, "gap": 0.012},
    {"labels": ("Plan A", "Plan B"), "outcome": "minutes_used", "base": 300.0, "gap": 40.0},
]
MULTI_SCENARIOS = [
    {"labels": ("Basic", "Standard", "Premium"), "outcome": "total_spend", "base": 50.0, "step": 8.0},
    {"labels": ("2022", "2023", "2024"), "outcome": "revenue", "base": 1000.0, "step": 150.0},
    {"labels": ("Zürich", "São Paulo", "Tokyo"), "outcome": "satisfaction", "base": 3.2, "step": 0.5},
]


def _first(sentence, pattern):
    m = re.search(pattern, sentence)
    return m.start() if m else None


def judge(report, means):
    """
    means: {group name: true mean}. For each sentence that mentions at least two
    groups together with their means, the order in which the names first appear
    must equal the order in which each group's own mean first appears.
    Returns (verdict, flagged sentences, number of groups checked).
    """
    flagged, paired = [], 0
    for sentence in re.split(r"(?<=[.!?])\s+", report):
        name_pos, value_pos = {}, {}
        for name, mean in means.items():
            p = _first(sentence, r"(?<!\w)" + re.escape(str(name)) + r"(?!\w)")
            texts = {f"{mean:.{k}f}" for k in (1, 2, 3, 4)}
            vals = [_first(sentence, r"(?<![\d.])" + re.escape(t) + r"0*(?!\d)") for t in texts]
            vals = [v for v in vals if v is not None]
            if p is not None and vals:
                name_pos[name], value_pos[name] = p, min(vals)
        if len(name_pos) < 2:
            continue
        if sorted(name_pos, key=name_pos.get) == sorted(value_pos, key=value_pos.get):
            paired += len(name_pos)
        else:
            flagged.append(sentence)
    if flagged:
        return "SWAP?", flagged, paired
    return ("ok" if paired else "unclear"), flagged, paired


def _dataset(store, df, tag):
    path = Path(f"/tmp/fid_{tag}.csv")
    df.to_csv(path, index=False)
    return load_dataset(path, session_id=f"s-fid-{tag}", name=f"fid_{tag}", store=store)


def _plan(ds, question_type, outcome, text):
    return AnalysisPlan(
        request_text=text, session_id="s-fid", question_type=question_type,
        causal_status=CausalStatus.OBSERVATIONAL, outcome_name=outcome, outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="row", data_sources=(ds.handle,), candidate_predictors=("group",),
    )


def two_group_case(store, sc, higher_index, repeat, seed):
    rng = np.random.default_rng(seed)
    a, b = sc["labels"]
    means = [sc["base"], sc["base"]]
    means[higher_index] += sc["gap"]
    sd = sc["gap"] * 0.8
    df = pd.DataFrame({
        "group": [a] * 120 + [b] * 120,
        "outcome": np.concatenate([rng.normal(means[0], sd, 120), rng.normal(means[1], sd, 120)]),
    })
    x, y = (a, b) if repeat % 2 == 0 else (b, a)
    question = f"Is {sc['outcome']} higher for {x} than for {y}?"
    ds = _dataset(store, df, f"two_{seed}")
    plan = _plan(ds, QuestionType.TWO_GROUP_COMPARISON, "outcome", question)
    plan = plan.record_method_decision(TWO_GROUP.select(ds, store, plan))
    result = TWO_GROUP.execute(ds, store, plan)
    plan = plan.__class__(**{**plan.__dict__, "outcome_name": sc["outcome"]})
    report_means = {result["group_a_label"]: result["group_a_mean"], result["group_b_label"]: result["group_b_mean"]}
    return plan, result, report_means


def multi_case(store, sc, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sc["labels"]))
    means = [sc["base"] + sc["step"] * int(i) for i in order]
    sd = sc["step"] * 0.6
    df = pd.DataFrame({
        "group": np.repeat(sc["labels"], 100),
        "outcome": np.concatenate([rng.normal(m, sd, 100) for m in means]),
    })
    ds = _dataset(store, df, f"multi_{seed}")
    plan = _plan(ds, QuestionType.MULTI_GROUP_COMPARISON, "outcome", f"Which groups differ in {sc['outcome']}?")
    plan = plan.record_method_decision(MULTI_GROUP.select(ds, store, plan))
    result = MULTI_GROUP.execute(ds, store, plan)
    plan = plan.__class__(**{**plan.__dict__, "outcome_name": sc["outcome"]})
    return plan, result, {k: float(v) for k, v in result["group_means"].items()}


def _call(fn, plan, result):
    try:
        return fn(plan, result)
    except Exception as e:
        if "429" in str(e) or "rate" in str(e).lower():
            time.sleep(25)
            return fn(plan, result)
        raise


def main():
    store = LocalDiskStore(base_dir=Path("/tmp/fidelity_store"))
    cases = []
    seed = 500
    for sc in TWO_GROUP_SCENARIOS:
        for higher in (0, 1):
            for r in range(REPEATS):
                seed += 1
                cases.append((f"two-group {sc['labels']} {sc['labels'][higher]} higher",
                              lambda s=sc, h=higher, rr=r, sd=seed: two_group_case(store, s, h, rr, sd), generate_report))
    for sc in MULTI_SCENARIOS:
        for r in range(REPEATS):
            seed += 1
            cases.append((f"multi-group {sc['labels']}", lambda s=sc, sd=seed: multi_case(store, s, sd), generate_multi_group_report))

    tally = {"ok": 0, "SWAP?": 0, "unclear": 0, "error": 0}
    for i, (label, build, report_fn) in enumerate(cases, 1):
        try:
            plan, result, means = build()
            report = _call(report_fn, plan, result)
            verdict, flagged, paired = judge(report, means)
            tally[verdict] += 1
            print(f"[{i}/{len(cases)}] {label} -> {verdict} ({paired} groups checked)")
            for s in flagged:
                print("     FLAGGED:", s)
        except Exception as e:
            tally["error"] += 1
            print(f"[{i}/{len(cases)}] {label} FAILED: {type(e).__name__}: {str(e)[:150]}")
        if i < len(cases):
            time.sleep(PAUSE_SECONDS)
    print("\nSummary:", tally, "| target: SWAP? = 0")


if __name__ == "__main__":
    main()