"""
Report fidelity check (M0.2a, D15 and D18). Calls the real report model on
results built by the real analysis code, in four conditions (which group is
higher, and how the question is phrased), several repeats each. It flags any
sentence where a group's mean sits closer to the OTHER group's name.

HEURISTIC: it reads free text, so it can miss a swap or flag a false one.
Read the flagged sentences yourself. Run from the agent folder:
    python -m eval.check_report_fidelity
"""
import re
import time
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import pandas as pd

from core.capabilities import TWO_GROUP
from core.dataset import LocalDiskStore
from core.ingestion import load_dataset
from core.plan import AnalysisPlan, QuestionType, CausalStatus, OutcomeType
from core.report import generate_report

load_dotenv()

REPEATS = 3
PAUSE_SECONDS = 6
PHRASINGS = ["Did West spend more than East?", "Did East spend more than West?"]


def build(store, higher, phrasing, seed):
    rng = np.random.default_rng(seed)
    east_mean, west_mean = (50.0, 60.0) if higher == "west" else (60.0, 50.0)
    df = pd.DataFrame({
        "region": ["East"] * 120 + ["West"] * 120,
        "total_spend": np.concatenate([rng.normal(east_mean, 8, 120), rng.normal(west_mean, 8, 120)]),
    })
    path = Path(f"/tmp/fidelity_{higher}_{seed}.csv")
    df.to_csv(path, index=False)
    ds = load_dataset(path, session_id=f"s-fid-{higher}-{seed}", name=f"fid_{higher}_{seed}", store=store)
    plan = AnalysisPlan(
        request_text=phrasing, session_id=f"s-fid-{higher}-{seed}",
        question_type=QuestionType.TWO_GROUP_COMPARISON, causal_status=CausalStatus.OBSERVATIONAL,
        outcome_name="total_spend", outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="customer", data_sources=(ds.handle,), candidate_predictors=("region",),
    )
    plan = plan.record_method_decision(TWO_GROUP.select(ds, store, plan))
    return plan, TWO_GROUP.execute(ds, store, plan)


def _positions(sentence, name):
    return [m.start() for m in re.finditer(r"\b" + re.escape(name) + r"\b", sentence)]


def judge(report, result):
    """Returns (verdict, flagged sentences, number of correctly paired means)."""
    means = {result["group_a_label"]: float(result["group_a_mean"]), result["group_b_label"]: float(result["group_b_mean"])}
    flagged, paired = [], 0
    for sentence in re.split(r"(?<=[.!?])\s+", report):
        for owner, mean in means.items():
            other = next(n for n in means if n != owner)
            for text in {f"{mean:.2f}", f"{mean:.1f}"}:
                for m in re.finditer(r"(?<![\d.])" + re.escape(text) + r"(?!\d)", sentence):
                    own, oth = _positions(sentence, owner), _positions(sentence, other)
                    if not own and not oth:
                        continue
                    d_own = min((abs(p - m.start()) for p in own), default=10 ** 9)
                    d_oth = min((abs(p - m.start()) for p in oth), default=10 ** 9)
                    if d_oth < d_own:
                        flagged.append(sentence)
                    else:
                        paired += 1
    if flagged:
        return "SWAP?", flagged, paired
    return ("ok" if paired else "unclear"), flagged, paired


def main():
    store = LocalDiskStore(base_dir=Path("/tmp/fidelity_store"))
    tally = {"ok": 0, "SWAP?": 0, "unclear": 0, "error": 0}
    seed = 100
    total = 2 * len(PHRASINGS) * REPEATS
    done = 0
    for higher in ("west", "east"):
        for phrasing in PHRASINGS:
            for _ in range(REPEATS):
                seed += 1
                done += 1
                try:
                    plan, result = build(store, higher, phrasing, seed)
                    try:
                        report = generate_report(plan, result)
                    except Exception as e:
                        if "429" in str(e) or "rate" in str(e).lower():
                            time.sleep(25)
                            report = generate_report(plan, result)
                        else:
                            raise
                    verdict, flagged, paired = judge(report, result)
                    tally[verdict] += 1
                    print(f"[{done}/{total}] {higher}-higher | \"{phrasing}\" -> {verdict} ({paired} means paired)")
                    for s in flagged:
                        print("     FLAGGED:", s)
                except Exception as e:
                    tally["error"] += 1
                    print(f"[{done}/{total}] FAILED: {type(e).__name__}: {str(e)[:150]}")
                if done < total:
                    time.sleep(PAUSE_SECONDS)
    print("\nSummary:", tally, "| target: SWAP? = 0")


if __name__ == "__main__":
    main()