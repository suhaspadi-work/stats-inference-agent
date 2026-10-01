from __future__ import annotations
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from langgraph.types import Command

from core.dataset import LocalDiskStore
from core.ingestion import load_dataset
from core.graph import build_agent_graph
from core.synthetic import SyntheticTwoGroupSpec

RESULTS_FILE = "eval_results.jsonl"
PAUSE_BETWEEN_SCENARIOS = 20  # seconds, to stay under Groq's 8,000 TPM cap
MAX_RETRIES = 4


class ServiceUnavailable(Exception):
    """The model service is overloaded, rate-limited, or timing out. Not an agent failure."""


def is_transient(msg: str) -> bool:
    markers = (
        "429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED", "UNAVAILABLE",
        "RateLimitError", "rate_limit_exceeded", "timed out", "Timeout", "timeout",
    )
    return any(m in msg for m in markers)


def with_retry(fn):
    """
    Retry transient errors with growing waits, honoring the server's own
    suggested wait time when it provides one (Groq's 429 messages include
    'Please try again in Ns'). Raises ServiceUnavailable if retries exhaust --
    this is a signal to stop the whole run cleanly, not a scenario failure.
    """
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if not is_transient(msg):
                raise
            if attempt == MAX_RETRIES - 1:
                raise ServiceUnavailable(f"Still failing after retries: {msg[:300]}")
            match = re.search(r"try again in ([\d.]+)s", msg)
            wait = float(match.group(1)) + 3 if match else 20 * (2 ** attempt)
            print(f"   transient error: {msg[:200]}")
            print(f"   retrying in {wait:.1f}s (attempt {attempt + 2}/{MAX_RETRIES})...")
            time.sleep(wait)


@dataclass(frozen=True)
class EvalScenario:
    """
    One evaluation case. A scenario builds its own dataset (so each one is
    reproducible independent of the others), and declares what a CORRECT
    agent run should look like -- the check function reads the real,
    structured final state and returns pass/fail plus a reason, never
    trusting the agent's own narrative about what it did.
    """
    name: str
    request_text: str
    build_dataset: Any  # Callable[[], tuple[pd.DataFrame, Optional[SyntheticTwoGroupSpec]]]
    check: Any  # Callable[[dict, Optional[SyntheticTwoGroupSpec]], tuple[bool, str]]


def run_scenario(scenario: EvalScenario, store, session_id: str) -> dict:
    """
    Runs one scenario through the real agent graph, auto-resolving every
    interrupt with a sensible default (approve plan freezes and dedupe
    proposals; this mirrors a cooperative, attentive human reviewer, not
    an adversarial one -- adversarial/rejection behavior is already
    covered by Phase 7's dedicated tests, not re-tested here).
    """
    df, spec = scenario.build_dataset()
    tmp_path = Path(f"/tmp/eval_{scenario.name}.csv")
    df.to_csv(tmp_path, index=False)
    dataset = load_dataset(tmp_path, session_id=session_id, name=scenario.name, store=store)

    app = build_agent_graph(store)
    config = {"configurable": {"thread_id": f"eval-{scenario.name}"}}

    result = with_retry(lambda: app.invoke(
        {"request_text": scenario.request_text, "session_id": session_id, "dataset": dataset}, config,
    ))

    # Auto-resolve interrupts: clarification gets a generic cooperative
    # answer, wrangling proposals get approved, plan freeze gets approved.
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        itype = payload.get("type")

        if itype == "clarification_needed":
            resume_value = "Please proceed with the most natural reading of my original question."
        elif itype == "wrangling_approval":
            resume_value = {"decisions": [
                {"index": op["index"], "action": "approve"} for op in payload["proposed_operations"]
            ]}
        elif itype == "plan_approval":
            resume_value = {"action": "approve"}
        else:
            raise ValueError(f"Unknown interrupt type in evaluation run: {itype}")

        result = with_retry(lambda: app.invoke(Command(resume=resume_value), config))

    passed, reason = scenario.check(result, spec)
    return {
        "scenario": scenario.name,
        "request": scenario.request_text,
        "passed": passed,
        "reason": reason,
        "plan_status": result.get("plan").status.value if result.get("plan") else None,
        "test_result": result.get("test_result"),
        "report": result.get("report"),
    }


def load_results() -> dict:
    done = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    done[row["scenario"]] = row
    return done


def run_eval_suite(scenarios: list[EvalScenario], store):
    """
    Resumable: already-completed scenarios (by name) are skipped on re-run,
    and each result is saved to disk the moment it finishes -- so a
    transient outage partway through never loses prior progress, and the
    same command re-run later simply picks up where it left off.
    """
    done = load_results()
    print(f"{len(done)} of {len(scenarios)} scenarios already completed; resuming.")

    for scenario in scenarios:
        if scenario.name in done:
            continue
        print(f"\n[{scenario.name}] {scenario.request_text}")
        try:
            row = run_scenario(scenario, store, session_id=f"eval-session-{scenario.name}")
        except ServiceUnavailable as e:
            print(f"\nSTOPPING: model service unavailable ({e})")
            print("Nothing was recorded for this scenario. Re-run later to resume from here.")
            break

        print(f"   {'PASS' if row['passed'] else 'FAIL'}: {row['reason']}")
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(row) + "\n")
        done[scenario.name] = row
        time.sleep(PAUSE_BETWEEN_SCENARIOS)

    total = len(done)
    passed = sum(1 for r in done.values() if r["passed"])
    print(f"\n{passed}/{total} scenarios passed ({len(scenarios) - total} not yet run).")
    return done