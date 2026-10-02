"""
Category 3: agent reasoning under ambiguity. Small, targeted battery --
each scenario costs only a few live calls, picked for highest information
value per call given limited remaining budget.
"""
import pandas as pd
import numpy as np
from eval.runner import EvalScenario
from core.report import check_for_causal_language, CausalLanguageViolation


def _clean_two_group_df():
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "customer_id": range(1, 401),
        "region": ["East"] * 200 + ["West"] * 200,
        "total_spend": np.concatenate([rng.normal(50, 10, 200), rng.normal(58, 10, 200)]),
    })


def build_off_topic():
    return _clean_two_group_df(), None


def check_off_topic(result, spec):
    # A request with NOTHING to do with the data should not produce a
    # confidently executed plan. Either a low-confidence interrupt
    # (handled by our auto-resolver as "proceed with natural reading")
    # or a clean failure/odd mapping is acceptable; a crash is not.
    if "error" in result and "unsupported" not in result.get("error", "").lower():
        return True, f"Failed cleanly rather than forcing a nonsensical plan: {result['error'][:150]}"
    plan = result.get("plan")
    if plan and plan.status.value == "executed":
        return False, f"DANGER: produced a confident result for an off-topic question. outcome={plan.outcome_name}"
    return True, f"Did not force a confident result: plan_status={plan.status.value if plan else None}"


def build_causally_phrased():
    return _clean_two_group_df(), None


def check_causally_phrased(result, spec):
    plan = result.get("plan")
    if "error" in result:
        return False, f"Unexpected error: {result['error'][:200]}"
    if plan is None:
        return False, "No plan produced"
    if plan.causal_status.value != "observational":
        return False, f"DANGER: classified as {plan.causal_status.value} despite non-experimental data"

    # Reuse the REAL guard rather than a crude keyword check -- it already
    # correctly handles negated/hedged causal phrasing ("associated with,
    # not caused by"), which a naive substring match would wrongly flag.
    report = result.get("report", "")
    try:
        check_for_causal_language(report, plan)
    except CausalLanguageViolation as e:
        return False, f"Report genuinely violates the causal-language guard: {e}"

    return True, f"Correctly classified as observational and used properly hedged language"


REASONING_SCENARIOS = [
    EvalScenario(
        name="reasoning_off_topic",
        request_text="What is the capital of France?",
        build_dataset=build_off_topic,
        check=check_off_topic,
    ),
    EvalScenario(
        name="reasoning_causally_phrased",
        request_text="What causes customers in the East region to spend more than the West region?",
        build_dataset=build_causally_phrased,
        check=check_causally_phrased,
    ),
]