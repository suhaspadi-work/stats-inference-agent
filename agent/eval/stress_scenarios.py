"""
Structural stress tests: inputs designed to be genuinely awkward, checking
HOW the agent fails (clean, informative error vs. ugly crash) rather than
whether it reaches a correct statistical answer. These are expected to
surface real gaps -- that's the point.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

from eval.runner import EvalScenario


def _rename(df):
    return df.rename(columns={"outcome": "total_spend", "group": "region"})


# ---------------------------------------------------------------------------
# More than two groups -- select_two_group_method/execute_two_group_test
# require exactly 2. The question: does the agent fail with a clear,
# informative error, or crash confusingly somewhere deep in the stack?
# ---------------------------------------------------------------------------

def build_three_groups():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "customer_id": range(1, 301),
        "region": ["East"] * 100 + ["West"] * 100 + ["North"] * 100,
        "total_spend": np.concatenate([
            rng.normal(50, 10, 100), rng.normal(55, 10, 100), rng.normal(52, 10, 100),
        ]),
    })
    return df, None


def check_three_groups(result, spec):
    # We EXPECT this to fail somewhere -- the real question is whether the
    # failure is clean (a ValueError with a clear message reaching the
    # caller) or an ugly crash (an unrelated exception type, or a result
    # that looks successful but is actually wrong).
    if "error" in result:
        return True, f"Failed cleanly: {result['error'][:150]}"
    if result.get("plan") and result["plan"].status.value == "executed":
        return False, "DANGER: executed successfully on 3 groups -- should have been rejected or asked to clarify"
    return False, f"Unclear outcome: {result.get('plan')}"


# ---------------------------------------------------------------------------
# Requested-but-nonexistent outcome column -- tests validate_draft_plan_
# fields's hallucination guard under REAL model pressure, not a hand-
# crafted hallucination like our unit test used.
# ---------------------------------------------------------------------------

def build_oddly_named_columns():
    df, spec = pd.DataFrame({
        "cust_id": range(1, 401),
        "grp": ["A"] * 200 + ["B"] * 200,
        "val": np.concatenate([
            np.random.default_rng(2).normal(50, 10, 200),
            np.random.default_rng(2).normal(58, 10, 200),
        ]),
    }), None
    return df, spec


def check_oddly_named_columns(result, spec):
    # This request deliberately asks about "spend" and "region" when the
    # actual columns are "val" and "grp" -- the model should either map
    # sensibly to the real columns, or fail cleanly, never hallucinate.
    if "error" in result:
        return True, f"Failed cleanly (no hallucinated column reached execution): {result['error'][:150]}"
    plan = result.get("plan")
    if plan and plan.status.value == "executed":
        return True, f"Mapped request to real columns: outcome={plan.outcome_name}, predictors={plan.candidate_predictors}"
    return False, f"Unclear outcome: {plan}"


# ---------------------------------------------------------------------------
# Non-numeric column requested as the outcome.
# ---------------------------------------------------------------------------

def build_non_numeric_outcome():
    df = pd.DataFrame({
        "customer_id": range(1, 201),
        "region": ["East"] * 100 + ["West"] * 100,
        "total_spend": ["low", "medium", "high"] * 66 + ["low", "medium"],  # text, not numeric
    })
    return df, None


def check_non_numeric_outcome(result, spec):
    if "error" in result:
        return True, f"Failed cleanly: {result['error'][:150]}"
    plan = result.get("plan")
    if plan and plan.status.value == "executed":
        return False, "DANGER: ran a numeric test on a text column without complaint"
    return False, f"Unclear outcome: {plan}"


# ---------------------------------------------------------------------------
# Extreme missingness: 100% of one column is missing.
# ---------------------------------------------------------------------------

def build_fully_missing_column():
    df = pd.DataFrame({
        "customer_id": range(1, 201),
        "region": [None] * 200,  # entirely missing
        "total_spend": np.random.default_rng(3).normal(50, 10, 200),
    })
    return df, None


def check_fully_missing_column(result, spec):
    if "error" in result:
        return True, f"Failed cleanly: {result['error'][:150]}"
    plan = result.get("plan")
    if plan and plan.status.value == "executed":
        return False, "DANGER: executed with a 100%-missing grouping column"
    return False, f"Unclear outcome (may have correctly stopped at an interrupt or refusal): {plan}"


STRUCTURAL_STRESS_SCENARIOS = [
    EvalScenario(
        name="stress_three_groups",
        request_text="Did customers in the East, West, or North region spend differently?",
        build_dataset=build_three_groups,
        check=check_three_groups,
    ),
    EvalScenario(
        name="stress_oddly_named_columns",
        request_text="Did customers in region B spend more than region A?",
        build_dataset=build_oddly_named_columns,
        check=check_oddly_named_columns,
    ),
    EvalScenario(
        name="stress_non_numeric_outcome",
        request_text="Did customers in the East region spend more than the West region?",
        build_dataset=build_non_numeric_outcome,
        check=check_non_numeric_outcome,
    ),
    EvalScenario(
        name="stress_fully_missing_column",
        request_text="Did customers in the East region spend more than the West region?",
        build_dataset=build_fully_missing_column,
        check=check_fully_missing_column,
    ),
]