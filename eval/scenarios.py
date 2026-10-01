from __future__ import annotations
import pandas as pd

from eval.runner import EvalScenario
from core.synthetic import generate_two_group_continuous, generate_two_group_continuous_with_quality_issues


def _rename_for_request(df: pd.DataFrame) -> pd.DataFrame:
    """All scenarios use the same column naming, matching what draft_plan_node needs to recognize."""
    return df.rename(columns={"outcome": "total_spend", "group": "region"})


# ---------------------------------------------------------------------------
# A/A scenarios: no real effect planted. Individually, each one is expected
# to occasionally call a result "significant" purely by chance (~5% of the
# time at alpha=0.05) -- that's correct behavior, not a bug. The real check
# happens in core.eval_summary once all of these have run: the AGGREGATE
# false-positive rate across all of them should land near 5%, not on any
# single trial. This mirrors Phase 3's test_aa_data_shows_no_significant_
# difference_on_average, but now exercising the full agent end-to-end
# instead of a raw scipy call.
# ---------------------------------------------------------------------------

def _make_aa_scenario(seed: int) -> EvalScenario:
    def build():
        df, spec = generate_two_group_continuous(n_per_group=200, mean_a=50.0, effect=0.0, std=10.0, seed=seed)
        return _rename_for_request(df), spec

    def check(result, spec):
        test_result = result.get("test_result")
        if test_result is None:
            return False, "No test_result in final state"
        # Both outcomes are "pass" here -- we're not judging this single
        # trial's correctness, only recording whether it rejected, for the
        # AGGREGATE check to use afterward.
        rejected = test_result["p_value"] < 0.05
        return True, f"p={test_result['p_value']:.4f}, rejected={rejected}"

    return EvalScenario(
        name=f"aa_seed_{seed}",
        request_text="Did customers in the East region spend more than the West region?",
        build_dataset=build,
        check=check,
    )


AA_SCENARIOS = [_make_aa_scenario(seed) for seed in range(100, 120)]  # 20 trials


# ---------------------------------------------------------------------------
# Planted-effect scenarios across varying effect sizes. Larger effects
# should be detected essentially every time; smaller ones may occasionally
# be missed -- that's statistically correct behavior (power < 100%), not an
# agent failure, so these checks use a generous but real tolerance rather
# than demanding every single one succeeds.
# ---------------------------------------------------------------------------

def _make_effect_scenario(name_suffix: str, effect: float, seed: int, expect_detection: bool) -> EvalScenario:
    def build():
        df, spec = generate_two_group_continuous(n_per_group=200, mean_a=50.0, effect=effect, std=10.0, seed=seed)
        return _rename_for_request(df), spec

    def check(result, spec):
        test_result = result.get("test_result")
        if test_result is None:
            return False, "No test_result in final state"
        observed = test_result["group_b_mean"] - test_result["group_a_mean"]
        close_to_truth = abs(observed - spec.true_effect) < 3.0
        detected = test_result["p_value"] < 0.05

        if not close_to_truth:
            return False, f"Observed effect {observed:.2f} too far from planted {spec.true_effect}"
        if expect_detection and not detected:
            return False, f"Expected detection of a clear effect ({effect}), got p={test_result['p_value']:.4f}"
        return True, f"observed={observed:.2f}, planted={effect}, p={test_result['p_value']:.4f}, detected={detected}"

    return EvalScenario(
        name=f"effect_{name_suffix}",
        request_text="Did customers in the East region spend more than the West region?",
        build_dataset=build,
        check=check,
    )


EFFECT_SCENARIOS = [
    _make_effect_scenario("small_3", effect=3.0, seed=200, expect_detection=False),   # may or may not reach significance
    _make_effect_scenario("medium_6", effect=6.0, seed=201, expect_detection=True),
    _make_effect_scenario("large_10", effect=10.0, seed=202, expect_detection=True),
    _make_effect_scenario("large_15", effect=15.0, seed=203, expect_detection=True),
]


# ---------------------------------------------------------------------------
# Messy-data scenarios: duplicates/missingness layered on a known effect,
# the full checklist-interrupt path gets exercised (auto-approved here,
# since the rejection path is already proven in test_7e_full_messy_data_
# proof.py), confirming the pipeline still reaches a correct statistical
# conclusion when real data-quality issues are present and get cleaned.
# ---------------------------------------------------------------------------

def _make_messy_scenario(name_suffix: str, n_duplicates: int, missingness_pct: float, seed: int) -> EvalScenario:
    def build():
        df, spec = generate_two_group_continuous_with_quality_issues(
            n_per_group=200, mean_a=50.0, effect=8.0, std=10.0, seed=seed,
            n_duplicate_rows=n_duplicates, missingness_pct=missingness_pct,
        )
        return _rename_for_request(df), spec

    def check(result, spec):
        test_result = result.get("test_result")
        if test_result is None:
            return False, "No test_result in final state"
        observed = test_result["group_b_mean"] - test_result["group_a_mean"]
        if abs(observed - spec.true_effect) > 3.0:
            return False, f"Observed effect {observed:.2f} too far from planted {spec.true_effect} despite cleaning"
        if test_result["p_value"] >= 0.05:
            return False, f"Expected detection after cleaning, got p={test_result['p_value']:.4f}"
        return True, f"Correctly recovered effect after cleaning (observed={observed:.2f}, p={test_result['p_value']:.4f})"

    return EvalScenario(
        name=f"messy_{name_suffix}",
        request_text="Did customers in the East region spend more than the West region?",
        build_dataset=build,
        check=check,
    )


MESSY_SCENARIOS = [
    _make_messy_scenario("duplicates_only", n_duplicates=5, missingness_pct=0.0, seed=300),
    _make_messy_scenario("missingness_only", n_duplicates=0, missingness_pct=0.08, seed=301),
]


ALL_SCENARIOS = AA_SCENARIOS + EFFECT_SCENARIOS + MESSY_SCENARIOS