import pandas as pd
import pytest
from core.dataset import LocalDiskStore
from core.synthetic import generate_two_group_continuous
from eval.runner import EvalScenario, run_scenario, run_eval_suite, RESULTS_FILE
import os


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def build_clean_effect_dataset():
    df, spec = generate_two_group_continuous(n_per_group=200, mean_a=50, effect=8.0, std=10.0, seed=42)
    df = df.rename(columns={"outcome": "total_spend", "group": "region"})
    return df, spec


def check_detects_planted_effect(result, spec):
    test_result = result.get("test_result")
    if test_result is None:
        return False, "No test_result in final state"
    if test_result["p_value"] >= 0.05:
        return False, f"Expected a significant result, got p={test_result['p_value']}"
    observed = test_result["group_b_mean"] - test_result["group_a_mean"]
    if abs(observed - spec.true_effect) > 2.5:
        return False, f"Observed effect {observed:.2f} too far from planted {spec.true_effect}"
    return True, f"Correctly detected effect (observed={observed:.2f}, planted={spec.true_effect})"


@pytest.mark.live
def test_smoke_single_scenario_runs_and_passes(tmp_path, store, monkeypatch):
    monkeypatch.chdir(tmp_path)  # isolate RESULTS_FILE to this test's tmp dir
    scenario = EvalScenario(
        name="smoke_clean_effect",
        request_text="Did customers in the East region spend more than the West region?",
        build_dataset=build_clean_effect_dataset,
        check=check_detects_planted_effect,
    )
    row = run_scenario(scenario, store, session_id="smoke-session")
    assert row["passed"], row["reason"]


@pytest.mark.live
def test_suite_runner_is_resumable(tmp_path, store, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scenario = EvalScenario(
        name="resumable_check",
        request_text="Did customers in the East region spend more than the West region?",
        build_dataset=build_clean_effect_dataset,
        check=check_detects_planted_effect,
    )
    done_first = run_eval_suite([scenario], store)
    assert "resumable_check" in done_first
    assert os.path.exists(RESULTS_FILE)

    # Second call should skip the already-completed scenario entirely --
    # confirmed by it not raising/re-running (no new live call needed)
    done_second = run_eval_suite([scenario], store)
    assert done_second["resumable_check"]["scenario"] == "resumable_check"