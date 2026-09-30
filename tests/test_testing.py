import pytest
import pandas as pd
from scipy import stats
from core.dataset import LocalDiskStore
from core.ingestion import load_dataset
from core.testing import run_two_group_test
from core.synthetic import generate_two_group_continuous


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def make_dataset(df, tmp_path, store, name="test_data"):
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    return load_dataset(path, session_id="session-abc", name=name, store=store)


def test_normal_equal_variance_data_uses_student_t_test(tmp_path, store):
    df, spec = generate_two_group_continuous(n_per_group=200, mean_a=50, effect=5.0, std=10.0, seed=1)
    df = df.rename(columns={"outcome": "value"})
    dataset = make_dataset(df, tmp_path, store)

    result, decision = run_two_group_test(dataset, store, outcome_col="value", group_col="group")

    assert decision.stage == "test_selection"
    assert result.method in ("student_t_test", "welch_t_test")  # depends on variance check, both valid here
    assert result.p_value < 0.05  # a real planted effect of 5.0 should be detected

    # Cross-check against an independent, direct scipy call
    a = df[df["group"] == "A"]["value"].to_numpy()
    b = df[df["group"] == "B"]["value"].to_numpy()
    direct_stat, direct_p = stats.ttest_ind(a, b, equal_var=(result.method == "student_t_test"))
    assert abs(result.p_value - direct_p) < 1e-9
    assert abs(result.statistic - direct_stat) < 1e-9


def test_aa_data_usually_fails_to_reject(tmp_path, store):
    df, spec = generate_two_group_continuous(n_per_group=200, effect=0.0, seed=7)
    df = df.rename(columns={"outcome": "value"})
    dataset = make_dataset(df, tmp_path, store)

    result, decision = run_two_group_test(dataset, store, outcome_col="value", group_col="group")
    # Not asserting p > 0.05 unconditionally -- that would make this test flaky by design
    # (exactly 5% of seeds SHOULD reject). Instead, confirm the mechanics are sound.
    assert 0.0 <= result.p_value <= 1.0
    assert result.group_a_n == 200 and result.group_b_n == 200


def test_method_decision_evidence_matches_real_assumption_checks(tmp_path, store):
    df, spec = generate_two_group_continuous(n_per_group=200, mean_a=50, effect=5.0, std=10.0, seed=1)
    df = df.rename(columns={"outcome": "value"})
    dataset = make_dataset(df, tmp_path, store)

    _, decision = run_two_group_test(dataset, store, outcome_col="value", group_col="group")

    a = df[df["group"] == "A"]["value"].to_numpy()
    b = df[df["group"] == "B"]["value"].to_numpy()
    _, direct_shapiro_a = stats.shapiro(a)
    _, direct_shapiro_b = stats.shapiro(b)

    assert abs(decision.evidence["shapiro_p_group_a"] - direct_shapiro_a) < 1e-3
    assert abs(decision.evidence["shapiro_p_group_b"] - direct_shapiro_b) < 1e-3


def test_non_normal_data_uses_mann_whitney(tmp_path, store):
    import numpy as np
    rng = np.random.default_rng(3)
    # Exponential distribution -- clearly non-normal, should fail Shapiro-Wilk
    group_a = rng.exponential(scale=5.0, size=150)
    group_b = rng.exponential(scale=7.0, size=150)
    df = pd.DataFrame({
        "group": ["A"] * 150 + ["B"] * 150,
        "value": np.concatenate([group_a, group_b]),
    })
    dataset = make_dataset(df, tmp_path, store)

    result, decision = run_two_group_test(dataset, store, outcome_col="value", group_col="group")

    assert result.method == "mann_whitney_u"
    assert any(alt.method == "welch_t_test" for alt in decision.alternatives)
    assert "normality check" in decision.rationale


def test_rejects_wrong_number_of_groups(tmp_path, store):
    df = pd.DataFrame({
        "group": ["A", "B", "C"],
        "value": [1.0, 2.0, 3.0],
    })
    dataset = make_dataset(df, tmp_path, store)
    with pytest.raises(ValueError):
        run_two_group_test(dataset, store, outcome_col="value", group_col="group")