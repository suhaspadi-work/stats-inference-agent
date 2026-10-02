import numpy as np
from scipy import stats
from core.synthetic import generate_two_group_continuous, generate_two_group_continuous_with_quality_issues


def test_generates_correct_group_sizes():
    df, spec = generate_two_group_continuous(n_per_group=100, effect=0.0)
    assert len(df) == 200
    assert (df["group"] == "A").sum() == 100
    assert (df["group"] == "B").sum() == 100


def test_aa_spec_reports_zero_true_effect():
    _, spec = generate_two_group_continuous(effect=0.0)
    assert spec.true_effect == 0.0
    assert spec.true_mean_a == spec.true_mean_b


def test_planted_effect_spec_reports_the_effect_correctly():
    _, spec = generate_two_group_continuous(mean_a=50.0, effect=5.0)
    assert spec.true_effect == 5.0
    assert spec.true_mean_b == 55.0


def test_generated_data_actually_reflects_the_planted_effect():
    """
    Not just checking the spec's bookkeeping -- checking the ACTUAL generated
    numbers land close to what was planted, using a real independent-samples
    t-test (scipy directly, not our own agent code) as the check.
    """
    df, spec = generate_two_group_continuous(n_per_group=500, mean_a=50.0, effect=5.0, std=10.0, seed=1)
    group_a = df[df["group"] == "A"]["outcome"]
    group_b = df[df["group"] == "B"]["outcome"]

    observed_effect = group_b.mean() - group_a.mean()
    assert abs(observed_effect - spec.true_effect) < 1.5

    stat, p_value = stats.ttest_ind(group_a, group_b)
    assert p_value < 0.05


def test_aa_data_shows_no_significant_difference_on_average():
    """
    The core A/A correctness check: run MANY A/A generations with different
    seeds and confirm the false-positive rate is close to the nominal 5%
    at alpha=0.05 -- not exactly 5% (that would be suspicious with this few
    trials), but clearly in a sane range, not wildly off.
    """
    rejections = 0
    n_trials = 200
    for seed in range(n_trials):
        df, spec = generate_two_group_continuous(n_per_group=100, effect=0.0, seed=seed)
        group_a = df[df["group"] == "A"]["outcome"]
        group_b = df[df["group"] == "B"]["outcome"]
        _, p_value = stats.ttest_ind(group_a, group_b)
        if p_value < 0.05:
            rejections += 1

    false_positive_rate = rejections / n_trials
    assert 0.01 < false_positive_rate < 0.12


def test_different_seeds_produce_different_data():
    df1, _ = generate_two_group_continuous(seed=1)
    df2, _ = generate_two_group_continuous(seed=2)
    assert not df1["outcome"].equals(df2["outcome"])


def test_same_seed_is_reproducible():
    df1, _ = generate_two_group_continuous(seed=99)
    df2, _ = generate_two_group_continuous(seed=99)
    assert df1["outcome"].equals(df2["outcome"])


def test_quality_issues_variant_preserves_ground_truth_spec():
    """The spec's true_effect must be identical to the clean version -- injection happens after, not during generation."""
    _, spec = generate_two_group_continuous_with_quality_issues(
        mean_a=50.0, effect=8.0, n_duplicate_rows=3, missingness_pct=0.05,
    )
    assert spec.true_effect == 8.0
    assert spec.true_mean_a == 50.0


def test_quality_issues_variant_injects_exact_duplicate_rows():
    df, _ = generate_two_group_continuous_with_quality_issues(
        n_per_group=200, n_duplicate_rows=3, missingness_pct=0.0,
    )
    assert len(df) == 400 + 3
    assert df.duplicated().sum() == 3


def test_quality_issues_variant_injects_missingness():
    df, _ = generate_two_group_continuous_with_quality_issues(
        n_per_group=200, n_duplicate_rows=0, missingness_pct=0.1, missingness_column="group",
    )
    missing_count = df["group"].isna().sum()
    assert missing_count == int(400 * 0.1)


def test_quality_issues_variant_with_both_issues_combined():
    df, spec = generate_two_group_continuous_with_quality_issues(
        n_per_group=200, mean_a=50.0, effect=8.0, n_duplicate_rows=3, missingness_pct=0.05,
    )
    assert len(df) == 403
    assert df.duplicated().sum() == 3
    assert df["group"].isna().sum() == int(403 * 0.05)


def test_underlying_effect_still_statistically_recoverable_despite_quality_issues():
    """
    The core correctness property: even with duplicates and missingness
    injected, a real statistical test on this data should still detect the
    planted effect -- the quality issues shouldn't corrupt the signal,
    only add noise/complexity the wrangling pipeline needs to handle.
    """
    df, spec = generate_two_group_continuous_with_quality_issues(
        n_per_group=200, mean_a=50.0, effect=8.0, std=10.0, seed=5,
        n_duplicate_rows=3, missingness_pct=0.0,
    )
    group_a = df[df["group"] == "A"]["outcome"].dropna()
    group_b = df[df["group"] == "B"]["outcome"].dropna()

    stat, p_value = stats.ttest_ind(group_a, group_b)
    assert p_value < 0.05
    observed_effect = group_b.mean() - group_a.mean()
    assert abs(observed_effect - spec.true_effect) < 2.0