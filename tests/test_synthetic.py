import numpy as np
from scipy import stats
from core.synthetic import generate_two_group_continuous


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
    # With n=500 per group, the observed effect should land close to the true 5.0
    assert abs(observed_effect - spec.true_effect) < 1.5

    stat, p_value = stats.ttest_ind(group_a, group_b)
    assert p_value < 0.05  # a real, planted effect this size should be detected


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
    # Nominal is 5%; with 200 trials, allow a reasonably wide band before
    # treating this as a real problem rather than sampling noise
    assert 0.01 < false_positive_rate < 0.12


def test_different_seeds_produce_different_data():
    df1, _ = generate_two_group_continuous(seed=1)
    df2, _ = generate_two_group_continuous(seed=2)
    assert not df1["outcome"].equals(df2["outcome"])


def test_same_seed_is_reproducible():
    df1, _ = generate_two_group_continuous(seed=99)
    df2, _ = generate_two_group_continuous(seed=99)
    assert df1["outcome"].equals(df2["outcome"])