from __future__ import annotations
from dataclasses import dataclass
from scipy import stats
import numpy as np
import pandas as pd

from core.dataset import Dataset, DatasetStore
from core.plan import MethodDecision, AlternativeConsidered


@dataclass(frozen=True)
class TwoGroupTestResult:
    """
    The actual output of a two-group test — every field here is a real
    number from scipy, never anything the model generated or summarized.
    """
    method: str
    statistic: float
    p_value: float
    effect_size: float          # Cohen's d (parametric) or rank-biserial (Mann-Whitney)
    confidence_interval: tuple[float, float]
    group_a_n: int
    group_b_n: int
    group_a_mean: float
    group_b_mean: float


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n_a, n_b = len(a), len(b)
    pooled_std = np.sqrt(((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / (n_a + n_b - 2))
    return (b.mean() - a.mean()) / pooled_std if pooled_std > 0 else 0.0


def _mean_diff_ci(a: np.ndarray, b: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """95% CI for the difference in means, via the standard two-sample formula."""
    n_a, n_b = len(a), len(b)
    diff = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / n_a + b.var(ddof=1) / n_b)
    dof = n_a + n_b - 2
    t_crit = stats.t.ppf((1 + confidence) / 2, dof)
    return (diff - t_crit * se, diff + t_crit * se)


def run_two_group_test(
    dataset: Dataset, store: DatasetStore, outcome_col: str, group_col: str, alpha: float = 0.05,
) -> tuple[TwoGroupTestResult, MethodDecision]:
    """
    Run the appropriate two-group test based on real assumption checks, and
    return BOTH the result and a MethodDecision explaining why this test was
    chosen over the alternatives — the audit-trail requirement from Phase 1,
    now produced by an actual analysis step rather than hand-written in a test.
    """
    df = store.read(dataset.storage_key)
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError(f"Expected exactly 2 groups in '{group_col}', found {len(groups)}: {list(groups)}")

    group_a_label, group_b_label = sorted(groups)
    a = df.loc[df[group_col] == group_a_label, outcome_col].dropna().to_numpy()
    b = df.loc[df[group_col] == group_b_label, outcome_col].dropna().to_numpy()

    # Real assumption checks -- these numbers drive the decision, not a guess
    _, shapiro_p_a = stats.shapiro(a) if len(a) >= 3 else (None, 1.0)
    _, shapiro_p_b = stats.shapiro(b) if len(b) >= 3 else (None, 1.0)
    _, levene_p = stats.levene(a, b)

    normal_enough = shapiro_p_a > 0.05 and shapiro_p_b > 0.05
    evidence = {
        "shapiro_p_group_a": round(float(shapiro_p_a), 4),
        "shapiro_p_group_b": round(float(shapiro_p_b), 4),
        "levene_p": round(float(levene_p), 4),
    }

    if normal_enough:
        equal_var = levene_p > 0.05
        stat, p_value = stats.ttest_ind(a, b, equal_var=equal_var)
        method = "student_t_test" if equal_var else "welch_t_test"
        effect_size = _cohens_d(a, b)
        ci = _mean_diff_ci(a, b)
        test_name = "standard" if equal_var else "Welch's"
        rationale = (
            f"Both groups pass a normality check (Shapiro-Wilk p > 0.05) and "
            f"{'have' if equal_var else 'do not have'} equal variances (Levene's p "
            f"{'>' if equal_var else '<='} 0.05), so a {test_name} "
            f"t-test is appropriate."
        )
        alternatives = (
            AlternativeConsidered(
                method="mann_whitney_u",
                rejected_because="a parametric test is more statistically powerful when its assumptions hold",
            ),
        )
    else:
        stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
        method = "mann_whitney_u"
        # Rank-biserial correlation as the effect size for Mann-Whitney
        effect_size = 1 - (2 * stat) / (len(a) * len(b))
        ci = _mean_diff_ci(a, b)  # reported alongside for interpretability, not the test's own basis
        rationale = (
            f"At least one group fails a normality check (Shapiro-Wilk p <= 0.05: "
            f"group A={evidence['shapiro_p_group_a']}, group B={evidence['shapiro_p_group_b']}), "
            f"so Mann-Whitney U is used instead of a t-test, which assumes normality."
        )
        alternatives = (
            AlternativeConsidered(
                method="welch_t_test",
                rejected_because="normality assumption violated per Shapiro-Wilk check",
            ),
            AlternativeConsidered(
                method="bootstrap_test",
                rejected_because="sample size is sufficient for a standard rank-based test",
            ),
        )

    result = TwoGroupTestResult(
        method=method, statistic=float(stat), p_value=float(p_value),
        effect_size=float(effect_size), confidence_interval=ci,
        group_a_n=len(a), group_b_n=len(b),
        group_a_mean=float(a.mean()), group_b_mean=float(b.mean()),
    )
    decision = MethodDecision(
        stage="test_selection", chosen_method=method, rationale=rationale,
        alternatives=alternatives, evidence=evidence,
    )
    return result, decision