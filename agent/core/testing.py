from __future__ import annotations
from dataclasses import dataclass
from scipy import stats
import numpy as np
import pandas as pd

from core.dataset import Dataset, DatasetStore
from core.plan import MethodDecision, AlternativeConsidered
import core.visualization as viz


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
    # Names of the two groups, in comparison order: group A is the first level
    # when sorted, group B the second.
    group_a_label: str = "Group A"
    group_b_label: str = "Group B"
    # Sign conventions (these are what the report facts assume):
    #   confidence_interval and Cohen's d are group B minus group A.
    #   The Mann-Whitney effect size (rank-biserial) is positive when B tends higher.
    #   The test statistic is computed on (A, B), so for the t tests it is
    #   positive when A is HIGHER. Do not show it to a report model as a direction.
@dataclass(frozen=True)
class TwoGroupMethodChoice:
    """The method-selection half of what run_two_group_test does, extracted so it can run before execution."""
    method: str
    decision: MethodDecision


def select_two_group_method(dataset: Dataset, store: DatasetStore, outcome_col: str, group_col: str) -> TwoGroupMethodChoice:
    """
    Run the real assumption checks and decide which test WILL be used,
    without actually running it yet. This lets the choice (and its
    MethodDecision rationale) be shown to a human for approval before
    freeze, rather than only being decided silently during execution.
    """
    df = store.read(dataset.storage_key)
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError(f"Expected exactly 2 groups in '{group_col}', found {len(groups)}: {list(groups)}")

    group_a_label, group_b_label = sorted(groups)
    a = df.loc[df[group_col] == group_a_label, outcome_col].dropna().to_numpy()
    b = df.loc[df[group_col] == group_b_label, outcome_col].dropna().to_numpy()

    if np.std(a) == 0 and np.std(b) == 0:
        raise ValueError(
            f"Cannot run a statistical test: the outcome column '{outcome_col}' has zero variance "
            f"in both groups (every value is identical). There is no variation to test."
        )

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
        method = "student_t_test" if equal_var else "welch_t_test"
        test_name = "standard" if equal_var else "Welch's"
        rationale = (
            f"Both groups pass a normality check (Shapiro-Wilk p > 0.05) and "
            f"{'have' if equal_var else 'do not have'} equal variances (Levene's p "
            f"{'>' if equal_var else '<='} 0.05), so a {test_name} t-test is appropriate."
        )
        alternatives = (AlternativeConsidered(
            method="mann_whitney_u",
            rejected_because="a parametric test is more statistically powerful when its assumptions hold",
        ),)
    else:
        method = "mann_whitney_u"
        rationale = (
            f"At least one group fails a normality check (Shapiro-Wilk p <= 0.05: "
            f"group A={evidence['shapiro_p_group_a']}, group B={evidence['shapiro_p_group_b']}), "
            f"so Mann-Whitney U is used instead of a t-test, which assumes normality."
        )
        alternatives = (
            AlternativeConsidered(method="welch_t_test", rejected_because="normality assumption violated per Shapiro-Wilk check"),
            AlternativeConsidered(method="bootstrap_test", rejected_because="sample size is sufficient for a standard rank-based test"),
        )

    boxplot = viz.boxplot_by_group({str(group_a_label): a, str(group_b_label): b}, outcome_col)
    qq_plots = [viz.qq_plot(a, str(group_a_label)), viz.qq_plot(b, str(group_b_label))]
    chart_data = (boxplot, *qq_plots)

    decision = MethodDecision(
        stage="test_selection", chosen_method=method, rationale=rationale,
        alternatives=alternatives, evidence=evidence, chart_data=chart_data,
    )
    return TwoGroupMethodChoice(method=method, decision=decision)


def execute_two_group_test(
    dataset: Dataset, store: DatasetStore, outcome_col: str, group_col: str, method: str,
) -> TwoGroupTestResult:
    """
    Execute a given, ALREADY-SELECTED two-group test method, without
    re-deciding which method to use. This is what execute_node calls --
    method selection happens once, before freeze (select_two_group_method),
    and execution simply carries out that pre-registered choice. Calling
    run_two_group_test here instead would silently re-select and record a
    SECOND, undocumented MethodDecision at execution time -- overwriting
    the one a human actually approved at freeze. This is a real bug found
    via Phase 7e's full end-to-end proof: it caused the unaddressed_issues
    caveat (correctly attached to the approved decision) to disappear from
    plan.latest_decision after execution.
    """
    df = store.read(dataset.storage_key)
    groups = sorted(df[group_col].dropna().unique())
    a = df.loc[df[group_col] == groups[0], outcome_col].dropna().to_numpy()
    b = df.loc[df[group_col] == groups[1], outcome_col].dropna().to_numpy()

    if method == "mann_whitney_u":
        stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
        effect_size = 1 - (2 * stat) / (len(a) * len(b))
    else:
        equal_var = method == "student_t_test"
        stat, p_value = stats.ttest_ind(a, b, equal_var=equal_var)
        effect_size = _cohens_d(a, b)

    ci = _mean_diff_ci(a, b)
    return TwoGroupTestResult(
        group_a_label=str(groups[0]), group_b_label=str(groups[1]),
        method=method, statistic=float(stat), p_value=float(p_value),
        effect_size=float(effect_size), confidence_interval=ci,
        group_a_n=len(a), group_b_n=len(b),
        group_a_mean=float(a.mean()), group_b_mean=float(b.mean()),
    )

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n_a, n_b = len(a), len(b)
    pooled_std = np.sqrt(((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / (n_a + n_b - 2))
    return float((b.mean() - a.mean()) / pooled_std) if pooled_std > 0 else 0.0


def _mean_diff_ci(a: np.ndarray, b: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """95% CI for the difference in means, via the standard two-sample formula."""
    n_a, n_b = len(a), len(b)
    diff = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / n_a + b.var(ddof=1) / n_b)
    dof = n_a + n_b - 2
    t_crit = stats.t.ppf((1 + confidence) / 2, dof)
    return (float(diff - t_crit * se), float(diff + t_crit * se))


def run_two_group_test(
    dataset: Dataset, store: DatasetStore, outcome_col: str, group_col: str, alpha: float = 0.05,
) -> tuple[TwoGroupTestResult, MethodDecision]:
    """
    Select the method (see select_two_group_method), then actually run it.
    Kept as one call for callers who don't need the pre-freeze split (e.g.
    Phase 5's own direct tests); the graph's select_method node uses
    select_two_group_method directly instead.
    """
    choice = select_two_group_method(dataset, store, outcome_col, group_col)

    df = store.read(dataset.storage_key)
    groups = sorted(df[group_col].dropna().unique())
    a = df.loc[df[group_col] == groups[0], outcome_col].dropna().to_numpy()
    b = df.loc[df[group_col] == groups[1], outcome_col].dropna().to_numpy()

    if choice.method == "mann_whitney_u":
        stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
        effect_size = 1 - (2 * stat) / (len(a) * len(b))
    else:
        equal_var = choice.method == "student_t_test"
        stat, p_value = stats.ttest_ind(a, b, equal_var=equal_var)
        effect_size = _cohens_d(a, b)

    ci = _mean_diff_ci(a, b)
    result = TwoGroupTestResult(
        group_a_label=str(groups[0]), group_b_label=str(groups[1]),
        method=choice.method, statistic=float(stat), p_value=float(p_value),
        effect_size=float(effect_size), confidence_interval=ci,
        group_a_n=len(a), group_b_n=len(b),
        group_a_mean=float(a.mean()), group_b_mean=float(b.mean()),
    )
    return result, choice.decision