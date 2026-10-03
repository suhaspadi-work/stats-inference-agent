"""
Multi-group comparison (3+ groups): one-way ANOVA or Kruskal-Wallis as
the omnibus test, chosen via the same Shapiro-Wilk/Levene's discipline
as the two-group capability -- extended from 2 to N groups. When the
omnibus test finds a significant difference, a matched post-hoc test
(Tukey HSD after ANOVA, Dunn's test after Kruskal-Wallis) identifies
which specific pairs of groups differ, since the omnibus p-value alone
only says "a difference exists somewhere," not where.
"""
from __future__ import annotations
from dataclasses import dataclass
from scipy import stats
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from core.dataset import Dataset, DatasetStore
from core.plan import MethodDecision, AlternativeConsidered
import core.visualization as viz


@dataclass(frozen=True)
class PairwiseComparison:
    group_a: str
    group_b: str
    mean_diff: float
    ci_lower: float
    ci_upper: float
    p_value: float
    significant: bool


@dataclass(frozen=True)
class MultiGroupTestResult:
    method: str  # "one_way_anova" or "kruskal_wallis"
    statistic: float
    p_value: float
    post_hoc_method: str  # "tukey_hsd" or "dunn" or "none" (omnibus not significant)
    pairwise_comparisons: tuple[PairwiseComparison, ...]
    group_ns: dict  # {group_label: n}
    group_means: dict  # {group_label: mean}
    chart_data: tuple[dict, ...]


@dataclass(frozen=True)
class MultiGroupMethodChoice:
    method: str
    decision: MethodDecision


def select_multi_group_method(dataset: Dataset, store: DatasetStore, outcome_col: str, group_col: str) -> MultiGroupMethodChoice:
    """
    Extends select_two_group_method's exact same assumption-check pattern
    to N >= 3 groups: Shapiro-Wilk per group, Levene's across all groups
    (scipy's levene() natively accepts N arrays), then ANOVA if normal
    and equal-variance, Kruskal-Wallis otherwise.
    """
    df = store.read(dataset.storage_key)
    groups = sorted(df[group_col].dropna().unique())
    if len(groups) < 3:
        raise ValueError(
            f"select_multi_group_method requires 3 or more groups in '{group_col}', found {len(groups)}: {groups}. "
            f"Use select_two_group_method for exactly 2 groups."
        )

    group_arrays = {str(g): df.loc[df[group_col] == g, outcome_col].dropna().to_numpy() for g in groups}

    for label, arr in group_arrays.items():
        if np.std(arr) == 0 and all(np.std(a) == 0 for a in group_arrays.values()):
            raise ValueError(
                f"Cannot run a statistical test: the outcome column '{outcome_col}' has zero variance "
                f"in all groups (every value is identical). There is no variation to test."
            )

    shapiro_ps = {}
    for label, arr in group_arrays.items():
        shapiro_ps[label] = float(stats.shapiro(arr)[1]) if len(arr) >= 3 else 1.0
    levene_p = float(stats.levene(*group_arrays.values())[1])

    normal_enough = all(p > 0.05 for p in shapiro_ps.values())
    equal_var = levene_p > 0.05

    evidence = {
        **{f"shapiro_p_{label}": round(p, 4) for label, p in shapiro_ps.items()},
        "levene_p": round(levene_p, 4),
    }

    if normal_enough and equal_var:
        method = "one_way_anova"
        rationale = (
            f"All {len(groups)} groups pass a normality check (Shapiro-Wilk p > 0.05) and "
            f"have equal variances (Levene's p > 0.05), so one-way ANOVA is appropriate."
        )
        alternatives = (AlternativeConsidered(
            method="kruskal_wallis",
            rejected_because="a parametric test is more statistically powerful when its assumptions hold",
        ),)
    else:
        method = "kruskal_wallis"
        failing = [label for label, p in shapiro_ps.items() if p <= 0.05]
        reason = (
            f"group(s) {failing} fail a normality check" if failing
            else f"groups have unequal variances (Levene's p={evidence['levene_p']})"
        )
        rationale = (
            f"{reason.capitalize()}, so Kruskal-Wallis is used instead of ANOVA, "
            f"which assumes normality and equal variances."
        )
        alternatives = (AlternativeConsidered(
            method="one_way_anova",
            rejected_because="normality or equal-variance assumption violated",
        ),)

    boxplot = viz.boxplot_by_group({k: v for k, v in group_arrays.items()}, outcome_col)
    qq_plots = [viz.qq_plot(arr, label) for label, arr in group_arrays.items()]
    chart_data = (boxplot, *qq_plots)

    decision = MethodDecision(
        stage="test_selection", chosen_method=method, rationale=rationale,
        alternatives=alternatives, evidence=evidence, chart_data=chart_data,
    )
    return MultiGroupMethodChoice(method=method, decision=decision)


def execute_multi_group_test(
    dataset: Dataset, store: DatasetStore, outcome_col: str, group_col: str, method: str,
) -> MultiGroupTestResult:
    """Executes an already-selected multi-group test, running the matched post-hoc test if the omnibus result is significant."""
    df = store.read(dataset.storage_key)
    groups = sorted(df[group_col].dropna().unique())
    group_arrays = {str(g): df.loc[df[group_col] == g, outcome_col].dropna().to_numpy() for g in groups}
    group_ns = {label: len(arr) for label, arr in group_arrays.items()}
    group_means = {label: float(arr.mean()) for label, arr in group_arrays.items()}

    if method == "one_way_anova":
        stat, p_value = stats.f_oneway(*group_arrays.values())
    else:
        stat, p_value = stats.kruskal(*group_arrays.values())

    pairwise = ()
    post_hoc_method = "none"
    if p_value < 0.05:
        if method == "one_way_anova":
            post_hoc_method = "tukey_hsd"
            flat_values = np.concatenate(list(group_arrays.values()))
            flat_labels = np.concatenate([[label] * len(arr) for label, arr in group_arrays.items()])
            tukey_result = pairwise_tukeyhsd(flat_values, flat_labels)
            pairwise = tuple(
                PairwiseComparison(
                    group_a=str(row[0]), group_b=str(row[1]), mean_diff=float(row[2]),
                    ci_lower=float(row[4]), ci_upper=float(row[5]), p_value=float(row[3]),
                    significant=bool(row[6]),
                )
                for row in tukey_result._results_table.data[1:]
            )
        else:
            post_hoc_method = "dunn"
            flat_values = np.concatenate(list(group_arrays.values()))
            flat_labels = np.concatenate([[label] * len(arr) for label, arr in group_arrays.items()])
            dunn_df = sp.posthoc_dunn(
                pd.DataFrame({"value": flat_values, "group": flat_labels}),
                val_col="value", group_col="group", p_adjust="holm",
            )
            labels = list(group_arrays.keys())
            pairwise_list = []
            for i, a in enumerate(labels):
                for b in labels[i + 1:]:
                    p = float(dunn_df.loc[a, b])
                    pairwise_list.append(PairwiseComparison(
                        group_a=a, group_b=b,
                        mean_diff=group_means[b] - group_means[a],
                        ci_lower=float("nan"), ci_upper=float("nan"),  # Dunn's test doesn't natively produce a CI
                        p_value=p, significant=p < 0.05,
                    ))
            pairwise = tuple(pairwise_list)

    chart_data = ()
    if pairwise:
        chart_data = (viz.post_hoc_comparison_chart(
            comparisons=[f"{c.group_a}-{c.group_b}" for c in pairwise],
            mean_diffs=[c.mean_diff for c in pairwise],
            ci_lower=[c.ci_lower for c in pairwise], ci_upper=[c.ci_upper for c in pairwise],
            significant=[c.significant for c in pairwise],
        ),)

    return MultiGroupTestResult(
        method=method, statistic=float(stat), p_value=float(p_value),
        post_hoc_method=post_hoc_method, pairwise_comparisons=pairwise,
        group_ns=group_ns, group_means=group_means, chart_data=chart_data,
    )