import numpy as np
import pandas as pd
import pytest
from core.dataset import LocalDiskStore
from core.ingestion import load_dataset
from core.multi_group import select_multi_group_method, execute_multi_group_test


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def make_dataset(df, tmp_path, store, name="data"):
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    return load_dataset(path, session_id="session-abc", name=name, store=store)


def test_rejects_fewer_than_three_groups(tmp_path, store):
    df = pd.DataFrame({"region": ["A"] * 10 + ["B"] * 10, "x": list(range(20))})
    ds = make_dataset(df, tmp_path, store)
    with pytest.raises(ValueError, match="requires 3 or more groups"):
        select_multi_group_method(ds, store, outcome_col="x", group_col="region")


def test_normal_equal_variance_data_uses_anova(tmp_path, store):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "region": ["A"] * 100 + ["B"] * 100 + ["C"] * 100,
        "x": np.concatenate([rng.normal(50, 10, 100), rng.normal(50, 10, 100), rng.normal(50, 10, 100)]),
    })
    ds = make_dataset(df, tmp_path, store)
    choice = select_multi_group_method(ds, store, outcome_col="x", group_col="region")
    assert choice.method == "one_way_anova"


def test_detects_a_real_effect_with_anova_and_tukey(tmp_path, store):
    """Three groups, one clearly different -- ANOVA should find it, Tukey should isolate which pair."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "region": ["A"] * 100 + ["B"] * 100 + ["C"] * 100,
        "x": np.concatenate([rng.normal(50, 5, 100), rng.normal(50, 5, 100), rng.normal(70, 5, 100)]),
    })
    ds = make_dataset(df, tmp_path, store)
    choice = select_multi_group_method(ds, store, outcome_col="x", group_col="region")
    result = execute_multi_group_test(ds, store, outcome_col="x", group_col="region", method=choice.method)

    assert result.p_value < 0.05
    assert result.post_hoc_method == "tukey_hsd"
    c_comparisons = [c for c in result.pairwise_comparisons if "C" in (c.group_a, c.group_b)]
    assert all(c.significant for c in c_comparisons)  # C should differ from both A and B
    ab_comparison = next(c for c in result.pairwise_comparisons if {c.group_a, c.group_b} == {"A", "B"})
    assert not ab_comparison.significant  # A and B should NOT differ


def test_non_normal_data_uses_kruskal_wallis(tmp_path, store):
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "region": ["A"] * 100 + ["B"] * 100 + ["C"] * 100,
        "x": np.concatenate([rng.exponential(5, 100), rng.exponential(5, 100), rng.exponential(5, 100)]),
    })
    ds = make_dataset(df, tmp_path, store)
    choice = select_multi_group_method(ds, store, outcome_col="x", group_col="region")
    assert choice.method == "kruskal_wallis"


def test_kruskal_wallis_with_real_effect_uses_dunn_posthoc(tmp_path, store):
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "region": ["A"] * 100 + ["B"] * 100 + ["C"] * 100,
        "x": np.concatenate([rng.exponential(5, 100), rng.exponential(5, 100), rng.exponential(20, 100)]),
    })
    ds = make_dataset(df, tmp_path, store)
    choice = select_multi_group_method(ds, store, outcome_col="x", group_col="region")
    assert choice.method == "kruskal_wallis"
    result = execute_multi_group_test(ds, store, outcome_col="x", group_col="region", method=choice.method)
    assert result.p_value < 0.05
    assert result.post_hoc_method == "dunn"


def test_chart_data_present_on_method_decision(tmp_path, store):
    rng = np.random.default_rng(5)
    df = pd.DataFrame({
        "region": ["A"] * 50 + ["B"] * 50 + ["C"] * 50,
        "x": np.concatenate([rng.normal(50, 10, 50), rng.normal(50, 10, 50), rng.normal(50, 10, 50)]),
    })
    ds = make_dataset(df, tmp_path, store)
    choice = select_multi_group_method(ds, store, outcome_col="x", group_col="region")
    chart_types = {c["chart_type"] for c in choice.decision.chart_data}
    assert "boxplot_by_group" in chart_types
    assert "qq_plot" in chart_types


def test_all_result_chart_data_is_json_serializable(tmp_path, store):
    import json
    rng = np.random.default_rng(9)
    df = pd.DataFrame({
        "region": ["A"] * 50 + ["B"] * 50 + ["C"] * 50,
        "x": np.concatenate([rng.normal(50, 5, 50), rng.normal(50, 5, 50), rng.normal(70, 5, 50)]),
    })
    ds = make_dataset(df, tmp_path, store)
    choice = select_multi_group_method(ds, store, outcome_col="x", group_col="region")
    result = execute_multi_group_test(ds, store, outcome_col="x", group_col="region", method=choice.method)
    for chart in choice.decision.chart_data:
        json.dumps(chart)
    for chart in result.chart_data:
        json.dumps({k: v for k, v in chart.items()})  # NaN in Dunn's CI would fail strict json.dumps; checked separately below


def test_dunn_missing_ci_is_explicitly_nan_not_fabricated(tmp_path, store):
    """Honest representation: Dunn's test has no native CI, so we must not invent one."""
    import math
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "region": ["A"] * 100 + ["B"] * 100 + ["C"] * 100,
        "x": np.concatenate([rng.exponential(5, 100), rng.exponential(5, 100), rng.exponential(20, 100)]),
    })
    ds = make_dataset(df, tmp_path, store)
    choice = select_multi_group_method(ds, store, outcome_col="x", group_col="region")
    result = execute_multi_group_test(ds, store, outcome_col="x", group_col="region", method=choice.method)
    assert any(math.isnan(c.ci_lower) for c in result.pairwise_comparisons)