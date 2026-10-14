"""
M0.5 step 2: a text column with more distinct values than HIGH_CARDINALITY_THRESHOLD has its individual
values withheld everywhere in the descriptive result (top values, mode, bar chart, pairwise chart) and in
the descriptive report prompt. Columns at or below the threshold and numeric columns are unchanged.
"""
import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from core.capabilities import DESCRIPTIVE
from core.dataset import LocalDiskStore
from core.descriptive import compute_descriptive_stats
from core.ingestion import HIGH_CARDINALITY_THRESHOLD, load_dataset
from core.plan import AnalysisPlan, CausalStatus, OutcomeType, QuestionType
from core.report import generate_descriptive_report
from tests.test_prompt_canaries import NAME_CANARIES, _base_df, _clean, fake_model  # noqa: F401 (fixture)

WITHHELD = "individual values are withheld"


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _dataset(df, tmp_path, store, name="data"):
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    return load_dataset(path, session_id="s", name=name, store=store)


def _by_name(result):
    return {c.name: c for c in result.column_stats}


def _charts(result, chart_type):
    return [c for c in result.chart_data if c["chart_type"] == chart_type]


def _labels(k, n=120):
    """k distinct labels with unequal counts (label i appears about n/(i+2) times) plus one missing value."""
    weights = np.array([1 / (i + 2) for i in range(k)])
    counts = np.maximum(1, np.round(weights / weights.sum() * n)).astype(int)
    values = [f"v{i:02d}" for i, c in enumerate(counts) for _ in range(c)]
    return values + [None]


# --- (a) the boundary ---------------------------------------------------------------------------

@pytest.mark.parametrize("k, withheld", [(HIGH_CARDINALITY_THRESHOLD, False), (HIGH_CARDINALITY_THRESHOLD + 1, True)])
def test_threshold_boundary(k, withheld, tmp_path, store):
    labels = _labels(k)
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"label": labels, "x": rng.normal(10, 2, len(labels))})
    assert df["label"].nunique(dropna=True) == k
    cols = _by_name(compute_descriptive_stats(_dataset(df, tmp_path, store), store))

    label = cols["label"]
    assert label.distinct_count == k
    if withheld:
        assert label.top_value_counts == () and label.mode is None
    else:
        counts = df["label"].dropna().value_counts()
        assert label.top_value_counts == tuple((str(v), int(c)) for v, c in counts.head(5).items())
        assert len(label.top_value_counts) == 5
        assert label.mode == str(counts.index[0])

    x, series = cols["x"], df["x"]
    assert x.distinct_count == series.nunique(dropna=True)
    assert x.mean == pytest.approx(series.mean()) and x.median == pytest.approx(series.median())
    assert x.std == pytest.approx(series.std())
    assert x.min == pytest.approx(series.min()) and x.max == pytest.approx(series.max())
    assert x.skewness == pytest.approx(scipy_stats.skew(series))
    assert x.mode is None and x.top_value_counts == ()


# --- (b) charts ------------------------------------------------------------------------------------

def _names(n=300):
    return [f"person-{i:03d}" for i in range(n)]


def test_no_bar_chart_for_a_high_cardinality_column(tmp_path, store):
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"name": _names(), "tier": rng.choice(["basic", "plus", "pro"], 300),
                       "x": rng.normal(0, 1, 300), "y": rng.normal(0, 1, 300)})
    result = compute_descriptive_stats(_dataset(df, tmp_path, store), store)
    assert [c["title"] for c in _charts(result, "bar_chart")] == ["Frequency of tier"]
    assert _charts(result, "pairwise_categorical_chart") == []  # only one low-cardinality text column
    for chart_type, count in (("histogram", 2), ("boxplot_single_column", 2), ("correlation_heatmap", 1),
                              ("scatter_matrix", 1), ("missingness_chart", 1)):
        assert len(_charts(result, chart_type)) == count, chart_type


def test_pairwise_chart_skips_a_high_cardinality_column_in_front(tmp_path, store):
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"name": _names(), "region": rng.choice(["East", "West"], 300),
                       "tier": rng.choice(["basic", "plus", "pro"], 300), "x": rng.normal(0, 1, 300)})
    result = compute_descriptive_stats(_dataset(df, tmp_path, store), store)
    (pairwise,) = _charts(result, "pairwise_categorical_chart")
    assert pairwise["title"] == "region by tier"
    assert pairwise["data"]["categories_a"] == ["East", "West"]
    assert pairwise["data"]["categories_b"] == ["basic", "plus", "pro"]


# --- (c) the canary data's descriptive result -------------------------------------------------------

def test_descriptive_result_of_the_canary_data_holds_no_name(tmp_path, store):
    df = _clean(_base_df())
    text = json.dumps(DESCRIPTIVE.compute(_dataset(df, tmp_path, store), store), default=str)
    for name in (*NAME_CANARIES, "name-0001", "name-0150", "name-0297"):
        assert name in set(df["customer_name"])
        assert name not in text, name
    assert "tier" in text and "basic" in text  # the low-cardinality column still has its values


# --- (d, e) the descriptive report prompt -------------------------------------------------------------

def _plan():
    return AnalysisPlan(request_text="Describe the data.", session_id="s", question_type=QuestionType.DESCRIPTIVE,
                        causal_status=CausalStatus.OBSERVATIONAL, outcome_name="x",
                        outcome_type=OutcomeType.CONTINUOUS, unit_of_analysis="row", data_sources=("d@v1",))


def _prompt(fake_model, result):
    prompts, _ = fake_model
    generate_descriptive_report(_plan(), result)
    (prompt,) = [p for p in prompts if p["stage"] == "report"]
    return prompt["text"]


def test_prompt_withholds_a_300_distinct_column(fake_model, tmp_path, store):
    rng = np.random.default_rng(6)
    names = [f"customer-{i:03d}" for i in range(300)]
    df = pd.DataFrame({"customer_name": names + names[:20], "x": rng.normal(0, 1, 320)})
    text = _prompt(fake_model, DESCRIPTIVE.compute(_dataset(df, tmp_path, store), store))
    assert "- customer_name (categorical): 300 distinct values; individual values are withheld" in text.splitlines()
    assert [n for n in names if n in text] == []


def test_prompt_keeps_the_top_values_of_a_5_distinct_column(fake_model, tmp_path, store):
    labels = ["a"] * 9 + ["b"] * 7 + ["c"] * 5 + ["d"] * 3 + ["e"] * 1
    df = pd.DataFrame({"letter": labels, "x": np.arange(len(labels), dtype=float)})
    text = _prompt(fake_model, DESCRIPTIVE.compute(_dataset(df, tmp_path, store), store))
    assert "- letter (categorical): most common values: a (9), b (7), c (5)" in text.splitlines()
    assert WITHHELD not in text


def test_an_older_result_without_distinct_count_keeps_the_old_line(fake_model):
    result = {"row_count": 4, "correlation_matrix": None, "column_stats": [
        {"name": "region", "is_numeric": False, "mode": "East", "top_value_counts": [("East", 3), ("West", 1)]}]}
    text = _prompt(fake_model, result)
    assert "- region (categorical): most common values: East (3), West (1)" in text.splitlines()
    assert WITHHELD not in text
