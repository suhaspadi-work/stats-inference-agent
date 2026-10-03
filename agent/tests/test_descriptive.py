import pandas as pd
import pytest
from core.dataset import LocalDiskStore
from core.ingestion import load_dataset
from core.descriptive import compute_descriptive_stats


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def make_dataset(df, tmp_path, store, name="data"):
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    return load_dataset(path, session_id="session-abc", name=name, store=store)


def test_numeric_column_stats_are_correct(tmp_path, store):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    ds = make_dataset(df, tmp_path, store)
    result = compute_descriptive_stats(ds, store)

    col = next(c for c in result.column_stats if c.name == "x")
    assert col.is_numeric
    assert col.mean == 3.0
    assert col.median == 3.0
    assert col.min == 1.0
    assert col.max == 5.0


def test_categorical_column_stats_are_correct(tmp_path, store):
    df = pd.DataFrame({"region": ["East", "East", "West"]})
    ds = make_dataset(df, tmp_path, store)
    result = compute_descriptive_stats(ds, store)

    col = next(c for c in result.column_stats if c.name == "region")
    assert not col.is_numeric
    assert col.mode == "East"
    assert dict(col.top_value_counts) == {"East": 2, "West": 1}


def test_correlation_matrix_present_with_two_numeric_columns(tmp_path, store):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})  # perfectly correlated
    ds = make_dataset(df, tmp_path, store)
    result = compute_descriptive_stats(ds, store)

    assert result.correlation_matrix is not None
    assert result.correlation_matrix["columns"] == ["x", "y"]
    assert abs(result.correlation_matrix["matrix"][0][1] - 1.0) < 1e-9  # x and y are perfectly correlated


def test_no_correlation_matrix_with_fewer_than_two_numeric_columns(tmp_path, store):
    df = pd.DataFrame({"region": ["East", "West"], "x": [1.0, 2.0]})
    ds = make_dataset(df, tmp_path, store)
    result = compute_descriptive_stats(ds, store)
    assert result.correlation_matrix is None


def test_chart_data_includes_expected_chart_types(tmp_path, store):
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0],
        "region": ["East", "West", "East", "West"], "tier": ["gold", "silver", "gold", "silver"],
    })
    ds = make_dataset(df, tmp_path, store)
    result = compute_descriptive_stats(ds, store)

    chart_types = {c["chart_type"] for c in result.chart_data}
    assert "histogram" in chart_types
    assert "bar_chart" in chart_types
    assert "boxplot_single_column" in chart_types
    assert "correlation_heatmap" in chart_types
    assert "scatter_matrix" in chart_types  # only 2 numeric columns, under the 5-column cap
    assert "missingness_chart" in chart_types
    assert "pairwise_categorical_chart" in chart_types  # 2 categorical columns present


def test_all_chart_data_is_json_serializable(tmp_path, store):
    import json
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "region": ["East", "West", "East"]})
    ds = make_dataset(df, tmp_path, store)
    result = compute_descriptive_stats(ds, store)
    for chart in result.chart_data:
        json.dumps(chart)  # raises if anything leaked through non-serializable