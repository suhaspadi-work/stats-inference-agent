import json
import numpy as np
import core.visualization as viz


def _assert_json_serializable(chart: dict):
    """The core guarantee: every chart dict must round-trip through JSON cleanly -- no numpy/pandas objects leaked."""
    json.dumps(chart)  # raises TypeError if anything non-serializable is present


def test_histogram_shape_and_serializability():
    chart = viz.histogram(np.array([1.0, 2.0, 3.0]), "total_spend")
    assert chart["chart_type"] == "histogram"
    assert chart["data"]["values"] == [1.0, 2.0, 3.0]
    _assert_json_serializable(chart)


def test_qq_plot_shape_and_serializability():
    chart = viz.qq_plot(np.random.default_rng(1).normal(50, 10, 30), "group A")
    assert chart["chart_type"] == "qq_plot"
    assert len(chart["data"]["theoretical_quantiles"]) == len(chart["data"]["sample_quantiles"])
    _assert_json_serializable(chart)


def test_boxplot_by_group_shape_and_serializability():
    chart = viz.boxplot_by_group({"East": np.array([1.0, 2.0]), "West": np.array([3.0, 4.0])}, "total_spend")
    assert chart["data"]["groups"]["East"] == [1.0, 2.0]
    assert chart["data"]["groups"]["West"] == [3.0, 4.0]
    _assert_json_serializable(chart)


def test_coefficient_plot_shape_and_serializability():
    chart = viz.coefficient_plot(["region", "age"], [2.5, 0.3], [1.0, -0.1], [4.0, 0.7])
    assert chart["data"]["predictors"] == ["region", "age"]
    _assert_json_serializable(chart)


def test_correlation_heatmap_shape_and_serializability():
    matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
    chart = viz.correlation_heatmap(["a", "b"], matrix)
    assert chart["data"]["matrix"] == [[1.0, 0.5], [0.5, 1.0]]
    _assert_json_serializable(chart)


def test_all_chart_functions_produce_json_serializable_output():
    """Catch-all: every chart type, with minimal plausible inputs, must be JSON-safe."""
    charts = [
        viz.histogram(np.array([1.0, 2.0]), "x"),
        viz.bar_chart(["a", "b"], [1, 2], "x"),
        viz.boxplot_by_group({"A": np.array([1.0]), "B": np.array([2.0])}, "x"),
        viz.boxplot_single_column(np.array([1.0, 2.0]), "x"),
        viz.qq_plot(np.array([1.0, 2.0, 3.0]), "x"),
        viz.scatter_with_fit(np.array([1.0]), np.array([2.0]), np.array([1.0]), np.array([2.0]), "x", "y"),
        viz.residual_plot(np.array([1.0]), np.array([0.1])),
        viz.residual_qq_plot(np.array([1.0, 2.0, 3.0])),
        viz.coefficient_plot(["a"], [1.0], [0.5], [1.5]),
        viz.residuals_vs_leverage(np.array([0.1]), np.array([0.5])),
        viz.correlation_heatmap(["a", "b"], np.array([[1.0, 0.2], [0.2, 1.0]])),
        viz.missingness_chart(["a", "b"], [0.1, 0.0]),
        viz.scatter_matrix(["a", "b"], {("a", "b"): ([1.0], [2.0])}),
        viz.pairwise_categorical_chart("a", "b", ["x"], ["y"], [[5]]),
        viz.post_hoc_comparison_chart(["A-B"], [1.0], [0.5], [1.5], [True]),
    ]
    for chart in charts:
        _assert_json_serializable(chart)
        assert "chart_type" in chart and "title" in chart and "data" in chart