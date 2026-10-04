import json
import numpy as np
import pandas as pd
import pytest
from core.dataset import LocalDiskStore
from core.ingestion import load_dataset
from core.regression import select_regression_method, execute_regression


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def make_dataset(df, tmp_path, store, name="data"):
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    return load_dataset(path, session_id="session-abc", name=name, store=store)


def run(df, tmp_path, store, predictors, outcome="y", overrides=None):
    ds = make_dataset(df, tmp_path, store)
    choice = select_regression_method(ds, store, outcome, predictors, encoding_overrides=overrides)
    result = execute_regression(ds, store, outcome, predictors, choice.method, choice.encodings)
    return choice, result


def known_data(n=500, seed=1):
    """y = 2 + 3*x1 - 1.5*x2 + noise(sd=1), with independent predictors."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 2 + 3 * x1 - 1.5 * x2 + rng.normal(0, 1, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def level_data(sizes: dict, effects: dict, seed=3, sd=1.0):
    rng = np.random.default_rng(seed)
    region = np.repeat(list(sizes), list(sizes.values()))
    y = np.array([effects[r] for r in region]) + rng.normal(0, sd, len(region))
    return pd.DataFrame({"region": region, "y": y})


ABC_SIZES = {"Alpha": 30, "Beta": 200, "Gamma": 70}
ABC_EFFECTS = {"Alpha": 0.0, "Beta": 5.0, "Gamma": 10.0}


def coef(result_or_fit, name):
    return next(c for c in result_or_fit.coefficients if c.name == name)


# ------------------------------------------------------------ core statistics

def test_recovers_known_coefficients(tmp_path, store):
    _, result = run(known_data(), tmp_path, store, ["x1", "x2"])
    assert abs(coef(result, "const").estimate - 2.0) < 0.3
    assert abs(coef(result, "x1").estimate - 3.0) < 0.3
    assert abs(coef(result, "x2").estimate - (-1.5)) < 0.3
    assert coef(result, "x1").ci_lower < 3.0 < coef(result, "x1").ci_upper
    assert coef(result, "x1").p_value < 0.001
    assert result.r_squared > 0.8


def test_method_choice_follows_the_breusch_pagan_evidence(tmp_path, store):
    """Homoscedastic by construction; asserts the choice is consistent with the recorded test, not a seed-dependent p-value."""
    choice, _ = run(known_data(), tmp_path, store, ["x1", "x2"])
    bp_p = choice.decision.evidence["breusch_pagan_p"]
    assert choice.method == ("ols" if bp_p > 0.05 else "ols_hc3_robust")


def test_heteroscedastic_data_uses_robust_standard_errors(tmp_path, store):
    rng = np.random.default_rng(2)
    n = 600
    x = rng.uniform(1, 10, n)
    y = 5 + 2 * x + rng.normal(0, 1, n) * x  # noise grows with x
    df = pd.DataFrame({"x": x, "y": y})
    choice, _ = run(df, tmp_path, store, ["x"])
    assert choice.method == "ols_hc3_robust"
    assert any(a.method == "ols" for a in choice.decision.alternatives)


def test_high_multicollinearity_is_flagged(tmp_path, store):
    rng = np.random.default_rng(4)
    n = 300
    x1 = rng.normal(0, 1, n)
    x2 = x1 + rng.normal(0, 0.05, n)  # nearly identical to x1
    y = 1 + x1 + rng.normal(0, 1, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    choice, _ = run(df, tmp_path, store, ["x1", "x2"])
    assert set(choice.decision.evidence["high_vif_columns"]) == {"x1", "x2"}
    assert "multicollinearity" in choice.decision.rationale


# ------------------------------------------------------------------- guards

def test_perfectly_collinear_predictors_rejected(tmp_path, store):
    df = known_data()
    df["x3"] = df["x1"] * 2
    ds = make_dataset(df, tmp_path, store)
    with pytest.raises(ValueError, match="perfectly collinear"):
        select_regression_method(ds, store, "y", ["x1", "x3"])


def test_rejects_non_numeric_outcome_and_missing_columns(tmp_path, store):
    df = known_data()
    df["label"] = ["a", "b"] * (len(df) // 2)
    ds = make_dataset(df, tmp_path, store)
    with pytest.raises(ValueError, match="cannot be used as the outcome"):
        select_regression_method(ds, store, "label", ["x1"])
    with pytest.raises(ValueError, match="not found"):
        select_regression_method(ds, store, "y", ["nope"])


def test_rejects_high_cardinality_categorical_predictor(tmp_path, store):
    df = known_data(n=100)
    df["customer"] = [f"c{i}" for i in range(100)]
    ds = make_dataset(df, tmp_path, store)
    with pytest.raises(ValueError, match="distinct values"):
        select_regression_method(ds, store, "y", ["customer"])


def test_rows_with_missing_values_are_dropped_and_counted(tmp_path, store):
    df = known_data(n=200)
    df.loc[:9, "x1"] = np.nan
    choice, result = run(df, tmp_path, store, ["x1", "x2"])
    assert result.n_dropped_missing == 10
    assert result.n_observations == 190
    assert choice.decision.evidence["n_dropped_missing"] == 10


# ----------------------------------------------------------------- encodings

def test_default_reference_is_the_largest_level_not_the_alphabetical_first(tmp_path, store):
    df = level_data(ABC_SIZES, ABC_EFFECTS)
    choice, result = run(df, tmp_path, store, ["region"])
    assert choice.encodings["region"]["scheme"] == "treatment"
    assert choice.encodings["region"]["reference"] == "Beta"  # n=200; "Alpha" would be the alphabetical default
    assert "largest group" in choice.decision.rationale
    assert abs(coef(result, "region[Alpha vs Beta]").estimate - (-5.0)) < 0.5
    assert abs(coef(result, "region[Gamma vs Beta]").estimate - 5.0) < 0.5


def test_user_can_override_the_reference_level(tmp_path, store):
    df = level_data(ABC_SIZES, ABC_EFFECTS)
    choice, result = run(df, tmp_path, store, ["region"], overrides={"region": {"reference": "Gamma"}})
    assert choice.encodings["region"]["reference"] == "Gamma"
    assert choice.encodings["region"]["source"] == "user"
    assert abs(coef(result, "region[Beta vs Gamma]").estimate - (-5.0)) < 0.5


def test_decision_records_rejected_encoding_alternatives(tmp_path, store):
    df = level_data(ABC_SIZES, ABC_EFFECTS)
    choice, _ = run(df, tmp_path, store, ["region"])
    methods = {a.method for a in choice.decision.alternatives}
    assert "effect_coding:region" in methods
    assert "ordinal_encoding:region" in methods
    ordinal = next(a for a in choice.decision.alternatives if a.method == "ordinal_encoding:region")
    assert "cannot be inferred" in ordinal.rejected_because


def test_effect_coding_alternative_is_the_same_model_with_different_coefficients(tmp_path, store):
    df = level_data(ABC_SIZES, ABC_EFFECTS)
    _, result = run(df, tmp_path, store, ["region"])
    effect = next(f for f in result.alternative_fits if f.label == "effect_coding")
    assert effect.same_fitted_model
    assert abs(effect.r_squared - result.r_squared) < 1e-9
    assert abs(effect.aic - result.aic) < 1e-6
    # level means are 0, 5, 10 so the average of levels is 5; effects are -5, 0, +5
    assert abs(coef(effect, "const").estimate - 5.0) < 0.5
    assert abs(coef(effect, "region[Alpha vs mean]").estimate - (-5.0)) < 0.5
    assert abs(coef(effect, "region[Gamma vs mean]").estimate - 5.0) < 0.5


def test_ordinal_encoding_requires_an_explicit_order(tmp_path, store):
    df = level_data(ABC_SIZES, ABC_EFFECTS)
    ds = make_dataset(df, tmp_path, store)
    with pytest.raises(ValueError, match="explicit"):
        select_regression_method(ds, store, "y", ["region"], encoding_overrides={"region": {"scheme": "ordinal"}})


def test_ordinal_order_must_cover_every_observed_level(tmp_path, store):
    rng = np.random.default_rng(5)
    size = np.repeat(["small", "medium", "large"], 60)
    y = rng.normal(0, 1, len(size))
    ds = make_dataset(pd.DataFrame({"size": size, "y": y}), tmp_path, store)
    with pytest.raises(ValueError, match="not in the supplied order"):
        select_regression_method(
            ds, store, "y", ["size"], encoding_overrides={"size": {"scheme": "ordinal", "order": ["small", "medium"]}},
        )


def test_ordinal_encoding_fits_one_slope_and_offers_the_dummy_model_as_an_alternative(tmp_path, store):
    rng = np.random.default_rng(6)
    size = np.repeat(["small", "medium", "large"], 80)
    step = np.repeat([0, 1, 2], 80)
    y = 2 * step + rng.normal(0, 0.5, len(size))
    df = pd.DataFrame({"size": size, "y": y})
    choice, result = run(
        df, tmp_path, store, ["size"],
        overrides={"size": {"scheme": "ordinal", "order": ["small", "medium", "large"]}},
    )
    assert abs(coef(result, "size[per step]").estimate - 2.0) < 0.3
    alt = next(f for f in result.alternative_fits if f.label == "treatment_instead_of_ordinal")
    assert not alt.same_fitted_model
    assert any(a.method == "treatment_coding:size" for a in choice.decision.alternatives)


def test_override_on_a_numeric_column_is_rejected(tmp_path, store):
    ds = make_dataset(known_data(), tmp_path, store)
    with pytest.raises(ValueError, match="not categorical"):
        select_regression_method(ds, store, "y", ["x1"], encoding_overrides={"x1": {"scheme": "effect"}})


def test_rare_levels_are_flagged_not_silently_merged(tmp_path, store):
    df = level_data({"Big": 150, "Mid": 80, "Tiny": 3}, {"Big": 0.0, "Mid": 2.0, "Tiny": 4.0})
    choice, result = run(df, tmp_path, store, ["region"])
    assert choice.decision.evidence["rare_levels"] == {"region": ["Tiny"]}
    assert "very few observations" in choice.decision.rationale
    assert any(c.name == "region[Tiny vs Big]" for c in result.coefficients)


# --------------------------------------------------------------------- charts

def test_chart_data_present_and_json_serializable(tmp_path, store):
    df = known_data()
    df["region"] = ["East", "West"] * (len(df) // 2)
    choice, result = run(df, tmp_path, store, ["x1", "x2", "region"])
    types = {c["chart_type"] for c in choice.decision.chart_data}
    assert {"residual_plot", "residual_qq_plot", "residuals_vs_leverage"} <= types
    assert result.chart_data[0]["chart_type"] == "coefficient_plot"
    for chart in (*choice.decision.chart_data, *result.chart_data):
        json.dumps(chart)
    json.dumps(choice.decision.evidence)


def test_single_numeric_predictor_adds_scatter_with_fit(tmp_path, store):
    choice, _ = run(known_data(), tmp_path, store, ["x1"])
    assert "scatter_with_fit" in {c["chart_type"] for c in choice.decision.chart_data}