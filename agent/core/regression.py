"""
Multi-predictor linear regression (OLS). Mirrors the select_*/execute_*
split used by the other capabilities: select_regression_method fits the
model to compute diagnostics, chooses between classical OLS and HC3-robust
standard errors (Breusch-Pagan decides), and chooses an encoding for each
categorical predictor, recording the reasoning and the rejected
alternatives in one MethodDecision BEFORE freeze. execute_regression then
refits with the approved method and encodings, and also refits under the
applicable alternative encodings so they can be compared side by side.

Encoding schemes:
  treatment - dummy coding against a reference level (default: the largest
              level, which gives the most stable comparisons)
  effect    - sum-to-zero coding; coefficients are each level's deviation
              from the average of all levels. Same fitted model as
              treatment, different coefficient meaning.
  ordinal   - one integer slope per step. Only used when the user supplies
              the category order; an order cannot be inferred from data.

Everything that crosses a state boundary is plain dicts/lists, never
custom objects.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

from core.dataset import Dataset, DatasetStore
from core.plan import MethodDecision, AlternativeConsidered
import core.visualization as viz

MAX_CATEGORICAL_LEVELS = 10
MIN_LEVEL_COUNT = 5
VIF_THRESHOLD = 5.0
SCHEMES = ("treatment", "effect", "ordinal")
METHODS = ("ols", "ols_hc3_robust")


@dataclass(frozen=True)
class CoefficientResult:
    name: str
    estimate: float
    std_error: float
    t_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    vif: float | None  # None for the intercept


@dataclass(frozen=True)
class AlternativeFit:
    label: str
    description: str
    same_fitted_model: bool  # True when only the coefficient parametrization differs
    r_squared: float
    adj_r_squared: float
    aic: float
    coefficients: tuple[CoefficientResult, ...]


@dataclass(frozen=True)
class RegressionResult:
    method: str  # "ols" or "ols_hc3_robust"
    n_observations: int
    n_dropped_missing: int
    r_squared: float
    adj_r_squared: float
    aic: float
    f_statistic: float
    f_p_value: float
    coefficients: tuple[CoefficientResult, ...]
    encodings: dict
    alternative_fits: tuple[AlternativeFit, ...]
    chart_data: tuple[dict, ...]


@dataclass(frozen=True)
class RegressionMethodChoice:
    method: str
    encodings: dict  # {column: {"scheme", "reference", "order", "source"}} -- plain dicts
    decision: MethodDecision


# ---------------------------------------------------------------- preparation

def _clean(df: pd.DataFrame, outcome_col: str, predictors: list[str]):
    """Validates inputs, drops incomplete rows, and profiles categorical predictors."""
    predictors = list(predictors)
    if not predictors:
        raise ValueError("Regression requires at least one predictor column.")
    if len(set(predictors)) != len(predictors):
        raise ValueError(f"Predictor list contains duplicates: {predictors}")
    missing = [c for c in [outcome_col, *predictors] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in the dataset: {missing}")
    if outcome_col in predictors:
        raise ValueError(f"The outcome column '{outcome_col}' cannot also be a predictor.")
    if not pd.api.types.is_numeric_dtype(df[outcome_col]):
        raise ValueError(
            f"Outcome column '{outcome_col}' has data type '{df[outcome_col].dtype}', "
            f"which cannot be used as the outcome of a linear regression."
        )

    sub = df[[outcome_col, *predictors]]
    before = len(sub)
    sub = sub.dropna()
    n_dropped = before - len(sub)
    if sub.empty:
        raise ValueError("No complete rows remain after dropping rows with missing values.")

    categorical = [p for p in predictors if not pd.api.types.is_numeric_dtype(sub[p])]
    level_counts = {}
    for p in categorical:
        counts = {str(k): int(v) for k, v in sub[p].astype(str).value_counts().items()}
        if len(counts) > MAX_CATEGORICAL_LEVELS:
            raise ValueError(
                f"Predictor '{p}' has {len(counts)} distinct values, more than the {MAX_CATEGORICAL_LEVELS} "
                f"supported for a categorical predictor. It may be an identifier or need grouping first."
            )
        if len(counts) < 2:
            raise ValueError(f"Predictor '{p}' has only one distinct value, so it cannot explain any variation.")
        level_counts[p] = counts
    return sub, n_dropped, categorical, level_counts


def _largest_level(counts: dict) -> str:
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _resolve_encodings(categorical: list[str], level_counts: dict, overrides: dict | None) -> dict:
    """Per-column encoding: user override if given, otherwise treatment coding against the largest level."""
    overrides = overrides or {}
    unknown = [c for c in overrides if c not in categorical]
    if unknown:
        raise ValueError(f"Encoding overrides were given for columns that are not categorical predictors: {unknown}")

    specs = {}
    for col in categorical:
        counts = level_counts[col]
        given = overrides.get(col)
        o = given or {}
        scheme = o.get("scheme", "treatment")
        if scheme not in SCHEMES:
            raise ValueError(f"Unknown encoding scheme '{scheme}' for '{col}'. Choose from {SCHEMES}.")
        source = "user" if given else "default"

        if scheme == "ordinal":
            order = o.get("order")
            if not order:
                raise ValueError(
                    f"Ordinal encoding for '{col}' requires an explicit 'order' (lowest to highest); "
                    f"an order cannot be inferred from the data."
                )
            order = [str(v) for v in order]
            if len(set(order)) != len(order):
                raise ValueError(f"The supplied order for '{col}' contains duplicates: {order}")
            absent = [lvl for lvl in counts if lvl not in order]
            if absent:
                raise ValueError(f"Levels {absent} of '{col}' are not in the supplied order {order}.")
            specs[col] = {"scheme": "ordinal", "reference": None, "order": order, "source": source}
        else:
            ref = str(o.get("reference", _largest_level(counts)))
            if ref not in counts:
                raise ValueError(f"Reference level '{ref}' is not a level of '{col}' (levels: {sorted(counts)}).")
            specs[col] = {"scheme": scheme, "reference": ref, "order": None, "source": source}
    return specs


def _design(sub: pd.DataFrame, outcome_col: str, predictors: list[str], specs: dict):
    """Builds (y, X) with an intercept. Column names state what each coefficient compares."""
    cols = {}
    for p in predictors:
        if p not in specs:
            cols[p] = sub[p].astype(float)
            continue
        spec = specs[p]
        s = sub[p].astype(str)
        levels = sorted(s.unique())
        if spec["scheme"] == "treatment":
            for lvl in levels:
                if lvl != spec["reference"]:
                    cols[f"{p}[{lvl} vs {spec['reference']}]"] = (s == lvl).astype(float)
        elif spec["scheme"] == "effect":
            for lvl in levels:
                if lvl != spec["reference"]:  # the reference level is the omitted level
                    cols[f"{p}[{lvl} vs mean]"] = (s == lvl).astype(float) - (s == spec["reference"]).astype(float)
        else:
            position = {lvl: i for i, lvl in enumerate(spec["order"])}
            cols[f"{p}[per step]"] = s.map(position).astype(float)
    X = pd.DataFrame(cols, index=sub.index)
    X = sm.add_constant(X, has_constant="add")
    y = sub[outcome_col].astype(float)
    return y, X


def _validate_fit(y: pd.Series, X: pd.DataFrame) -> None:
    if len(y) <= X.shape[1] + 1:
        raise ValueError(
            f"Not enough observations ({len(y)}) to fit {X.shape[1]} parameters after dropping rows with missing values."
        )
    if float(y.std()) == 0.0:
        raise ValueError(
            f"Cannot fit a regression: outcome column '{y.name}' has zero variance (every value is identical)."
        )
    if np.linalg.matrix_rank(X.values) < X.shape[1]:
        raise ValueError(
            "Cannot fit a regression: two or more predictors are perfectly collinear "
            "(one is an exact function of others), or a predictor is constant."
        )


def _fit(y, X, robust: bool):
    model = sm.OLS(y, X)
    return model.fit(cov_type="HC3") if robust else model.fit()


def _vifs(X: pd.DataFrame) -> dict:
    return {
        col: float(variance_inflation_factor(X.values, i))
        for i, col in enumerate(X.columns) if col != "const"
    }


def _coef_results(res, X: pd.DataFrame) -> tuple:
    vifs = _vifs(X)
    ci = res.conf_int()
    return tuple(
        CoefficientResult(
            name=str(name), estimate=float(res.params[name]), std_error=float(res.bse[name]),
            t_statistic=float(res.tvalues[name]), p_value=float(res.pvalues[name]),
            ci_lower=float(ci.loc[name, 0]), ci_upper=float(ci.loc[name, 1]),
            vif=None if name == "const" else vifs[name],
        )
        for name in X.columns
    )


# ------------------------------------------------------------ method selection

def _encoding_alternatives_for_decision(col: str, spec: dict) -> list:
    if spec["scheme"] == "treatment":
        return [
            AlternativeConsidered(
                method=f"effect_coding:{col}",
                rejected_because=(
                    "gives the same fitted model, but coefficients become deviations from the average of all "
                    "levels, which is harder to explain than a comparison against a named baseline"
                ),
            ),
            AlternativeConsidered(
                method=f"ordinal_encoding:{col}",
                rejected_because="no category order was supplied, and an order cannot be inferred from the data",
            ),
        ]
    if spec["scheme"] == "effect":
        return [AlternativeConsidered(
            method=f"treatment_coding:{col}",
            rejected_because="effect coding was explicitly requested for this column",
        )]
    return [AlternativeConsidered(
        method=f"treatment_coding:{col}",
        rejected_because=(
            "an order was supplied, so one slope per step is used; it needs fewer parameters than separate level "
            "coefficients but assumes equal spacing between steps"
        ),
    )]


def select_regression_method(
    dataset: Dataset, store: DatasetStore, outcome_col: str, predictors: list[str],
    encoding_overrides: dict | None = None,
) -> RegressionMethodChoice:
    df = store.read(dataset.storage_key)
    sub, n_dropped, categorical, level_counts = _clean(df, outcome_col, predictors)
    encodings = _resolve_encodings(categorical, level_counts, encoding_overrides)
    y, X = _design(sub, outcome_col, list(predictors), encodings)
    _validate_fit(y, X)

    ols = _fit(y, X, robust=False)
    residuals = np.asarray(ols.resid)
    fitted = np.asarray(ols.fittedvalues)

    bp_p = float(het_breuschpagan(residuals, np.asarray(X.values))[1])
    shapiro_p = float(stats.shapiro(residuals)[1]) if len(residuals) >= 3 else 1.0
    vifs = _vifs(X)
    high_vif = sorted(c for c, v in vifs.items() if v > VIF_THRESHOLD)
    rare_levels = {
        col: sorted(lvl for lvl, n in counts.items() if n < MIN_LEVEL_COUNT)
        for col, counts in level_counts.items()
    }
    rare_levels = {c: lv for c, lv in rare_levels.items() if lv}

    robust = bp_p <= 0.05
    if robust:
        method = "ols_hc3_robust"
        rationale = (
            f"The Breusch-Pagan test (p={bp_p:.4f}) indicates the residual variance is not constant, "
            f"so ordinary least squares is used with heteroscedasticity-robust (HC3) standard errors, "
            f"which keep p-values and confidence intervals reliable."
        )
        alternatives = [AlternativeConsidered(
            method="ols",
            rejected_because="Breusch-Pagan indicates non-constant residual variance, so classical standard errors would be unreliable",
        )]
    else:
        method = "ols"
        rationale = (
            f"The Breusch-Pagan test (p={bp_p:.4f}) finds no evidence of non-constant residual variance, "
            f"so ordinary least squares with classical standard errors is appropriate."
        )
        alternatives = [AlternativeConsidered(
            method="ols_hc3_robust",
            rejected_because="no evidence of heteroscedasticity, so classical standard errors are valid and more efficient",
        )]

    for col in categorical:
        spec = encodings[col]
        counts = level_counts[col]
        if spec["scheme"] == "ordinal":
            rationale += f" '{col}' is coded as an ordered scale ({' < '.join(spec['order'])}) using the order you supplied."
        elif spec["scheme"] == "effect":
            rationale += f" '{col}' uses effect coding: each coefficient is a level's deviation from the average of all levels."
        else:
            ref = spec["reference"]
            why = (
                f"the largest group (n={counts[ref]}), which gives the most stable comparisons"
                if spec["source"] == "default" else "the level you chose"
            )
            rationale += f" '{col}' uses treatment coding against '{ref}', {why}; each coefficient is a difference from that level."
        alternatives.extend(_encoding_alternatives_for_decision(col, spec))

    if col_rare := [c for c in rare_levels]:
        rationale += (
            f" Level(s) with very few observations (under {MIN_LEVEL_COUNT}) in {col_rare} give unstable "
            f"coefficients and may need grouping."
        )
    if shapiro_p <= 0.05:
        rationale += (
            f" Residuals depart from normality (Shapiro-Wilk p={shapiro_p:.4f}); coefficient estimates "
            f"remain valid, but p-values and intervals are less reliable with small samples."
        )
    if high_vif:
        rationale += (
            f" Predictor(s) {high_vif} show high multicollinearity (VIF above {VIF_THRESHOLD:g}), "
            f"so their individual coefficients should be interpreted with caution."
        )

    evidence = {
        "n_observations": int(len(y)),
        "n_dropped_missing": int(n_dropped),
        "n_parameters": int(X.shape[1]),
        "breusch_pagan_p": round(bp_p, 4),
        "shapiro_p_residuals": round(shapiro_p, 4),
        "max_vif": round(max(vifs.values()), 4) if vifs else 0.0,
        "high_vif_columns": high_vif,
        "rare_levels": rare_levels,
        "encodings": {
            col: {**encodings[col], "levels": level_counts[col]} for col in categorical
        },
    }

    infl = ols.get_influence()
    charts = [
        viz.residual_plot(fitted, residuals),
        viz.residual_qq_plot(residuals),
        viz.residuals_vs_leverage(np.asarray(infl.hat_matrix_diag), np.asarray(infl.resid_studentized_internal)),
    ]
    if len(predictors) == 1 and predictors[0] not in encodings:
        xcol = predictors[0]
        x_vals = X[xcol].to_numpy()
        fit_x = np.array([x_vals.min(), x_vals.max()])
        fit_y = float(ols.params["const"]) + float(ols.params[xcol]) * fit_x
        charts.insert(0, viz.scatter_with_fit(x_vals, y.to_numpy(), fit_x, fit_y, xcol, outcome_col))

    decision = MethodDecision(
        stage="test_selection", chosen_method=method, rationale=rationale,
        alternatives=tuple(alternatives), evidence=evidence, chart_data=tuple(charts),
    )
    return RegressionMethodChoice(method=method, encodings=encodings, decision=decision)


# -------------------------------------------------------------------- execution

def _alternative_encodings(encodings: dict, level_counts: dict) -> list:
    """Applicable alternative encodings as (label, description, same_fitted_model, specs)."""
    alts = []
    schemes = {s["scheme"] for s in encodings.values()}

    if "treatment" in schemes:
        specs = {c: ({**s, "scheme": "effect"} if s["scheme"] == "treatment" else s) for c, s in encodings.items()}
        alts.append((
            "effect_coding",
            "Same model with treatment-coded columns switched to effect coding: coefficients become deviations "
            "from the average of all levels. R-squared, AIC and predictions are identical.",
            True, specs,
        ))
    if "effect" in schemes:
        specs = {c: ({**s, "scheme": "treatment"} if s["scheme"] == "effect" else s) for c, s in encodings.items()}
        alts.append((
            "treatment_coding",
            "Same model with effect-coded columns switched to treatment coding against a reference level. "
            "R-squared, AIC and predictions are identical.",
            True, specs,
        ))
    if "ordinal" in schemes:
        specs = {
            c: ({"scheme": "treatment", "reference": _largest_level(level_counts[c]), "order": None, "source": "alternative"}
                if s["scheme"] == "ordinal" else s)
            for c, s in encodings.items()
        }
        alts.append((
            "treatment_instead_of_ordinal",
            "Ordered columns refit as separate level comparisons instead of one slope per step. This is a "
            "genuinely different model (more parameters), so compare adjusted R-squared and AIC.",
            False, specs,
        ))
    return alts


def execute_regression(
    dataset: Dataset, store: DatasetStore, outcome_col: str, predictors: list[str],
    method: str, encodings: dict | None = None,
) -> RegressionResult:
    if method not in METHODS:
        raise ValueError(f"Unknown regression method '{method}'. Choose from {METHODS}.")
    df = store.read(dataset.storage_key)
    sub, n_dropped, categorical, level_counts = _clean(df, outcome_col, predictors)
    if encodings is None:
        encodings = _resolve_encodings(categorical, level_counts, None)
    elif set(encodings) != set(categorical):
        raise ValueError(
            f"Encodings were given for {sorted(encodings)} but the categorical predictors are {sorted(categorical)}."
        )

    robust = method == "ols_hc3_robust"
    y, X = _design(sub, outcome_col, list(predictors), encodings)
    _validate_fit(y, X)
    res = _fit(y, X, robust=robust)
    coefficients = _coef_results(res, X)

    alternative_fits = []
    for label, description, same_model, alt_specs in _alternative_encodings(encodings, level_counts):
        ya, Xa = _design(sub, outcome_col, list(predictors), alt_specs)
        _validate_fit(ya, Xa)
        ra = _fit(ya, Xa, robust=robust)
        alternative_fits.append(AlternativeFit(
            label=label, description=description, same_fitted_model=same_model,
            r_squared=float(ra.rsquared), adj_r_squared=float(ra.rsquared_adj), aic=float(ra.aic),
            coefficients=_coef_results(ra, Xa),
        ))

    non_intercept = [c for c in coefficients if c.name != "const"]
    chart = viz.coefficient_plot(
        [c.name for c in non_intercept], [c.estimate for c in non_intercept],
        [c.ci_lower for c in non_intercept], [c.ci_upper for c in non_intercept],
    )
    return RegressionResult(
        method=method, n_observations=int(len(y)), n_dropped_missing=int(n_dropped),
        r_squared=float(res.rsquared), adj_r_squared=float(res.rsquared_adj), aic=float(res.aic),
        f_statistic=float(np.squeeze(res.fvalue)), f_p_value=float(np.squeeze(res.f_pvalue)),
        coefficients=coefficients, encodings=encodings,
        alternative_fits=tuple(alternative_fits), chart_data=(chart,),
    )