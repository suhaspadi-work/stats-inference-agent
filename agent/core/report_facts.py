"""
Report facts. Turns a stored analysis result into a labeled block of facts
computed by code. The report model restates these facts in prose; it never
works out which group is higher or in which direction a difference runs.
Pure functions with no imports from the report code.

Structured helpers (multi_group_pairs, regression_terms) return plain dicts so
tests can check direction and size without parsing text.

Judgement calls, easy to change:
  MAX_LISTED: at most this many pairs or terms are listed in the narrative;
              the facts always state the full count and say when some are not listed.
  The intercept is not interpreted (zero is often not a meaningful value).
  A numeric column whose only values are 0 and 1 reads as "1 versus 0".
  Coefficients are never ranked by importance (different units).
"""
from __future__ import annotations
import math

MAX_LISTED = 8


def _decimals_for(values, minimum: int = 2) -> int:
    """Decimal places so that the smallest-scale numbers in a block keep about three significant digits."""
    nonzero = []
    for v in values:
        if v is None:
            continue
        v = float(v)
        if math.isfinite(v) and v != 0:
            nonzero.append(abs(v))
    if not nonzero:
        return minimum
    return max(minimum, min(8, 2 - math.floor(math.log10(max(nonzero)))))


def _fmt(x, d: int = 2) -> str:
    x = float(x)
    places = max(d, 4) if (x != 0 and abs(x) < 0.01) else d
    if x != 0 and abs(x) < 0.5 * 10 ** (-places):
        return f"{x:.3g}"
    return f"{x:.{places}f}"


def _signed(x, d: int = 2) -> str:
    x = float(x)
    if x > 0:
        return "+" + _fmt(x, d)
    if x < 0:
        return "-" + _fmt(abs(x), d)
    return _fmt(x, d)


def _p(p) -> str:
    p = float(p)
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}"


# ------------------------------------------------------------------ two-group

def two_group_facts(test_result: dict, outcome_name: str, predictor_name: str) -> str:
    """
    Differences are always group B minus group A, where B is the second group in
    sorted order. Labels come from the result; older results fall back to
    "Group A" and "Group B".
    """
    a = test_result.get("group_a_label", "Group A")
    b = test_result.get("group_b_label", "Group B")
    a_mean = float(test_result["group_a_mean"])
    b_mean = float(test_result["group_b_mean"])
    diff = b_mean - a_mean
    lo, hi = (float(v) for v in test_result["confidence_interval"])
    excludes_zero = lo > 0 or hi < 0
    d = _decimals_for([a_mean, b_mean, diff, lo, hi])

    if diff > 0:
        higher = f"Higher mean: {b}, by {_fmt(diff, d)}"
    elif diff < 0:
        higher = f"Higher mean: {a}, by {_fmt(-diff, d)}"
    else:
        higher = "Means are equal"

    if test_result["method"] == "mann_whitney_u":
        effect = f"Effect size (rank-biserial correlation, positive when {b} tends higher): {float(test_result['effect_size']):.3f}"
    else:
        effect = f"Effect size (Cohen's d, positive when {b} is higher): {float(test_result['effect_size']):.3f}"

    lines = [
        f"Comparison of {outcome_name} between {a} and {b} (grouped by {predictor_name})",
        f"{a}: n={int(test_result['group_a_n'])}, mean={_fmt(a_mean, d)}",
        f"{b}: n={int(test_result['group_b_n'])}, mean={_fmt(b_mean, d)}",
        higher,
        f"Difference ({b} minus {a}): {_signed(diff, d)}, 95% CI ({_fmt(lo, d)}, {_fmt(hi, d)})",
        f"The interval excludes zero: {'yes' if excludes_zero else 'no'}",
        f"Method: {test_result['method']}",
        f"p-value: {_p(test_result['p_value'])}",
        effect,
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------- multi-group

def multi_group_pairs(test_result: dict) -> list[dict]:
    """
    One dict per post-hoc pair: which group is higher, by how much (from the group
    means, which are authoritative), the interval for (higher minus lower) when the
    post-hoc test produced a consistent one, and the adjusted p-value.
    """
    means = {str(k): float(v) for k, v in test_result["group_means"].items()}
    out = []
    for c in test_result.get("pairwise_comparisons", []):
        a, b = str(c["group_a"]), str(c["group_b"])
        diff = means[b] - means[a]
        ci = None
        lo, hi = c.get("ci_lower"), c.get("ci_upper")
        if lo is not None and hi is not None and math.isfinite(float(lo)) and math.isfinite(float(hi)):
            reported = float(c.get("mean_diff", diff))
            if abs(reported - diff) <= 1e-6 * max(1.0, abs(diff)):
                ci = (float(lo), float(hi)) if diff >= 0 else (-float(hi), -float(lo))
        if diff > 0:
            higher, lower, size = b, a, diff
        elif diff < 0:
            higher, lower, size = a, b, -diff
        else:
            higher, lower, size = None, None, 0.0
        out.append({
            "group_a": a, "group_b": b, "higher": higher, "lower": lower, "size": size,
            "ci_higher_minus_lower": ci, "p_value": float(c["p_value"]), "significant": bool(c["significant"]),
        })
    return out


def _pair_sentence(p: dict, d: int) -> str:
    if p["higher"] is None:
        base = f"{p['group_a']} and {p['group_b']} have equal means"
    else:
        base = f"{p['higher']} is higher than {p['lower']} by {_fmt(p['size'], d)}"
    if p["ci_higher_minus_lower"] is not None:
        lo, hi = p["ci_higher_minus_lower"]
        base += f" (95% CI {_fmt(lo, d)} to {_fmt(hi, d)})"
    return f"{base}; adjusted p-value {_p(p['p_value'])}"


def multi_group_facts(test_result: dict, outcome_name: str, predictor_name: str, max_listed: int = MAX_LISTED) -> str:
    ns = {str(k): int(v) for k, v in test_result["group_ns"].items()}
    means = {str(k): float(v) for k, v in test_result["group_means"].items()}
    ranked = sorted(means, key=lambda g: (-means[g], g))
    pairs = multi_group_pairs(test_result)
    d = _decimals_for(list(means.values()) + [p["size"] for p in pairs])

    lines = [
        f"Comparison of {outcome_name} across {len(means)} groups of {predictor_name}",
        "Groups ranked by mean, highest first:",
    ]
    for i, g in enumerate(ranked, 1):
        lines.append(f"  {i}. {g}: n={ns[g]}, mean={_fmt(means[g], d)}")
    lines.append(f"Omnibus test: {test_result['method']}, p-value: {_p(test_result['p_value'])}")

    if not pairs:
        lines.append("No post-hoc comparisons were run (they only run when the omnibus test is significant).")
        return "\n".join(lines)

    sig = sorted((p for p in pairs if p["significant"]), key=lambda p: (p["p_value"], p["group_a"], p["group_b"]))
    non = [p for p in pairs if not p["significant"]]
    lines.append(
        f"Post-hoc test: {test_result['post_hoc_method']}; {len(sig)} of {len(pairs)} pairs differ significantly "
        f"(p < 0.05, adjusted for multiple comparisons)"
    )
    for p in sig[:max_listed]:
        lines.append("  - " + _pair_sentence(p, d))
    if len(sig) > max_listed:
        lines.append(f"  ... and {len(sig) - max_listed} more significant pairs not listed")
    if non:
        names = [f"{p['group_a']} and {p['group_b']}" for p in non[:max_listed]]
        more = f", and {len(non) - max_listed} more pairs" if len(non) > max_listed else ""
        lines.append("Not significantly different: " + "; ".join(names) + more)
    return "\n".join(lines)


# ------------------------------------------------------------------ regression

def regression_terms(test_result: dict) -> list[dict]:
    """One dict per non-intercept term, with direction and detectability from the coefficient's own fields."""
    out = []
    for c in test_result["coefficients"]:
        kind = c.get("kind", "numeric")
        if kind == "intercept" or c["name"] == "const":
            continue
        est, lo, hi = float(c["estimate"]), float(c["ci_lower"]), float(c["ci_upper"])
        out.append({
            "name": c["name"], "kind": kind, "column": c.get("column"), "level": c.get("level"),
            "reference": c.get("reference"), "estimate": est, "size": abs(est),
            "direction": "higher" if est > 0 else ("lower" if est < 0 else "equal"),
            "ci": (lo, hi), "p_value": float(c["p_value"]), "detectable": lo > 0 or hi < 0,
            "vif": c.get("vif"),
        })
    return out


def _term_sentence(t: dict, outcome: str, d: int) -> str:
    col = t["column"] or t["name"]
    if t["direction"] == "equal":
        return f"{col}: no association (estimate is exactly 0)"
    size, dirn, kind = _fmt(t["size"], d), t["direction"], t["kind"]
    if kind == "binary":
        core = f"{col}: cases with {col} = 1 have {size} {dirn} {outcome} than cases with {col} = 0"
    elif kind == "level":
        core = f"{col}: {t['level']} has {size} {dirn} {outcome} than {t['reference']}"
    elif kind == "level_vs_mean":
        core = f"{col}: {t['level']} has {size} {dirn} {outcome} than the average of all levels"
    elif kind == "ordinal":
        core = f"{col}: each step up the ordered scale is associated with {size} {dirn} {outcome}"
    else:
        core = f"{col}: each additional unit is associated with {size} {dirn} {outcome}"
    lo, hi = t["ci"]
    return f"{core} (95% CI for the coefficient {_fmt(lo, d)} to {_fmt(hi, d)}; p-value {_p(t['p_value'])})"


def regression_facts(test_result: dict, outcome_name: str, max_listed: int = MAX_LISTED) -> str:
    terms = regression_terms(test_result)
    d = _decimals_for([t["estimate"] for t in terms] + [b for t in terms for b in t["ci"]])
    detect = sorted((t for t in terms if t["detectable"]), key=lambda t: (t["p_value"], t["name"]))
    other = [t for t in terms if not t["detectable"]]
    robust = test_result["method"] == "ols_hc3_robust"

    lines = [
        f"Regression of {outcome_name} on {len(terms)} predictor terms",
        f"Observations used: {int(test_result['n_observations'])}; rows left out for missing values: {int(test_result['n_dropped_missing'])}",
        f"R-squared: {float(test_result['r_squared']):.3f} (adjusted {float(test_result['adj_r_squared']):.3f}); "
        f"overall F-test p-value: {_p(test_result['f_p_value'])}",
        "Standard errors: " + ("robust to non-constant variance (HC3)" if robust else "classical"),
        f"Each association below holds the other predictors fixed. {len(detect)} of {len(terms)} terms are "
        f"statistically detectable (95% CI excludes zero).",
    ]
    if detect:
        lines.append("Detectable associations, smallest p-value first:")
        for t in detect[:max_listed]:
            lines.append("  - " + _term_sentence(t, outcome_name, d))
        if len(detect) > max_listed:
            lines.append(f"  ... and {len(detect) - max_listed} more detectable terms not listed")
    if other:
        names = [t["name"] for t in other[:max_listed]]
        more = f", and {len(other) - max_listed} more" if len(other) > max_listed else ""
        lines.append("Not statistically detectable: " + ", ".join(names) + more)

    cautions = []
    high = [t["name"] for t in terms if t["vif"] is not None and t["vif"] > 5]
    if high:
        cautions.append(f"These terms are strongly correlated with other predictors (VIF above 5), so their individual coefficients are less reliable: {', '.join(high)}")
    if int(test_result["n_dropped_missing"]):
        cautions.append(f"{int(test_result['n_dropped_missing'])} rows were left out because of missing values")
    if len(terms) >= 2:
        cautions.append("Coefficient sizes are in each predictor's own units, so they are not a ranking of importance")
    if cautions:
        lines.append("Cautions:")
        lines.extend(f"  - {c}" for c in cautions)
    lines.append("Intercept: not interpreted, because zero may not be a meaningful value for the predictors")
    return "\n".join(lines)