"""
Report facts (M0.2a). Turns a stored analysis result into a labeled block of
facts computed by code. The report model restates these facts in prose; it
never works out which group is higher, or in which direction a difference
runs. Pure functions with no imports from the report code, so they are
trivial to test and every capability can have its own.
"""
from __future__ import annotations


def _fmt(x: float) -> str:
    x = float(x)
    if x != 0 and abs(x) < 0.01:
        return f"{x:.4f}"
    return f"{x:.2f}"


def _signed(x: float) -> str:
    x = float(x)
    if x > 0:
        return "+" + _fmt(x)
    if x < 0:
        return "-" + _fmt(abs(x))
    return _fmt(x)


def _p(p: float) -> str:
    p = float(p)
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}"


def two_group_facts(test_result: dict, outcome_name: str, predictor_name: str) -> str:
    """
    Facts for a two-group result. Differences are always group B minus group A,
    where B is the second group in sorted order. The labels come from the result;
    older results without labels fall back to "Group A" and "Group B".
    """
    a = test_result.get("group_a_label", "Group A")
    b = test_result.get("group_b_label", "Group B")
    a_mean = float(test_result["group_a_mean"])
    b_mean = float(test_result["group_b_mean"])
    diff = b_mean - a_mean
    lo, hi = (float(v) for v in test_result["confidence_interval"])
    excludes_zero = lo > 0 or hi < 0

    if diff > 0:
        higher = f"Higher mean: {b}, by {_fmt(diff)}"
    elif diff < 0:
        higher = f"Higher mean: {a}, by {_fmt(-diff)}"
    else:
        higher = "Means are equal"

    if test_result["method"] == "mann_whitney_u":
        effect = f"Effect size (rank-biserial correlation, positive when {b} tends higher): {float(test_result['effect_size']):.3f}"
    else:
        effect = f"Effect size (Cohen's d, positive when {b} is higher): {float(test_result['effect_size']):.3f}"

    lines = [
        f"Comparison of {outcome_name} between {a} and {b} (grouped by {predictor_name})",
        f"{a}: n={int(test_result['group_a_n'])}, mean={_fmt(a_mean)}",
        f"{b}: n={int(test_result['group_b_n'])}, mean={_fmt(b_mean)}",
        higher,
        f"Difference ({b} minus {a}): {_signed(diff)}, 95% CI ({_fmt(lo)}, {_fmt(hi)})",
        f"The interval excludes zero: {'yes' if excludes_zero else 'no'}",
        f"Method: {test_result['method']}",
        f"p-value: {_p(test_result['p_value'])}",
        effect,
    ]
    return "\n".join(lines)