"""
Decision records for test selection (M0.2 steps 5 to 7, G15, G30). Pure
functions: no graph, no model calls, no data access. Each record is built
from the MethodDecision the selector already produced. Every pass/fail is
read from the selector's own flags, which it computed from the unrounded
p-values; the 4-decimal p-values are only quoted in the reasons.

This module must not import core.capabilities: capabilities imports it.
"""
from __future__ import annotations
import copy

from core.decision_record import Candidate, DecisionRecord
from core.plan import MethodDecision

TEST_SELECTION_POINT = "stage2.test_selection"

# Mirrors the 0.05 threshold used in the selectors; used only for the words in
# the reasons. Source-text guards in the tests check that the selectors agree.
ALPHA = 0.05

# Below this many observations the selectors skip Shapiro-Wilk and assign p = 1.0.
_MIN_SHAPIRO_N = 3

_ROUNDED_NOTE = " [rounded; the selector used the unrounded value]"

TWO_GROUP_METHODS = ("student_t_test", "welch_t_test", "mann_whitney_u")
_TWO_GROUP_REQUIRED = (
    "shapiro_p_group_a", "shapiro_p_group_b", "levene_p",
    "n_group_a", "n_group_b", "group_a_label", "group_b_label",
    "group_a_passes_normality", "group_b_passes_normality", "variances_equal",
)

MULTI_GROUP_METHODS = ("one_way_anova", "kruskal_wallis")
_MULTI_GROUP_REQUIRED = (
    "group_labels", "n_by_group", "normality_p_by_group",
    "passes_normality_by_group", "levene_p", "variances_equal",
)
_POST_HOC = {"one_way_anova": "tukey_hsd", "kruskal_wallis": "dunn"}


def _p(p: float) -> str:
    """'p = 0.2100', or 'p < 0.0001' when the 4-decimal value is 0, with a note when it reads exactly 0.0500."""
    p = float(p)
    if p == 0:
        return "p < 0.0001"
    text = f"p = {p:.4f}"
    return text + _ROUNDED_NOTE if f"{p:.4f}" == f"{ALPHA:.4f}" else text


def _require(ev: dict, required: tuple[str, ...], builder: str) -> None:
    missing = [k for k in required if k not in ev]
    if missing:
        raise ValueError(f"{builder}: evidence is missing key(s): {', '.join(missing)}")


def _require_method(method: str, allowed: tuple[str, ...], builder: str) -> None:
    if method not in allowed:
        raise ValueError(f"{builder}: unknown method {method!r}; expected one of {allowed}")


class _Group:
    def __init__(self, label, n, shapiro_p, passes):
        self.label, self.n, self.shapiro_p = str(label), int(n), float(shapiro_p)
        self.passes = bool(passes)

    @property
    def checked(self) -> bool:
        return self.n >= _MIN_SHAPIRO_N

    def describe(self) -> str:
        if not self.checked:
            return f"{self.label} n = {self.n}: normality not checked, treated as passing"
        return f"{self.label} n={self.n}: Shapiro-Wilk {_p(self.shapiro_p)}"

    def describe_failure(self) -> str:
        return f"{self.label} fails the normality check (n={self.n}, Shapiro-Wilk {_p(self.shapiro_p)})"


def _levene_text(levene_p: float, equal: bool) -> str:
    return f"Levene's {_p(levene_p)} is {'above' if equal else 'at or below'} {ALPHA}"


def _record(decision: MethodDecision, candidates, family: str, groups, extra_evidence=None) -> DecisionRecord:
    evidence = copy.deepcopy(dict(decision.evidence))
    evidence["family"] = family
    evidence["small_groups_unchecked"] = [g.label for g in groups if not g.checked]
    evidence.update(extra_evidence or {})
    # chart_data is deliberately not copied: the charts hold raw values (M0.5 guard decision).
    return DecisionRecord(
        decision_point=TEST_SELECTION_POINT, chosen=decision.chosen_method, decided_by="code_rule",
        candidates=tuple(candidates), rationale=decision.rationale, evidence=evidence,
    )


# ------------------------------------------------------------------ two-group

def _two_group_reasons(groups, levene_p, equal_var_ok, chosen):
    normality_ok = all(g.passes for g in groups)
    both_pass = f"both groups pass the normality check ({'; '.join(g.describe() for g in groups)})"
    failures = " and ".join(g.describe_failure() for g in groups if not g.passes)
    levene = _levene_text(levene_p, equal_var_ok)

    def student():
        if not normality_ok:
            reason = f"not eligible: assumes normality, but {failures}"
            if not equal_var_ok:
                reason += f"; it also assumes equal variances, but {levene}"
            return False, reason
        if not equal_var_ok:
            return False, f"not eligible: assumes equal variances, but {levene}, so Welch's t-test is used"
        if chosen == "student_t_test":
            return True, f"chosen: {both_pass} and {levene}"
        return True, "eligible; not chosen by the selection rule"

    def welch():
        if not normality_ok:
            return False, f"not eligible: assumes normality, but {failures}"
        if chosen == "welch_t_test" and not equal_var_ok:
            return True, f"chosen: {both_pass} and {levene}, so variances are treated as unequal"
        if equal_var_ok:
            return True, f"eligible, but the variances are equal ({levene}), so the standard t-test is used"
        return True, "eligible; not chosen by the selection rule"

    def mann_whitney():
        if chosen == "mann_whitney_u" and not normality_ok:
            return True, f"chosen: {failures}, so a rank-based test that assumes no distribution is used"
        if normality_ok:
            return True, (f"eligible, since it assumes no distribution, but {both_pass}, "
                          f"so the selection rule prefers a t test")
        return True, "eligible; not chosen by the selection rule"

    return {"student_t_test": student(), "welch_t_test": welch(), "mann_whitney_u": mann_whitney()}


def two_group_selection_record(decision: MethodDecision) -> DecisionRecord:
    ev = decision.evidence
    _require(ev, _TWO_GROUP_REQUIRED, "two_group_selection_record")
    _require_method(decision.chosen_method, TWO_GROUP_METHODS, "two_group_selection_record")

    groups = (
        _Group(ev["group_a_label"], ev["n_group_a"], ev["shapiro_p_group_a"], ev["group_a_passes_normality"]),
        _Group(ev["group_b_label"], ev["n_group_b"], ev["shapiro_p_group_b"], ev["group_b_passes_normality"]),
    )
    levene_p = float(ev["levene_p"])
    reasons = _two_group_reasons(groups, levene_p, bool(ev["variances_equal"]), decision.chosen_method)

    shapiro = {"shapiro_p_group_a": float(ev["shapiro_p_group_a"]),
               "shapiro_p_group_b": float(ev["shapiro_p_group_b"])}
    candidate_evidence = {
        "student_t_test": {**shapiro, "levene_p": levene_p},
        "welch_t_test": dict(shapiro),
        "mann_whitney_u": {},
    }
    candidates = [
        Candidate(option=m, eligible=reasons[m][0], reason=reasons[m][1], evidence=candidate_evidence[m])
        for m in TWO_GROUP_METHODS
    ]
    return _record(decision, candidates, "two_group", groups)


# ---------------------------------------------------------------- multi-group

def _multi_group_reasons(groups, levene_p, equal_var_ok, chosen):
    n = len(groups)
    normality_ok = all(g.passes for g in groups)
    all_pass = f"all {n} groups pass the normality check ({'; '.join(g.describe() for g in groups)})"
    failures = " and ".join(g.describe_failure() for g in groups if not g.passes)
    levene = _levene_text(levene_p, equal_var_ok)

    def anova():
        if not (normality_ok and equal_var_ok):
            parts = []
            if not normality_ok:
                parts.append(f"assumes normality, but {failures}")
            if not equal_var_ok:
                parts.append(f"assumes equal variances, but {levene}")
            return False, "not eligible: " + "; ".join(parts)
        if chosen == "one_way_anova":
            return True, (f"chosen: {all_pass} and {levene}; if the omnibus test is significant, "
                          f"post-hoc comparisons use Tukey HSD")
        return True, "eligible; not chosen by the selection rule"

    def kruskal():
        if chosen == "kruskal_wallis":
            causes = []
            if not normality_ok:
                causes.append(failures)
            if not equal_var_ok:
                causes.append(levene)
            cause = "; ".join(causes) if causes else "selected by the selection rule"
            return True, (f"chosen: {cause}; if the omnibus test is significant, "
                          f"post-hoc comparisons use Dunn's test with Holm adjustment")
        if chosen == "one_way_anova" and normality_ok and equal_var_ok:
            return True, (f"eligible, since it assumes no distribution, but {all_pass} and {levene}, "
                          f"so the selection rule prefers ANOVA")
        return True, "eligible; not chosen by the selection rule"

    return {"one_way_anova": anova(), "kruskal_wallis": kruskal()}


def multi_group_selection_record(decision: MethodDecision) -> DecisionRecord:
    ev = decision.evidence
    _require(ev, _MULTI_GROUP_REQUIRED, "multi_group_selection_record")
    _require_method(decision.chosen_method, MULTI_GROUP_METHODS, "multi_group_selection_record")

    labels = [str(label) for label in ev["group_labels"]]
    per_group = ("n_by_group", "normality_p_by_group", "passes_normality_by_group")
    absent = [f"{key}[{label!r}]" for key in per_group for label in labels if label not in ev[key]]
    if absent:
        raise ValueError(f"multi_group_selection_record: evidence is missing entries: {', '.join(absent)}")

    groups = [
        _Group(label, ev["n_by_group"][label], ev["normality_p_by_group"][label],
               ev["passes_normality_by_group"][label])
        for label in labels
    ]
    levene_p = float(ev["levene_p"])
    reasons = _multi_group_reasons(groups, levene_p, bool(ev["variances_equal"]), decision.chosen_method)

    candidate_evidence = {
        "one_way_anova": {"normality_p_by_group": {g.label: g.shapiro_p for g in groups}, "levene_p": levene_p},
        "kruskal_wallis": {},
    }
    candidates = [
        Candidate(option=m, eligible=reasons[m][0], reason=reasons[m][1], evidence=candidate_evidence[m])
        for m in MULTI_GROUP_METHODS
    ]
    return _record(decision, candidates, "multi_group", groups,
                   {"post_hoc_if_omnibus_significant": _POST_HOC[decision.chosen_method]})


# ----------------------------------------------------------------- regression

REGRESSION_METHOD_POINT = "stage3.regression_method"
ENCODING_POINT = "stage3.encoding"
REGRESSION_METHODS = ("ols", "ols_hc3_robust")
ENCODING_SCHEMES = ("treatment", "effect", "ordinal")
_REGRESSION_REQUIRED = (
    "n_observations", "n_parameters", "breusch_pagan_p", "heteroscedasticity_detected", "encodings",
)
_DEFAULT_ENCODING = "default: treatment coding against the largest level"
_NO_ORDER = "no category order was supplied, and an order cannot be inferred from the data"


def _regression_method_record(decision: MethodDecision) -> DecisionRecord:
    ev = decision.evidence
    bp = float(ev["breusch_pagan_p"])
    n = int(ev["n_observations"])
    hetero = bool(ev["heteroscedasticity_detected"])
    chosen = decision.chosen_method

    if hetero:
        ols = (False, f"not eligible: Breusch-Pagan {_p(bp)} is at or below {ALPHA}, so the residual variance "
                      f"is not constant and classical standard errors would be unreliable")
    elif chosen == "ols":
        ols = (True, f"chosen: Breusch-Pagan {_p(bp)} (n={n}) is above {ALPHA}, so classical standard errors "
                     f"are valid and more efficient")
    else:
        ols = (True, "eligible; not chosen by the selection rule")

    if chosen == "ols_hc3_robust" and hetero:
        robust = (True, f"chosen: Breusch-Pagan {_p(bp)} (n={n}) is at or below {ALPHA}, so the residual variance "
                        f"is not constant and heteroscedasticity-robust (HC3) standard errors are used")
    elif chosen == "ols" and not hetero:
        robust = (True, f"eligible, and also valid when variance is constant, but Breusch-Pagan {_p(bp)} is above "
                        f"{ALPHA}, so classical standard errors are used, which are more efficient")
    else:
        robust = (True, "eligible; not chosen by the selection rule")

    reasons = {"ols": ols, "ols_hc3_robust": robust}
    candidates = tuple(
        Candidate(option=m, eligible=reasons[m][0], reason=reasons[m][1], evidence={"breusch_pagan_p": bp})
        for m in REGRESSION_METHODS
    )
    evidence = copy.deepcopy(dict(ev))
    evidence["family"] = "regression"
    # chart_data is deliberately not copied: the charts hold raw values (M0.5 guard decision).
    return DecisionRecord(
        decision_point=REGRESSION_METHOD_POINT, chosen=chosen, decided_by="code_rule",
        candidates=candidates, rationale=decision.rationale, evidence=evidence,
    )


def _encoding_record(column: str, spec: dict) -> DecisionRecord:
    scheme = spec["scheme"]
    if scheme not in ENCODING_SCHEMES:
        raise ValueError(f"regression_selection_records: unknown encoding scheme {scheme!r} for {column!r}")
    levels = dict(spec.get("levels") or {})
    k = len(levels)
    ref = spec.get("reference")
    ref_n = int(levels[ref]) if ref is not None and ref in levels else None
    order = list(spec.get("order") or [])
    by_user = spec.get("source") == "user"

    if scheme == "treatment":
        if by_user:
            rationale = f"treatment coding against '{ref}', the level you chose"
            fact = f"treatment coding against '{ref}' (n={ref_n}), the level you chose, of {k} levels"
        else:
            rationale = (f"treatment coding against '{ref}', the largest of {k} levels (n={ref_n}); this is a "
                         f"fixed default rule and no ordering evidence was considered (G33)")
            fact = f"treatment coding against '{ref}', the largest of {k} levels (n={ref_n})"
    elif scheme == "effect":
        rationale = "effect coding was explicitly requested for this column"
        fact = f"effect coding was explicitly requested for this column; coefficients are deviations from the average of all {k} levels"
    else:
        rationale = f"ordinal coding using the supplied order {' < '.join(order)}"
        fact = f"ordinal coding using the supplied order {' < '.join(order)}; one slope per step, assuming equal spacing"

    ordinal_note = "not chosen because an order was supplied, so one slope per step is used and equal spacing is assumed"
    reasons = {}
    for option in ENCODING_SCHEMES:
        if option == scheme:
            reasons[option] = (True, f"chosen: {fact}")
        elif option == "ordinal":
            reasons[option] = (False, _NO_ORDER)
        elif scheme == "ordinal":
            reasons[option] = (True, f"eligible; {ordinal_note}")
        elif option == "effect":
            reasons[option] = (True, "eligible, and gives the same fitted model, but coefficients become deviations "
                                     "from the average of all levels; not chosen because no effect coding was "
                                     "requested for this column")
        else:  # treatment, while effect was chosen
            reasons[option] = (True, "eligible, and gives the same fitted model, but coefficients would be "
                                     "differences from one reference level; not chosen because effect coding was "
                                     "requested for this column")

    candidates = tuple(
        Candidate(option=o, eligible=reasons[o][0], reason=reasons[o][1]) for o in ENCODING_SCHEMES
    )
    return DecisionRecord(
        decision_point=ENCODING_POINT, chosen=scheme,
        decided_by="human" if by_user else "code_rule",
        overridden_by_human=by_user,
        original_choice=_DEFAULT_ENCODING if by_user else None,
        candidates=candidates, rationale=rationale,
        evidence={
            "column": column, "scheme": scheme, "reference": ref, "source": spec.get("source"),
            "n_levels": k, "reference_n": ref_n, "levels": copy.deepcopy(levels),
        },
    )


def regression_selection_records(decision: MethodDecision) -> tuple[DecisionRecord, ...]:
    """The stage3.regression_method record, then one stage3.encoding record per categorical column."""
    ev = decision.evidence
    _require(ev, _REGRESSION_REQUIRED, "regression_selection_records")
    _require_method(decision.chosen_method, REGRESSION_METHODS, "regression_selection_records")
    records = [_regression_method_record(decision)]
    records.extend(_encoding_record(str(col), spec) for col, spec in ev["encodings"].items())
    return tuple(records)
