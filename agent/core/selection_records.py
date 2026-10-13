"""
Decision records for test selection (M0.2 step 5, G15). Pure functions: no
graph, no model calls, no data access. Each record is built from the
MethodDecision the selector already produced; eligibility and every reason
are derived from the numbers in its evidence, never from canned text.

This module must not import core.capabilities: capabilities imports it.
"""
from __future__ import annotations
import copy

from core.decision_record import Candidate, DecisionRecord
from core.plan import MethodDecision

TEST_SELECTION_POINT = "stage2.test_selection"

# Mirrors the 0.05 threshold used in testing.select_two_group_method;
# a test in tests/test_selection_records.py guards that they agree.
ALPHA = 0.05

# Below this many observations the selector skips Shapiro-Wilk and assigns p = 1.0.
_MIN_SHAPIRO_N = 3

TWO_GROUP_METHODS = ("student_t_test", "welch_t_test", "mann_whitney_u")
_TWO_GROUP_REQUIRED = (
    "shapiro_p_group_a", "shapiro_p_group_b", "levene_p",
    "n_group_a", "n_group_b", "group_a_label", "group_b_label",
)


def _p(p: float) -> str:
    """'p = 0.2100', or 'p < 0.0001' when the 4-decimal value is 0."""
    p = float(p)
    return "p < 0.0001" if p == 0 else f"p = {p:.4f}"


class _Group:
    def __init__(self, label, n, shapiro_p):
        self.label, self.n, self.shapiro_p = str(label), int(n), float(shapiro_p)

    @property
    def checked(self) -> bool:
        return self.n >= _MIN_SHAPIRO_N

    @property
    def passes(self) -> bool:
        return not self.checked or self.shapiro_p > ALPHA

    def describe(self) -> str:
        if not self.checked:
            return f"{self.label} n = {self.n}: normality not checked, treated as passing"
        return f"{self.label} n={self.n}: Shapiro-Wilk {_p(self.shapiro_p)}"

    def describe_failure(self) -> str:
        return f"{self.label} fails the normality check (n={self.n}, Shapiro-Wilk {_p(self.shapiro_p)})"


def _two_group_reasons(groups, levene_p, chosen):
    normality_ok = all(g.passes for g in groups)
    equal_var_ok = levene_p > ALPHA
    both_pass = f"both groups pass the normality check ({'; '.join(g.describe() for g in groups)})"
    failures = " and ".join(g.describe_failure() for g in groups if not g.passes)
    levene_above = f"Levene's {_p(levene_p)} is above {ALPHA}"
    levene_at_or_below = f"Levene's {_p(levene_p)} is at or below {ALPHA}"

    def student():
        if not normality_ok:
            reason = f"not eligible: assumes normality, but {failures}"
            if not equal_var_ok:
                reason += f"; it also assumes equal variances, but {levene_at_or_below}"
            return False, reason
        if not equal_var_ok:
            return False, (f"not eligible: assumes equal variances, but {levene_at_or_below}, "
                           f"so Welch's t-test is used")
        if chosen == "student_t_test":
            return True, f"chosen: {both_pass} and {levene_above}"
        return True, "eligible; not chosen by the selection rule"

    def welch():
        if not normality_ok:
            return False, f"not eligible: assumes normality, but {failures}"
        if chosen == "welch_t_test" and not equal_var_ok:
            return True, f"chosen: {both_pass} and {levene_at_or_below}, so variances are treated as unequal"
        if equal_var_ok:
            return True, (f"eligible, but the variances are equal ({levene_above}), "
                          f"so the standard t-test is used")
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
    missing = [k for k in _TWO_GROUP_REQUIRED if k not in ev]
    if missing:
        raise ValueError(f"two_group_selection_record: evidence is missing key(s): {', '.join(missing)}")
    if decision.chosen_method not in TWO_GROUP_METHODS:
        raise ValueError(
            f"two_group_selection_record: unknown method {decision.chosen_method!r}; "
            f"expected one of {TWO_GROUP_METHODS}"
        )

    groups = (
        _Group(ev["group_a_label"], ev["n_group_a"], ev["shapiro_p_group_a"]),
        _Group(ev["group_b_label"], ev["n_group_b"], ev["shapiro_p_group_b"]),
    )
    levene_p = float(ev["levene_p"])
    reasons = _two_group_reasons(groups, levene_p, decision.chosen_method)

    shapiro = {"shapiro_p_group_a": float(ev["shapiro_p_group_a"]),
               "shapiro_p_group_b": float(ev["shapiro_p_group_b"])}
    candidate_evidence = {
        "student_t_test": {**shapiro, "levene_p": levene_p},
        "welch_t_test": dict(shapiro),
        "mann_whitney_u": {},
    }
    candidates = tuple(
        Candidate(option=m, eligible=reasons[m][0], reason=reasons[m][1], evidence=candidate_evidence[m])
        for m in TWO_GROUP_METHODS
    )

    evidence = copy.deepcopy(dict(ev))
    evidence["family"] = "two_group"
    evidence["small_groups_unchecked"] = [g.label for g in groups if not g.checked]

    # chart_data is deliberately not copied: the charts hold raw values (M0.5 guard decision).
    return DecisionRecord(
        decision_point=TEST_SELECTION_POINT, chosen=decision.chosen_method, decided_by="code_rule",
        candidates=candidates, rationale=decision.rationale, evidence=evidence,
    )
