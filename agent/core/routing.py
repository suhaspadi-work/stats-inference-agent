"""
Routing from variable types and level counts (M0.3, G1, G2). Pure: no graph,
no model calls, no data access, and no import of core.nodes.

route_for_plan reconciles the classifier's question type with what the data
supports: a comparison goes to two_group or multi_group by the grouping
column's level count, driver_analysis goes to regression, descriptive is
unchanged, and association or outcomes the analyses cannot use are reported
as unsupported with a message for the user. Nothing calls it yet.
"""
from __future__ import annotations

from core.capabilities import GROUPING_MAX_LEVELS, capability_for_question_type, evaluate_all
from core.model_safe_view import ModelSafeView

COMPARISON_TYPES = ("two_group_comparison", "ab_test", "multi_group_comparison")
_TWO_GROUP_TYPES = ("two_group_comparison", "ab_test")


def _result(capability, status, question_type, classifier_question_type, reason, message, applicability):
    return {
        "capability": capability,
        "status": status,
        "question_type": question_type,
        "classifier_question_type": classifier_question_type,
        "reason": reason,
        "message": message,
        "evidence": {"applicability": applicability, "grouping_max_levels": GROUPING_MAX_LEVELS},
    }


def _hint(evidence: dict, outcome: str, predictor_hint: str, numeric_hint: str) -> str:
    if "predictor_count" in evidence:
        return predictor_hint
    if "max_levels" in evidence:
        return f"Choose a column with 2 to 10 groups, or ask what drives '{outcome}' to use it as a predictor."
    if "outcome_dtype" in evidence or "outcome_type" in evidence:
        return numeric_hint
    if "outcome" in evidence:  # the outcome is also listed as a predictor
        return f"Remove '{outcome}' from the predictors; the outcome can't also be a predictor."
    if "missing_columns" in evidence:
        return "Pick columns that exist in the dataset."
    return ""


def _message(prefix: str, reason: str, hint: str) -> str:
    return f"{prefix}{reason} {hint}".strip()


def route_for_plan(plan, profile: ModelSafeView) -> dict:
    qt = plan.question_type.value
    outcome = plan.outcome_name

    if qt == "descriptive":
        return _result("descriptive", "ok", qt, qt,
                       "question type 'descriptive' routes to the descriptive diagnostics", None, {})

    results = {analysis.name: applicability for analysis, applicability in evaluate_all(plan, profile)}
    applicability = {name: {"eligible": bool(a.eligible), "reason": str(a.reason)} for name, a in results.items()}

    if qt == "association":
        return _result(None, "unsupported", qt, qt, "association has no analysis yet (G1)",
                       f"Association questions are not supported yet. To see which variables relate to "
                       f"'{outcome}', ask what drives '{outcome}'.", applicability)

    if qt in COMPARISON_TYPES:
        routed = capability_for_question_type(qt).name
        eligible = [name for name in ("two_group", "multi_group") if results[name].eligible]
        if len(eligible) == 1 or routed in eligible:
            chosen = routed if routed in eligible else eligible[0]
            if chosen == "two_group":
                new_qt = qt if qt in _TWO_GROUP_TYPES else "two_group_comparison"
            else:
                new_qt = "multi_group_comparison"
            if chosen == routed:
                reason = (f"question type '{qt}' routes to {chosen}, and the data supports it: "
                          f"{results[chosen].reason}")
                return _result(chosen, "ok", new_qt, qt, reason, None, applicability)
            ev = results[chosen].evidence
            reason = (f"the data supports {chosen} ('{ev.get('group_column')}' has {ev.get('levels')} levels), "
                      f"not the {routed} the question type routes to")
            return _result(chosen, "reconciled", new_qt, qt, reason, None, applicability)
        failed = results[routed]
        hint = _hint(failed.evidence, outcome,
                     f"A group comparison needs exactly one grouping column; for several predictors, "
                     f"ask what drives '{outcome}'.",
                     "Comparisons need a continuous numeric outcome; other outcomes are not supported yet.")
        return _result(None, "unsupported", qt, qt, failed.reason,
                       _message("This comparison can't be run: ", failed.reason, hint), applicability)

    if qt == "driver_analysis":
        regression = results["regression"]
        if regression.eligible:
            return _result("regression", "ok", qt, qt,
                           f"question type 'driver_analysis' routes to regression, and the data supports it: "
                           f"{regression.reason}", None, applicability)
        hint = _hint(regression.evidence, outcome, "Name at least one predictor.",
                     "Regression needs a continuous numeric outcome; other outcomes are not supported yet.")
        return _result(None, "unsupported", qt, qt, regression.reason,
                       _message("This analysis can't be run: ", regression.reason, hint), applicability)

    raise ValueError(f"route_for_plan: unknown question type {qt!r}")
