"""
Decision records for the planner's draft-or-skip choice and its routing
(M0.2 step 10, M0.3). Pure functions: no graph, no model calls, and no import
of core.nodes. The route itself comes from core.routing.route_for_plan, which
reconciles the classifier's question type with the data; these records state
that route and the code-computed applicability of every analysis next to it.

  planner.draft_plan   descriptive questions skip the model's column choice
  planner.route        which capability runs, or that none can (unsupported)
"""
from __future__ import annotations
import copy
from typing import Any, Optional

from core.capabilities import capability_for_question_type, evaluate_all
from core.decision_record import Candidate, DecisionRecord
from core.model_safe_view import ModelSafeView
from core.routing import route_for_plan

DRAFT_PLAN_POINT = "planner.draft_plan"
ROUTE_POINT = "planner.route"

_DESCRIPTIVE_SKIPPED_STEPS = ["select_method", "freeze", "execute"]


def _plain(value: Any) -> Any:
    """Copy into plain dicts and lists (tuples become lists)."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return copy.deepcopy(value)


def skipped_draft_record(plan) -> DecisionRecord:
    skip_reason = ("chosen: descriptive questions describe the whole dataset, so the model is not asked "
                   "to pick an outcome, predictors or unit")
    return DecisionRecord(
        decision_point=DRAFT_PLAN_POINT, chosen="skipped", decided_by="code_rule",
        candidates=(
            Candidate(option="ask_model", eligible=False,
                      reason="not eligible: the question is descriptive, so there is no single outcome "
                             "column or predictor set to choose"),
            Candidate(option="skip", eligible=True, reason=skip_reason),
        ),
        rationale=skip_reason,
        evidence={
            "question_type": plan.question_type.value,
            "placeholders": {
                "outcome_name": plan.outcome_name,
                "outcome_type": plan.outcome_type.value,
                "unit_of_analysis": plan.unit_of_analysis,
            },
            "placeholders_chosen_by": "nobody; fixed values used because no model choice was made",
        },
    )


def route_record(plan, profile: ModelSafeView, route: Optional[dict] = None) -> DecisionRecord:
    """The planner.route record for the route chosen by core.routing.route_for_plan (computed when not given)."""
    if route is None:
        route = route_for_plan(plan, profile)
    qt, classifier_qt, status = route["question_type"], route["classifier_question_type"], route["status"]

    if route["capability"] == "descriptive":
        reason = (f"chosen: question type '{qt}' routes to the descriptive diagnostics, which skip method "
                  f"selection, plan approval and execution")
        # evaluate_all is deliberately not called: the placeholder outcome would give a misleading
        # "column not found" reason for every analysis.
        return DecisionRecord(
            decision_point=ROUTE_POINT, chosen="descriptive", decided_by="code_rule",
            candidates=(
                Candidate(option="descriptive", eligible=True, reason=reason),
                Candidate(option="inference", eligible=False,
                          reason="not eligible: the question is descriptive, so no hypothesis test applies"),
            ),
            rationale=route["reason"],
            evidence={"question_type": qt, "capability": "descriptive",
                      "skipped_steps": list(_DESCRIPTIVE_SKIPPED_STEPS),
                      "classifier_question_type": classifier_qt, "status": status},
        )

    chosen = route["capability"]
    routed = capability_for_question_type(classifier_qt).name
    candidates = []
    chosen_is_eligible = False
    for analysis, applicability in evaluate_all(plan, profile):
        applies = f"applicability: {applicability.reason}"
        if status == "unsupported":
            reason = f"not chosen: no analysis can run this question; {applies}"
        elif analysis.name == chosen:
            chosen_is_eligible = bool(applicability.eligible)
            if status == "reconciled":
                reason = (f"chosen: the data supports this analysis, although question type '{classifier_qt}' "
                          f"routes to {routed}; {applies}")
            else:
                reason = f"chosen: question type '{classifier_qt}' routes to this analysis; {applies}"
        elif status == "reconciled":
            reason = f"not chosen: the data supports {chosen}; {applies}"
        else:
            reason = f"not chosen: question type '{classifier_qt}' routes to {chosen}; {applies}"
        candidates.append(Candidate(option=analysis.name, eligible=bool(applicability.eligible),
                                    reason=reason, evidence=_plain(applicability.evidence)))

    evidence = {
        "question_type": qt,
        "classifier_question_type": classifier_qt,
        "status": status,
        "capability": chosen,
        "skipped_steps": list(_DESCRIPTIVE_SKIPPED_STEPS) if status == "unsupported" else [],
        "chosen_is_eligible": chosen_is_eligible,
        "grouping_max_levels": route["evidence"]["grouping_max_levels"],
    }
    if status == "unsupported":
        evidence["message"] = route["message"]
    return DecisionRecord(
        decision_point=ROUTE_POINT, chosen=chosen if chosen is not None else "unsupported", decided_by="code_rule",
        candidates=tuple(candidates), rationale=route["reason"], evidence=evidence,
    )
