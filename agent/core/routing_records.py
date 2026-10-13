"""
Decision records for the planner's draft-or-skip choice and its routing
(M0.2 step 10). Pure functions: no graph, no model calls, and no import of
core.nodes. Routing is still by question type (graph.route_after_draft_plan
and capabilities.capability_for_plan); these records state that rule and the
code-computed applicability of every analysis next to it.

  planner.draft_plan   descriptive questions skip the model's column choice
  planner.route        which capability the question type routes to
"""
from __future__ import annotations
import copy
from typing import Any

from core.capabilities import capability_for_plan, evaluate_all
from core.decision_record import Candidate, DecisionRecord
from core.model_safe_view import ModelSafeView

DRAFT_PLAN_POINT = "planner.draft_plan"
ROUTE_POINT = "planner.route"

# Mirrors the association note in core/classification_records.py. Association
# routes to the two-group capability until M3.3; the guard tests in
# tests/test_routing_records.py fail if that changes, so this text is revisited.
_ASSOCIATION_NOTE = "no association capability yet (G1); falls into the two-group code"

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


def route_record(plan, profile: ModelSafeView) -> DecisionRecord:
    capability = capability_for_plan(plan)
    qt = plan.question_type.value

    if capability.kind == "diagnostic":
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
            rationale=reason,
            evidence={"question_type": qt, "capability": capability.name,
                      "skipped_steps": list(_DESCRIPTIVE_SKIPPED_STEPS)},
        )

    chosen = capability.name
    candidates = []
    chosen_is_eligible = False
    for analysis, applicability in evaluate_all(plan, profile):
        if analysis.name == chosen:
            reason = f"chosen: question type '{qt}' routes to this analysis; applicability: {applicability.reason}"
            chosen_is_eligible = bool(applicability.eligible)
        else:
            reason = f"not chosen: question type '{qt}' routes to {chosen}; applicability: {applicability.reason}"
        candidates.append(Candidate(option=analysis.name, eligible=bool(applicability.eligible),
                                    reason=reason, evidence=_plain(applicability.evidence)))

    evidence = {"question_type": qt, "capability": chosen, "skipped_steps": [],
                "chosen_is_eligible": chosen_is_eligible}
    if qt == "association":
        evidence["borrowed_route"] = _ASSOCIATION_NOTE
    rationale = next(c.reason for c in candidates if c.option == chosen)
    return DecisionRecord(
        decision_point=ROUTE_POINT, chosen=chosen, decided_by="code_rule",
        candidates=tuple(candidates), rationale=rationale, evidence=evidence,
    )
