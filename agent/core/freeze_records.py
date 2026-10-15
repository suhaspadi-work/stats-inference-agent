"""
Decision records for plan approval at freeze (M0.2 step 9). Pure functions:
no graph, no model calls, and no import of core.nodes.

  planner.plan_approval   what the human decided about the drafted plan (M1.3c: an accepted edit is not frozen;
                          the method is selected again; an edit that breaks a rule is recorded as "refused")
  planner.predictors      a new predictors record when the human edits them,
                          superseding the model's draft (decided_by human)
"""
from __future__ import annotations
from typing import Optional

from core.decision_record import Candidate, DecisionRecord
from core.draft_plan_records import PREDICTORS_POINT, predictor_candidates
from core.model_safe_view import ModelSafeView

PLAN_APPROVAL_POINT = "planner.plan_approval"

_APPROVAL_CANDIDATES = (
    Candidate(option="approve", eligible=True, reason="freeze the plan as drafted and run the selected method"),
    Candidate(option="edit", eligible=True,
              reason="change the predictors; they are checked against the predictor rules and the method is "
                     "selected again before anything is frozen"),
    Candidate(option="reject", eligible=True, reason="do not freeze or run; the run stops"),
)


def _names(predictors) -> list[str]:
    return [str(p) for p in (predictors or ())]


def _joined(names: list[str]) -> str:
    return ", ".join(names) if names else "none"


def _approval_evidence(plan, action, selection_record_id) -> dict:
    return {
        "plan_id": plan.id,
        "question_type": plan.question_type.value,
        "causal_status": plan.causal_status.value,
        "outcome_name": plan.outcome_name,
        "outcome_type": plan.outcome_type.value,
        "predictors_before": _names(plan.candidate_predictors),
        "method_at_approval": plan.primary_method,
        "selection_record_id": selection_record_id,
        "received_action": action,
        "frozen": action not in ("reject", "edit"),
        "reselects": action == "edit",
    }


def plan_approval_record(plan, decision: dict, selection_record_id: Optional[str],
                         predictors_record_id: Optional[str] = None) -> DecisionRecord:
    """An accepted edit is not frozen (M1.3c): the plan goes back through method selection (reselects True)."""
    decision = decision or {}
    action = decision.get("action")
    before = _names(plan.candidate_predictors)
    evidence = _approval_evidence(plan, action, selection_record_id)
    overridden, original = False, None

    if action == "approve":
        chosen, decided_by, rationale = "approve", "human", "approved as drafted"
    elif action == "edit":
        chosen, decided_by, rationale = "edit", "human", ("predictors edited before freezing; the method is selected "
                                                          "again for the new predictors before the plan is frozen")
        overridden = True
        original = f"approve as drafted: predictors [{', '.join(before)}]"
        evidence["predictors_after"] = _names(decision.get("candidate_predictors", plan.candidate_predictors))
        evidence["predictors_record_id"] = predictors_record_id
    elif action == "reject":
        chosen, decided_by = "reject", "human"
        rationale = decision.get("reason", "no reason given")
    else:
        chosen, decided_by = "approve", "code_rule"
        rationale = f"action '{action}' is not one of approve, edit or reject; the node freezes the plan (G36)"

    return DecisionRecord(
        decision_point=PLAN_APPROVAL_POINT, chosen=chosen, decided_by=decided_by,
        overridden_by_human=overridden, original_choice=original,
        candidates=_APPROVAL_CANDIDATES, rationale=str(rationale), evidence=evidence,
    )


def _verdict_reasons(refusals) -> str:
    return "; ".join(f"{v['column'] or v['rule']}: {v['reason']}" for v in refusals)


def refused_plan_approval_record(plan, decision: dict, refusal: dict, selection_record_id: Optional[str]) -> DecisionRecord:
    """An edit at plan approval that breaks a rule (M1.3c): a human decision; the plan-approval pause re-opens."""
    decision = decision or {}
    reasons = _verdict_reasons(refusal["refusals"])
    submitted = decision.get("candidate_predictors")
    evidence = {
        **_approval_evidence(plan, decision.get("action"), selection_record_id),
        "frozen": False,
        "reselects": False,
        "submitted_predictors": list(submitted) if isinstance(submitted, (list, tuple)) else [],
        "refusals": list(refusal["refusals"]),
        "attempt": refusal["attempt"],
    }
    return DecisionRecord(
        decision_point=PLAN_APPROVAL_POINT, chosen="refused", decided_by="human",
        candidates=(*_APPROVAL_CANDIDATES,
                    Candidate(option="edit with your predictors", eligible=False, reason=reasons)),
        rationale="the edited predictors do not fit the rules: " + reasons, evidence=evidence,
    )


def edited_predictors_record(plan, new_predictors, profile: ModelSafeView, supersedes_id: Optional[str],
                             selection_record_id: Optional[str]) -> DecisionRecord:
    before, after = _names(plan.candidate_predictors), _names(new_predictors)
    return DecisionRecord(
        decision_point=PREDICTORS_POINT, chosen=_joined(after), decided_by="human",
        overridden_by_human=True, original_choice=_joined(before), supersedes=supersedes_id,
        candidates=predictor_candidates(profile, plan.outcome_name, after),
        rationale=("predictors edited at plan approval; the method and diagnostics were computed for the "
                   "previous predictors and were not recomputed (G5)"),
        evidence={
            "predictors_before": before, "predictors_after": after,
            "method_selected_before_edit": plan.primary_method,
            "selection_record_id": selection_record_id,
            "diagnostics_recomputed": False,
            "n_columns": len(profile.columns), "n_chosen": len(after),
        },
    )
