"""
Decision records for the boundary gate and the exploratory round (M1.3b). Pure: no graph, no model calls,
and no import of core.nodes.

  planner.stage_gate    after the main analysis: run the exploratory round, skip it, refused (the user's
                        predictors break a rule; the gate re-opens), or not offered
  planner.round_start   the predictors of the exploratory regression, every column with its reason
"""
from __future__ import annotations

from core.decision_record import Candidate, DecisionRecord
from core.round_two import MAX_OTHER_PREDICTORS

STAGE_GATE_POINT = "planner.stage_gate"
ROUND_START_POINT = "planner.round_start"


def _verdict_text(refusals) -> str:
    return "; ".join(f"{v['column'] or v['rule']}: {v['reason']}" for v in refusals)


def stage_gate_record(plan, proposal: dict, decision, choice: str, offered: bool, capability: str = None, *,
                      chosen_predictors=None, refused_attempts: int = 0, refusal: dict = None) -> DecisionRecord:
    """
    choice is "run", "skip", "refused" or "not_offered"; an unrecognised action is recorded as a skip by
    code_rule. chosen_predictors are the predictors that will be used (the proposal unless the user chose
    others); refusal is {"submitted", "refusals", "attempt"} for a refused choice.
    """
    action = (decision or {}).get("action") if decision is not None else None
    proposed = list(proposal["predictors"])
    overridden, original = False, None
    run_reason = (f"run the exploratory regression: {', '.join(proposed)}" if offered
                  else "no usable extra predictor")
    candidates = [Candidate(option="run", eligible=offered, reason=run_reason)]

    if choice == "not_offered":
        decided_by, rationale = "code_rule", "no other usable predictor in the data"
    elif choice == "refused":
        decided_by = "human"
        rationale = "the chosen predictors do not fit the rules: " + _verdict_text(refusal["refusals"])
        candidates.append(Candidate(option="run with your predictors", eligible=False,
                                    reason=_verdict_text(refusal["refusals"])))
    elif action not in ("run", "skip"):
        decided_by = "code_rule"
        rationale = f"action '{action}' is not one of run or skip; the node skips the exploratory round"
    elif choice == "run" and chosen_predictors is not None and list(chosen_predictors) != proposed:
        decided_by = "human"
        rationale = "the user chose their own predictors (they fit the rules)"
        overridden, original = True, f"run with the proposed predictors: {', '.join(proposed)}"
    else:
        decided_by = "human"
        rationale = ("the user chose to run the exploratory regression" if choice == "run"
                     else "the user chose to end the run after the main analysis")
    candidates.append(Candidate(option="skip", eligible=True, reason="end the run after the main analysis"))

    if choice == "run":
        used = list(chosen_predictors) if chosen_predictors is not None else proposed
    else:
        used = []
    evidence = {
        "finished_round": 1,
        "capability": capability,
        "question_type": plan.question_type.value,
        "method": plan.primary_method,
        "outcome": plan.outcome_name,
        "group_column": proposed[0],
        "proposed_predictors": proposed,
        "excluded_columns": list(proposal["excluded"]),
        "exploratory": True,
        "received_action": action,
        "chosen_predictors": used,
        "refused_attempts": int(refused_attempts),
    }
    if choice == "refused":
        evidence.update(submitted_predictors=list(refusal["submitted"]), refusals=list(refusal["refusals"]),
                        attempt=refusal["attempt"])
    return DecisionRecord(
        decision_point=STAGE_GATE_POINT, chosen=choice, decided_by=decided_by,
        overridden_by_human=overridden, original_choice=original,
        candidates=tuple(candidates), rationale=rationale, evidence=evidence,
    )


def round_start_record(plan2, profile, proposal: dict, from_round: int) -> DecisionRecord:
    predictors = list(plan2.candidate_predictors)
    chosen = set(predictors)
    included = set(proposal["included"])

    def reason(name):
        if name in chosen and name not in included:
            return "chosen by you; it fits the rules"
        return proposal["reasons"][name]

    return DecisionRecord(
        decision_point=ROUND_START_POINT, chosen=", ".join(predictors), decided_by="code_rule",
        candidates=tuple(
            Candidate(option=c.name, eligible=c.name in chosen, reason=reason(c.name),
                      evidence={"dtype": str(c.dtype), "distinct_count": int(c.distinct_count),
                                "missing_pct": float(c.missing_pct)})
            for c in profile.columns
        ),
        rationale="an exploratory regression adjusting the comparison for other columns",
        evidence={
            "round": from_round + 1,
            "from_round": from_round,
            "exploratory": True,
            "capability": "regression",
            "question_type": "driver_analysis",
            "outcome": plan2.outcome_name,
            "group_column": predictors[0],
            "max_other_predictors": MAX_OTHER_PREDICTORS,
            "predictors": predictors,
            "excluded": [c for c in proposal["excluded"] if c not in chosen],
            "chosen_by": "user" if predictors != list(proposal["predictors"]) else "code_rule",
        },
    )
