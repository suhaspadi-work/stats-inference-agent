"""
Decision records for the boundary gate and the exploratory round (M1.3b). Pure: no graph, no model calls,
and no import of core.nodes.

  planner.stage_gate    after the main analysis: run the exploratory round, skip it, or not offered
  planner.round_start   the predictors of the exploratory regression, every column with its reason
"""
from __future__ import annotations

from core.decision_record import Candidate, DecisionRecord
from core.round_two import MAX_OTHER_PREDICTORS

STAGE_GATE_POINT = "planner.stage_gate"
ROUND_START_POINT = "planner.round_start"


def stage_gate_record(plan, proposal: dict, decision, choice: str, offered: bool, capability: str = None) -> DecisionRecord:
    """choice is "run", "skip" or "not_offered"; an unrecognised action is recorded as a skip by code_rule."""
    action = (decision or {}).get("action") if decision is not None else None
    if choice == "not_offered":
        decided_by, rationale = "code_rule", "no other usable predictor in the data"
    elif action not in ("run", "skip"):
        decided_by = "code_rule"
        rationale = f"action '{action}' is not one of run or skip; the node skips the exploratory round"
    else:
        decided_by = "human"
        rationale = ("the user chose to run the exploratory regression" if choice == "run"
                     else "the user chose to end the run after the main analysis")
    run_reason = (f"run the exploratory regression: {', '.join(proposal['predictors'])}" if offered
                  else "no usable extra predictor")
    return DecisionRecord(
        decision_point=STAGE_GATE_POINT, chosen=choice, decided_by=decided_by,
        candidates=(
            Candidate(option="run", eligible=offered, reason=run_reason),
            Candidate(option="skip", eligible=True, reason="end the run after the main analysis"),
        ),
        rationale=rationale,
        evidence={
            "finished_round": 1,
            "capability": capability,
            "question_type": plan.question_type.value,
            "method": plan.primary_method,
            "outcome": plan.outcome_name,
            "group_column": proposal["predictors"][0],
            "proposed_predictors": list(proposal["predictors"]),
            "excluded_columns": list(proposal["excluded"]),
            "exploratory": True,
            "received_action": action,
        },
    )


def round_start_record(plan2, profile, proposal: dict, from_round: int) -> DecisionRecord:
    predictors = list(proposal["predictors"])
    included = set(proposal["included"]) | {predictors[0]}
    return DecisionRecord(
        decision_point=ROUND_START_POINT, chosen=", ".join(predictors), decided_by="code_rule",
        candidates=tuple(
            Candidate(option=c.name, eligible=c.name in included, reason=proposal["reasons"][c.name],
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
            "excluded": list(proposal["excluded"]),
        },
    )
