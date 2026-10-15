"""
Decision records for the stage plan (M1.1). Pure functions: no graph, no model calls, and no import of
core.nodes. The plan comes from core.stage_plan.build_stage_plan.

  planner.stage_plan        which steps the run takes, with every step as a candidate
  planner.step.<step>       run or skip, one record per step (none for an unsupported plan)
"""
from __future__ import annotations

from core.decision_record import Candidate, DecisionRecord
from core.stage_plan import plan_steps

STAGE_PLAN_POINT = "planner.stage_plan"
STEP_POINTS = {
    "clean": "planner.step.clean",
    "describe": "planner.step.describe",
    "inference": "planner.step.inference",
    "modeling": "planner.step.modeling",
}

_ELIGIBLE_STATES = ("planned", "offered")


def stage_plan_record(plan: dict, classification, issues) -> DecisionRecord:
    steps = plan_steps(plan)
    if plan["status"] == "unsupported":
        chosen = "unsupported"
    else:
        chosen = ", ".join(s["step"] for s in steps if s["run"]) or "none"
    return DecisionRecord(
        decision_point=STAGE_PLAN_POINT,
        chosen=chosen,
        decided_by="code_rule",
        candidates=tuple(
            Candidate(option=s["step"], eligible=s["state"] in _ELIGIBLE_STATES,
                      reason=f"{s['state']}: {s['reason']}", evidence=dict(s["evidence"]))
            for s in steps
        ),
        rationale=f"question type '{plan['question_type']}': "
                  + ", ".join(f"{s['step']} {s['state']}" for s in steps),
        evidence={
            "question_type": plan["question_type"],
            "status": plan["status"],
            "classifier_confidence": classification.confidence,
            "issue_count": len(issues),
            "issue_types": sorted({i.issue_type for i in issues}),
            "message": plan["message"],
        },
    )


def step_records(plan: dict) -> tuple[DecisionRecord, ...]:
    if plan["status"] == "unsupported":
        return ()
    stage_of = {s["step"]: stage["stage"] for stage in plan["stages"] for s in stage["steps"]}
    records = []
    for s in plan_steps(plan):
        detail = f"{s['state']}: {s['reason']}"
        run_prefix = "chosen: " if s["run"] else "not chosen: "
        skip_prefix = "not chosen: " if s["run"] else "chosen: "
        records.append(DecisionRecord(
            decision_point=STEP_POINTS[s["step"]],
            chosen="run" if s["run"] else "skip",
            decided_by="code_rule",
            candidates=(
                Candidate(option="run", eligible=s["state"] in _ELIGIBLE_STATES, reason=run_prefix + detail),
                Candidate(option="skip", eligible=s["state"] != "planned", reason=skip_prefix + detail),
            ),
            rationale=s["reason"],
            evidence={"stage": stage_of[s["step"]], "step": s["step"], "state": s["state"], "run": s["run"],
                      "exploratory": s["exploratory"], **s["evidence"]},
        ))
    return tuple(records)
