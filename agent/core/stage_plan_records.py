"""
Decision records for the stage plan (M1.1). Pure functions: no graph, no model calls, and no import of
core.nodes. The plan comes from core.stage_plan.build_stage_plan.

  planner.stage_plan        which steps the run takes, with every step as a candidate
  planner.step.<step>       run or skip, one record per step (none for an unsupported plan)
  planner.stage_plan_approval  the human's answer at the stage-plan gate (M1.3a)
  planner.stage_replan      the plan recomputed after the Stage 1 loop (M2.4a)
"""
from __future__ import annotations

from core.capabilities import is_numeric_dtype_name
from core.decision_record import Candidate, DecisionRecord
from core.stage_plan import plan_steps
from core.stage_plan_gate import adjustable_steps, changed_steps

STAGE_PLAN_POINT = "planner.stage_plan"
STEP_POINTS = {
    "clean": "planner.step.clean",
    "describe": "planner.step.describe",
    "inference": "planner.step.inference",
    "modeling": "planner.step.modeling",
}

_ELIGIBLE_STATES = ("planned", "offered")


def stage_plan_record(plan: dict, classification, issues, tables=None) -> DecisionRecord:
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
            # M2.1: every table of the upload, {"name", "rows", "columns", "role": "analysis" or "carried"}
            "tables": [dict(t) for t in (tables or [])],
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


STAGE_PLAN_APPROVAL_POINT = "planner.stage_plan_approval"

_STATUS_CHOICE = {"approved": "approve", "approved_with_changes": "approve_with_changes", "rejected": "reject"}


def _verdict_reasons(refusals) -> str:
    return "; ".join(f"{v.get('step') or v['rule']}: {v['reason']}" for v in refusals)


def stage_plan_approval_record(plan_before: dict, decision: dict, overrides: dict, status: str,
                               refusal: dict = None) -> DecisionRecord:
    """
    The human's answer at the stage-plan gate (M1.3a). plan_before is the plan as shown. status "refused"
    (M1.3c) records run choices that break a rule; refusal is {"submitted", "refusals", "attempt"}.
    """
    decision = decision or {}
    action = decision.get("action")
    steps = plan_steps(plan_before)
    run_before = {s["step"]: s["run"] for s in steps}
    run_after = {name: overrides.get(name, run) for name, run in run_before.items()}
    changed = changed_steps(plan_before, overrides)
    adjustable = adjustable_steps(plan_before)
    overridden, original = False, None

    extra_candidates, extra_evidence = (), {}
    if status == "refused":
        chosen, decided_by = "refused", "human"
        reasons = _verdict_reasons(refusal["refusals"])
        rationale = "the chosen run flags do not fit the rules: " + reasons
        extra_candidates = (Candidate(option="approve with your choices", eligible=False, reason=reasons),)
        submitted = decision.get("run")
        extra_evidence = {"submitted": dict(submitted) if isinstance(submitted, dict) else {},
                          "refusals": list(refusal["refusals"]), "attempt": refusal["attempt"]}
    elif action not in ("approve", "reject"):
        chosen, decided_by = "approve", "code_rule"
        rationale = f"action '{action}' is not one of approve or reject; the node approves the plan as proposed (G36)"
    elif status == "rejected":
        chosen, decided_by = "reject", "human"
        rationale = decision.get("reason") or "no reason given"
    elif status == "approved_with_changes":
        chosen, decided_by = "approve_with_changes", "human"
        rationale = ", ".join(f"{name} switched {'on' if run_after[name] else 'off'}" for name in changed)
        overridden = True
        original = ("approve the plan as proposed: "
                    + (", ".join(name for name, run in run_before.items() if run) or "no steps"))
    else:
        chosen, decided_by, rationale = _STATUS_CHOICE[status], "human", "approved as proposed"

    return DecisionRecord(
        decision_point=STAGE_PLAN_APPROVAL_POINT, chosen=chosen, decided_by=decided_by,
        overridden_by_human=overridden, original_choice=original,
        candidates=(
            Candidate(option="approve", eligible=True, reason="run the steps as proposed"),
            Candidate(option="approve_with_changes", eligible=bool(adjustable), reason="switch describe on or off"),
            Candidate(option="reject", eligible=True, reason="stop before any cleaning; the run ends"),
            *extra_candidates,
        ),
        rationale=str(rationale),
        evidence={
            "question_type": plan_before["question_type"],
            "run_before": run_before,
            "run_after": run_after,
            "overridden_steps": changed,
            "adjustable_steps": adjustable,
            "exploratory_offers": [s["step"] for s in steps if s["state"] == "offered" and s["exploratory"]],
            "received_action": action,
            **extra_evidence,
        },
    )


STAGE_REPLAN_POINT = "planner.stage_replan"


def stage_replan_record(changed: list, rounds: int, ended_by, profile) -> DecisionRecord:
    """
    M2.4a: the stage plan recomputed after the Stage 1 loop (core.stage_plan.replan_stage_plan). A record of its own,
    never another planner.stage_plan: a question's records start at the last planner.stage_plan. changed is
    [{"step", "before_state", "after_state", "reason"}]; counts and step names only.
    """
    names = ", ".join(c["step"] for c in changed)
    chosen = "changed" if changed else "unchanged"
    reason_changed = f"the table Stage 1 left changes {names}" if changed else "the table Stage 1 left changes no step"
    return DecisionRecord(
        decision_point=STAGE_REPLAN_POINT, chosen=chosen, decided_by="code_rule",
        candidates=(
            Candidate(option="changed", eligible=bool(changed), reason=reason_changed),
            Candidate(option="unchanged", eligible=not changed,
                      reason="every step stays as planned" if not changed else f"not eligible: {reason_changed}"),
        ),
        rationale=(f"after {rounds} cleaning round(s) the plan was recomputed from the analysis table: "
                   + (names + " changed" if changed else "no step changed")),
        evidence={
            "rounds": int(rounds), "ended_by": ended_by,
            "changed_steps": [dict(c) for c in changed],
            "column_count": int(profile.column_count),
            "numeric_column_count": sum(1 for c in profile.columns if is_numeric_dtype_name(c.dtype)),
        },
    )
