"""
The stage-plan gate (M1.3a): the user reviews the stage plan before anything is changed. Pure: no graph,
no model call, no import of core.nodes, and the inputs are never mutated.

The gate pauses only when some step is offered. Planned steps cannot be switched off, unavailable and
not_applicable steps cannot be switched on, and only the steps in ADJUSTABLE_STEPS can be changed. Since M1.3c
a choice that breaks a rule is not raised: check_run_overrides returns every problem at once as verdicts
{"step", "rule", "reason", "explanation", "evidence"}, the node records the refusal and the gate re-opens.
"""
from __future__ import annotations
import copy

from core.stage_plan import STEP_NAMES, plan_steps

# Inference and modeling offers become selectable in M1.3b (after the main analysis).
ADJUSTABLE_STEPS = ("describe",)
APPROVAL_STATUSES = ("approved", "approved_with_changes", "rejected")
_NOT_SELECTABLE_NOTE = "you will be asked after the main analysis, if the data allows it"


def gate_needed(plan: dict) -> bool:
    return any(s["state"] == "offered" for s in plan_steps(plan))


def adjustable_steps(plan: dict) -> list[str]:
    return [s["step"] for s in plan_steps(plan) if s["state"] == "offered" and s["step"] in ADJUSTABLE_STEPS]


def gate_payload(plan: dict, request_text: str, refused=None) -> dict:
    """refused is the previous refusal ({"submitted", "refusals", "attempt"}), shown when the gate re-opens."""
    adjustable = adjustable_steps(plan)
    return {
        "type": "stage_plan_approval",
        "question": "Review which steps this run will take before anything is changed.",
        "request_text": request_text,
        "question_type": plan["question_type"],
        "stages": [
            {
                "stage": stage["stage"],
                "name": stage["name"],
                "steps": [
                    {
                        "step": s["step"], "state": s["state"], "run": s["run"], "exploratory": s["exploratory"],
                        "reason": s["reason"], "adjustable": s["step"] in adjustable,
                        "note": _NOT_SELECTABLE_NOTE if s["state"] == "offered" and s["step"] not in adjustable else None,
                    }
                    for s in stage["steps"]
                ],
            }
            for stage in plan["stages"]
        ],
        "adjustable_steps": adjustable,
        "allowed_decisions": ["approve", "reject"],
        "refused": refused,
    }


def _step_verdict(step, rule, reason, explanation, evidence=None) -> dict:
    return {"step": step, "rule": rule, "reason": reason, "explanation": explanation, "evidence": dict(evidence or {})}


def check_run_overrides(plan: dict, overrides) -> dict:
    """
    The user's run choices, {step: bool}, checked against the plan: {"ok", "overrides", "refusals"}, with every
    problem collected at once. Only an adjustable (offered) step can be switched, and only to true or false.
    """
    if overrides is None:
        return {"ok": True, "overrides": {}, "refusals": []}
    if not isinstance(overrides, dict):
        return {"ok": False, "overrides": {}, "refusals": [_step_verdict(
            None, "not_a_dict", 'run choices must be an object like {"describe": true}',
            'run choices must be an object like {"describe": true}')]}
    steps = {s["step"]: s for s in plan_steps(plan)}
    adjustable = adjustable_steps(plan)
    refusals = []
    for name, value in overrides.items():
        step = steps.get(name) if name in STEP_NAMES else None
        if step is None:
            refusals.append(_step_verdict(name, "unknown_step", "no such step",
                                          f"'{name}' is not a step of this plan; the steps are clean, describe, "
                                          "inference, modeling"))
            continue
        evidence = {"state": step["state"], "run": step["run"], **step["evidence"]}
        if step["state"] == "planned":
            refusals.append(_step_verdict(name, "planned_step", "needed by your question",
                                          f"'{name}' is needed to answer your question, so it cannot be switched off",
                                          evidence))
        elif step["state"] == "unavailable":
            available_from = step["evidence"].get("available_from")
            refusals.append(_step_verdict(name, "unavailable", f"not built yet ({available_from})",
                                          f"'{name}' would apply to your question but is not built yet; it arrives "
                                          f"with {available_from}", evidence))
        elif step["state"] == "not_applicable":
            refusals.append(_step_verdict(name, "not_applicable", "does not apply to this question", step["reason"],
                                          evidence))
        elif name not in adjustable:
            refusals.append(_step_verdict(name, "chosen_later", "chosen after the main analysis",
                                          "you will be asked about this step after the main analysis, if the data "
                                          "allows it", evidence))
        elif not isinstance(value, bool):
            refusals.append(_step_verdict(name, "not_boolean", "must be true or false",
                                          f"the choice for '{name}' must be true or false", evidence))
    if refusals:
        return {"ok": False, "overrides": {}, "refusals": refusals}
    return {"ok": True, "overrides": dict(overrides), "refusals": []}


def changed_steps(plan: dict, overrides: dict) -> list[str]:
    """Steps, in plan order, whose run value an override actually changes."""
    return [s["step"] for s in plan_steps(plan) if s["step"] in overrides and overrides[s["step"]] != s["run"]]


def apply_gate_decision(plan: dict, overrides: dict, status: str, reason=None) -> dict:
    if status not in APPROVAL_STATUSES:
        raise ValueError(f"Unknown approval status '{status}'.")
    changed = changed_steps(plan, overrides)
    new_plan = copy.deepcopy(plan)
    for s in plan_steps(new_plan):
        if s["step"] in overrides:
            s["run"] = overrides[s["step"]]
    new_plan["approval"] = {"status": status, "overridden_steps": changed, "reason": reason}
    return new_plan
