"""
The stage-plan gate (M1.3a): the user reviews the stage plan before anything is changed. Pure: no graph,
no model call, no import of core.nodes, and the inputs are never mutated.

The gate pauses only when some step is offered. Planned steps cannot be switched off, unavailable and
not_applicable steps cannot be switched on, and only the steps in ADJUSTABLE_STEPS can be changed.
"""
from __future__ import annotations
import copy

from core.stage_plan import plan_steps

# Inference and modeling offers become selectable in M1.3b (after the main analysis).
ADJUSTABLE_STEPS = ("describe",)
APPROVAL_STATUSES = ("approved", "approved_with_changes", "rejected")
_NOT_SELECTABLE_NOTE = "offered after the main analysis; not selectable yet"


def gate_needed(plan: dict) -> bool:
    return any(s["state"] == "offered" for s in plan_steps(plan))


def adjustable_steps(plan: dict) -> list[str]:
    return [s["step"] for s in plan_steps(plan) if s["state"] == "offered" and s["step"] in ADJUSTABLE_STEPS]


def gate_payload(plan: dict, request_text: str) -> dict:
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
    }


def validate_run_overrides(plan: dict, overrides) -> dict[str, bool]:
    """The user's run choices as {step: bool}; raises ValueError for any step that cannot be changed here."""
    if overrides is None:
        return {}
    adjustable = adjustable_steps(plan)
    if not isinstance(overrides, dict):
        raise ValueError("Run choices must be true or false.")
    bad = [k for k in overrides if k not in adjustable]
    if bad:
        raise ValueError(f"Cannot change step(s) {sorted(map(str, bad))} here. Adjustable steps: {adjustable}.")
    if not all(isinstance(v, bool) for v in overrides.values()):
        raise ValueError("Run choices must be true or false.")
    return dict(overrides)


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
