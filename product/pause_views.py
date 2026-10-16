"""
What each pause shows and what each answer sends (M1.5 step P2a). Pure: no streamlit. A view function turns an
interrupt payload into the plain values a screen shows; an answer builder turns a screen's choices into the resume
value the backend expects. Every payload key read here is listed in pause_types (REQUIRED_PAYLOAD_KEYS or
NESTED_KEYS), so agent/tests/test_product_contract.py catches a payload change.
"""

STATE_LABELS = {
    "planned": "will run",
    "offered": "your choice",
    "unavailable": "not built yet",
    "not_applicable": "not needed",
}

GATE_CHOICES = ("proposed", "own", "skip")

# an offered step that cannot be toggled here (exploratory modeling) is chosen at the boundary gate
OFFERED_LATER_LABEL = "asked after the main analysis"

RESELECTED_NOTE = "You edited the predictors, so the method was selected again for the new predictors."


def method_label(method_id) -> str:
    """The glossary's display name of a method, the id itself when it has none, "unknown method" for None."""
    if method_id is None:
        return "unknown method"
    try:
        from core.method_glossary import GLOSSARY
    except ImportError:
        return str(method_id)
    entry = GLOSSARY.get(method_id)
    return entry["name"] if entry is not None else str(method_id)


def refusal_lines(refused) -> list:
    """[] when nothing was refused; otherwise a header and one line per verdict of any of the three gates."""
    if not refused:
        return []
    lines = [f"Your last choice was refused (attempt {refused.get('attempt')}):"]
    for verdict in refused.get("refusals") or []:
        label = verdict.get("column") or verdict.get("step") or verdict.get("rule")
        text = verdict.get("explanation") or verdict.get("reason")
        lines.append(f"{label}: {text}")
    return lines


def _reason(reason) -> str:
    return (reason or "").strip() or "No reason given"


# --- stage_plan_approval ---------------------------------------------------------------------------------------------

def _state_label(step) -> str:
    if step["state"] == "offered" and not step["adjustable"]:
        return OFFERED_LATER_LABEL
    return STATE_LABELS.get(step["state"], step["state"])


def stage_plan_view(payload) -> dict:
    # "note" is read with .get: the payload sets it only for an offered step that cannot be toggled
    rows = [{"step": s["step"], "state": s["state"], "state_label": _state_label(s), "run": s["run"],
             "reason": s["reason"], "adjustable": s["adjustable"], "note": s.get("note")}
            for stage in payload["stages"] for s in stage["steps"]]
    return {"request_text": payload["request_text"], "question_type": payload["question_type"], "rows": rows,
            "adjustable": list(payload["adjustable_steps"]), "refusal": refusal_lines(payload["refused"])}


def approve_stage_plan(view, toggles=None) -> dict:
    """Sends only the adjustable steps whose toggle differs from the current run flag."""
    current = {r["step"]: r["run"] for r in view["rows"]}
    run = {step: bool(value) for step, value in (toggles or {}).items()
           if step in view["adjustable"] and bool(value) != current.get(step)}
    return {"action": "approve", **({"run": run} if run else {})}


def reject_stage_plan(reason) -> dict:
    return {"action": "reject", "reason": _reason(reason)}


# --- plan_approval ---------------------------------------------------------------------------------------------------

def plan_approval_view(payload) -> dict:
    return {
        "request_text": payload["request_text"],
        "outcome": payload["outcome_name"],
        "outcome_type": payload["outcome_type"],
        "predictors": ", ".join(payload["candidate_predictors"]),
        "data_type": payload["causal_status"],
        "method_label": method_label(payload["method"]),
        "reselected_note": RESELECTED_NOTE if payload["reselected"] else None,
        "rationale": payload["classification_rationale"],
        "stage1_report": payload["stage1_report"],
        "refusal": refusal_lines(payload["refused"]),
    }


def approve_plan() -> dict:
    return {"action": "approve"}


def reject_plan(reason) -> dict:
    return {"action": "reject", "reason": _reason(reason)}


# --- stage_gate -------------------------------------------------------------------------------------------------------

def _finished_lines(finished) -> list:
    lines = [f"Round {finished['round']}: {method_label(finished['method'])}."]
    if finished.get("p_value") is not None:
        lines.append(f"p = {finished['p_value']:.3g}")
    return lines


def stage_gate_view(payload) -> dict:
    offer = payload["offer"]
    proposed = list(offer["proposed_predictors"])
    group_column = proposed[0]
    return {
        "finished": _finished_lines(payload["finished"]),
        "group_column": group_column,
        "proposed": proposed,
        "options": [{"column": o["column"], "proposed": o["proposed"], "reason": o["reason"]}
                    for o in offer["selectable"]],
        "default_own": [p for p in proposed if p != group_column],
        "left_out": [{"column": column, "reason": v["reason"], "explanation": v["explanation"]}
                     for column, v in offer["unsuitable"].items()],
        "note": offer["note"],
        "refusal": refusal_lines(payload["refused"]),
    }


def gate_answer(view, choice, chosen=None) -> dict:
    """proposed: run the proposal; own: the group column first, then the chosen columns in order; skip."""
    if choice == "proposed":
        return {"action": "run"}
    if choice == "own":
        others = [c for c in (chosen or []) if c != view["group_column"]]
        return {"action": "run", "predictors": [view["group_column"], *others]}
    if choice == "skip":
        return {"action": "skip"}
    raise ValueError(f"Unknown boundary-gate choice '{choice}'; expected one of {', '.join(GATE_CHOICES)}.")


# --- wrangling_approval and clarification_needed (as app.py reads them today) ------------------------------------------

def wrangling_view(payload) -> dict:
    return {"note": payload.get("auto_approved_note"),
            "operations": [{"index": op["index"], "op_type": op["op_type"], "rationale": op["rationale"],
                            "alternative": op["alternative"], "params": op["params"]}
                           for op in payload["proposed_operations"]]}


def clarification_view(payload) -> dict:
    return {"question": payload["question"]}


_VIEWS = {
    "clarification_needed": clarification_view,
    "stage_plan_approval": stage_plan_view,
    "wrangling_approval": wrangling_view,
    "plan_approval": plan_approval_view,
    "stage_gate": stage_gate_view,
}


def view_for(payload) -> dict:
    view = _VIEWS.get(payload.get("type"))
    return view(payload) if view is not None else {"unsupported": True, "type": payload.get("type")}
