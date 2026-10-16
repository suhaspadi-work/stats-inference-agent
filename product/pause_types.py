"""
The pauses the product handles and the payload keys it reads from each (M1.5 step P1). Pure: no streamlit, no
import of the agent. agent/tests/test_product_contract.py runs the real graph and fails when the backend emits a
pause the product does not list here (or the product lists one the backend never emits), or when a payload lacks
a key listed here.
"""

HANDLED_PAUSE_TYPES = ("clarification_needed", "stage_plan_approval", "wrangling_approval", "plan_approval",
                       "stage_gate")

REQUIRED_PAYLOAD_KEYS = {
    "clarification_needed": ("type", "question"),
    "stage_plan_approval": ("type", "request_text", "question_type", "stages", "adjustable_steps",
                            "allowed_decisions", "refused"),
    "wrangling_approval": ("type", "proposed_operations"),
    "plan_approval": ("type", "request_text", "outcome_name", "outcome_type", "candidate_predictors",
                      "causal_status", "classification_rationale", "allowed_decisions", "refused", "method",
                      "reselected", "stage1_report"),
    "stage_gate": ("type", "request_text", "finished", "offer", "allowed_decisions", "refused"),
}


# Nested structures the views read (product/pause_views.py): {(pause type, path): keys}. The path is followed key
# by key from the payload; a list along the way is entered item by item, and "*" means every value of a dict. The
# keys must be in the dict reached, or in every item when it is a list. A None along the path has nothing to check.
_REFUSED = ("attempt", "refusals")
_VERDICT = ("rule", "reason", "explanation")
NESTED_KEYS = {
    ("stage_plan_approval", ("stages",)): ("steps",),
    ("stage_plan_approval", ("stages", "steps")): ("step", "state", "run", "reason", "adjustable"),
    ("stage_plan_approval", ("refused",)): _REFUSED,
    ("stage_plan_approval", ("refused", "refusals")): _VERDICT,
    ("wrangling_approval", ("proposed_operations",)): ("index", "op_type", "params", "rationale", "alternative"),
    ("plan_approval", ("refused",)): _REFUSED,
    ("plan_approval", ("refused", "refusals")): _VERDICT,
    ("stage_gate", ("finished",)): ("round", "method", "p_value"),
    ("stage_gate", ("offer",)): ("proposed_predictors", "selectable", "unsuitable", "note"),
    ("stage_gate", ("offer", "selectable")): ("column", "proposed", "reason"),
    ("stage_gate", ("offer", "unsuitable", "*")): ("reason", "explanation"),
    ("stage_gate", ("refused",)): _REFUSED,
    ("stage_gate", ("refused", "refusals")): _VERDICT,
}


def nested_targets(payload, path) -> list:
    """The dicts a NESTED_KEYS path reaches in a payload: one per list item along the way, none past a None."""
    current = [payload]
    for key in path:
        reached = []
        for value in current:
            if value is None:
                continue
            child = list(value.values()) if key == "*" else value.get(key)
            if isinstance(child, list):
                reached.extend(child)
            else:
                reached.append(child)
        current = reached
    return [value for value in current if value is not None]


def widget_key(kind, serial, *parts) -> str:
    """"<kind>-<serial>" then "-<part>" for each part, so a re-opened pause (a new serial) starts clean."""
    return "-".join([str(kind), str(serial), *(str(p) for p in parts)])
