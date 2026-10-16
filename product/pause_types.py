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


def widget_key(kind, serial, *parts) -> str:
    """"<kind>-<serial>" then "-<part>" for each part, so a re-opened pause (a new serial) starts clean."""
    return "-".join([str(kind), str(serial), *(str(p) for p in parts)])
