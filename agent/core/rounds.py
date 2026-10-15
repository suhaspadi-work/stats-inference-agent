"""
Results per round (M1.3b). report_node appends one entry to state["rounds"] for every finished round;
plan, test_result and report keep meaning the latest round. An exploratory round (an analysis chosen
after seeing the main result) gets a note in front of its report. Pure: no graph and no import of core.nodes.
"""
from __future__ import annotations

EXPLORATORY_NOTE = ("Exploratory analysis: this was not part of your original question, so it was chosen after "
                    "seeing the main result. Its results are associations to explore, and they are not corrected "
                    "for multiple testing.")


def with_exploratory_note(text: str) -> str:
    return EXPLORATORY_NOTE + "\n\n" + text


def make_round_entry(index: int, plan, capability_name: str, result: dict, report_text: str, exploratory: bool) -> dict:
    return {
        "round": index,
        "exploratory": bool(exploratory),
        "capability": capability_name,
        "question_type": plan.question_type.value,
        "method": plan.primary_method,
        "plan": plan,
        "result": result,
        "report": report_text,
    }
