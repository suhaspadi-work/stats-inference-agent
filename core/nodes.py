from __future__ import annotations
from typing import Callable
from langgraph.types import interrupt

from core.agent_state import AgentState
from core.classifier import classify_request
from core.plan import ClassificationDecision

CONFIDENCE_THRESHOLD = 0.6


def classify_node(
    state: AgentState,
    classify_fn: Callable[[str], ClassificationDecision] = classify_request,
) -> dict:
    """
    Classify the request. If confidence is below CONFIDENCE_THRESHOLD, this
    is the #5 gap from our design review made real: interrupt and wait for
    a human to clarify, rather than silently guessing. The clarification
    interrupt resumes with the human's answer, which is appended to the
    original request and re-classified -- not blindly trusted as a new
    classification itself.

    `classify_fn` is injectable so the interrupt/routing LOGIC can be tested
    deterministically without a live model call; production use relies on
    the default (the real classify_request).
    """
    decision = classify_fn(state["request_text"])

    if decision.confidence < CONFIDENCE_THRESHOLD:
        clarification_answer = interrupt({
            "type": "clarification_needed",
            "question": (
                f"Your request could reasonably be read more than one way "
                f"(confidence: {decision.confidence:.2f}). {decision.rationale} "
                f"Could you clarify what you're asking?"
            ),
            "original_request": state["request_text"],
            "tentative_classification": decision.question_type.value,
        })
        combined_request = f"{state['request_text']}\n\nClarification: {clarification_answer}"
        decision = classify_fn(combined_request)

    return {"classification": decision}