from __future__ import annotations
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

from core.plan import ClassificationDecision, AlternativeConsidered, QuestionType, CausalStatus


class _AlternativeOut(BaseModel):
    method: str = Field(description="A question_type or causal_status that was considered and rejected")
    rejected_because: str = Field(description="Concrete reason this alternative was not chosen")


class _ClassificationOut(BaseModel):
    """
    The structured shape the model must fill in. Using a Pydantic schema
    bound directly to the model (rather than parsing free text) means the
    model's output is GUARANTEED to have these fields in these types —
    no regex, no "hope it formatted correctly."
    """
    question_type: QuestionType
    causal_status: CausalStatus
    rationale: str = Field(description="Why this question_type and causal_status fit this specific request")
    alternatives: list[_AlternativeOut] = Field(
        description="Other question_types genuinely considered and rejected, with concrete reasons"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="How confident you are in this classification. Use a LOW score (below 0.6) "
                    "if the request is genuinely ambiguous between two readings.",
    )


_SYSTEM_PROMPT = """You are the request classifier for a statistical inference agent.

Given a natural-language analysis request, classify it along two axes:

1. question_type — one of:
   - descriptive: just describe/summarize data, no comparison or test
   - two_group_comparison: compare exactly two named groups on an outcome
   - multi_group_comparison: compare three or more groups
   - ab_test: compare two groups where one was a deliberate experimental treatment
   - association: does X relate to Y, no causal or "driver" framing
   - driver_analysis: "what drives/causes/leads to Y" — asking what predicts or explains an outcome

2. causal_status — one of:
   - experimental: the request describes or implies random assignment to groups (e.g. an A/B test)
   - observational: the data was not randomized; groups/predictors are naturally occurring

You must also state your confidence (0.0-1.0). Use a genuinely low score (below 0.6) when the
request could reasonably be read two different ways — do not default to high confidence just to
seem decisive. A low-confidence classification will be sent to a human for clarification rather
than acted on directly, so it is always safe and correct to report low confidence when warranted.

List real alternatives you considered and rejected, not token alternatives added for form."""


def classify_request(request_text: str, model_name: str = "openai/gpt-oss-120b") -> ClassificationDecision:
    """
    Call the model with structured output to classify a natural-language
    analysis request. Returns a real ClassificationDecision — the model's
    output is validated against _ClassificationOut's schema before we ever
    touch it, so a malformed response fails loudly here, not silently later.
    """
    model = init_chat_model(model_name, model_provider="groq")
    structured_model = model.with_structured_output(_ClassificationOut)

    result: _ClassificationOut = structured_model.invoke([
        ("system", _SYSTEM_PROMPT),
        ("human", request_text),
    ])

    return ClassificationDecision(
        question_type=result.question_type,
        causal_status=result.causal_status,
        rationale=result.rationale,
        alternatives=tuple(
            AlternativeConsidered(method=a.method, rejected_because=a.rejected_because)
            for a in result.alternatives
        ),
        confidence=result.confidence,
        evidence={},  # the classifier's "evidence" is the rationale itself; no numeric evidence at this stage
    )