from __future__ import annotations
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

from core.plan import ClassificationDecision, AlternativeConsidered, QuestionType, CausalStatus
from core.model_safe_view import ModelSafeView
from core.plan import OutcomeType
from core.detection import CandidateIssue


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
class _DraftPlanOut(BaseModel):
    outcome_name: str = Field(description="The column name to use as the outcome/dependent variable")
    outcome_type: OutcomeType
    unit_of_analysis: str = Field(description="What one row represents, e.g. 'customer', 'transaction'")
    candidate_predictors: list[str] = Field(
        description="Column names that are plausible predictors/grouping variables for this question"
    )
    rationale: str = Field(description="Why these specific columns fit this request")


def draft_plan_fields(request_text: str, profile: ModelSafeView, model_name: str = "openai/gpt-oss-120b") -> _DraftPlanOut:
    """
    Ask the model to pick an outcome column, its type, the unit of analysis,
    and candidate predictors -- using ONLY the column names/types/patterns
    from `profile` (never raw data, per the privacy boundary). The caller is
    responsible for validating the chosen names actually exist in the
    dataset before trusting this output (see validate_draft_plan_fields).
    """
    model = init_chat_model(model_name, model_provider="groq")
    structured_model = model.with_structured_output(_DraftPlanOut)

    column_summary = "\n".join(
        f"- {c.name} (type: {c.dtype}, missing: {c.missing_pct:.1%}"
        + (f", {c.detected_format}" if c.detected_format else "")
        + ")"
        for c in profile.columns
    )
    system_prompt = (
        "You are planning a statistical analysis. Given a request and the dataset's "
        "columns (names, types, and detected patterns only -- you do not see actual values), "
        "choose the outcome column, its type, the unit of analysis, and candidate predictor "
        "columns. You MUST choose outcome_name and every candidate_predictor from the exact "
        "column names listed below -- never invent a column name.\n\n"
        f"Available columns:\n{column_summary}"
    )

    return structured_model.invoke([
        ("system", system_prompt),
        ("human", request_text),
    ])


def validate_draft_plan_fields(fields: _DraftPlanOut, profile: ModelSafeView) -> None:
    """
    Structural guard against hallucinated column names -- raises if the
    model picked a column that doesn't actually exist in the profile.
    """
    valid_names = {c.name for c in profile.columns}
    chosen = [fields.outcome_name, *fields.candidate_predictors]
    invalid = [name for name in chosen if name not in valid_names]
    if invalid:
        raise ValueError(f"Model chose column(s) not present in the dataset: {invalid}")

_ALLOWED_OP_TYPES = {"cast", "rename", "filter", "dedupe"}


class _ProposedOperationOut(BaseModel):
    issue_index: int = Field(description="Index into the candidate issues list this operation addresses")
    op_type: str = Field(description=f"One of: {sorted(_ALLOWED_OP_TYPES)}")
    params: dict = Field(description="Parameters for this op_type, matching core.wrangling's propose_* function signatures")
    rationale: str = Field(description="Why this operation is the right fix for this specific issue")
    alternative_approach: str = Field(description="One concrete alternative that was considered and rejected")
    alternative_rejected_because: str


class _WranglingProposalsOut(BaseModel):
    proposals: list[_ProposedOperationOut]


def propose_wrangling_operations(
    issues: list[CandidateIssue], model_name: str = "openai/gpt-oss-120b"
) -> list[_ProposedOperationOut]:
    """
    Given ALL detected issues at once, propose one operation per issue in a
    single call -- not one call per issue. The model must only choose from
    the existing operation whitelist (cast/rename/filter/dedupe); there is
    no 'impute' operation, so a missingness issue should be proposed as a
    filter (dropping rows with 'not_null') rather than something unsupported.
    """
    if not issues:
        return []

    model = init_chat_model(model_name, model_provider="groq")
    structured_model = model.with_structured_output(_WranglingProposalsOut)

    issues_summary = "\n".join(
        f"[{i}] {issue.issue_type}: {issue.description} (columns: {', '.join(issue.affected_columns)}, "
        f"evidence: {issue.evidence})"
        for i, issue in enumerate(issues)
    )

    system_prompt = (
        "You are proposing data-cleaning operations for detected data-quality issues. "
        f"You MUST choose op_type from exactly these options: {sorted(_ALLOWED_OP_TYPES)}. "
        "There is no imputation operation available -- if missingness needs addressing, "
        "propose a 'filter' with condition 'not_null' on the affected column to drop those rows. "
        "For duplicate rows, propose a 'dedupe'. "
        "Propose exactly one operation per issue listed below, referencing its index.\n\n"
        f"Detected issues:\n{issues_summary}"
    )

    result = structured_model.invoke([
        ("system", system_prompt),
        ("human", "Propose a fix for each detected issue."),
    ])

    invalid = [p.op_type for p in result.proposals if p.op_type not in _ALLOWED_OP_TYPES]
    if invalid:
        raise ValueError(f"Model proposed unsupported operation type(s): {invalid}")

    return result.proposals