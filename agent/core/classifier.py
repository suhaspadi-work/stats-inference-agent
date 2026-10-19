from __future__ import annotations
from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr
from langchain.chat_models import init_chat_model

from core.plan import ClassificationDecision, AlternativeConsidered, QuestionType, CausalStatus
from core.wrangling import describe_operation
from core.model_safe_view import ModelSafeView, assert_model_safe
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
    assert_model_safe(profile, "profile")
    model = init_chat_model(model_name, model_provider="groq")
    structured_model = model.with_structured_output(_DraftPlanOut)

    column_summary = "\n".join(
        f"- {c.name} (type: {c.dtype}, distinct values: {c.distinct_count}, missing: {c.missing_pct:.1%}"
        + (f", {c.detected_format}" if c.detected_format else "")
        + ")"
        for c in profile.columns
    )
    system_prompt = (
        "You are planning a statistical analysis. Given a request and the dataset's "
        "columns (names, types, and detected patterns only -- you do not see actual values), "
        "choose the outcome column, its type, the unit of analysis, and candidate predictor "
        "columns. You MUST choose outcome_name and every candidate_predictor from the exact "
        "column names listed below -- never invent a column name. For questions about what "
        "predicts, drives or is associated with an outcome, choose every plausible predictor "
        "column. Never choose an identifier column (one where nearly every value is distinct, "
        "such as an ID) as a predictor, and never use the outcome column as a predictor.\n\n"
        f"Available columns:\n{column_summary}"
    )

    return structured_model.invoke([
        ("system", system_prompt),
        ("human", request_text),
    ])


_CONTINUOUS_COMPATIBLE_DTYPES = {"int64", "float64", "int32", "float32"}


def validate_draft_plan_fields(fields: _DraftPlanOut, profile: ModelSafeView) -> None:
    """
    Structural guard against two distinct failure modes: a hallucinated
    column name (not present in the dataset at all), and a type mismatch
    (a column that exists, but whose actual dtype can't support the chosen
    OutcomeType). The second check exists because a model can correctly
    identify that a column named "total_spend" is the right one to analyze
    while still being wrong about whether it's numeric -- the column's own
    declared dtype in the profile is the ground truth here, not the
    model's assumption from the column's name or the user's phrasing.
    """
    valid_names = {c.name for c in profile.columns}
    chosen = [fields.outcome_name, *fields.candidate_predictors]
    invalid = [name for name in chosen if name not in valid_names]
    if invalid:
        raise ValueError(f"Model chose column(s) not present in the dataset: {invalid}")

    if fields.outcome_type == OutcomeType.CONTINUOUS:
        outcome_col = next(c for c in profile.columns if c.name == fields.outcome_name)
        if outcome_col.dtype not in _CONTINUOUS_COMPATIBLE_DTYPES:
            raise ValueError(
                f"Outcome column '{fields.outcome_name}' was classified as continuous, but its "
                f"actual data type is '{outcome_col.dtype}', which cannot support a continuous "
                f"statistical test. This may mean the column is text/categorical, or the wrong "
                f"column was selected for this analysis."
            )

        
class _ProposedOperationOut(BaseModel):
    issue_index: int = Field(description="Index into the detected issues list this choice addresses")
    candidate_id: str = Field(description="The id of the chosen option for this issue, exactly as listed")
    rationale: str = Field(description="Why this option is the right fix for this specific issue")
    alternative_candidate_id: str = Field(description="The id of a different option of the same issue that was "
                                                      "considered and not chosen (leave_as_is is allowed)")
    alternative_rejected_because: str
    # M2.3b: for a joinable_table issue, whether the question needs the other table; "apply" for every other issue
    recommendation: Literal["apply", "skip"] = Field(
        default="apply", description="For a joinable_table issue: 'apply' when the question needs the other table, "
                                     "'skip' when it does not; 'apply' for every other issue")
    # M2.3a: True when this proposal came from the one corrected answer (not part of the schema the model sees)
    _retried: bool = PrivateAttr(default=False)


class _WranglingProposalsOut(BaseModel):
    proposals: list[_ProposedOperationOut]


def _options(candidates) -> list:
    """The options shown to the model: eligible candidates other than leave_as_is, as id and plain words only."""
    return [c for c in candidates if c["eligible"] and c["id"] != "leave_as_is"]


def propose_wrangling_operations(
    issues: list[CandidateIssue], candidates: list, request_text: str | None = None,
    model_name: str = "openai/gpt-oss-120b",
) -> list[_ProposedOperationOut]:
    """
    M2.3a: the code enumerates the candidate fixes of every issue; the model chooses one candidate per issue, in a
    single call, writes the rationale and names a different candidate as the alternative. The prompt holds the
    issues and each option's id with a plain description of the operation (no preview, no count, no fill value).
    The choice is validated here. M2.3b: the prompt also holds the user's question (when given), and a
    joinable_table issue's description names the carried table's columns (names only, never a value).
    """
    if not issues:
        return []
    if len(candidates) != len(issues):
        raise ValueError("There must be one candidate list per issue.")

    options = [[{"id": c["id"], "description": describe_operation(c["op_type"], c["params"])} for c in _options(cands)]
               for cands in candidates]
    assert_model_safe(issues, "issues")
    assert_model_safe(options, "options")
    if request_text is not None:
        assert_model_safe(request_text, "request_text")
    model = init_chat_model(model_name, model_provider="groq")
    structured_model = model.with_structured_output(_WranglingProposalsOut)

    blocks = []
    for i, issue in enumerate(issues):
        lines = [f"[{i}] {issue.issue_type}: {issue.description} (columns: {', '.join(issue.affected_columns)}, "
                 f"evidence: {issue.evidence})", "    options:"]
        lines += [f"      {o['id']}: {o['description']}" for o in options[i]]
        blocks.append("\n".join(lines))
    system_prompt = (
        "You are choosing data-cleaning operations for detected data-quality issues. For each issue, choose "
        "exactly one of its listed options by its id (candidate_id) and explain why. Then name one different "
        "option id of the same issue as the alternative you considered and did not choose (alternative_candidate_id); "
        "'leave_as_is' (keep the data as it is) is also a valid alternative. Choose exactly one option per issue, "
        "referencing its index. Copy each option id exactly as written, including every part of it."
        + (" For a joinable_table issue, say in 'recommendation' whether the question needs the other table "
           "('apply') or not ('skip'), and in 'rationale' why; you must still choose the best join option so the user "
           "can decide." if any(i.issue_type == "joinable_table" for i in issues) else "")
        + (f"\n\nThe user's question: {request_text}" if request_text is not None else "")
        + "\n\nDetected issues:\n" + "\n".join(blocks)
    )

    messages = [("system", system_prompt), ("human", "Choose a fix for each detected issue.")]
    result = structured_model.invoke(messages)
    try:
        _validate_choices(result.proposals, issues, candidates)
        return result.proposals
    except ValueError as first:
        # one correction: the error names only option ids and indexes (no value, no number from the data)
        retry = messages + [("human", f"Your previous answer was rejected: {first}. Choose again for every issue, "
                                      "copying each option id exactly as listed.")]
    result = structured_model.invoke(retry)
    try:
        _validate_choices(result.proposals, issues, candidates)
    except ValueError as second:
        raise ValueError(f"{second} (the model was asked once to correct its choice and did not).") from second
    for proposal in result.proposals:
        proposal._retried = True
    return result.proposals


def _validate_choices(proposals, issues, candidates) -> None:
    seen = set()
    for p in proposals:
        if not isinstance(p.issue_index, int) or not 0 <= p.issue_index < len(issues):
            raise ValueError(f"The model chose for issue index {p.issue_index}, which does not exist.")
        if p.issue_index in seen:
            raise ValueError(f"The model chose more than once for issue {p.issue_index}.")
        seen.add(p.issue_index)
        ids = {c["id"] for c in candidates[p.issue_index]}
        eligible = {c["id"] for c in _options(candidates[p.issue_index])}
        if p.candidate_id not in eligible:
            raise ValueError(f"The model chose '{p.candidate_id}' for issue {p.issue_index}, which is not one of its "
                             f"eligible options: {sorted(eligible)}.")
        if p.alternative_candidate_id not in ids or p.alternative_candidate_id == p.candidate_id:
            raise ValueError(f"The model's alternative '{p.alternative_candidate_id}' for issue {p.issue_index} must be "
                             "a different option of the same issue.")
    missing = sorted(set(range(len(issues))) - seen)
    if missing:
        raise ValueError(f"The model made no choice for issue(s) {missing}.")