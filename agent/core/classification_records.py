"""
Decision records for request classification (M0.2 step 3). Pure functions:
no graph and no model calls. Candidate lists come from code (the QuestionType
and CausalStatus enums and the capability routing table); the model only
supplies the choice, its rationale, and reasons for alternatives it named.
"""
from __future__ import annotations
from typing import Optional

from core.capabilities import capability_for_question_type
from core.decision_record import Candidate, DecisionRecord
from core.plan import CausalStatus, ClassificationDecision, QuestionType

QUESTION_TYPE_POINT = "intake.question_type"
CAUSAL_STATUS_POINT = "intake.causal_status"
CLARIFICATION_GATE_POINT = "intake.clarification_gate"
CLARIFICATION_POINT = "intake.clarification"

# Question types that route to a capability built for a different question.
# The guard test in tests/test_classification_records.py fails if association
# gets its own capability, so this reason is revisited then.
_BORROWED_ROUTES = {
    "association": "no association capability yet (G1); falls into the two-group code",
}

_CHOSEN_REASON = "chosen by the classifier; see rationale"
_NO_COMMENT_REASON = "the classifier did not comment on this option"


def _model_reasons(decision: ClassificationDecision) -> dict[str, str]:
    """First rejected_because per alternative method, keyed by lower-cased method name."""
    reasons: dict[str, str] = {}
    for alt in decision.alternatives:
        reasons.setdefault(alt.method.strip().lower(), alt.rejected_because)
    return reasons


def _option_candidate(option: str, chosen: str, eligible: bool, model_reasons: dict[str, str],
                      evidence: dict, ineligible_reason: Optional[str] = None) -> Candidate:
    if ineligible_reason is not None:
        # The code-determined reason wins; a model comment on this option is kept in evidence.
        reason, generated_by = ineligible_reason, "code"
        if option != chosen and option in model_reasons:
            evidence = {**evidence, "model_rejected_because": model_reasons[option]}
    elif option == chosen:
        reason, generated_by = _CHOSEN_REASON, "code"
    elif option in model_reasons:
        reason, generated_by = model_reasons[option], "model"
    else:
        reason, generated_by = _NO_COMMENT_REASON, "code"
    return Candidate(option=option, eligible=eligible, reason=reason,
                     evidence=evidence, generated_by=generated_by)


def _base_evidence(decision: ClassificationDecision, pass_number: int) -> dict:
    return {"confidence": float(decision.confidence), "pass": int(pass_number)}


def question_type_record(decision: ClassificationDecision, pass_number: int,
                         supersedes: Optional[str] = None) -> DecisionRecord:
    chosen = decision.question_type.value
    model_reasons = _model_reasons(decision)
    candidates = []
    for qt in QuestionType:
        option = qt.value
        routes_to = capability_for_question_type(option).name
        ineligible = _BORROWED_ROUTES.get(option)
        candidates.append(_option_candidate(
            option, chosen, eligible=ineligible is None, model_reasons=model_reasons,
            evidence={"routes_to": routes_to}, ineligible_reason=ineligible,
        ))

    known = {qt.value for qt in QuestionType} | {cs.value for cs in CausalStatus}
    evidence = _base_evidence(decision, pass_number)
    unmatched = [
        {"method": a.method, "rejected_because": a.rejected_because}
        for a in decision.alternatives
        if a.method.strip().lower() not in known
    ]
    if unmatched:
        evidence["unmatched_model_alternatives"] = unmatched

    return DecisionRecord(
        decision_point=QUESTION_TYPE_POINT, chosen=chosen, decided_by="model",
        candidates=tuple(candidates), rationale=decision.rationale,
        evidence=evidence, supersedes=supersedes,
    )


def causal_status_record(decision: ClassificationDecision, pass_number: int,
                         supersedes: Optional[str] = None) -> DecisionRecord:
    chosen = decision.causal_status.value
    model_reasons = _model_reasons(decision)
    candidates = tuple(
        _option_candidate(cs.value, chosen, eligible=True, model_reasons=model_reasons, evidence={})
        for cs in CausalStatus
    )
    return DecisionRecord(
        decision_point=CAUSAL_STATUS_POINT, chosen=chosen, decided_by="model",
        candidates=candidates, rationale=decision.rationale,
        evidence=_base_evidence(decision, pass_number), supersedes=supersedes,
    )


def _fmt_pair(a: float, b: float) -> tuple[str, str]:
    """Two decimals, unless that would print two different numbers as the same value."""
    fa, fb = f"{a:.2f}", f"{b:.2f}"
    if fa == fb and a != b:
        return repr(a), repr(b)
    return fa, fb


def clarification_gate_record(confidence: float, threshold: float) -> DecisionRecord:
    confidence, threshold = float(confidence), float(threshold)
    c, t = _fmt_pair(confidence, threshold)
    below = confidence < threshold
    if below:
        ask_reason = f"confidence {c} is below the threshold {t}"
        proceed_reason = f"confidence {c} does not meet the threshold {t}"
    else:
        ask_reason = f"confidence {c} is not below the threshold {t}"
        proceed_reason = f"confidence {c} meets the threshold {t}"
    return DecisionRecord(
        decision_point=CLARIFICATION_GATE_POINT,
        chosen="ask_human" if below else "proceed",
        decided_by="code_rule",
        candidates=(
            Candidate(option="ask_human", eligible=True, reason=ask_reason),
            Candidate(option="proceed", eligible=True, reason=proceed_reason),
        ),
        rationale=ask_reason if below else proceed_reason,
        evidence={"confidence": confidence, "threshold": threshold},
    )


def clarification_record(answer: str) -> DecisionRecord:
    return DecisionRecord(
        decision_point=CLARIFICATION_POINT, chosen="clarification_provided",
        decided_by="human", rationale=str(answer),
    )
