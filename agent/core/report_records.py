"""
The decision record for an analysis report's narrative (M1.4a, G52). Pure: no graph, no model calls, and no
import of core.nodes. The descriptive report is not guarded and writes no record.

  report.narrative   whose text became the report: the model's, the model's after one correction, or a
                     fallback written by code from the computed facts
"""
from __future__ import annotations

from core.decision_record import Candidate, DecisionRecord

REPORT_NARRATIVE_POINT = "report.narrative"

_RATIONALES = {
    "model": "the model's text passed the check",
    "model_retry": "the retried text passed the check after one correction",
    "facts_fallback": "both model texts used causal language, so the report was written from the computed facts",
    "minimal_fallback": "the fact-based text tripped the guard as well, so the minimal text was used",
}


def report_narrative_record(round_index: int, capability: str, exploratory: bool, attempts: list,
                            chosen: str) -> DecisionRecord:
    """attempts: [{"attempt", "passed", "violation": {"matched", "sentence"} or None}]; chosen as in _RATIONALES."""
    if chosen not in _RATIONALES:
        raise ValueError(f"Unknown report narrative choice '{chosen}'.")
    first = attempts[0]
    second = attempts[1] if len(attempts) > 1 else None
    first_reason = ("the model's text passed the causal-language check" if first["passed"]
                    else f"the model's text used '{first['violation']['matched']}' in: {first['violation']['sentence']}")
    if second is None:
        retry_reason = "not needed: the first text passed"
    elif second["passed"]:
        retry_reason = "the retried text passed the check"
    else:
        retry_reason = f"the retried text used '{second['violation']['matched']}' in: {second['violation']['sentence']}"
    return DecisionRecord(
        decision_point=REPORT_NARRATIVE_POINT, chosen=chosen, decided_by="code_rule",
        candidates=(
            Candidate(option="model", eligible=bool(first["passed"]), reason=first_reason),
            Candidate(option="model_retry", eligible=bool(second and second["passed"]), reason=retry_reason),
            Candidate(option="facts_fallback", eligible=True,
                      reason="always available: written by code from the computed facts"),
            Candidate(option="minimal_fallback", eligible=chosen == "minimal_fallback",
                      reason="last resort: used only when the fact-based fallback itself trips the guard"),
        ),
        rationale=_RATIONALES[chosen],
        evidence={"round": round_index, "capability": capability, "exploratory": bool(exploratory),
                  "attempts": [dict(a) for a in attempts]},
    )
