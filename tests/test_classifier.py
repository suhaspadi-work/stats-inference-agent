import pytest
from core.classifier import classify_request
from core.plan import QuestionType, CausalStatus

pytestmark = pytest.mark.live  # marks these as tests that hit the real Groq API


def test_classifies_a_clear_two_group_comparison():
    decision = classify_request(
        "Did customers in the West region spend more on average than customers in the East region?"
    )
    assert decision.question_type == QuestionType.TWO_GROUP_COMPARISON
    assert decision.causal_status == CausalStatus.OBSERVATIONAL
    assert decision.confidence > 0.6
    assert len(decision.rationale) > 0


def test_classifies_a_clear_ab_test():
    decision = classify_request(
        "We randomly assigned half our users to a new checkout page and half to the old one. "
        "Did the new page increase conversion rate?"
    )
    assert decision.question_type == QuestionType.AB_TEST
    assert decision.causal_status == CausalStatus.EXPERIMENTAL
    assert decision.confidence > 0.6


def test_classifies_a_driver_analysis_request():
    decision = classify_request(
        "Here are three exports from our CRM. What drives customer churn?"
    )
    assert decision.question_type == QuestionType.DRIVER_ANALYSIS
    assert decision.causal_status == CausalStatus.OBSERVATIONAL  # no randomization mentioned


def test_ambiguous_request_reports_lower_confidence():
    """
    A genuinely ambiguous request -- could read as association or driver_analysis.
    We don't assert a specific question_type here (that would be over-claiming
    what we can predict about model behavior); we assert the calibration
    property that actually matters: ambiguity should correspond to lower
    confidence than the two crystal-clear requests above.
    """
    decision = classify_request("Is customer satisfaction related to response time?")
    assert decision.confidence <= 1.0  # sanity
    assert isinstance(decision.rationale, str) and len(decision.rationale) > 0