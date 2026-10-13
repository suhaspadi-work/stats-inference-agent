import json

import numpy as np
import pytest

from core.capabilities import capability_for_question_type
from core.classification_records import (
    causal_status_record, clarification_gate_record, clarification_record, question_type_record,
)
from core.decision_log import log_update
from core.plan import AlternativeConsidered, CausalStatus, ClassificationDecision, QuestionType


def _decision(qt=QuestionType.TWO_GROUP_COMPARISON, cs=CausalStatus.OBSERVATIONAL,
              alternatives=(), confidence=0.85):
    return ClassificationDecision(
        question_type=qt, causal_status=cs, rationale="Two named regions on one outcome.",
        alternatives=tuple(alternatives), confidence=confidence, evidence={},
    )


def _by_option(record):
    return {c.option: c for c in record.candidates}


# --- question type ---------------------------------------------------------

EXPECTED_ROUTES = {
    "descriptive": "descriptive",
    "two_group_comparison": "two_group",
    "multi_group_comparison": "multi_group",
    "ab_test": "two_group",
    "association": "two_group",
    "driver_analysis": "regression",
}


def test_all_question_types_are_candidates_with_routes_and_eligibility():
    r = question_type_record(_decision(), pass_number=1)
    cands = _by_option(r)
    assert set(cands) == {qt.value for qt in QuestionType} == set(EXPECTED_ROUTES)
    for option, c in cands.items():
        assert c.evidence["routes_to"] == EXPECTED_ROUTES[option]
        assert c.eligible is (option != "association")
    assert r.decision_point == "intake.question_type"
    assert r.chosen == "two_group_comparison"
    assert r.decided_by == "model"
    assert r.rationale == "Two named regions on one outcome."


def test_association_is_ineligible_with_code_reason():
    c = _by_option(question_type_record(_decision(), pass_number=1))["association"]
    assert c.eligible is False
    assert c.reason == "no association capability yet (G1); falls into the two-group code"
    assert c.generated_by == "code"


def test_guard_association_still_borrows_the_two_group_capability():
    """If this fails, association got its own capability: revisit the hardcoded reason in classification_records."""
    assert capability_for_question_type("association") is capability_for_question_type("two_group_comparison")


def test_chosen_and_uncommented_reasons():
    cands = _by_option(question_type_record(_decision(), pass_number=1))
    assert cands["two_group_comparison"].reason == "chosen by the classifier; see rationale"
    assert cands["two_group_comparison"].generated_by == "code"
    assert cands["descriptive"].reason == "the classifier did not comment on this option"
    assert cands["descriptive"].generated_by == "code"


def test_model_alternatives_attach_case_insensitively_as_model_generated():
    alts = [
        AlternativeConsidered(method="Multi_Group_Comparison", rejected_because="only two regions are named"),
        AlternativeConsidered(method="ab_test", rejected_because="no randomized treatment is described"),
    ]
    cands = _by_option(question_type_record(_decision(alternatives=alts), pass_number=1))
    assert cands["multi_group_comparison"].reason == "only two regions are named"
    assert cands["multi_group_comparison"].generated_by == "model"
    assert cands["ab_test"].reason == "no randomized treatment is described"
    assert cands["ab_test"].generated_by == "model"


def test_model_comment_on_association_kept_in_evidence():
    alts = [AlternativeConsidered(method="association", rejected_because="the user names groups")]
    c = _by_option(question_type_record(_decision(alternatives=alts), pass_number=1))["association"]
    assert c.reason.startswith("no association capability yet (G1)")
    assert c.evidence["model_rejected_because"] == "the user names groups"


def test_unmatched_model_alternatives_land_in_evidence():
    alts = [
        AlternativeConsidered(method="time_series", rejected_because="no time column"),
        AlternativeConsidered(method="experimental", rejected_because="no randomization"),
        AlternativeConsidered(method="ab_test", rejected_because="no treatment"),
    ]
    r = question_type_record(_decision(alternatives=alts), pass_number=1)
    assert r.evidence["unmatched_model_alternatives"] == [
        {"method": "time_series", "rejected_because": "no time column"},
    ]


def test_no_unmatched_key_when_everything_matches():
    r = question_type_record(_decision(), pass_number=1)
    assert "unmatched_model_alternatives" not in r.evidence


# --- causal status ---------------------------------------------------------

def test_causal_status_has_two_eligible_candidates():
    alts = [AlternativeConsidered(method="Experimental", rejected_because="regions are not assigned at random")]
    r = causal_status_record(_decision(alternatives=alts), pass_number=1)
    cands = _by_option(r)
    assert set(cands) == {"experimental", "observational"}
    assert all(c.eligible is True for c in cands.values())
    assert r.decision_point == "intake.causal_status"
    assert r.chosen == "observational"
    assert r.decided_by == "model"
    assert cands["observational"].reason == "chosen by the classifier; see rationale"
    assert cands["experimental"].reason == "regions are not assigned at random"
    assert cands["experimental"].generated_by == "model"


# --- evidence and supersedes -----------------------------------------------

@pytest.mark.parametrize("builder", [question_type_record, causal_status_record])
def test_confidence_is_plain_float_and_pass_recorded(builder):
    r = builder(_decision(confidence=np.float64(0.85)), pass_number=2)
    assert type(r.evidence["confidence"]) is float
    assert r.evidence["confidence"] == 0.85
    assert r.evidence["pass"] == 2


@pytest.mark.parametrize("builder", [question_type_record, causal_status_record])
def test_supersedes_only_when_passed(builder):
    assert builder(_decision(), pass_number=1).supersedes is None
    assert builder(_decision(), pass_number=2, supersedes="abc-123").supersedes == "abc-123"


# --- clarification gate -----------------------------------------------------

def test_gate_ask_human_branch_carries_numbers():
    r = clarification_gate_record(0.3, 0.6)
    assert r.decision_point == "intake.clarification_gate"
    assert r.chosen == "ask_human"
    assert r.decided_by == "code_rule"
    cands = _by_option(r)
    assert cands["ask_human"].reason == "confidence 0.30 is below the threshold 0.60"
    assert "0.30" in cands["proceed"].reason and "0.60" in cands["proceed"].reason
    assert all(c.eligible is True for c in cands.values())
    assert r.evidence == {"confidence": 0.3, "threshold": 0.6}


def test_gate_proceed_branch_carries_numbers():
    r = clarification_gate_record(0.9, 0.6)
    assert r.chosen == "proceed"
    cands = _by_option(r)
    assert cands["proceed"].reason == "confidence 0.90 meets the threshold 0.60"
    assert "0.90" in cands["ask_human"].reason and "0.60" in cands["ask_human"].reason


def test_gate_at_threshold_proceeds():
    assert clarification_gate_record(0.6, 0.6).chosen == "proceed"


def test_gate_reason_never_prints_different_numbers_as_equal():
    r = clarification_gate_record(0.599, 0.6)
    assert r.chosen == "ask_human"
    assert "0.599" in _by_option(r)["ask_human"].reason


# --- clarification ----------------------------------------------------------

def test_clarification_record():
    r = clarification_record("I mean West versus East only")
    assert r.decision_point == "intake.clarification"
    assert r.chosen == "clarification_provided"
    assert r.decided_by == "human"
    assert r.rationale == "I mean West versus East only"


# --- every record passes log_update ----------------------------------------

def test_every_record_passes_log_update():
    alts = [AlternativeConsidered(method="ab_test", rejected_because="no treatment"),
            AlternativeConsidered(method="something_else", rejected_because="n/a")]
    d = _decision(alternatives=alts, confidence=np.float64(0.4))
    records = [
        question_type_record(d, pass_number=1),
        causal_status_record(d, pass_number=1),
        clarification_gate_record(d.confidence, 0.6),
        clarification_record("answer"),
        question_type_record(d, pass_number=2, supersedes="x"),
        causal_status_record(d, pass_number=2, supersedes="y"),
    ]
    update = log_update(*records)
    assert len(update["decision_log"]) == 6
    json.dumps(update)
