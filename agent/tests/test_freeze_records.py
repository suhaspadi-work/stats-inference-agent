import json

import pytest

from core.classifier import _DraftPlanOut
from core.decision_log import log_update
from core.draft_plan_records import draft_plan_records, predictor_candidates
from core.freeze_records import edited_predictors_record, plan_approval_record
from core.model_safe_view import ColumnSummary, ModelSafeView
from core.plan import (
    AlternativeConsidered, AnalysisPlan, CausalStatus, MethodDecision, OutcomeType, QuestionType,
)


def _plan(predictors=("region",)):
    plan = AnalysisPlan(
        request_text="Did West spend more than East?", session_id="session-abc",
        question_type=QuestionType.TWO_GROUP_COMPARISON, causal_status=CausalStatus.OBSERVATIONAL,
        outcome_name="total_spend", outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="customer", data_sources=("customers@v1",),
        candidate_predictors=tuple(predictors),
    )
    return plan.record_method_decision(MethodDecision(
        stage="test_selection", chosen_method="welch_t_test", rationale="test",
        alternatives=(AlternativeConsidered(method="mann_whitney_u", rejected_because="test"),),
    ))


def _profile():
    return ModelSafeView(
        dataset_handle="customers@v1", row_count=100, column_count=3,
        columns=(
            ColumnSummary(name="region", dtype="object", missing_pct=0.0, missing_count=0, distinct_count=2),
            ColumnSummary(name="plan_type", dtype="object", missing_pct=0.02, missing_count=2, distinct_count=3),
            ColumnSummary(name="total_spend", dtype="float64", missing_pct=0.0, missing_count=0, distinct_count=100),
        ),
    )


def _by_option(record):
    return {c.option: c for c in record.candidates}


# --- plan_approval_record -----------------------------------------------------

def test_approve():
    plan = _plan()
    r = plan_approval_record(plan, {"action": "approve"}, "sel-1")
    assert r.decision_point == "planner.plan_approval"
    assert (r.chosen, r.decided_by, r.rationale) == ("approve", "human", "approved as drafted")
    assert r.overridden_by_human is False and r.original_choice is None
    assert r.evidence == {
        "plan_id": plan.id, "question_type": "two_group_comparison", "causal_status": "observational",
        "outcome_name": "total_spend", "outcome_type": "continuous", "predictors_before": ["region"],
        "method_at_approval": "welch_t_test", "selection_record_id": "sel-1",
        "received_action": "approve", "frozen": True,
    }


def test_edit():
    r = plan_approval_record(_plan(), {"action": "edit", "candidate_predictors": ["region", "plan_type"]},
                             "sel-1", predictors_record_id="pred-2")
    assert (r.chosen, r.decided_by) == ("edit", "human")
    assert r.overridden_by_human is True
    assert r.original_choice == "approve as drafted: predictors [region]"
    assert r.rationale == "predictors edited before freezing"
    assert r.evidence["predictors_before"] == ["region"]
    assert r.evidence["predictors_after"] == ["region", "plan_type"]
    assert r.evidence["predictors_record_id"] == "pred-2"
    assert r.evidence["frozen"] is True


@pytest.mark.parametrize("decision, reason", [
    ({"action": "reject", "reason": "Wrong predictors"}, "Wrong predictors"),
    ({"action": "reject"}, "no reason given"),
])
def test_reject(decision, reason):
    r = plan_approval_record(_plan(), decision, None)
    assert (r.chosen, r.decided_by, r.rationale) == ("reject", "human", reason)
    assert r.evidence["frozen"] is False


@pytest.mark.parametrize("decision, received", [({"action": "maybe"}, "maybe"), ({}, None)])
def test_unrecognized_or_missing_action_freezes_by_code_rule(decision, received):
    r = plan_approval_record(_plan(), decision, "sel-1")
    assert (r.chosen, r.decided_by) == ("approve", "code_rule")
    assert "G36" in r.rationale and f"'{received}'" in r.rationale
    assert r.evidence["received_action"] == received
    assert r.evidence["frozen"] is True


@pytest.mark.parametrize("decision", [
    {"action": "approve"}, {"action": "edit", "candidate_predictors": []}, {"action": "reject"},
    {"action": "maybe"}, {},
])
def test_candidates_and_edit_only_fields(decision):
    r = plan_approval_record(_plan(), decision, "sel-1", predictors_record_id="pred-2")
    assert [(c.option, c.eligible, c.generated_by) for c in r.candidates] == [
        ("approve", True, "code"), ("edit", True, "code"), ("reject", True, "code")]
    is_edit = decision.get("action") == "edit"
    assert ("predictors_after" in r.evidence) is is_edit
    assert ("predictors_record_id" in r.evidence) is is_edit
    assert isinstance(r.evidence["predictors_before"], list)
    json.dumps(r.to_dict())


# --- edited_predictors_record --------------------------------------------------

def test_edited_predictors_record():
    plan, profile = _plan(), _profile()
    r = edited_predictors_record(plan, ("region", "plan_type"), profile, "draft-pred-1", "sel-1")
    assert r.decision_point == "planner.predictors"
    assert (r.chosen, r.decided_by) == ("region, plan_type", "human")
    assert r.overridden_by_human is True
    assert r.original_choice == "region"
    assert r.supersedes == "draft-pred-1"
    assert "not recomputed (G5)" in r.rationale
    assert r.evidence == {
        "predictors_before": ["region"], "predictors_after": ["region", "plan_type"],
        "method_selected_before_edit": "welch_t_test", "selection_record_id": "sel-1",
        "diagnostics_recomputed": False, "n_columns": 3, "n_chosen": 2,
    }
    c = _by_option(r)
    assert [x.option for x in r.candidates] == ["region", "plan_type", "total_spend"]
    assert c["total_spend"].eligible is False
    assert c["total_spend"].reason == "the outcome cannot also be a predictor"
    assert c["plan_type"].reason == "chosen by the model; see rationale"
    assert c["plan_type"].evidence == {"dtype": "object", "distinct_count": 3, "missing_pct": 0.02}


def test_empty_edit_gives_none():
    r = edited_predictors_record(_plan(), (), _profile(), None, None)
    assert r.chosen == "none"
    assert r.evidence["predictors_after"] == [] and r.evidence["n_chosen"] == 0
    assert r.supersedes is None


def test_empty_original_gives_none():
    r = edited_predictors_record(_plan(predictors=()), ("region",), _profile(), None, None)
    assert r.original_choice == "none"


def test_predictor_candidates_match_the_draft_record():
    profile = _profile()
    fields = _DraftPlanOut(outcome_name="total_spend", outcome_type=OutcomeType.CONTINUOUS,
                           unit_of_analysis="customer", candidate_predictors=["region"], rationale="r")
    draft_predictors = draft_plan_records(fields, profile)[2]
    assert predictor_candidates(profile, "total_spend", ["region"]) == draft_predictors.candidates
    edited = edited_predictors_record(_plan(), ["region"], profile, None, None)
    assert edited.candidates == draft_predictors.candidates


# --- plain data, no mutation ---------------------------------------------------

def test_records_pass_log_update_and_inputs_are_not_mutated():
    plan, profile = _plan(), _profile()
    decision = {"action": "edit", "candidate_predictors": ["region", "plan_type"]}
    before = (repr(plan), repr(profile), repr(decision))
    edited = edited_predictors_record(plan, tuple(decision["candidate_predictors"]), profile, "p", "s")
    approval = plan_approval_record(plan, decision, "s", predictors_record_id=edited.id)
    update = log_update(edited, approval, plan_approval_record(plan, {}, None))
    json.dumps(update)
    assert (repr(plan), repr(profile), repr(decision)) == before
