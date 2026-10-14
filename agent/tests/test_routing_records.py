import json

import pytest

import core.routing_records as routing_module
from core.capabilities import capability_for_question_type
from core.decision_log import log_update
from core.routing_records import route_record, skipped_draft_record
from core.model_safe_view import ColumnSummary, ModelSafeView
from core.plan import AnalysisPlan, CausalStatus, OutcomeType, QuestionType
from core.routing import route_for_plan


def make_plan(question_type, predictors, outcome="total_spend", unit="customer"):
    return AnalysisPlan(
        request_text="x", session_id="session-abc",
        question_type=question_type, causal_status=CausalStatus.OBSERVATIONAL,
        outcome_name=outcome, outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis=unit, data_sources=("x@v1",),
        candidate_predictors=tuple(predictors),
    )


def make_profile(columns):
    """columns: {name: (dtype, distinct_count)}"""
    return ModelSafeView(
        dataset_handle="x@v1", row_count=100, column_count=len(columns),
        columns=tuple(
            ColumnSummary(name=n, dtype=d, missing_count=0, missing_pct=0.0, distinct_count=k)
            for n, (d, k) in columns.items()
        ),
    )


PROFILE = make_profile({
    "region": ("object", 2), "tier": ("object", 3), "tenure": ("float64", 80), "total_spend": ("float64", 100),
})


def _by_option(record):
    return {c.option: c for c in record.candidates}


# --- skipped draft --------------------------------------------------------------

def test_skipped_draft_record():
    plan = make_plan(QuestionType.DESCRIPTIVE, (), outcome="(whole dataset)", unit="row")
    r = skipped_draft_record(plan)
    assert r.decision_point == "planner.draft_plan"
    assert (r.chosen, r.decided_by) == ("skipped", "code_rule")
    c = _by_option(r)
    assert [x.option for x in r.candidates] == ["ask_model", "skip"]
    assert c["ask_model"].eligible is False and c["ask_model"].reason.startswith("not eligible: the question is descriptive")
    assert c["skip"].eligible is True and c["skip"].reason == r.rationale
    assert r.rationale.startswith("chosen: descriptive questions describe the whole dataset")
    assert r.evidence == {
        "question_type": "descriptive",
        "placeholders": {"outcome_name": "(whole dataset)", "outcome_type": "continuous", "unit_of_analysis": "row"},
        "placeholders_chosen_by": "nobody; fixed values used because no model choice was made",
    }
    assert all(x.generated_by == "code" for x in r.candidates)


# --- descriptive route ------------------------------------------------------------

def test_descriptive_route_does_not_evaluate_analyses(monkeypatch):
    def must_not_be_called(*args, **kwargs):
        raise AssertionError("evaluate_all must not run for a descriptive plan")
    monkeypatch.setattr(routing_module, "evaluate_all", must_not_be_called)

    plan = make_plan(QuestionType.DESCRIPTIVE, (), outcome="(whole dataset)")
    r = route_record(plan, PROFILE, route_for_plan(plan, PROFILE))  # M0.3: the route is passed in
    assert r.decision_point == "planner.route"
    assert (r.chosen, r.decided_by) == ("descriptive", "code_rule")
    c = _by_option(r)
    assert [x.option for x in r.candidates] == ["descriptive", "inference"]
    assert c["descriptive"].eligible is True
    assert c["descriptive"].reason.startswith("chosen: question type 'descriptive' routes to the descriptive diagnostics")
    # M0.3: the rationale is the route's reason, and the evidence names the classifier's question type and status
    assert r.rationale == "question type 'descriptive' routes to the descriptive diagnostics"
    assert c["inference"].eligible is False
    assert r.evidence == {"question_type": "descriptive", "capability": "descriptive",
                          "skipped_steps": ["select_method", "freeze", "execute"],
                          "classifier_question_type": "descriptive", "status": "ok"}


# --- inference routes -------------------------------------------------------------

def test_two_group_route_with_a_two_level_column():
    plan = make_plan(QuestionType.TWO_GROUP_COMPARISON, ("region",))
    route = route_for_plan(plan, PROFILE)
    r = route_record(plan, PROFILE, route)  # M0.3: the route is passed in
    c = _by_option(r)
    assert r.chosen == "two_group"
    assert [x.option for x in r.candidates] == ["two_group", "multi_group", "regression"]
    assert [c[o].eligible for o in ("two_group", "multi_group", "regression")] == [True, False, True]
    assert c["two_group"].reason == ("chosen: question type 'two_group_comparison' routes to this analysis; "
                                     "applicability: 'region' has exactly 2 levels and the outcome is numeric.")
    assert c["multi_group"].reason == ("not chosen: question type 'two_group_comparison' routes to two_group; "
                                       "applicability: 'region' has 2 levels; a multi-group comparison needs 3 or more.")
    assert c["regression"].reason.startswith("not chosen: question type 'two_group_comparison' routes to two_group;")
    assert r.rationale == route["reason"]  # M0.3: the rationale is the route's reason
    assert r.evidence == {"question_type": "two_group_comparison", "classifier_question_type": "two_group_comparison",
                          "status": "ok", "capability": "two_group", "skipped_steps": [],
                          "chosen_is_eligible": True, "grouping_max_levels": 10}


def test_multi_group_route_with_a_three_level_column():
    r = route_record(make_plan(QuestionType.MULTI_GROUP_COMPARISON, ("tier",)), PROFILE)
    c = _by_option(r)
    assert r.chosen == "multi_group"
    assert [c[o].eligible for o in ("two_group", "multi_group", "regression")] == [False, True, True]
    assert r.evidence["chosen_is_eligible"] is True
    assert "routes to multi_group" in c["two_group"].reason


def test_driver_analysis_routes_to_regression():
    r = route_record(make_plan(QuestionType.DRIVER_ANALYSIS, ("region", "tenure")), PROFILE)
    c = _by_option(r)
    assert r.chosen == "regression"
    assert c["regression"].eligible is True
    assert c["regression"].reason.startswith("chosen: question type 'driver_analysis' routes to this analysis;")
    assert c["two_group"].eligible is False and "exactly one grouping column; got 2" in c["two_group"].reason


def test_g2_two_group_question_with_a_three_level_predictor():
    # M0.3: the route is reconciled to multi_group instead of choosing the ineligible two_group
    plan = make_plan(QuestionType.TWO_GROUP_COMPARISON, ("tier",))
    r = route_record(plan, PROFILE, route_for_plan(plan, PROFILE))
    c = _by_option(r)
    assert r.chosen == "multi_group"
    assert r.evidence["status"] == "reconciled"
    assert c["two_group"].eligible is False
    assert "'tier' has 3 levels; a two-group comparison needs exactly 2." in c["two_group"].reason
    assert r.evidence["chosen_is_eligible"] is True
    assert c["multi_group"].eligible is True
    assert c["multi_group"].reason.startswith(
        "chosen: the data supports this analysis, although question type 'two_group_comparison' routes to two_group;")


def test_ab_test_routes_to_two_group():
    r = route_record(make_plan(QuestionType.AB_TEST, ("region",)), PROFILE)
    assert r.chosen == "two_group"
    assert "borrowed_route" not in r.evidence


def test_association_is_unsupported():
    # M0.3: association has no analysis, so the route is unsupported (no borrowed two-group route any more)
    plan = make_plan(QuestionType.ASSOCIATION, ("region",))
    r = route_record(plan, PROFILE, route_for_plan(plan, PROFILE))
    assert r.chosen == "unsupported"
    assert r.evidence["status"] == "unsupported"
    assert "borrowed_route" not in r.evidence
    assert r.evidence["message"].startswith("Association questions are not supported yet.")


def test_guard_association_still_borrows_the_two_group_capability():
    """If this fails, association got its own capability: revisit the association rule in core/routing.py."""
    assert capability_for_question_type("association") is capability_for_question_type("two_group_comparison")


# --- plain data, no mutation -----------------------------------------------------

def test_candidate_evidence_is_plain_and_records_pass_log_update():
    plans = [
        make_plan(QuestionType.TWO_GROUP_COMPARISON, ("region",)),
        make_plan(QuestionType.TWO_GROUP_COMPARISON, ("missing_col",)),  # missing-column evidence is a list
        make_plan(QuestionType.DRIVER_ANALYSIS, ("region", "tenure")),
        make_plan(QuestionType.DESCRIPTIVE, (), outcome="(whole dataset)"),
    ]
    before = [repr(p) for p in plans] + [repr(PROFILE)]
    records = [route_record(p, PROFILE) for p in plans] + [skipped_draft_record(plans[-1])]
    for r in records:
        for c in r.candidates:
            assert not any(isinstance(v, tuple) for v in c.evidence.values())
    missing = _by_option(records[1])["two_group"].evidence
    assert missing == {"missing_columns": ["missing_col"]}
    json.dumps(log_update(*records))
    assert [repr(p) for p in plans] + [repr(PROFILE)] == before

