import pytest
from core.plan import (
    AnalysisPlan, QuestionType, CausalStatus, OutcomeType, PlanStatus,
    MethodDecision, AlternativeConsidered,
)


@pytest.fixture
def draft_plan():
    return AnalysisPlan(
        request_text="Did customers in the West region spend more than the East region?",
        session_id="session-abc",
        question_type=QuestionType.TWO_GROUP_COMPARISON,
        causal_status=CausalStatus.OBSERVATIONAL,
        outcome_name="total_spend",
        outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="customer",
        data_sources=("customers@v2",),
    )


def test_cannot_freeze_without_a_method_decision(draft_plan):
    with pytest.raises(ValueError):
        draft_plan.freeze()


def test_freeze_requires_method_decision_recorded(draft_plan):
    decision = MethodDecision(
        stage="test_selection", chosen_method="welch_t_test",
        rationale="Roughly equal variances observed",
        alternatives=(AlternativeConsidered(method="mann_whitney_u", rejected_because="less powerful if normal"),),
        evidence={"shapiro_p_group_a": 0.34},
    )
    refined = draft_plan.refine(candidate_predictors=("region",)).record_method_decision(decision)
    assert refined.primary_method == "welch_t_test"

    frozen = refined.freeze()
    assert frozen.status == PlanStatus.FROZEN


def test_cannot_refine_after_freeze(draft_plan):
    decision = MethodDecision(
        stage="test_selection", chosen_method="welch_t_test",
        rationale="test", alternatives=(),
    )
    frozen = draft_plan.record_method_decision(decision).freeze()
    with pytest.raises(ValueError):
        frozen.refine(alpha=0.10)


def test_deviation_and_superseding_method_decision(draft_plan):
    first = MethodDecision(
        stage="test_selection", chosen_method="welch_t_test",
        rationale="Roughly equal variances observed", alternatives=(),
        evidence={"shapiro_p_group_a": 0.34},
    )
    frozen = draft_plan.record_method_decision(first).freeze()

    second = MethodDecision(
        stage="test_selection", chosen_method="mann_whitney_u",
        rationale="Full-dataset check showed non-normal residuals",
        alternatives=(AlternativeConsidered(method="welch_t_test", rejected_because="normality violated"),),
        evidence={"shapiro_p_group_b_full_data": 0.02},
        supersedes=first.id,
    )
    deviated = frozen.record_deviation(
        description="Switched from Welch's t-test to Mann-Whitney U",
        reason="Assumption check on the full dataset invalidated the original choice",
    ).record_method_decision(second)

    assert len(deviated.deviations) == 1
    assert deviated.primary_method == "mann_whitney_u"
    assert deviated.latest_decision.supersedes == first.id
    assert len(deviated.method_decisions) == 2


def test_cannot_freeze_causal_claims_on_observational_data(draft_plan):
    decision = MethodDecision(
        stage="test_selection", chosen_method="welch_t_test",
        rationale="test", alternatives=(),
    )
    bad_plan = draft_plan.refine(allowed_claims="causal").record_method_decision(decision)
    with pytest.raises(ValueError):
        bad_plan.freeze()

def test_record_method_decision_tolerates_list_from_checkpoint_deserialization(draft_plan):
    """
    Regression test for a real bug found via the full end-to-end graph test:
    LangGraph's checkpointer returns lists where tuples were originally set,
    after a pause/resume across an interrupt. record_method_decision must
    not assume method_decisions is still a tuple by the time it's called.
    """
    from dataclasses import replace
    decision1 = MethodDecision(stage="test_selection", chosen_method="welch_t_test", rationale="x", alternatives=())
    plan_with_one = draft_plan.record_method_decision(decision1)

    # Simulate what a checkpoint round-trip does: tuple becomes list
    plan_as_if_checkpointed = replace(plan_with_one, method_decisions=list(plan_with_one.method_decisions))

    decision2 = MethodDecision(stage="test_selection", chosen_method="mann_whitney_u", rationale="y", alternatives=())
    result = plan_as_if_checkpointed.record_method_decision(decision2)  # must not raise

    assert len(result.method_decisions) == 2
    assert result.primary_method == "mann_whitney_u"

def test_mark_executed_descriptive_works_without_freeze():
    """EDA plans skip freeze entirely -- this transition must work from DRAFT status directly."""
    plan = AnalysisPlan(
        request_text="Describe this data", session_id="session-abc",
        question_type=QuestionType.DESCRIPTIVE, causal_status=CausalStatus.OBSERVATIONAL,
        outcome_name="(whole dataset)", outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="row", data_sources=("test@v1",),
    )
    executed = plan.mark_executed_descriptive()
    assert executed.status == PlanStatus.EXECUTED


def test_mark_executed_descriptive_rejects_non_descriptive_plans():
    """The guard must be narrow -- this is not a general freeze-bypass for any plan type."""
    plan = AnalysisPlan(
        request_text="Compare spend", session_id="session-abc",
        question_type=QuestionType.TWO_GROUP_COMPARISON, causal_status=CausalStatus.OBSERVATIONAL,
        outcome_name="total_spend", outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="customer", data_sources=("test@v1",),
    )
    with pytest.raises(ValueError, match="only valid for DESCRIPTIVE plans"):
        plan.mark_executed_descriptive()    