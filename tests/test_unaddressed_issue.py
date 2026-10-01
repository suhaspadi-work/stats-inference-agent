from dataclasses import replace
from core.plan import AnalysisPlan, UnaddressedIssue, QuestionType, CausalStatus, OutcomeType


def make_base_plan():
    return AnalysisPlan(
        request_text="Did West spend more than East?", session_id="session-abc",
        question_type=QuestionType.TWO_GROUP_COMPARISON, causal_status=CausalStatus.OBSERVATIONAL,
        outcome_name="total_spend", outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="customer", data_sources=("customers@v1",),
    )


def test_record_unaddressed_issue_appends_correctly():
    plan = make_base_plan()
    assert len(plan.unaddressed_issues) == 0

    issue = UnaddressedIssue(
        issue_type="duplicate_rows",
        description="12 exact duplicate rows detected on customer_id",
        proposed_operation="dedupe(subset=['customer_id'])",
        rejection_reason="User wants to investigate duplicates manually first",
        affected_columns=("customer_id",),
    )
    updated = plan.record_unaddressed_issue(issue)

    assert len(updated.unaddressed_issues) == 1
    assert updated.unaddressed_issues[0].issue_type == "duplicate_rows"
    assert updated.unaddressed_issues[0].rejection_reason == "User wants to investigate duplicates manually first"
    # Original plan object is untouched -- same immutability discipline as everything else
    assert len(plan.unaddressed_issues) == 0


def test_multiple_unaddressed_issues_accumulate():
    plan = make_base_plan()
    issue_1 = UnaddressedIssue(
        issue_type="duplicate_rows", description="x", proposed_operation="dedupe",
        rejection_reason="r1", affected_columns=("customer_id",),
    )
    issue_2 = UnaddressedIssue(
        issue_type="high_missingness", description="y", proposed_operation="impute",
        rejection_reason="r2", affected_columns=("region",),
    )
    updated = plan.record_unaddressed_issue(issue_1).record_unaddressed_issue(issue_2)

    assert len(updated.unaddressed_issues) == 2
    assert [i.issue_type for i in updated.unaddressed_issues] == ["duplicate_rows", "high_missingness"]


def test_record_unaddressed_issue_tolerates_list_from_checkpoint_deserialization():
    """
    Same regression class as test_record_method_decision_tolerates_list_...
    -- a checkpoint round-trip can hand back a list where a tuple was
    originally set. This must not break appending a new issue.
    """
    plan = make_base_plan()
    issue_1 = UnaddressedIssue(
        issue_type="duplicate_rows", description="x", proposed_operation="dedupe",
        rejection_reason="r1", affected_columns=("customer_id",),
    )
    plan_with_one = plan.record_unaddressed_issue(issue_1)

    plan_as_if_checkpointed = replace(plan_with_one, unaddressed_issues=list(plan_with_one.unaddressed_issues))

    issue_2 = UnaddressedIssue(
        issue_type="high_missingness", description="y", proposed_operation="impute",
        rejection_reason="r2", affected_columns=("region",),
    )
    result = plan_as_if_checkpointed.record_unaddressed_issue(issue_2)  # must not raise

    assert len(result.unaddressed_issues) == 2