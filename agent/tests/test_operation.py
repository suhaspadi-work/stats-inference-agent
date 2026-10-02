import pytest
from core.operation import (
    Operation, OperationImpact, ApprovalStatus,
    WranglingRationale, AlternativeApproachConsidered,
)


@pytest.fixture
def pending_op_with_rationale():
    rationale = WranglingRationale(
        rationale="Exact duplicate rows on customer_id detected; safe to drop.",
        alternatives=(
            AlternativeApproachConsidered(
                approach="keep_and_flag",
                rejected_because="duplicates carry no additional information here",
            ),
        ),
        evidence={"duplicate_row_count": 3, "duplicate_pct": 0.03},
    )
    return Operation(
        op_type="dedupe", session_id="session-abc",
        input_handles=("customers@v1",), params={"subset": ["customer_id"]},
        approval_status=ApprovalStatus.PENDING, rationale=rationale,
    )


def test_rationale_attaches_and_survives_lifecycle(pending_op_with_rationale):
    op = pending_op_with_rationale
    assert op.rationale is not None
    assert len(op.rationale.alternatives) == 1

    approved = op.approve()
    assert approved.rationale is op.rationale  # unchanged through approval

    finished = approved.with_result(
        impact=OperationImpact(rows_before=100, rows_after=97, columns_before=5, columns_after=5),
        output_handle="customers@v2",
    )
    assert finished.rationale is op.rationale  # still present after execution


def test_cannot_approve_a_rejected_operation(pending_op_with_rationale):
    rejected = pending_op_with_rationale.reject(reason="Investigate first")
    assert rejected.approval_status == ApprovalStatus.REJECTED
    assert rejected.rationale is not None  # rationale survives rejection too

    with pytest.raises(ValueError):
        rejected.approve()


def test_failed_state_is_distinct_from_rejected(pending_op_with_rationale):
    approved = pending_op_with_rationale.approve()
    failed = approved.mark_failed("KeyError: 'customer_id' not found in subscriptions@v1")

    assert failed.approval_status == ApprovalStatus.FAILED
    assert failed.failure_reason == "KeyError: 'customer_id' not found in subscriptions@v1"
    assert failed.rejected_reason is None  # never conflated with rejection


def test_only_approved_operations_can_be_marked_failed(pending_op_with_rationale):
    with pytest.raises(ValueError):
        pending_op_with_rationale.mark_failed("should not be reachable from PENDING")


def test_id_stable_across_all_transitions(pending_op_with_rationale):
    op = pending_op_with_rationale
    approved = op.approve()
    finished = approved.with_result(
        impact=OperationImpact(rows_before=100, rows_after=97, columns_before=5, columns_after=5),
        output_handle="customers@v2",
    )
    assert op.id == approved.id == finished.id