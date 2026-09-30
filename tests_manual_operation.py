from core.operation import (
    Operation, OperationImpact, ApprovalStatus,
    WranglingRationale, AlternativeApproachConsidered,
)

# A dedupe operation, proposed WITH a rationale — this is the new piece:
# why dedupe at all, and why this approach over alternatives
rationale = WranglingRationale(
    rationale="Exact duplicate rows on customer_id detected; safe to drop since "
              "these appear to be re-submitted form entries, not distinct events.",
    alternatives=(
        AlternativeApproachConsidered(
            approach="keep_and_flag",
            rejected_because="duplicates carry no additional information here, "
                              "unlike a case where repeats might represent repeat purchases",
        ),
    ),
    evidence={"duplicate_row_count": 3, "duplicate_pct": 0.03},
)

op = Operation(
    op_type="dedupe",
    session_id="session-abc",
    input_handles=("customers@v1",),
    params={"subset": ["customer_id"]},
    approval_status=ApprovalStatus.PENDING,
    rationale=rationale,
)
print("initial status:", op.approval_status)
print("rationale attached:", op.rationale.rationale)
print("alternatives considered:", len(op.rationale.alternatives))

# Reject path — unchanged behavior, still works
rejected = op.reject(reason="Not sure duplicates are actually errors — investigate first")
print("after reject:", rejected.approval_status, "-", rejected.rejected_reason)
print("rationale survives rejection:", rejected.rationale is not None)

try:
    rejected.approve()
except ValueError as e:
    print("correctly blocked:", e)

# Approve -> execute -> record impact, same as before
approved = op.approve()
print("after approve:", approved.approval_status)

impact = OperationImpact(
    rows_before=100, rows_after=97, columns_before=5, columns_after=5,
    details={"duplicates_removed": 3},
)
finished = approved.with_result(impact=impact, output_handle="customers@v2")
print("final status:", finished.approval_status)
print("final impact:", finished.impact)
print("rationale still attached after execution:", finished.rationale.rationale[:30], "...")

# NEW: the FAILED state — a different operation, approved, but execution itself errors
join_rationale = WranglingRationale(
    rationale="Join customers to subscriptions on customer_id to bring in plan_type",
    alternatives=(),
    evidence={},
)
join_op = Operation(
    op_type="join",
    session_id="session-abc",
    input_handles=("customers@v2", "subscriptions@v1"),
    params={"keys": ["customer_id"], "how": "left"},
    approval_status=ApprovalStatus.PENDING,
    rationale=join_rationale,
)
join_approved = join_op.approve()
print("\njoin approved:", join_approved.approval_status)

# Try to mark a PENDING (not approved) operation as failed -- should be blocked
try:
    join_op.mark_failed("test — should not be reachable from PENDING")
except ValueError as e:
    print("correctly blocked marking non-approved op as failed:", e)

# Simulate execution actually failing (e.g. the join key doesn't exist in subscriptions)
join_failed = join_approved.mark_failed("KeyError: 'customer_id' not found in subscriptions@v1")
print("join status after failure:", join_failed.approval_status)
print("failure_reason:", join_failed.failure_reason)
print("rejected_reason (should be None, this was NOT a rejection):", join_failed.rejected_reason)
print("id stayed the same throughout:", op.id == rejected.id == approved.id == finished.id)