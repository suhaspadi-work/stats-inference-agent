from core.operation import Operation, OperationImpact, ApprovalStatus

# A dedupe operation, proposed but not yet approved
op = Operation(
    op_type="dedupe",
    session_id="session-abc",
    input_handles=("customers@v1",),
    params={"subset": ["customer_id"]},
    approval_status=ApprovalStatus.PENDING,
)
print("initial status:", op.approval_status)
print("initial impact:", op.impact)

# Try rejecting it
rejected = op.reject(reason="Not sure duplicates are actually errors — investigate first")
print("after reject:", rejected.approval_status, "-", rejected.rejected_reason)

# Confirm you cannot approve something already rejected
try:
    rejected.approve()
except ValueError as e:
    print("correctly blocked:", e)

# Now the approve -> execute -> record-impact path
approved = op.approve()
print("after approve:", approved.approval_status)

impact = OperationImpact(
    rows_before=100, rows_after=97, columns_before=5, columns_after=5,
    details={"duplicates_removed": 3},
)
finished = approved.with_result(impact=impact, output_handle="customers@v2")
print("final status:", finished.approval_status)
print("final impact:", finished.impact)
print("final output_handle:", finished.output_handle)
print("id stayed the same throughout:", op.id == rejected.id == approved.id == finished.id)