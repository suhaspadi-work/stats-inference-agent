from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import uuid


class ApprovalStatus(str, Enum):
    NOT_REQUIRED = "not_required"   # e.g. a rename — no meaningful risk of harm
    PENDING = "pending"             # requires human sign-off, not yet given
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class OperationImpact:
    """
    What actually happened when the operation ran, in concrete, checkable
    numbers — never a model's description of what it did.
    """
    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    warnings: tuple[str, ...] = field(default_factory=tuple)
    # op-specific extra facts, e.g. {"unmatched_left": 112, "fan_out": "none"}
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Operation:
    """
    A single, reviewable wrangling step. Immutable once created — the agent
    proposes one of these, a policy or a human approves/rejects it, and only
    then does executing code actually run and produce a new Dataset version.
    """
    op_type: str                        # e.g. "cast", "filter", "dedupe", "join"
    session_id: str
    input_handles: tuple[str, ...]      # Dataset.handle values this reads from
    params: dict[str, Any]              # op-specific, validated by the tool that builds this
    approval_status: ApprovalStatus
    impact: Optional[OperationImpact] = None   # None until the op has actually executed
    output_handle: Optional[str] = None        # the Dataset.handle this produced, once executed
    rejected_reason: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def with_result(self, impact: OperationImpact, output_handle: str) -> "Operation":
        """Return a new Operation recording that execution happened and what it produced."""
        return Operation(
            op_type=self.op_type,
            session_id=self.session_id,
            input_handles=self.input_handles,
            params=self.params,
            approval_status=self.approval_status,
            impact=impact,
            output_handle=output_handle,
            id=self.id,
            created_at=self.created_at,
        )

    def approve(self) -> "Operation":
        if self.approval_status == ApprovalStatus.REJECTED:
            raise ValueError(f"Operation {self.id} was already rejected; cannot approve.")
        return Operation(
            op_type=self.op_type, session_id=self.session_id,
            input_handles=self.input_handles, params=self.params,
            approval_status=ApprovalStatus.APPROVED,
            impact=self.impact, output_handle=self.output_handle,
            id=self.id, created_at=self.created_at,
        )

    def reject(self, reason: str) -> "Operation":
        return Operation(
            op_type=self.op_type, session_id=self.session_id,
            input_handles=self.input_handles, params=self.params,
            approval_status=ApprovalStatus.REJECTED,
            impact=self.impact, output_handle=self.output_handle,
            rejected_reason=reason,
            id=self.id, created_at=self.created_at,
        )