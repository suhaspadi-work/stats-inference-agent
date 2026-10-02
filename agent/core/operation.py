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
    FAILED = "failed"               # approved, but execution itself threw an error —
                                     # distinct from REJECTED: this is a system fault,
                                     # not a human decision, and must not be conflated
                                     # with "a human said no"


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
    details: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1


@dataclass(frozen=True)
class AlternativeApproachConsidered:
    """One wrangling approach that was considered at this step and not chosen."""
    approach: str
    rejected_because: str
    schema_version: int = 1


@dataclass(frozen=True)
class WranglingRationale:
    """
    Why THIS wrangling approach was chosen for THIS step, and what else was
    considered — the same audit-trail discipline as MethodDecision, applied
    one stage earlier in the pipeline (cleaning/wrangling rather than testing).
    e.g. "impute median rather than drop rows, because missingness looked
    random and dropping would lose 40% of the data."
    """
    rationale: str
    alternatives: tuple[AlternativeApproachConsidered, ...] = field(default_factory=tuple)
    evidence: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1


@dataclass(frozen=True)
class Operation:
    """
    A single, reviewable wrangling step. Immutable once created — the agent
    proposes one of these (with its rationale), a policy or a human
    approves/rejects it, and only then does executing code actually run and
    produce a new Dataset version.
    """
    op_type: str                        # e.g. "cast", "filter", "dedupe", "join"
    session_id: str
    input_handles: tuple[str, ...]      # Dataset.handle values this reads from
    params: dict[str, Any]              # op-specific, validated by the tool that builds this
    approval_status: ApprovalStatus
    rationale: Optional[WranglingRationale] = None   # why this approach, vs. alternatives
    impact: Optional[OperationImpact] = None         # None until the op has actually executed
    output_handle: Optional[str] = None              # the Dataset.handle this produced, once executed
    rejected_reason: Optional[str] = None
    failure_reason: Optional[str] = None             # populated only on ApprovalStatus.FAILED
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: int = 1

    def with_result(self, impact: OperationImpact, output_handle: str) -> "Operation":
        """Return a new Operation recording that execution happened and what it produced."""
        return Operation(
            op_type=self.op_type, session_id=self.session_id,
            input_handles=self.input_handles, params=self.params,
            approval_status=self.approval_status, rationale=self.rationale,
            impact=impact, output_handle=output_handle,
            id=self.id, created_at=self.created_at,
        )

    def approve(self) -> "Operation":
        if self.approval_status == ApprovalStatus.REJECTED:
            raise ValueError(f"Operation {self.id} was already rejected; cannot approve.")
        return Operation(
            op_type=self.op_type, session_id=self.session_id,
            input_handles=self.input_handles, params=self.params,
            approval_status=ApprovalStatus.APPROVED, rationale=self.rationale,
            impact=self.impact, output_handle=self.output_handle,
            id=self.id, created_at=self.created_at,
        )

    def reject(self, reason: str) -> "Operation":
        return Operation(
            op_type=self.op_type, session_id=self.session_id,
            input_handles=self.input_handles, params=self.params,
            approval_status=ApprovalStatus.REJECTED, rationale=self.rationale,
            impact=self.impact, output_handle=self.output_handle,
            rejected_reason=reason,
            id=self.id, created_at=self.created_at,
        )

    def mark_failed(self, failure_reason: str) -> "Operation":
        """
        Record that this operation was approved and attempted, but execution
        itself failed (e.g. a column didn't exist). Distinct from reject():
        this is a system fault discovered during execution, not a human
        decision made before execution.
        """
        if self.approval_status != ApprovalStatus.APPROVED:
            raise ValueError("Only an APPROVED operation can transition to FAILED.")
        return Operation(
            op_type=self.op_type, session_id=self.session_id,
            input_handles=self.input_handles, params=self.params,
            approval_status=ApprovalStatus.FAILED, rationale=self.rationale,
            impact=self.impact, output_handle=self.output_handle,
            failure_reason=failure_reason,
            id=self.id, created_at=self.created_at,
        )