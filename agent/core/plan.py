from __future__ import annotations
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import uuid


class PlanStatus(str, Enum):
    DRAFT = "draft"        # built from the request alone, before profiling
    REFINED = "refined"    # updated once real columns/types/missingness are known
    FROZEN = "frozen"      # locked — no test may run until this state is reached
    EXECUTED = "executed"  # the frozen plan has been carried out


class QuestionType(str, Enum):
    DESCRIPTIVE = "descriptive"
    TWO_GROUP_COMPARISON = "two_group_comparison"
    MULTI_GROUP_COMPARISON = "multi_group_comparison"
    AB_TEST = "ab_test"
    ASSOCIATION = "association"
    DRIVER_ANALYSIS = "driver_analysis"


class CausalStatus(str, Enum):
    EXPERIMENTAL = "experimental"    # randomized — causal language permitted
    OBSERVATIONAL = "observational"  # not randomized — associations only


class OutcomeType(str, Enum):
    CONTINUOUS = "continuous"
    BINARY = "binary"


@dataclass(frozen=True)
class Deviation:
    """A logged record of the plan changing after being frozen, and why."""
    description: str
    reason: str
    at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: int = 1

@dataclass(frozen=True)
class UnaddressedIssue:
    """
    A real data-quality issue that was detected and proposed as a fix, but
    the human rejected the operation. Carried forward on the plan so every
    downstream stage -- method selection, execution, reporting -- can
    account for it rather than silently treating the data as clean.
    """
    issue_type: str              # e.g. "duplicate_rows", "high_missingness"
    description: str             # what was detected, in concrete terms
    proposed_operation: str      # what fix was proposed and rejected
    rejection_reason: str        # the human's stated reason, if given
    affected_columns: tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = 1


@dataclass(frozen=True)
class AlternativeConsidered:
    """One method that was considered at a decision point and not chosen, with the concrete reason."""
    method: str
    rejected_because: str
    schema_version: int = 1


@dataclass(frozen=True)
class MethodDecision:
    """
    A single, reviewable record of choosing a method at some stage of the
    analysis. Every method selection produces one of these — not just the
    first choice, but every subsequent change too, so the full reasoning
    chain is visible: why this method for this problem, what else was
    considered and rejected, and — if this decision replaces an earlier
    one — why the new choice is the better fit given what was learned.
    """
    stage: str                                     # e.g. "test_selection", "regression_diagnostics"
    chosen_method: str
    rationale: str                                 # why THIS one, for THIS problem
    alternatives: tuple[AlternativeConsidered, ...]
    evidence: dict[str, Any] = field(default_factory=dict)  # e.g. {"shapiro_p": 0.01} — real values only
    supersedes: Optional[str] = None               # id of the MethodDecision this replaces, if any
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: int = 1

@dataclass(frozen=True)
class ClassificationDecision:
    """
    The record of classifying a request BEFORE any AnalysisPlan is drafted.
    Same audit-trail shape as MethodDecision: what was chosen, what else was
    considered, and the evidence/confidence behind it. A low-confidence
    classification is what triggers a human-clarification interrupt rather
    than a silent guess (see Phase 6 in the blueprint).
    """
    question_type: QuestionType
    causal_status: CausalStatus
    rationale: str
    alternatives: tuple[AlternativeConsidered, ...]
    confidence: float  # 0.0-1.0; below a configured threshold triggers a clarification interrupt
    evidence: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: int = 1

@dataclass(frozen=True)
class AnalysisPlan:
    """
    The single object every stage of the agent reads from. Nothing runs a
    hypothesis test or a model until this reaches FROZEN — that freeze is
    what prevents the agent from trying methods until something looks
    significant (the same discipline as pre-registration in real research).

    Method choice is never a bare string: every selection (and every later
    change of selection) is recorded as a MethodDecision, so the reasoning
    behind "why this method" and "why we switched" is always inspectable.
    """
    request_text: str
    session_id: str
    question_type: QuestionType
    causal_status: CausalStatus
    outcome_name: str
    outcome_type: OutcomeType
    unit_of_analysis: str
    data_sources: tuple[str, ...]          # Dataset handles involved
    status: PlanStatus = PlanStatus.DRAFT

    # Filled in progressively — None/empty is legitimate at DRAFT/REFINED
    candidate_predictors: Optional[tuple[str, ...]] = None
    classification_decision: Optional["ClassificationDecision"] = None
    method_decisions: tuple[MethodDecision, ...] = field(default_factory=tuple)
    alpha: float = 0.05
    multiple_testing_policy: Optional[str] = None
    allowed_claims: str = "associations only"  # overridden explicitly if causal_status is EXPERIMENTAL

    deviations: tuple[Deviation, ...] = field(default_factory=tuple)
    unaddressed_issues: tuple[UnaddressedIssue, ...] = field(default_factory=tuple)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: int = 1

    @property
    def primary_method(self) -> Optional[str]:
        """The current, active method — the most recent decision recorded."""
        if not self.method_decisions:
            return None
        return self.method_decisions[-1].chosen_method

    @property
    def latest_decision(self) -> Optional[MethodDecision]:
        return self.method_decisions[-1] if self.method_decisions else None

    @classmethod
    def from_classification(
        cls,
        request_text: str,
        session_id: str,
        classification: "ClassificationDecision",
        outcome_name: str,
        outcome_type: OutcomeType,
        unit_of_analysis: str,
        data_sources: tuple[str, ...],
    ) -> "AnalysisPlan":
        """
        Build a fresh, DRAFT AnalysisPlan whose question_type/causal_status
        are traceable back to a real ClassificationDecision, not just passed
        in as bare values with no record of why they were chosen.
        """
        return cls(
            request_text=request_text,
            session_id=session_id,
            question_type=classification.question_type,
            causal_status=classification.causal_status,
            outcome_name=outcome_name,
            outcome_type=outcome_type,
            unit_of_analysis=unit_of_analysis,
            data_sources=data_sources,
            classification_decision=classification,
        )

    def _replace(self, **changes) -> "AnalysisPlan":
        return replace(self, **changes)

    def refine(self, **updates) -> "AnalysisPlan":
        if self.status == PlanStatus.FROZEN:
            raise ValueError("Cannot refine a frozen plan — record a Deviation instead.")
        return self._replace(status=PlanStatus.REFINED, **updates)

    def record_method_decision(self, decision: MethodDecision) -> "AnalysisPlan":
        """
        Append a method decision — either the first choice for a stage, or a
        later one that supersedes an earlier decision after new evidence.

        method_decisions is normalized to a tuple before appending: if this
        plan round-tripped through a LangGraph checkpoint (pause/resume
        across an interrupt), JSON-based serialization returns lists where
        tuples originally were -- this keeps the method robust regardless
        of which form the plan arrives in.
        """
        return self._replace(method_decisions=tuple(self.method_decisions) + (decision,))

    def freeze(self) -> "AnalysisPlan":
        if self.primary_method is None:
            raise ValueError("Cannot freeze a plan with no method decision recorded.")
        if self.causal_status == CausalStatus.OBSERVATIONAL and self.allowed_claims != "associations only":
            raise ValueError("Observational data cannot be frozen with causal claims allowed.")
        return self._replace(status=PlanStatus.FROZEN)

    def record_deviation(self, description: str, reason: str) -> "AnalysisPlan":
        if self.status != PlanStatus.FROZEN and self.status != PlanStatus.EXECUTED:
            raise ValueError("Deviations are only meaningful after a plan has been frozen.")
        new_deviation = Deviation(description=description, reason=reason)
        return self._replace(deviations=tuple(self.deviations) + (new_deviation,))

    def record_unaddressed_issue(self, issue: UnaddressedIssue) -> "AnalysisPlan":
        """
        Record a detected data-quality issue that was proposed as a fix and
        rejected. method_decisions-style defensive tuple() normalization
        applies here too, in case this plan round-tripped through a
        checkpoint before this call (see record_method_decision).
        """
        return self._replace(unaddressed_issues=tuple(self.unaddressed_issues) + (issue,))

    def mark_executed(self) -> "AnalysisPlan":
        if self.status != PlanStatus.FROZEN:
            raise ValueError("Cannot execute a plan that was never frozen.")
        return self._replace(status=PlanStatus.EXECUTED)