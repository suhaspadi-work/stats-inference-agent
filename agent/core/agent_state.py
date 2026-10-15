from __future__ import annotations
import operator
from typing import Annotated, Optional, TypedDict

from core.dataset import Dataset
from core.model_safe_view import ModelSafeView
from core.plan import AnalysisPlan
from core.plan import AnalysisPlan, ClassificationDecision


class AgentState(TypedDict, total=False):
    """
    The single object that flows through every node in the graph. TypedDict
    with total=False means any key may be absent early on -- a node only
    ever adds keys, it never needs every key populated from the start.

    This is intentionally a thin, mutable-looking container (LangGraph's own
    convention) even though everything INSIDE it -- Dataset, AnalysisPlan,
    ClassificationDecision -- remains the immutable, audit-trailed objects
    from Phase 1. The graph replaces whole values in this dict; it never
    mutates the dataclasses themselves.
    """
    request_text: str
    session_id: str
    classification: ClassificationDecision  # set by the classify node
    # First-pass result held while the human is asked to clarify; None once resolved
    pending_classification: Optional[ClassificationDecision]
    clarification_answer: str  # the human's answer at the clarification interrupt

    dataset: Dataset               # set once, by intake -- before this node, it doesn't exist
    profile: ModelSafeView         # set by the profile node
    stage_plan: dict               # set by plan_stages (M1.1): the steps of the three stages, core.stage_plan

    plan: AnalysisPlan             # set by classify, updated by draft_plan, updated again by freeze

    # Populated only when the classify node decides confidence is too low
    # to proceed automatically -- presence of this key IS the interrupt signal
    clarification_needed: Optional[str]

    # Populated only after execute runs -- the real TwoGroupTestResult
    test_result: Optional[dict]

    # Final plain-language output, populated by the report node
    report: Optional[str]

    pending_unaddressed_issues: tuple  # set by wrangle_review if issues exist before the plan is created
    # Held between wrangle_propose and wrangle_review: the detected issues, the proposed
    # Operations (dataclasses, the G4 caveat), each operation's issue index and the
    # auto-approved indices. None once resolved.
    pending_wrangling: Optional[dict]

    autonomy_mode: str  # "analyst" (default behavior: interrupt on everything) or "stakeholder"

    descriptive_result: dict  # set by eda_node for DESCRIPTIVE question types
    diagnostics_result: dict  # the DIAGNOSE result (M0.6); set by the Stage 1 controller from M1 and M2; nothing sets it yet

    # The route chosen by draft_plan_node (capability, status, question types, reason, message,
    # evidence), from core.routing.route_for_plan; None for older states.
    route: Optional[dict]

    # Every DecisionRecord, as plain dicts, in the order made. Appended by
    # LangGraph (operator.add): a node returns only its new records via
    # core.decision_log.log_update and can never overwrite earlier ones.
    decision_log: Annotated[list[dict], operator.add]

    # One entry per finished round (M1.3b), appended by report_node (core.rounds.make_round_entry);
    # plan, test_result and report are the latest round.
    rounds: Annotated[list[dict], operator.add]
    exploratory_round: bool  # set by start_round in an exploratory round (M1.3b step 2); absent otherwise
    next_round: Optional[dict]  # set by stage_gate: {"run": bool, "predictors": [...]} (M1.3b)
    # set by stage_gate when the user's predictors break a rule: {"submitted", "refusals", "attempt"}; the gate
    # re-opens and shows it. None otherwise (M1.3b step 2b).
    stage_gate_refusal: Optional[dict]
    stage_plan_refusal: Optional[dict]  # run choices refused at the stage-plan gate, shown when it re-opens (M1.3c)
    plan_approval_refusal: Optional[dict]  # an edit refused at plan approval, shown when the pause re-opens (M1.3c)
    reselect_after_edit: Optional[dict]  # set by an accepted edit: select_method runs again, then freeze (M1.3c)
