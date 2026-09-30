from __future__ import annotations
from typing import Optional, TypedDict

from core.dataset import Dataset
from core.model_safe_view import ModelSafeView
from core.plan import AnalysisPlan


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

    dataset: Dataset               # set once, by intake -- before this node, it doesn't exist
    profile: ModelSafeView         # set by the profile node

    plan: AnalysisPlan             # set by classify, updated by draft_plan, updated again by freeze

    # Populated only when the classify node decides confidence is too low
    # to proceed automatically -- presence of this key IS the interrupt signal
    clarification_needed: Optional[str]

    # Populated only after execute runs -- the real TwoGroupTestResult
    test_result: Optional[dict]

    # Final plain-language output, populated by the report node
    report: Optional[str]