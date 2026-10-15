"""
The registry of decision points (M0.2 step 12), read by the guard tests.

Default-deny: every node in the real graph must appear in exactly one of
DECISION_NODES (nodes that choose something and the decision points they may
emit) and NON_CHOOSING_NODES (nodes that choose nothing, with the reason). No
test can know by itself whether a new node makes a choice, so a node added to
the graph without being listed here fails the guard until someone decides.

Every string is imported from the module that defines it; none is redefined
here. This module must not import core.nodes or core.graph.
"""
from __future__ import annotations

from core.classification_records import (
    CAUSAL_STATUS_POINT, CLARIFICATION_GATE_POINT, CLARIFICATION_POINT, QUESTION_TYPE_POINT,
)
from core.cleaning_records import (
    APPROVAL_POINT as CLEAN_APPROVAL_POINT,
    EXECUTION_POINT as CLEAN_EXECUTION_POINT,
    FIX_POINT, SCAN_POINT,
)
from core.describe_records import DESCRIBE_EXECUTION_POINT
from core.draft_plan_records import OUTCOME_POINT, OUTCOME_TYPE_POINT, PREDICTORS_POINT, UNIT_POINT
from core.execution_records import POST_HOC_POINT
from core.freeze_records import PLAN_APPROVAL_POINT
from core.routing_records import DRAFT_PLAN_POINT, ROUTE_POINT
from core.selection_records import ENCODING_POINT, REGRESSION_METHOD_POINT, TEST_SELECTION_POINT
from core.stage_plan_records import STAGE_PLAN_APPROVAL_POINT, STAGE_PLAN_POINT, STEP_POINTS

# Each choosing graph node and the decision points it may emit.
DECISION_NODES: dict[str, tuple[str, ...]] = {
    "classify": (QUESTION_TYPE_POINT, CAUSAL_STATUS_POINT, CLARIFICATION_GATE_POINT),
    "clarify": (CLARIFICATION_POINT,),
    "reclassify": (QUESTION_TYPE_POINT, CAUSAL_STATUS_POINT),
    "plan_stages": (STAGE_PLAN_POINT, STEP_POINTS["clean"], STEP_POINTS["describe"], STEP_POINTS["inference"],
                    STEP_POINTS["modeling"]),
    "stage_plan_gate": (STAGE_PLAN_APPROVAL_POINT,),
    "wrangle_propose": (SCAN_POINT, FIX_POINT),
    "wrangle_review": (CLEAN_APPROVAL_POINT, CLEAN_EXECUTION_POINT),
    "describe": (DESCRIBE_EXECUTION_POINT,),
    "draft_plan": (DRAFT_PLAN_POINT, OUTCOME_POINT, OUTCOME_TYPE_POINT, PREDICTORS_POINT, UNIT_POINT, ROUTE_POINT),
    "select_method": (TEST_SELECTION_POINT, REGRESSION_METHOD_POINT, ENCODING_POINT),
    "freeze": (PLAN_APPROVAL_POINT, PREDICTORS_POINT),
    "execute": (POST_HOC_POINT,),
}

# Graph nodes that choose nothing, and why.
NON_CHOOSING_NODES: dict[str, str] = {
    "profile": "computes the model-safe profile; no option is chosen",
    "eda": ("runs the descriptive statistics; the one choice around it, skipping method selection "
            "and freeze, is recorded by planner.route"),
    "report": "restates facts computed by code; the model writes prose but no option is chosen",
}

# The stage of every graph node (M1.2). The stage is a tag on a node, not a LangGraph subgraph;
# "analysis" is stage 2 or 3 by the route. M1.3 places gates and M1.4 groups records by it.
STAGES: tuple[str, ...] = ("intake", "stage1", "analysis", "report")
NODE_STAGES: dict[str, str] = {
    "classify": "intake", "clarify": "intake", "reclassify": "intake", "profile": "intake", "plan_stages": "intake",
    "stage_plan_gate": "intake",
    "wrangle_propose": "stage1", "wrangle_review": "stage1", "describe": "stage1", "eda": "stage1",
    "draft_plan": "analysis", "select_method": "analysis", "freeze": "analysis", "execute": "analysis",
    "report": "report",
}

# Analyses that write no record at run time, and why.
ANALYSES_WITHOUT_EXECUTION_RECORD: dict[str, str] = {
    "two_group": "the method is fixed before the freeze; nothing is chosen at run time",
    "regression": "the method is fixed before the freeze; nothing is chosen at run time",
}

# Decision records may only carry charts of summary numbers; the decision-time
# charts hold raw values and stay on the old MethodDecision (M0.5).
ALLOWED_RECORD_CHART_TYPES: tuple[str, ...] = ("post_hoc_comparison_chart",)


def registered_points() -> set[str]:
    """Every decision point some node in DECISION_NODES may emit."""
    return {point for points in DECISION_NODES.values() for point in points}
