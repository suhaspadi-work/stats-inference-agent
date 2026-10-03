from __future__ import annotations
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from core.agent_state import AgentState
from core.dataset import DatasetStore
from core.nodes import (
    classify_node, make_profile_node, make_wrangle_node, make_draft_plan_node,
    make_select_method_node, freeze_node, make_execute_node, make_eda_node, make_report_node,
)


def route_after_freeze(state: AgentState) -> str:
    """
    After freeze_node resumes, the plan is either FROZEN (approve/edit) or
    still not frozen (reject -- freeze_node deliberately never calls
    plan.freeze() on that path). Route accordingly rather than letting
    execute_node's own status check be the only thing standing between a
    rejected plan and an attempted execution.
    """
    if state["plan"].status.value == "frozen":
        return "execute"
    return END

def route_after_draft_plan(state: AgentState) -> str:
    """
    DESCRIPTIVE plans skip method-selection and freeze entirely -- there is
    no inferential method choice for a human to approve, so routing
    straight to EDA avoids an approval gate that would protect nothing.
    Every other question_type continues through the normal
    select_method -> freeze -> execute chain.
    """
    if state["plan"].question_type.value == "descriptive":
        return "eda"
    return "select_method"


def build_agent_graph(store: DatasetStore):
    """
    Wires every node from Phase 6 into one complete graph:
    intake (implicit -- caller provides request_text/dataset in initial state)
      -> classify (clarification interrupt handled internally)
      -> profile
      -> draft_plan
      -> freeze (approval interrupt; routes to execute or END)
      -> execute
      -> report
      -> END
    """
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_node)
    graph.add_node("profile", make_profile_node(store))
    graph.add_node("wrangle", make_wrangle_node(store))
    graph.add_node("draft_plan", make_draft_plan_node())
    graph.add_node("eda", make_eda_node(store))
    graph.add_node("select_method", make_select_method_node(store))
    graph.add_node("freeze", freeze_node)
    graph.add_node("execute", make_execute_node(store))
    graph.add_node("report", make_report_node())

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "profile")
    graph.add_edge("profile", "wrangle")
    graph.add_edge("wrangle", "draft_plan")
    graph.add_conditional_edges("draft_plan", route_after_draft_plan, {"eda": "eda", "select_method": "select_method"})
    graph.add_edge("select_method", "freeze")
    graph.add_edge("eda", "report")
    graph.add_conditional_edges("freeze", route_after_freeze, {"execute": "execute", END: END})
    graph.add_edge("execute", "report")
    graph.add_edge("report", END)

    return graph.compile(checkpointer=InMemorySaver())