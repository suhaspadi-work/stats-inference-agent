from __future__ import annotations
from langgraph.graph import StateGraph, START, END

from core.agent_state import AgentState
from core.checkpoint_types import make_checkpointer
from core.dataset import DatasetStore
from core.event_sink import with_event_log
from core.nodes import (
    classify_node, clarify_node, reclassify_node, route_after_classify,
    make_profile_node, make_plan_stages_node, make_wrangle_propose_node, make_wrangle_review_node, route_after_wrangle_propose,
    make_draft_plan_node,
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

def route_after_plan_stages(state: AgentState) -> str:
    """An unsupported question ends the run after the stage plan, before any cleaning (M1.1)."""
    if state["stage_plan"]["status"] == "unsupported":
        return END
    return "wrangle_propose"


def route_after_draft_plan(state: AgentState) -> str:
    """
    DESCRIPTIVE plans skip method-selection and freeze entirely -- there is
    no inferential method choice for a human to approve, so routing
    straight to EDA avoids an approval gate that would protect nothing.
    Every other question_type continues through the normal
    select_method -> freeze -> execute chain.

    Since M0.3 the route that draft_plan_node stores in state decides: an unsupported route ends the run
    before method selection, with clarification_needed set. Without a route, the question type decides as before.
    """
    route = state.get("route")
    if route is not None and route["status"] == "unsupported":
        return END
    if route is not None:
        return "eda" if route["capability"] == "descriptive" else "select_method"
    if state["plan"].question_type.value == "descriptive":
        return "eda"
    return "select_method"


def build_agent_graph(store: DatasetStore, *, classify_fn=None, draft_fn=None, propose_fn=None,
                      event_log_dir=None, checkpointer=None):
    """
    Wires every node from Phase 6 into one complete graph:
    intake (implicit -- caller provides request_text/dataset in initial state)
      -> classify (first pass, records question type, causal status and the clarification gate)
         -> if confidence is low: clarify (clarification interrupt) -> reclassify (second pass)
      -> profile
      -> plan_stages (M1.1: which steps of the three stages run, by code; an unsupported question ends here)
      -> wrangle_propose (detect issues, one model call for fixes)
         -> if any issues: wrangle_review (checklist interrupt, then execute)
      -> draft_plan
      -> freeze (approval interrupt; routes to execute or END)
      -> execute
      -> report
      -> END

    classify_fn, draft_fn and propose_fn exist so tests can run the real graph without the model;
    left as None, each node uses its real model call. When event_log_dir is given, every node's new
    decision records are also written to <event_log_dir>/<thread id>/events.jsonl as the node returns.
    The graph compiles with the given checkpointer, or by default an in-memory one whose serializer allows the state types (G4).
    """
    graph = StateGraph(AgentState)

    def add_node(name, fn):
        graph.add_node(name, fn if event_log_dir is None else with_event_log(name, fn, event_log_dir))

    if classify_fn is None:
        add_node("classify", classify_node)
    else:
        add_node("classify", lambda state: classify_node(state, classify_fn=classify_fn))
    add_node("clarify", clarify_node)
    if classify_fn is None:
        add_node("reclassify", reclassify_node)
    else:
        add_node("reclassify", lambda state: reclassify_node(state, classify_fn=classify_fn))
    add_node("profile", make_profile_node(store))
    add_node("plan_stages", make_plan_stages_node(store))
    if propose_fn is None:
        add_node("wrangle_propose", make_wrangle_propose_node(store))
    else:
        add_node("wrangle_propose", make_wrangle_propose_node(store, propose_fn=propose_fn))
    add_node("wrangle_review", make_wrangle_review_node(store))
    if draft_fn is None:
        add_node("draft_plan", make_draft_plan_node())
    else:
        add_node("draft_plan", make_draft_plan_node(draft_fn=draft_fn))
    add_node("eda", make_eda_node(store))
    add_node("select_method", make_select_method_node(store))
    add_node("freeze", freeze_node)
    add_node("execute", make_execute_node(store))
    add_node("report", make_report_node())

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_after_classify, {"profile": "profile", "clarify": "clarify"})
    graph.add_edge("clarify", "reclassify")
    graph.add_edge("reclassify", "profile")
    graph.add_edge("profile", "plan_stages")
    graph.add_conditional_edges("plan_stages", route_after_plan_stages, {"wrangle_propose": "wrangle_propose", END: END})
    graph.add_conditional_edges("wrangle_propose", route_after_wrangle_propose,
                                {"wrangle_review": "wrangle_review", "draft_plan": "draft_plan"})
    graph.add_edge("wrangle_review", "draft_plan")
    graph.add_conditional_edges("draft_plan", route_after_draft_plan,
                                {"eda": "eda", "select_method": "select_method", END: END})
    graph.add_edge("select_method", "freeze")
    graph.add_edge("eda", "report")
    graph.add_conditional_edges("freeze", route_after_freeze, {"execute": "execute", END: END})
    graph.add_edge("execute", "report")
    graph.add_edge("report", END)

    return graph.compile(checkpointer=checkpointer if checkpointer is not None else make_checkpointer())