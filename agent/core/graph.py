from __future__ import annotations
from langgraph.graph import StateGraph, START, END

from core.agent_state import AgentState
from core.checkpoint_types import make_checkpointer
from core.dataset import DatasetStore
from core.event_sink import with_event_log
from core.nodes import (
    classify_node, clarify_node, reclassify_node, route_after_classify,
    make_profile_node, make_plan_stages_node, make_wrangle_propose_node, make_wrangle_review_node, route_after_wrangle_propose,
    make_describe_node, make_stage1_report_node, route_after_cleaning,
    route_after_plan_stages, stage_plan_gate_node, route_after_stage_plan_gate,
    route_after_report, make_stage_gate_node, route_after_stage_gate, make_start_round_node,
    make_draft_plan_node,
    make_select_method_node, make_freeze_node, make_execute_node, make_eda_node, make_report_node,
)


def route_after_freeze(state: AgentState) -> str:
    """
    After freeze_node resumes: a refused edit re-opens the plan-approval pause; an accepted edit goes back
    through select_method (M1.3c); an approved plan is FROZEN and runs; a rejected plan is never frozen
    (freeze_node deliberately never calls plan.freeze() on that path) and the run ends. Routing here means
    execute_node's own status check is not the only thing standing between a rejected plan and execution.
    """
    if state.get("plan_approval_refusal") is not None:
        return "freeze"
    if state.get("reselect_after_edit") is not None:
        return "select_method"
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
         -> if some step is offered: stage_plan_gate (M1.3a: approval interrupt; describe on or off, or reject)
            (M1.3c: run choices that break a rule are refused and the gate re-opens: stage_plan_gate -> stage_plan_gate)
      -> wrangle_propose (detect issues, one model call for fixes)
         -> if any issues: wrangle_review (checklist interrupt, then execute)
      -> describe (M1.2: when the stage plan says run and the question is not descriptive; no model call)
      -> stage1_report (M1.4b: the Stage 1 report, composed by code from the records; shown at plan approval)
      -> draft_plan
      -> freeze (approval interrupt; routes to execute or END)
         (M1.3c: a refused edit re-opens the pause: freeze -> freeze; an accepted edit re-selects the method:
          freeze -> select_method -> freeze)
      -> execute
      -> report
         -> M1.3b: after a comparison whose stage plan offered modeling: stage_gate (run or skip the exploratory round)
            (a predictor choice that breaks a rule is refused and the gate re-opens: stage_gate -> stage_gate)
         -> if run: start_round (an exploratory regression plan) -> select_method -> freeze -> execute -> report
      -> END (after the exploratory round's report, or when nothing is offered)

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
    add_node("stage_plan_gate", stage_plan_gate_node)
    if propose_fn is None:
        add_node("wrangle_propose", make_wrangle_propose_node(store))
    else:
        add_node("wrangle_propose", make_wrangle_propose_node(store, propose_fn=propose_fn))
    add_node("wrangle_review", make_wrangle_review_node(store))
    add_node("describe", make_describe_node(store))
    add_node("stage1_report", make_stage1_report_node())
    if draft_fn is None:
        add_node("draft_plan", make_draft_plan_node())
    else:
        add_node("draft_plan", make_draft_plan_node(draft_fn=draft_fn))
    add_node("eda", make_eda_node(store))
    add_node("select_method", make_select_method_node(store))
    add_node("freeze", make_freeze_node(store))
    add_node("execute", make_execute_node(store))
    add_node("report", make_report_node())
    add_node("stage_gate", make_stage_gate_node(store))
    add_node("start_round", make_start_round_node(store))

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_after_classify, {"profile": "profile", "clarify": "clarify"})
    graph.add_edge("clarify", "reclassify")
    graph.add_edge("reclassify", "profile")
    graph.add_edge("profile", "plan_stages")
    graph.add_conditional_edges("plan_stages", route_after_plan_stages,
                                {"stage_plan_gate": "stage_plan_gate", "wrangle_propose": "wrangle_propose", END: END})
    graph.add_conditional_edges("stage_plan_gate", route_after_stage_plan_gate,
                                {"stage_plan_gate": "stage_plan_gate", "wrangle_propose": "wrangle_propose", END: END})
    graph.add_conditional_edges("wrangle_propose", route_after_wrangle_propose,
                                {"wrangle_review": "wrangle_review", "describe": "describe", "stage1_report": "stage1_report"})
    graph.add_conditional_edges("wrangle_review", route_after_cleaning,
                                {"describe": "describe", "stage1_report": "stage1_report"})
    graph.add_edge("describe", "stage1_report")
    graph.add_edge("stage1_report", "draft_plan")
    graph.add_conditional_edges("draft_plan", route_after_draft_plan,
                                {"eda": "eda", "select_method": "select_method", END: END})
    graph.add_edge("select_method", "freeze")
    graph.add_edge("eda", "report")
    graph.add_conditional_edges("freeze", route_after_freeze,
                                {"freeze": "freeze", "select_method": "select_method", "execute": "execute", END: END})
    graph.add_edge("execute", "report")
    graph.add_conditional_edges("report", route_after_report, {"stage_gate": "stage_gate", END: END})
    graph.add_conditional_edges("stage_gate", route_after_stage_gate,
                                {"start_round": "start_round", "stage_gate": "stage_gate", END: END})
    graph.add_edge("start_round", "select_method")

    return graph.compile(checkpointer=checkpointer if checkpointer is not None else make_checkpointer())