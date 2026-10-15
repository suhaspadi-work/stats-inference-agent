"""
M0.3 step 2: the route from core.routing.route_for_plan is computed by draft_plan_node, stored in state,
reconciles the plan's question type before approval, stops unsupported questions before method
selection, and is read by the select, execute, eda and report nodes. Model functions are faked.
"""
import json

import numpy as np
import pandas as pd
import pytest
from langgraph.graph import END
from langgraph.types import Command

import core.routing as routing_core
from core.capabilities import MULTI_GROUP, TWO_GROUP
from core.classifier import _DraftPlanOut
from core.dataset import LocalDiskStore
from core.graph import build_agent_graph, route_after_draft_plan
from core.ingestion import profile_dataset
from core.nodes import (
    make_draft_plan_node, make_eda_node, make_execute_node, make_report_node, make_select_method_node,
)
from core.plan import AnalysisPlan, CausalStatus, ClassificationDecision, OutcomeType, QuestionType
from core.routing import route_for_plan
from tests.test_decision_point_guard import _dataset, confident, fake_draft, must_not_propose

STOP = ["execute", "eda"]


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _frame(seed=1, means=(50, 55, 65)):
    rng = np.random.default_rng(seed)
    n = 60
    return pd.DataFrame({
        "region": ["East", "West"] * (3 * n // 2),
        "tier": [t for t in ("basic", "plus", "pro") for _ in range(n)],
        "tenure": rng.normal(24, 8, 3 * n),
        "converted": rng.integers(0, 2, 3 * n),
        "total_spend": np.concatenate([rng.normal(m, 5, n) for m in means]),
    })


@pytest.fixture
def data(tmp_path, store):
    ds = _dataset(_frame(), tmp_path, store, "wiring")
    return ds, profile_dataset(ds, store=store)


def _classification(question_type):
    return ClassificationDecision(question_type=question_type, causal_status=CausalStatus.OBSERVATIONAL,
                                  rationale="fake", alternatives=(), confidence=0.9, evidence={})


def _draft(data, question_type, draft_fn):
    ds, profile = data
    state = {"request_text": "x", "session_id": "session-abc", "dataset": ds, "profile": profile,
             "classification": _classification(question_type)}
    return make_draft_plan_node(draft_fn=draft_fn)(state)


def _binary_draft(request_text, profile):
    return _DraftPlanOut(outcome_name="converted", outcome_type=OutcomeType.BINARY, unit_of_analysis="customer",
                         candidate_predictors=["region"], rationale="conversion by region")


def _points(update):
    return [e["decision_point"] for e in update["decision_log"]]


# --- draft_plan_node ----------------------------------------------------------------

def test_ok_route_is_stored_and_the_plan_is_unchanged(data):
    update = _draft(data, QuestionType.TWO_GROUP_COMPARISON, fake_draft(["region"]))
    route = update["route"]
    json.dumps(route)
    assert route["status"] == "ok" and route["capability"] == "two_group"
    assert update["plan"].question_type is QuestionType.TWO_GROUP_COMPARISON
    assert update["decision_log"][-1]["chosen"] == "two_group"
    assert "clarification_needed" not in update


def test_reconciled_route_updates_the_plan_but_not_the_classification(data):
    update = _draft(data, QuestionType.TWO_GROUP_COMPARISON, fake_draft(["tier"]))
    route, plan = update["route"], update["plan"]
    assert route["status"] == "reconciled"
    assert route["classifier_question_type"] == "two_group_comparison"
    assert plan.question_type is QuestionType.MULTI_GROUP_COMPARISON
    assert plan.classification_decision.question_type is QuestionType.TWO_GROUP_COMPARISON
    record = update["decision_log"][-1]
    assert record["decision_point"] == "planner.route" and record["chosen"] == "multi_group"
    assert (record["evidence"]["classifier_question_type"], record["evidence"]["question_type"],
            record["evidence"]["status"]) == ("two_group_comparison", "multi_group_comparison", "reconciled")
    chosen = next(c for c in record["candidates"] if c["option"] == "multi_group")
    assert chosen["eligible"] is True
    assert "clarification_needed" not in update


@pytest.mark.parametrize("question_type, draft_fn", [
    (QuestionType.ASSOCIATION, fake_draft(["region", "tenure"])),
    (QuestionType.TWO_GROUP_COMPARISON, _binary_draft),  # a conversion-style 0/1 outcome (G3, G44)
])
def test_unsupported_route_sets_clarification_needed(data, question_type, draft_fn):
    update = _draft(data, question_type, draft_fn)
    route = update["route"]
    assert route["status"] == "unsupported" and route["capability"] is None
    assert update["clarification_needed"] == route["message"]
    assert update["plan"].status.value != "frozen"
    record = update["decision_log"][-1]
    assert record["chosen"] == "unsupported"
    assert record["evidence"]["skipped_steps"] == ["select_method", "freeze", "execute"]
    assert record["evidence"]["message"] == route["message"]
    assert all(c["reason"].startswith("not chosen: no analysis can run this question; ")
               for c in record["candidates"])
    assert _points(update)[:4] == ["planner.outcome", "planner.outcome_type", "planner.predictors",
                                   "planner.unit_of_analysis"]


def test_binary_outcome_message_cites_g3(data):
    update = _draft(data, QuestionType.TWO_GROUP_COMPARISON, _binary_draft)
    assert "(G3)" in update["clarification_needed"]


def test_descriptive_route_skips_evaluate_all(data, monkeypatch):
    def must_not_be_called(*args, **kwargs):
        raise AssertionError("evaluate_all must not run for a descriptive plan")
    monkeypatch.setattr(routing_core, "evaluate_all", must_not_be_called)
    update = _draft(data, QuestionType.DESCRIPTIVE, fake_draft([]))
    assert (update["route"]["capability"], update["route"]["status"]) == ("descriptive", "ok")
    assert _points(update) == ["planner.draft_plan", "planner.route"]


# --- route_after_draft_plan ---------------------------------------------------------------

def _plan(question_type, predictors=("region",)):
    return AnalysisPlan(request_text="x", session_id="session-abc", question_type=question_type,
                        causal_status=CausalStatus.OBSERVATIONAL, outcome_name="total_spend",
                        outcome_type=OutcomeType.CONTINUOUS, unit_of_analysis="customer",
                        data_sources=("x@v1",), candidate_predictors=tuple(predictors))


def test_route_after_draft_plan_follows_the_route():
    plan = _plan(QuestionType.TWO_GROUP_COMPARISON)
    assert route_after_draft_plan({"plan": plan, "route": {"status": "unsupported", "capability": None}}) == END
    assert route_after_draft_plan({"plan": plan, "route": {"status": "ok", "capability": "descriptive"}}) == "eda"
    assert route_after_draft_plan({"plan": plan, "route": {"status": "reconciled", "capability": "multi_group"}}) \
        == "select_method"


def test_route_after_draft_plan_without_a_route_behaves_as_before():
    assert route_after_draft_plan({"plan": _plan(QuestionType.DESCRIPTIVE, ())}) == "eda"
    assert route_after_draft_plan({"plan": _plan(QuestionType.TWO_GROUP_COMPARISON)}) == "select_method"


# --- the four readers ------------------------------------------------------------------------

def _real_plan(ds, question_type, predictors):
    return AnalysisPlan(request_text="x", session_id="session-abc", question_type=question_type,
                        causal_status=CausalStatus.OBSERVATIONAL, outcome_name="total_spend",
                        outcome_type=OutcomeType.CONTINUOUS, unit_of_analysis="customer",
                        data_sources=(ds.handle,), candidate_predictors=tuple(predictors))


def _state(ds, plan, **extra):
    return {"plan": plan, "dataset": ds, "request_text": "x", "session_id": "session-abc", **extra}


MULTI_ROUTE = {"status": "reconciled", "capability": "multi_group"}


def test_select_method_reads_the_route(data, store):
    ds, _ = data
    # The plan's own question type would pick two_group; the route names multi_group.
    plan = _real_plan(ds, QuestionType.TWO_GROUP_COMPARISON, ["tier"])
    update = make_select_method_node(store)(_state(ds, plan, route=MULTI_ROUTE))
    assert update["decision_log"][0]["evidence"]["family"] == "multi_group"
    assert update["plan"].primary_method in ("one_way_anova", "kruskal_wallis")
    # Without a route the plan's question type decides, as before.
    plain = make_select_method_node(store)(_state(ds, _real_plan(ds, QuestionType.TWO_GROUP_COMPARISON, ["region"])))
    assert plain["decision_log"][0]["evidence"]["family"] == "two_group"


def test_execute_reads_the_route(data, store):
    ds, _ = data
    plan = _real_plan(ds, QuestionType.TWO_GROUP_COMPARISON, ["tier"])
    frozen = plan.record_method_decision(MULTI_GROUP.select(ds, store, plan)).freeze()
    update = make_execute_node(store)(_state(ds, frozen, route=MULTI_ROUTE))
    assert update["test_result"]["result_type"] == "multi_group"
    assert [e["decision_point"] for e in update["decision_log"]] == ["stage2.post_hoc"]
    two = _real_plan(ds, QuestionType.TWO_GROUP_COMPARISON, ["region"])
    plain = make_execute_node(store)(_state(ds, two.record_method_decision(TWO_GROUP.select(ds, store, two)).freeze()))
    assert plain["test_result"]["result_type"] == "two_group"


def test_eda_reads_the_route(data, store):
    ds, _ = data
    plan = _real_plan(ds, QuestionType.DESCRIPTIVE, [])
    with pytest.raises(ValueError, match="non-descriptive"):
        make_eda_node(store)(_state(ds, plan, route=MULTI_ROUTE))
    update = make_eda_node(store)(_state(ds, plan, route={"status": "ok", "capability": "descriptive"}))
    assert update["descriptive_result"]["row_count"] == 180
    assert make_eda_node(store)(_state(ds, plan))["descriptive_result"]["row_count"] == 180  # no route, as before


def test_report_reads_the_route(data):
    ds, _ = data
    node = make_report_node(report_fn=lambda plan, result: "two-group stub")
    # A descriptive plan would take the descriptive branch; the route sends it to the analysis branch.
    descriptive_plan = _real_plan(ds, QuestionType.DESCRIPTIVE, [])
    update = node(_state(ds, descriptive_plan, route={"status": "ok", "capability": "two_group"},
                         test_result={"result_type": "two_group", "p_value": 0.01}))
    assert update["report"] == "two-group stub"
    plain = node(_state(ds, _real_plan(ds, QuestionType.TWO_GROUP_COMPARISON, ["region"]),
                        test_result={"result_type": "two_group", "p_value": 0.01}))
    assert plain["report"] == "two-group stub"  # no route, as before


# --- the real graph ---------------------------------------------------------------------------

def test_draft_plan_edges(tmp_path):
    edges = {(e.source, e.target): e.conditional
             for e in build_agent_graph(LocalDiskStore(base_dir=tmp_path)).get_graph().edges}
    assert edges[("draft_plan", "eda")] is True
    assert edges[("draft_plan", "select_method")] is True
    assert edges[("draft_plan", "__end__")] is True


def _run(app, ds, thread, resumes):
    config = {"configurable": {"thread_id": thread}}
    queue, pauses = list(resumes), []
    result = app.invoke({"request_text": "x", "session_id": "session-abc", "dataset": ds}, config, interrupt_after=STOP)
    while "__interrupt__" in result:
        pauses.append(result["__interrupt__"][0].value)
        result = app.invoke(Command(resume=queue.pop(0)), config, interrupt_after=STOP)
    assert queue == []
    snapshot = app.get_state(config)
    return snapshot.values, pauses, snapshot.next


def test_full_run_reconciled_to_multi_group(data, store):
    ds, _ = data
    app = build_agent_graph(store, classify_fn=confident(QuestionType.TWO_GROUP_COMPARISON),
                            draft_fn=fake_draft(["tier"]), propose_fn=must_not_propose)
    state, pauses, next_nodes = _run(app, ds, "reconciled", [{"action": "approve"}] * 2)  # M1.3a: gate first
    assert [p["type"] for p in pauses] == ["stage_plan_approval", "plan_approval"]
    assert pauses[1]["question_type"] == "multi_group_comparison"
    assert state["test_result"]["result_type"] == "multi_group"
    points = [e["decision_point"] for e in state["decision_log"]]
    assert "stage2.post_hoc" in points
    assert next_nodes == ("report",)


@pytest.mark.parametrize("question_type, draft_fn, text, last_point, gate_pauses", [
    # M1.1: association now ends at the stage plan, before cleaning and the draft plan
    (QuestionType.ASSOCIATION, fake_draft(["region", "tenure"]), "Association questions are not supported yet",
     "planner.stage_plan", 0),
    # M1.3a: the binary two-group question passes the stage-plan gate first
    (QuestionType.TWO_GROUP_COMPARISON, _binary_draft, "(G3)", "planner.route", 1),
])
def test_full_run_unsupported_ends_before_method_selection(data, store, question_type, draft_fn, text, last_point,
                                                          gate_pauses):
    ds, _ = data
    app = build_agent_graph(store, classify_fn=confident(question_type), draft_fn=draft_fn,
                            propose_fn=must_not_propose)
    state, pauses, next_nodes = _run(app, ds, f"unsupported-{question_type.value}", [{"action": "approve"}] * gate_pauses)
    assert [p["type"] for p in pauses] == ["stage_plan_approval"] * gate_pauses and next_nodes == ()
    assert text in state["clarification_needed"]
    points = [e["decision_point"] for e in state["decision_log"]]
    assert points[-1] == last_point
    assert not any(p in ("stage2.test_selection", "stage3.regression_method") for p in points)
