"""
M1.3a: the stage-plan gate. The user reviews the stage plan before anything is changed: describe can be
switched on or off, every other change is refused, and rejecting ends the run. The approved run flags are
written back into the stage plan, so the M1.2 router acts on them.
"""
import copy
import json
import re

import numpy as np
import pandas as pd
import pytest
from langgraph.graph import END
from langgraph.types import Command

from core.dataset import LocalDiskStore
from core.decision_log import log_update
from core.decision_points import DECISION_NODES, NODE_STAGES, registered_points
from core.detection import CandidateIssue
from core.graph import build_agent_graph
from core.nodes import route_after_plan_stages, route_after_stage_plan_gate
from core.plan import QuestionType
from core.stage_plan import build_stage_plan, plan_steps
from core.stage_plan_gate import (
    ADJUSTABLE_STEPS, adjustable_steps, apply_gate_decision, gate_needed, gate_payload, validate_run_overrides,
)
from core.stage_plan_records import STAGE_PLAN_APPROVAL_POINT, stage_plan_approval_record
from tests.test_decision_point_guard import (
    _dataset, _drive, _two_group_df, _with_duplicate, confident, fake_classify, fake_draft, fake_propose,
    must_not_propose,
)
from tests.test_stage_plan import FOUR, classification

RUNNABLE = (QuestionType.TWO_GROUP_COMPARISON, QuestionType.AB_TEST, QuestionType.MULTI_GROUP_COMPARISON,
            QuestionType.DRIVER_ANALYSIS)
GATE = "stage_plan_approval"
STOP = ["execute", "eda"]


def plan_for(question_type, issues=()):
    return build_stage_plan(classification(question_type), FOUR, list(issues))


def step(plan, name):
    return next(s for s in plan_steps(plan) if s["step"] == name)


def cannot(bad, adjustable):
    return re.escape(f"Cannot change step(s) {bad} here. Adjustable steps: {adjustable}.")


# --- pure helpers ----------------------------------------------------------------------------------

def test_adjustable_steps_constant():
    assert ADJUSTABLE_STEPS == ("describe",)


@pytest.mark.parametrize("question_type", RUNNABLE)
def test_gate_needed_and_adjustable_for_runnable_types(question_type):
    plan = plan_for(question_type)
    assert gate_needed(plan) is True
    assert adjustable_steps(plan) == ["describe"]


@pytest.mark.parametrize("question_type", [QuestionType.DESCRIPTIVE, QuestionType.ASSOCIATION])
def test_no_gate_for_descriptive_or_association(question_type):
    plan = plan_for(question_type)
    assert gate_needed(plan) is False
    assert adjustable_steps(plan) == []


def test_validate_run_overrides_accepts():
    plan = plan_for(QuestionType.TWO_GROUP_COMPARISON)
    assert validate_run_overrides(plan, None) == {}
    assert validate_run_overrides(plan, {"describe": False}) == {"describe": False}


@pytest.mark.parametrize("question_type, overrides, message", [
    (QuestionType.TWO_GROUP_COMPARISON, {"clean": False}, cannot(["clean"], ["describe"])),          # planned
    (QuestionType.DRIVER_ANALYSIS, {"inference": True}, cannot(["inference"], ["describe"])),        # unavailable
    (QuestionType.TWO_GROUP_COMPARISON, {"modeling": True}, cannot(["modeling"], ["describe"])),     # exploratory
    (QuestionType.TWO_GROUP_COMPARISON, {"forecast": True}, cannot(["forecast"], ["describe"])),     # unknown
    (QuestionType.TWO_GROUP_COMPARISON, {"describe": "yes"}, re.escape("Run choices must be true or false.")),
    (QuestionType.TWO_GROUP_COMPARISON, ["describe"], re.escape("Run choices must be true or false.")),
], ids=["planned", "unavailable", "exploratory", "unknown", "not_bool", "not_dict"])
def test_validate_run_overrides_refuses(question_type, overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_run_overrides(plan_for(question_type), overrides)


def _issue():
    return CandidateIssue(issue_type="duplicate_rows", description="d", affected_columns=("region",), evidence={})


def test_a_planned_step_cannot_be_switched_off():
    plan = plan_for(QuestionType.TWO_GROUP_COMPARISON, [_issue()])
    assert step(plan, "clean")["state"] == "planned"
    with pytest.raises(ValueError, match=cannot(["clean"], ["describe"])):
        validate_run_overrides(plan, {"clean": False})


def test_apply_gate_decision():
    plan = plan_for(QuestionType.TWO_GROUP_COMPARISON)
    before = copy.deepcopy(plan)
    new = apply_gate_decision(plan, {"describe": False}, "approved_with_changes")
    assert plan == before
    assert step(new, "describe")["run"] is False
    assert {s["step"]: s for s in plan_steps(new) if s["step"] != "describe"} == \
        {s["step"]: s for s in plan_steps(plan) if s["step"] != "describe"}
    assert new["approval"] == {"status": "approved_with_changes", "overridden_steps": ["describe"], "reason": None}
    assert {k: v for k, v in new.items() if k != "approval"}.keys() == plan.keys()
    json.dumps(new)

    no_op = apply_gate_decision(plan, {"describe": True}, "approved")
    assert no_op["approval"]["overridden_steps"] == []
    rejected = apply_gate_decision(plan, {}, "rejected", "wrong question")
    assert rejected["approval"] == {"status": "rejected", "overridden_steps": [], "reason": "wrong question"}


# --- payload -----------------------------------------------------------------------------------------

def test_gate_payload():
    plan = plan_for(QuestionType.TWO_GROUP_COMPARISON)
    payload = gate_payload(plan, "Did West spend more?")
    assert set(payload) == {"type", "question", "request_text", "question_type", "stages", "adjustable_steps",
                            "allowed_decisions"}
    assert (payload["type"], payload["question"], payload["request_text"], payload["question_type"]) == (
        GATE, "Review which steps this run will take before anything is changed.", "Did West spend more?",
        "two_group_comparison")
    assert payload["allowed_decisions"] == ["approve", "reject"] and payload["adjustable_steps"] == ["describe"]
    assert [(s["stage"], s["name"]) for s in payload["stages"]] == [(st["stage"], st["name"]) for st in plan["stages"]]
    steps = {s["step"]: s for stage in payload["stages"] for s in stage["steps"]}
    for stage in payload["stages"]:
        assert set(stage) == {"stage", "name", "steps"}
        for s in stage["steps"]:
            assert set(s) == {"step", "state", "run", "exploratory", "reason", "adjustable", "note"}
    assert {n: s["adjustable"] for n, s in steps.items()} == {
        "clean": False, "describe": True, "inference": False, "modeling": False}
    assert {n: s["note"] for n, s in steps.items()} == {
        "clean": None, "describe": None, "inference": None,
        "modeling": "offered after the main analysis; not selectable yet"}
    json.dumps(payload)


def test_gate_payload_for_a_driver_plan():
    steps = {s["step"]: s for stage in gate_payload(plan_for(QuestionType.DRIVER_ANALYSIS), "x")["stages"]
             for s in stage["steps"]}
    assert {n: s["adjustable"] for n, s in steps.items()} == {
        "clean": False, "describe": True, "inference": False, "modeling": False}
    assert all(s["note"] is None for s in steps.values())


# --- record ------------------------------------------------------------------------------------------

def _candidates(r):
    return [(c.option, c.eligible, c.reason) for c in r.candidates]


CANDIDATES = [("approve", True, "run the steps as proposed"),
              ("approve_with_changes", True, "switch describe on or off"),
              ("reject", True, "stop before any cleaning; the run ends")]


def test_record_approve():
    plan = plan_for(QuestionType.TWO_GROUP_COMPARISON)
    r = stage_plan_approval_record(plan, {"action": "approve"}, {}, "approved")
    assert (r.decision_point, r.chosen, r.decided_by) == (STAGE_PLAN_APPROVAL_POINT, "approve", "human")
    assert (r.overridden_by_human, r.original_choice, r.rationale) == (False, None, "approved as proposed")
    assert _candidates(r) == CANDIDATES
    assert r.evidence == {
        "question_type": "two_group_comparison",
        "run_before": {"clean": False, "describe": True, "inference": True, "modeling": False},
        "run_after": {"clean": False, "describe": True, "inference": True, "modeling": False},
        "overridden_steps": [], "adjustable_steps": ["describe"], "exploratory_offers": ["modeling"],
        "received_action": "approve",
    }


def test_record_describe_switched_off():
    plan = plan_for(QuestionType.TWO_GROUP_COMPARISON)
    r = stage_plan_approval_record(plan, {"action": "approve", "run": {"describe": False}}, {"describe": False},
                                   "approved_with_changes")
    assert (r.chosen, r.decided_by, r.overridden_by_human) == ("approve_with_changes", "human", True)
    assert r.original_choice == "approve the plan as proposed: describe, inference"
    assert r.rationale == "describe switched off"
    assert r.evidence["run_after"]["describe"] is False and r.evidence["overridden_steps"] == ["describe"]


def test_record_describe_switched_on_after_a_default_off():
    plan = plan_for(QuestionType.TWO_GROUP_COMPARISON)
    step(plan, "describe")["run"] = False
    r = stage_plan_approval_record(plan, {"action": "approve", "run": {"describe": True}}, {"describe": True},
                                   "approved_with_changes")
    assert (r.chosen, r.rationale) == ("approve_with_changes", "describe switched on")
    assert r.original_choice == "approve the plan as proposed: inference"
    assert (r.evidence["run_before"]["describe"], r.evidence["run_after"]["describe"]) == (False, True)


@pytest.mark.parametrize("decision, rationale", [
    ({"action": "reject", "reason": "wrong outcome"}, "wrong outcome"),
    ({"action": "reject"}, "no reason given"),
])
def test_record_reject(decision, rationale):
    r = stage_plan_approval_record(plan_for(QuestionType.DRIVER_ANALYSIS), decision, {}, "rejected")
    assert (r.chosen, r.decided_by, r.overridden_by_human, r.original_choice) == ("reject", "human", False, None)
    assert r.rationale == rationale and r.evidence["received_action"] == "reject"
    assert r.evidence["exploratory_offers"] == []


def test_record_unrecognised_action():
    r = stage_plan_approval_record(plan_for(QuestionType.TWO_GROUP_COMPARISON), {"action": "maybe"}, {}, "approved")
    assert (r.chosen, r.decided_by) == ("approve", "code_rule")
    assert r.rationale == "action 'maybe' is not one of approve or reject; the node approves the plan as proposed (G36)"
    assert r.evidence["received_action"] == "maybe"


def test_records_serialize_with_unique_ids():
    plan = plan_for(QuestionType.TWO_GROUP_COMPARISON)
    records = [stage_plan_approval_record(plan, {"action": "approve"}, {}, "approved"),
               stage_plan_approval_record(plan, {"action": "reject"}, {}, "rejected"),
               stage_plan_approval_record(plan, {"action": "maybe"}, {}, "approved")]
    assert len({r.id for r in records}) == 3
    json.dumps(log_update(*records))


def test_approve_with_changes_is_not_eligible_without_an_adjustable_step():
    r = stage_plan_approval_record(plan_for(QuestionType.DESCRIPTIVE), {"action": "approve"}, {}, "approved")
    assert r.candidates[1].eligible is False


# --- the node through the real graph ---------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _points(values):
    return [e["decision_point"] for e in values["decision_log"]]


def test_gate_pauses_before_any_cleaning_or_model_call(tmp_path, store):
    allowed = {"after_gate": False}

    def propose(issues):
        assert allowed["after_gate"], "the proposer ran before the gate was answered"
        return fake_propose(issues)

    def draft(request_text, profile):
        assert allowed["after_gate"], "the draft ran before the gate was answered"
        return fake_draft(["region"])(request_text, profile)

    app = build_agent_graph(store, classify_fn=confident(QuestionType.TWO_GROUP_COMPARISON), draft_fn=draft,
                            propose_fn=propose)
    config = {"configurable": {"thread_id": "gate"}}
    ds = _dataset(_with_duplicate(_two_group_df(np.random.default_rng(41))), tmp_path, store, "gate")
    result = app.invoke({"request_text": "Did West spend more?", "session_id": "session-abc", "dataset": ds},
                        config, interrupt_after=STOP)
    payload = result["__interrupt__"][0].value
    state = app.get_state(config).values
    assert payload == gate_payload(state["stage_plan"], "Did West spend more?")
    assert not any(p.startswith("stage1.") for p in _points(state)) and state["dataset"].version == 1

    allowed["after_gate"] = True
    result = app.invoke(Command(resume={"action": "approve"}), config, interrupt_after=STOP)
    assert result["__interrupt__"][0].value["type"] == "wrangling_approval"
    assert app.get_state(config).values["stage_plan"]["approval"]["status"] == "approved"


def _gate_run(tmp_path, store, gate, name):
    app = build_agent_graph(store, classify_fn=confident(QuestionType.TWO_GROUP_COMPARISON),
                            draft_fn=fake_draft(["region"]), propose_fn=must_not_propose)
    state = {"request_text": "x", "session_id": "session-abc",
             "dataset": _dataset(_two_group_df(np.random.default_rng(42)), tmp_path, store, name)}
    resumes = [] if gate.get("action") == "reject" else [{"action": "approve"}]
    return _drive(app, state, name, resumes, {}, gate=gate)


def test_switching_describe_off(tmp_path, store):
    values, pauses, _ = _gate_run(tmp_path, store, {"action": "approve", "run": {"describe": False}}, "off")
    assert pauses == [GATE, "plan_approval"]
    assert values["stage_plan"]["approval"]["status"] == "approved_with_changes"
    assert step(values["stage_plan"], "describe")["run"] is False
    record = next(e for e in values["decision_log"] if e["decision_point"] == STAGE_PLAN_APPROVAL_POINT)
    assert (record["chosen"], record["overridden_by_human"]) == ("approve_with_changes", True)
    assert "stage1.describe.execution" not in _points(values) and "descriptive_result" not in values
    assert values["test_result"]


def test_keeping_describe_on(tmp_path, store):
    values, _, _ = _gate_run(tmp_path, store, {"action": "approve", "run": {"describe": True}}, "on")
    assert values["stage_plan"]["approval"]["status"] == "approved"
    assert "stage1.describe.execution" in _points(values)


def test_an_invalid_override_raises_and_records_nothing(tmp_path, store):
    app = build_agent_graph(store, classify_fn=confident(QuestionType.TWO_GROUP_COMPARISON),
                            draft_fn=fake_draft(["region"]), propose_fn=must_not_propose)
    config = {"configurable": {"thread_id": "bad"}}
    ds = _dataset(_two_group_df(np.random.default_rng(43)), tmp_path, store, "bad")
    app.invoke({"request_text": "x", "session_id": "session-abc", "dataset": ds}, config, interrupt_after=STOP)
    with pytest.raises(ValueError, match=re.escape("Cannot change step(s) ['clean'] here.")):
        app.invoke(Command(resume={"action": "approve", "run": {"clean": False}}), config, interrupt_after=STOP)
    values = app.get_state(config).values
    assert STAGE_PLAN_APPROVAL_POINT not in _points(values)
    assert "approval" not in values["stage_plan"]


def test_reject_ends_the_run(tmp_path, store):
    values, pauses, next_nodes = _gate_run(tmp_path, store, {"action": "reject", "reason": "wrong data"}, "reject")
    assert (pauses, next_nodes) == ([GATE], ())
    assert values["clarification_needed"] == "Stage plan rejected: wrong data"
    assert values["stage_plan"]["approval"]["status"] == "rejected"
    points = _points(values)
    assert points[-1] == STAGE_PLAN_APPROVAL_POINT
    record = values["decision_log"][-1]
    assert (record["chosen"], record["decided_by"]) == ("reject", "human")
    assert not any(p.startswith(("stage1.", "stage2.", "stage3.")) for p in points)
    assert not any(p in points for p in ("planner.outcome", "planner.route", "planner.plan_approval"))


# --- pause order -------------------------------------------------------------------------------------------

def _start(ds, **extra):
    return {"request_text": "x", "session_id": "session-abc", "dataset": ds, **extra}


def _messy(rng):
    df = _two_group_df(rng)
    df["notes"] = [None if i % 6 == 0 else "ok" for i in range(len(df))]
    return _with_duplicate(df)


CLEAN_BOTH = {"decisions": [{"index": 0, "action": "approve"}, {"index": 1, "action": "approve"}]}


def test_pause_order_comparison_without_issues(tmp_path, store):
    app = build_agent_graph(store, classify_fn=confident(QuestionType.TWO_GROUP_COMPARISON),
                            draft_fn=fake_draft(["region"]), propose_fn=must_not_propose)
    _, pauses, _ = _drive(app, _start(_dataset(_two_group_df(np.random.default_rng(44)), tmp_path, store, "p1")),
                          "p1", [{"action": "approve"}], {})
    assert pauses == [GATE, "plan_approval"]


def test_pause_order_comparison_with_duplicates(tmp_path, store):
    app = build_agent_graph(store, classify_fn=confident(QuestionType.TWO_GROUP_COMPARISON),
                            draft_fn=fake_draft(["region"]), propose_fn=fake_propose)
    ds = _dataset(_with_duplicate(_two_group_df(np.random.default_rng(45))), tmp_path, store, "p2")
    _, pauses, _ = _drive(app, _start(ds), "p2", [{"decisions": [{"index": 0, "action": "approve"}]},
                                                  {"action": "approve"}], {})
    assert pauses == [GATE, "wrangling_approval", "plan_approval"]


def test_pause_order_with_a_clarification(tmp_path, store):
    app = build_agent_graph(store, classify_fn=fake_classify(QuestionType.TWO_GROUP_COMPARISON),
                            draft_fn=fake_draft(["region"]), propose_fn=fake_propose)
    _, pauses, _ = _drive(app, _start(_dataset(_messy(np.random.default_rng(46)), tmp_path, store, "p3")), "p3",
                          ["I mean the two regions, East and West.", CLEAN_BOTH, {"action": "approve"}], {})
    assert pauses == ["clarification_needed", GATE, "wrangling_approval", "plan_approval"]


def test_pause_order_driver_analysis(tmp_path, store):
    rng = np.random.default_rng(47)
    n = 120
    tenure = rng.normal(24, 8, n)
    df = pd.DataFrame({"tenure": tenure, "tier": rng.choice(["basic", "plus", "pro"], n),
                       "total_spend": 20 + 2 * tenure + rng.normal(0, 5, n)})
    app = build_agent_graph(store, classify_fn=confident(QuestionType.DRIVER_ANALYSIS),
                            draft_fn=fake_draft(["tenure", "tier"]), propose_fn=must_not_propose)
    _, pauses, _ = _drive(app, _start(_dataset(df, tmp_path, store, "p4")), "p4", [{"action": "approve"}], {})
    assert pauses == [GATE, "plan_approval"]


@pytest.mark.parametrize("question_type, next_nodes", [(QuestionType.DESCRIPTIVE, ("report",)),
                                                        (QuestionType.ASSOCIATION, ())])
def test_no_gate_for_descriptive_or_association_runs(question_type, next_nodes, tmp_path, store):
    app = build_agent_graph(store, classify_fn=confident(question_type), draft_fn=fake_draft(["region"]),
                            propose_fn=must_not_propose)
    values, pauses, nxt = _drive(app, _start(_dataset(_two_group_df(np.random.default_rng(48)), tmp_path, store,
                                                      question_type.value)), question_type.value, [], {})
    assert (pauses, nxt) == ([], next_nodes)
    assert STAGE_PLAN_APPROVAL_POINT not in _points(values)


def test_stakeholder_mode_still_pauses_at_the_gate(tmp_path, store):
    app = build_agent_graph(store, classify_fn=confident(QuestionType.TWO_GROUP_COMPARISON),
                            draft_fn=fake_draft(["region"]), propose_fn=fake_propose)
    ds = _dataset(_with_duplicate(_two_group_df(np.random.default_rng(49))), tmp_path, store, "p5")
    _, pauses, _ = _drive(app, _start(ds, autonomy_mode="stakeholder"), "p5", [{"action": "approve"}], {})
    assert pauses == [GATE, "plan_approval"]  # the dedupe is auto-approved; the gate is not


# --- routing, edges and registry ------------------------------------------------------------------------------

def test_route_after_plan_stages():
    assert route_after_plan_stages({"stage_plan": plan_for(QuestionType.ASSOCIATION)}) == END
    assert route_after_plan_stages({"stage_plan": plan_for(QuestionType.TWO_GROUP_COMPARISON)}) == "stage_plan_gate"
    assert route_after_plan_stages({"stage_plan": plan_for(QuestionType.DESCRIPTIVE)}) == "wrangle_propose"


def test_route_after_stage_plan_gate():
    plan = plan_for(QuestionType.TWO_GROUP_COMPARISON)
    assert route_after_stage_plan_gate({"stage_plan": apply_gate_decision(plan, {}, "approved")}) == "wrangle_propose"
    assert route_after_stage_plan_gate({"stage_plan": apply_gate_decision(plan, {}, "rejected")}) == END


def test_graph_edges(tmp_path):
    edges = {(e.source, e.target): e.conditional
             for e in build_agent_graph(LocalDiskStore(base_dir=tmp_path)).get_graph().edges}
    for target in ("stage_plan_gate", "wrangle_propose", "__end__"):
        assert edges[("plan_stages", target)] is True, target
    assert edges[("stage_plan_gate", "wrangle_propose")] is True
    assert edges[("stage_plan_gate", "__end__")] is True


def test_registry():
    assert len(registered_points()) == 26 and len(DECISION_NODES) == 12
    assert DECISION_NODES["stage_plan_gate"] == (STAGE_PLAN_APPROVAL_POINT,)
    assert NODE_STAGES["stage_plan_gate"] == "intake"
