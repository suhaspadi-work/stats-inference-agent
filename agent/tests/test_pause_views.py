"""
M1.5 step P2a: the pure view logic of every pause (product/pause_views.py). Views are checked on the real
payloads of the contract run; every answer builder's output is fed to the real graph, so the answer shapes cannot
drift from what the nodes accept.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "product"))

import pause_views as pv  # noqa: E402
from pause_types import HANDLED_PAUSE_TYPES  # noqa: E402

from core.dataset import LocalDiskStore  # noqa: E402
from core.graph import build_agent_graph  # noqa: E402
from core.method_glossary import GLOSSARY  # noqa: E402
from core.plan import QuestionType  # noqa: E402
from tests.test_decision_point_guard import _dataset, confident, fake_draft, fake_propose  # noqa: E402
from tests.test_product_contract import _messy, payloads  # noqa: E402,F401 (fixture)
from tests.test_prompt_canaries import fake_model  # noqa: E402,F401 (fixture)
from tests.test_stage1_report import _drive  # noqa: E402


def _of(payloads, kind):
    return [p for p in payloads if p["type"] == kind]


# --- refusal_lines and method_label ------------------------------------------------------------------------------------

def test_refusal_lines():
    assert pv.refusal_lines(None) == []
    refused = {"attempt": 2, "refusals": [
        {"step": "clean", "rule": "planned", "reason": "r1", "explanation": "clean always runs"},
        {"column": "customer_id", "rule": "identifier", "reason": "r2", "explanation": "looks like an id"},
        {"column": None, "rule": "none_chosen", "reason": "r3", "explanation": "choose at least one"},
        {"column": "tier", "rule": "x", "reason": "only a reason", "explanation": None},
    ]}
    assert pv.refusal_lines(refused) == [
        "Your last choice was refused (attempt 2):", "clean: clean always runs", "customer_id: looks like an id",
        "none_chosen: choose at least one", "tier: only a reason"]


def test_method_label():
    assert pv.method_label("student_t_test") == GLOSSARY["student_t_test"]["name"]
    assert pv.method_label("made_up") == "made_up"
    assert pv.method_label(None) == "unknown method"


# --- the views on the real payloads -----------------------------------------------------------------------------------

def test_every_real_payload_has_a_view(payloads):
    assert {p["type"] for p in payloads} == set(HANDLED_PAUSE_TYPES)
    for p in payloads:
        view = pv.view_for(p)
        assert "unsupported" not in view, p["type"]


def test_stage_plan_view(payloads):
    for p in _of(payloads, "stage_plan_approval"):
        view = pv.stage_plan_view(p)
        steps = [s for stage in p["stages"] for s in stage["steps"]]
        assert [r["step"] for r in view["rows"]] == [s["step"] for s in steps]
        assert all(r["adjustable"] == (r["step"] in p["adjustable_steps"]) for r in view["rows"])
        for r in view["rows"]:
            later = r["state"] == "offered" and not r["adjustable"]
            assert r["state_label"] == (pv.OFFERED_LATER_LABEL if later else pv.STATE_LABELS[r["state"]]), r["step"]
        assert view["adjustable"] == p["adjustable_steps"] and view["request_text"] == p["request_text"]
        if p["refused"]:
            assert view["refusal"][0] == f"Your last choice was refused (attempt {p['refused']['attempt']}):"
            assert len(view["refusal"]) == 1 + len(p["refused"]["refusals"])
        else:
            assert view["refusal"] == []
    assert sum(bool(p["refused"]) for p in _of(payloads, "stage_plan_approval")) == 1


def test_the_modeling_row_is_asked_later(payloads):
    for p in _of(payloads, "stage_plan_approval"):
        rows = {r["step"]: r for r in pv.stage_plan_view(p)["rows"]}
        step = next(s for stage in p["stages"] for s in stage["steps"] if s["step"] == "modeling")
        assert (rows["modeling"]["state"], rows["modeling"]["adjustable"]) == ("offered", False)
        assert rows["modeling"]["state_label"] == "asked after the main analysis"
        assert rows["modeling"]["note"] == step["note"] and step["note"]
        assert rows["describe"]["adjustable"] and rows["describe"]["state_label"] == "your choice"
        assert rows["describe"]["note"] is None


def test_the_two_offered_labels():
    def row(adjustable, **extra):
        step = {"step": "x", "state": "offered", "run": False, "reason": "r", "adjustable": adjustable, **extra}
        payload = {"request_text": "q", "question_type": "t", "stages": [{"steps": [step]}],
                   "adjustable_steps": ["x"] if adjustable else [], "refused": None}
        return pv.stage_plan_view(payload)["rows"][0]
    assert row(True)["state_label"] == "your choice" and row(True)["note"] is None
    later = row(False, note="chosen after the main analysis")
    assert (later["state_label"], later["note"]) == ("asked after the main analysis", "chosen after the main analysis")


def test_plan_approval_view(payloads):
    plans = _of(payloads, "plan_approval")
    for p in plans:
        view = pv.plan_approval_view(p)
        assert view["reselected_note"] == (pv.RESELECTED_NOTE if p["reselected"] else None)
        assert view["predictors"] == ", ".join(p["candidate_predictors"])
        assert view["method_label"] == GLOSSARY[p["method"]]["name"]
        assert view["stage1_report"] == p["stage1_report"]
        assert (view["refusal"] != []) == bool(p["refused"])
    assert sum(p["reselected"] for p in plans) >= 1 and sum(not p["reselected"] for p in plans) >= 1
    assert any(pv.plan_approval_view(p)["stage1_report"]["title"] for p in plans if p["stage1_report"])


def test_stage_gate_view(payloads):
    gates = _of(payloads, "stage_gate")
    for p in gates:
        view = pv.stage_gate_view(p)
        offer = p["offer"]
        assert view["group_column"] == offer["proposed_predictors"][0]
        assert view["options"] == offer["selectable"]
        assert view["default_own"] == offer["proposed_predictors"][1:]
        assert [x["column"] for x in view["left_out"]] == list(offer["unsuitable"])
        assert view["finished"][0] == f"Round {p['finished']['round']}: {pv.method_label(p['finished']['method'])}."
        assert view["finished"][1] == f"p = {p['finished']['p_value']:.3g}"
        assert (view["refusal"] != []) == bool(p["refused"])
    assert any(p["refused"] for p in gates) and any(not p["refused"] for p in gates)


def test_wrangling_and_clarification_views(payloads):
    (w,) = _of(payloads, "wrangling_approval")
    view = pv.wrangling_view(w)
    assert [o["index"] for o in view["operations"]] == [o["index"] for o in w["proposed_operations"]]
    assert view["note"] == w.get("auto_approved_note")
    (c,) = _of(payloads, "clarification_needed")
    assert pv.clarification_view(c) == {"question": c["question"]}


def test_view_for_an_unknown_type():
    assert pv.view_for({"type": "made_up_pause"}) == {"unsupported": True, "type": "made_up_pause"}


# --- the answer builders ---------------------------------------------------------------------------------------------

def _stage_view():
    return {"rows": [{"step": "clean", "run": True, "adjustable": False},
                     {"step": "describe", "run": True, "adjustable": True}], "adjustable": ["describe"]}


def test_approve_stage_plan():
    view = _stage_view()
    assert pv.approve_stage_plan(view, {}) == {"action": "approve"}
    assert pv.approve_stage_plan(view, None) == {"action": "approve"}
    assert pv.approve_stage_plan(view, {"describe": True}) == {"action": "approve"}
    assert pv.approve_stage_plan(view, {"describe": False}) == {"action": "approve", "run": {"describe": False}}
    assert pv.approve_stage_plan(view, {"clean": False}) == {"action": "approve"}  # not adjustable: ignored


def test_reject_answers():
    assert pv.reject_stage_plan("  ") == {"action": "reject", "reason": "No reason given"}
    assert pv.reject_stage_plan(None) == {"action": "reject", "reason": "No reason given"}
    assert pv.reject_plan(" wrong outcome ") == {"action": "reject", "reason": "wrong outcome"}
    assert pv.approve_plan() == {"action": "approve"}


def test_gate_answer():
    view = {"group_column": "region"}
    assert pv.gate_answer(view, "proposed") == {"action": "run"}
    assert pv.gate_answer(view, "own", ["tier", "tenure"]) == {"action": "run", "predictors": ["region", "tier", "tenure"]}
    assert pv.gate_answer(view, "own", ["tier", "region"]) == {"action": "run", "predictors": ["region", "tier"]}
    assert pv.gate_answer(view, "own", []) == {"action": "run", "predictors": ["region"]}
    assert pv.gate_answer(view, "skip") == {"action": "skip"}
    with pytest.raises(ValueError, match="'later'"):
        pv.gate_answer(view, "later")


# --- the real backend accepts every answer the builders make -----------------------------------------------------------

def _builder_run(tmp_path, name, gate_answers):
    """A two-group run on messy data whose every answer comes from the builders; gate_answers maps a stage_gate
    view to its answer, in order."""
    store = LocalDiskStore(base_dir=tmp_path / name)
    app = build_agent_graph(store, classify_fn=confident(QuestionType.TWO_GROUP_COMPARISON),
                            draft_fn=fake_draft(["region"]), propose_fn=fake_propose)
    state = {"request_text": "Did West spend more?", "session_id": "session-abc",
             "dataset": _dataset(_messy(), tmp_path, store, name)}
    gates = list(gate_answers)

    def stage_plan(payload):
        view = pv.stage_plan_view(payload)
        describe = next(r["run"] for r in view["rows"] if r["step"] == "describe")
        return pv.approve_stage_plan(view, {"describe": not describe, "clean": False})

    def gate(payload):
        return gates.pop(0)(pv.stage_gate_view(payload))
    values, pauses, _ = _drive(app, state, name, stage_plan_approval=stage_plan,
                               plan_approval=lambda payload: pv.approve_plan(), stage_gate=gate)
    assert gates == []
    return values, pauses


def _records(values, point):
    return [r for r in values["decision_log"] if r["decision_point"] == point]


def test_the_backend_accepts_the_proposed_run(fake_model, tmp_path):
    values, _ = _builder_run(tmp_path, "proposed", [lambda v: pv.gate_answer(v, "proposed")])
    (approval,) = _records(values, "planner.stage_plan_approval")
    assert approval["chosen"] != "refused"
    # describe is on by default; the builder toggled it off, and the backend applied it
    describe = next(s for stage in values["stage_plan"]["stages"] for s in stage["steps"] if s["step"] == "describe")
    assert describe["run"] is False and _records(values, "stage1.describe.execution") == []
    assert [e["round"] for e in values["rounds"]] == [1, 2]
    assert [g["chosen"] for g in _records(values, "planner.stage_gate")] == ["run"]


def test_the_backend_accepts_own_predictors(fake_model, tmp_path):
    def own(view):
        return pv.gate_answer(view, "own", [view["options"][0]["column"], view["group_column"]])
    values, _ = _builder_run(tmp_path, "own", [own])
    assert len(values["rounds"]) == 2
    (gate,) = _records(values, "planner.stage_gate")
    assert gate["chosen"] == "run"
    assert values["plan"].candidate_predictors[0] == values["rounds"][0]["plan"].candidate_predictors[0]


def test_an_empty_own_choice_is_refused_then_skip(fake_model, tmp_path):
    values, pauses = _builder_run(tmp_path, "empty", [lambda v: pv.gate_answer(v, "own", []),
                                                     lambda v: pv.gate_answer(v, "skip")])
    assert [g["chosen"] for g in _records(values, "planner.stage_gate")] == ["refused", "skip"]
    assert len(values["rounds"]) == 1
    second = [p["payload"] for p in pauses if p["type"] == "stage_gate"][1]
    assert pv.stage_gate_view(second)["refusal"][0] == "Your last choice was refused (attempt 1):"
