"""
The pause screens (product/pauses.py) rendered headless with AppTest on the real payloads: each renders, shows
what it should, and returns exactly what the pure answer builders return for the same choices.
"""
import copy

import pytest
from streamlit.testing.v1 import AppTest

import pause_views as pv
from pause_types import widget_key

SERIAL = 1
REFUSED_HEADER = "Your last choice was refused (attempt"


def _script():
    import streamlit as st
    from pauses import render_pause
    st.session_state["answer"] = render_pause(st.session_state["payload"], st.session_state["values"],
                                              st.session_state["serial"])


def _app(pause, serial=SERIAL):
    at = AppTest.from_function(_script)
    at.session_state["payload"] = pause["payload"]
    at.session_state["values"] = pause["values"]
    at.session_state["serial"] = serial
    at.run()
    assert not at.exception, at.exception
    return at


def _answer(at):
    return at.session_state["answer"]


def _of(real_pauses, kind):
    return [p for p in real_pauses if p["payload"]["type"] == kind]


def _key(kind, *parts):
    return widget_key(kind, SERIAL, *parts)


def _refusals(at):
    return [e.value for e in at.error if e.value.startswith(REFUSED_HEADER)]


def _texts(at):
    return [m.value for m in at.markdown] + [c.value for c in at.caption]


def test_every_real_payload_renders(real_pauses):
    assert {p["payload"]["type"] for p in real_pauses} == {"clarification_needed", "stage_plan_approval",
                                                          "wrangling_approval", "plan_approval", "stage_gate"}
    for pause in real_pauses:
        at = _app(pause)
        assert _answer(at) is None, pause["payload"]["type"]


# --- stage_plan_approval -------------------------------------------------------------------------------------------------

def _stage_plans(real_pauses):
    plans = _of(real_pauses, "stage_plan_approval")
    refused = [p for p in plans if p["payload"]["refused"]]
    clean = [p for p in plans if not p["payload"]["refused"]]
    assert refused and clean
    return refused[0], clean[0]


def test_stage_plan_screen(real_pauses):
    refused, clean = _stage_plans(real_pauses)
    view = pv.stage_plan_view(clean["payload"])
    describe = next(r for r in view["rows"] if r["step"] == "describe")
    modeling = next(r for r in view["rows"] if r["step"] == "modeling")
    at = _app(clean)
    assert at.checkbox(key=_key("sp_toggle", "describe")).value == describe["run"]
    assert [c.key for c in at.checkbox] == [_key("sp_toggle", r["step"]) for r in view["rows"] if r["adjustable"]]
    assert modeling["note"] in _texts(at) and "**modeling**: asked after the main analysis" in _texts(at)
    assert _refusals(at) == []
    assert _refusals(_app(refused))[0].startswith(REFUSED_HEADER)


def test_stage_plan_answers(real_pauses):
    _, clean = _stage_plans(real_pauses)
    view = pv.stage_plan_view(clean["payload"])
    describe = next(r for r in view["rows"] if r["step"] == "describe")

    at = _app(clean)
    at.button(key=_key("sp_approve")).click().run()
    assert _answer(at) == {"action": "approve"} == pv.approve_stage_plan(view, {"describe": describe["run"]})

    at = _app(clean)
    at.checkbox(key=_key("sp_toggle", "describe")).set_value(not describe["run"]).run()
    at.button(key=_key("sp_approve")).click().run()
    assert _answer(at) == {"action": "approve", "run": {"describe": not describe["run"]}}
    assert _answer(at) == pv.approve_stage_plan(view, {"describe": not describe["run"]})

    at = _app(clean)
    at.text_input(key=_key("sp_reason")).input("not now").run()
    at.button(key=_key("sp_reject")).click().run()
    assert _answer(at) == {"action": "reject", "reason": "not now"} == pv.reject_stage_plan("not now")


# --- plan_approval -------------------------------------------------------------------------------------------------------

def test_plan_approval_screen(real_pauses):
    plans = _of(real_pauses, "plan_approval")
    for pause in plans:
        at = _app(pause)
        p = pause["payload"]
        infos = [i.value for i in at.info]
        assert infos == ([pv.RESELECTED_NOTE] if p["reselected"] else [])
        assert "Editing the predictors here is not available yet." in _texts(at)
        assert bool(_refusals(at)) == bool(p["refused"])
        if p["stage1_report"]:
            (expander,) = at.expander
            assert expander.label == p["stage1_report"]["title"]
            headings = [f"**{s['heading']}**" for s in p["stage1_report"]["sections"]]
            assert all(h in [m.value for m in expander.markdown] for h in headings)
    assert any(p["payload"]["reselected"] for p in plans) and any(p["payload"]["refused"] for p in plans)


def test_plan_approval_answers(real_pauses):
    pause = _of(real_pauses, "plan_approval")[0]
    at = _app(pause)
    at.button(key=_key("plan_approve")).click().run()
    assert _answer(at) == {"action": "approve"} == pv.approve_plan()
    at = _app(pause)
    at.text_input(key=_key("plan_reason")).input("wrong outcome").run()
    at.button(key=_key("plan_reject")).click().run()
    assert _answer(at) == {"action": "reject", "reason": "wrong outcome"} == pv.reject_plan("wrong outcome")


# --- stage_gate ------------------------------------------------------------------------------------------------------------

def _gates(real_pauses):
    gates = _of(real_pauses, "stage_gate")
    refused = [p for p in gates if p["payload"]["refused"]]
    clean = [p for p in gates if not p["payload"]["refused"]]
    assert refused and clean
    return refused[0], clean[0]


def test_stage_gate_screen(real_pauses):
    refused, clean = _gates(real_pauses)
    view = pv.stage_gate_view(clean["payload"])
    at = _app(clean)
    radio = at.radio(key=_key("gate_choice"))
    assert radio.options[0] == f"Run the exploratory analysis with the proposed predictors: {', '.join(view['proposed'])}"
    assert radio.value == radio.options[0]
    assert clean["values"]["report"] and clean["values"]["report"] in [m.value for m in at.markdown]
    assert all(line in [m.value for m in at.markdown] for line in view["finished"])
    (expander,) = at.expander
    assert expander.label == "Columns left out, and why"
    assert [m.value for m in expander.markdown] == [f"**{x['column']}**: {x['explanation']}" for x in view["left_out"]]
    assert _refusals(at) == []
    assert _refusals(_app(refused))[0].startswith(REFUSED_HEADER)


def test_stage_gate_answers(real_pauses):
    _, clean = _gates(real_pauses)
    view = pv.stage_gate_view(clean["payload"])

    at = _app(clean)
    at.button(key=_key("gate_submit")).click().run()
    assert _answer(at) == {"action": "run"} == pv.gate_answer(view, "proposed")

    at = _app(clean)
    at.radio(key=_key("gate_choice")).set_value("Run it with my own predictors").run()
    own = at.multiselect(key=_key("gate_own"))
    columns = [o["column"] for o in view["options"]]
    assert own.value == [c for c in view["default_own"] if c in columns]
    assert f"{view['group_column']} always stays in the model" in _texts(at)
    chosen = list(reversed(columns))
    own.set_value(chosen).run()
    at.button(key=_key("gate_submit")).click().run()
    assert _answer(at) == {"action": "run", "predictors": [view["group_column"], *chosen]}
    assert _answer(at) == pv.gate_answer(view, "own", chosen)
    assert _answer(at)["predictors"].count(view["group_column"]) == 1

    at = _app(clean)
    at.radio(key=_key("gate_choice")).set_value("Skip the exploratory analysis").run()
    at.button(key=_key("gate_submit")).click().run()
    assert _answer(at) == {"action": "skip"} == pv.gate_answer(view, "skip")


def test_widget_keys_do_not_leak_between_pauses(real_pauses):
    _, clean = _gates(real_pauses)
    at = _app(clean, serial=1)
    at.radio(key=widget_key("gate_choice", 1)).set_value("Skip the exploratory analysis").run()
    assert at.radio(key=widget_key("gate_choice", 1)).value == "Skip the exploratory analysis"
    at.session_state["serial"] = 2
    at.run()
    radio = at.radio(key=widget_key("gate_choice", 2))
    assert radio.value == radio.options[0]


# --- wrangling_approval and clarification_needed ---------------------------------------------------------------------------

def test_wrangling_screen(real_pauses):
    (pause,) = _of(real_pauses, "wrangling_approval")
    indices = [op["index"] for op in pause["payload"]["proposed_operations"]]
    assert len(indices) >= 2

    at = _app(pause)
    at.button(key=_key("wrangle_submit")).click().run()
    assert _answer(at) == {"decisions": [{"index": i, "action": "approve"} for i in indices]}

    at = _app(pause)
    at.radio(key=_key("wrangle_action", indices[0])).set_value("reject").run()
    at.text_input(key=_key("wrangle_reason", indices[0])).input("the gaps are meaningful").run()
    at.radio(key=_key("wrangle_action", indices[1])).set_value("edit").run()
    assert any("isn't built yet" in c.value for c in at.caption)
    at.button(key=_key("wrangle_submit")).click().run()
    assert _answer(at)["decisions"][:2] == [
        {"index": indices[0], "action": "reject", "reason": "the gaps are meaningful"},
        {"index": indices[1], "action": "approve"}]


def test_clarification_screen(real_pauses):
    (pause,) = _of(real_pauses, "clarification_needed")
    at = _app(pause)
    assert pause["payload"]["question"] in [m.value for m in at.markdown]
    at.button(key=_key("clarification_submit")).click().run()
    assert _answer(at) is None
    at.text_input(key=_key("clarification_input")).input("East versus West").run()
    at.button(key=_key("clarification_submit")).click().run()
    assert _answer(at) == "East versus West"


def test_an_unsupported_pause():
    at = _app({"payload": {"type": "made_up_pause", "x": 1}, "values": {}})
    assert [w.value for w in at.warning] == ["Interrupt type 'made_up_pause' isn't wired up in the UI yet."]
    assert len(at.json) == 1 and _answer(at) is None


# --- hand-built variants of real payloads (M1.5 step P3b2) ------------------------------------------------------------

def test_the_auto_approved_note(real_pauses):
    (pause,) = _of(real_pauses, "wrangling_approval")
    payload = {**copy.deepcopy(pause["payload"]), "auto_approved_note": "1 additional operation(s) were auto-approved."}
    at = _app({"payload": payload, "values": pause["values"]})
    assert [i.value for i in at.info] == ["1 additional operation(s) were auto-approved."]


def test_a_gate_with_no_left_out_columns(real_pauses):
    _, clean = _gates(real_pauses)
    payload = copy.deepcopy(clean["payload"])
    payload["offer"]["unsuitable"] = {}
    at = _app({"payload": payload, "values": clean["values"]})
    assert "Columns left out, and why" not in [e.label for e in at.expander]
    assert _app(clean).expander[0].label == "Columns left out, and why"
