"""
M1.5 step P1: the product's pure run bookkeeping (product/run_state.py) and widget keys (product/pause_types.py).
The product modules import no streamlit, so they load here from the sibling product/ folder.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "product"))

from pause_types import widget_key  # noqa: E402
from run_state import (  # noqa: E402
    explanation_key, get_explanation, next_pause_serial, record_once, remember_explanation, reset_run, store_result,
)


# --- widget_key -----------------------------------------------------------------------------------------------------

def test_widget_key():
    assert widget_key("wrangle_action", 3, 0) == "wrangle_action-3-0"
    assert widget_key("approve", 1) == "approve-1"
    assert widget_key("step", 2, "describe", "run") == "step-2-describe-run"
    assert widget_key("x", 4, None, 1.5, True) == "x-4-None-1.5-True"


# --- next_pause_serial -------------------------------------------------------------------------------------------------

def test_next_pause_serial():
    state = {}
    assert [next_pause_serial(state) for _ in range(3)] == [1, 2, 3]
    assert state["pause_serial"] == 3


# --- record_once ---------------------------------------------------------------------------------------------------------

def test_record_once():
    state, calls = {}, []
    assert record_once(state, lambda: calls.append(1)) is True
    assert record_once(state, lambda: calls.append(2)) is False
    assert calls == [1] and state["recorded_run"] is True


def test_a_failing_recorder_leaves_the_flag_unset():
    state, calls = {}, []

    def boom():
        raise RuntimeError("disk full")
    with pytest.raises(RuntimeError, match="disk full"):
        record_once(state, boom)
    assert "recorded_run" not in state
    assert record_once(state, lambda: calls.append(1)) is True and calls == [1]


def test_record_once_is_per_state():
    a, b, calls = {}, {}, []
    assert record_once(a, lambda: calls.append("a")) and record_once(b, lambda: calls.append("b"))
    assert calls == ["a", "b"]


# --- reset_run -----------------------------------------------------------------------------------------------------------

def test_reset_run():
    state = {"graph_result": {}, "current_question": "q", "recorded_run": True, "pause_serial": 7, "dataset": "d"}
    reset_run(state)
    assert state == {"pause_serial": 7, "dataset": "d"}
    empty = {}
    reset_run(empty)
    assert empty == {}


def test_the_finished_screen_records_once_per_run():
    state, calls = {"graph_result": {"report": "r1"}, "current_question": "q1"}, []
    for _ in range(3):  # three Streamlit reruns of the finished screen
        record_once(state, lambda: calls.append(state["current_question"]))
    assert calls == ["q1"]
    reset_run(state)  # "Ask another question"
    state.update(graph_result={"report": "r2"}, current_question="q2")
    for _ in range(2):
        record_once(state, lambda: calls.append(state["current_question"]))
    assert calls == ["q1", "q2"]


# --- store_result (M1.5 step P2b) ----------------------------------------------------------------------------------------

def test_store_result_with_an_interrupt():
    state, paused = {}, {"__interrupt__": ("pause",)}
    assert store_result(state, paused) == 1
    assert state["graph_result"] is paused and state["pause_serial"] == 1


def test_store_result_without_an_interrupt():
    state, finished = {"pause_serial": 4}, {"report": "done"}
    assert store_result(state, finished) is None
    assert state["graph_result"] is finished and state["pause_serial"] == 4


def test_store_result_replaces_and_counts():
    state = {}
    results = [{"__interrupt__": ("a",)}, {"__interrupt__": ("b",)}, {"report": "r"}, {"__interrupt__": ("c",)}]
    serials = [store_result(state, r) for r in results]
    assert serials == [1, 2, None, 3]
    assert state["graph_result"] is results[-1]


# --- the explanation cache (M1.5 step P3a) ----------------------------------------------------------------------------------

def test_explanation_key():
    assert explanation_key(1, "ols", None) == explanation_key(1, "ols", "") == explanation_key(1, "ols", "   ") == "1|ols|"
    assert explanation_key(1, "ols", "  Why?  ") == "1|ols|Why?"
    keys = {explanation_key(1, "ols", "q"), explanation_key(2, "ols", "q"), explanation_key(1, "dunn", "q"),
            explanation_key(1, "ols", "other")}
    assert len(keys) == 4


def test_remember_and_get_explanation():
    a, b = {}, {}
    key = explanation_key(1, "ols", None)
    result = {"answer": "It fits a line.", "source": "model"}
    assert remember_explanation(a, key, result) is result
    assert get_explanation(a, key) is result
    assert get_explanation(a, explanation_key(2, "ols", None)) is None
    assert get_explanation(b, key) is None and "explanations" not in b


def test_reset_run_forgets_the_message_and_its_explanations():
    state = {"graph_result": {}, "current_question": "q", "recorded_run": True, "message_id": 7,
             "explanations": {"1|ols|": {}}, "pause_serial": 3}
    reset_run(state)
    assert state == {"pause_serial": 3}
