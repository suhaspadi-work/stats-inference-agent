"""
M1.5 step P3b1: one on-request explanation per (round, method, question) (product/explain_flow.py): the model is
called only for a new request, the answer is cached, it is recorded once, and a failed record keeps the answer.
Fakes only: no model.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "product"))

from explain_flow import explain_once  # noqa: E402
from run_state import explanation_key  # noqa: E402


class Explainer:
    def __init__(self, error=None):
        self.calls, self.error = [], error

    def __call__(self, method_id, round_index, question):
        self.calls.append((method_id, round_index, question))
        if self.error is not None:
            raise self.error
        return {"method_id": method_id, "round": round_index, "question": question, "answer": f"about {method_id}",
                "source": "model"}


class Recorder:
    def __init__(self, error=None):
        self.calls, self.error = [], error

    def __call__(self, result):
        self.calls.append(result)
        if self.error is not None:
            raise self.error


def test_a_first_request():
    state, explain, record = {}, Explainer(), Recorder()
    result, computed = explain_once(state, (1, "ols", None), explain, record)
    assert computed is True and explain.calls == [("ols", 1, None)]
    assert record.calls == [result] and result["saved"] is True
    assert state["explanations"][explanation_key(1, "ols", None)] is result


def test_the_same_request_again_is_cached():
    state, explain, record = {}, Explainer(), Recorder()
    first, _ = explain_once(state, (1, "ols", "Why?"), explain, record)
    again, computed = explain_once(state, (1, "ols", "Why?"), explain, record)
    assert again is first and computed is False
    assert len(explain.calls) == 1 and len(record.calls) == 1


def test_a_blank_and_a_none_question_share_one_call():
    state, explain = {}, Explainer()
    explain_once(state, (1, "ols", None), explain)
    _, computed = explain_once(state, (1, "ols", "   "), explain)
    assert computed is False and len(explain.calls) == 1


def test_different_requests_each_call():
    state, explain = {}, Explainer()
    for request in ((1, "ols", None), (2, "ols", None), (1, "dunn", None), (1, "ols", "other")):
        assert explain_once(state, request, explain)[1] is True
    assert len(explain.calls) == 4 and len(state["explanations"]) == 4


def test_an_explain_error_keeps_nothing():
    state, record = {}, Recorder()
    with pytest.raises(RuntimeError, match="rate limit"):
        explain_once(state, (1, "ols", None), Explainer(RuntimeError("rate limit")), record)
    assert explanation_key(1, "ols", None) not in (state.get("explanations") or {})
    assert record.calls == []
    explain = Explainer()
    result, computed = explain_once(state, (1, "ols", None), explain, record)
    assert computed is True and explain.calls == [("ols", 1, None)] and record.calls == [result]


def test_a_record_error_keeps_the_answer():
    state, explain, record = {}, Explainer(), Recorder(OSError("database is locked"))
    result, computed = explain_once(state, (1, "ols", None), explain, record)
    assert computed is True and result["answer"] == "about ols" and result["saved"] is False
    assert state["explanations"][explanation_key(1, "ols", None)] is result
    again, computed = explain_once(state, (1, "ols", None), explain, record)
    assert again is result and computed is False and len(record.calls) == 1 and len(explain.calls) == 1


def test_without_a_recorder():
    state = {}
    result, computed = explain_once(state, (1, "ols", None), Explainer())
    assert computed is True and "saved" not in result


def test_states_are_independent():
    a, b, explain = {}, {}, Explainer()
    explain_once(a, (1, "ols", None), explain)
    _, computed = explain_once(b, (1, "ols", None), explain)
    assert computed is True and len(explain.calls) == 2
