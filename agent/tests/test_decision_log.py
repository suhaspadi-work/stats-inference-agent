import json
import operator
import typing

import numpy as np
import pandas as pd
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from core.agent_state import AgentState
from core.decision_log import log_update
from core.decision_record import Candidate, DecisionRecord


def _record(point: str = "stage2.test_selection", **kwargs) -> DecisionRecord:
    return DecisionRecord(decision_point=point, chosen="welch_t_test", decided_by="code_rule", **kwargs)


def _assert_plain(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert type(k) is str
            _assert_plain(v)
    elif isinstance(obj, list):
        for v in obj:
            _assert_plain(v)
    else:
        assert type(obj) in (str, int, float, bool, type(None)), f"not plain: {type(obj)}"


# --- state annotation -----------------------------------------------------

def test_decision_log_annotation_resolves_to_operator_add():
    hints = typing.get_type_hints(AgentState, include_extras=True)
    assert hints["decision_log"].__metadata__ == (operator.add,)


# --- log_update output ----------------------------------------------------

def test_output_plain_json_safe_and_ordered():
    a, b, c = _record("p.a"), _record("p.b"), _record("p.c")
    update = log_update(a, b, c)
    assert list(update) == ["decision_log"]
    assert [e["decision_point"] for e in update["decision_log"]] == ["p.a", "p.b", "p.c"]
    assert update["decision_log"][0] == a.to_dict()
    json.dumps(update)
    _assert_plain(update)


def test_no_records_gives_empty_list():
    assert log_update() == {"decision_log": []}


def test_numpy_scalars_become_python_types():
    r = _record(
        evidence={"n": np.int64(42), "p": np.float64(0.03), "sig": np.bool_(True), "nested": [np.int64(1)]},
        candidates=(Candidate(option="welch_t_test", eligible=True, reason="r",
                              evidence={"ratio": np.float64(3.1), "ok": np.bool_(False), "k": np.int64(2)}),),
        chart_data=({"type": "bar", "values": [np.float64(1.5), np.int64(2)], "flag": np.bool_(True)},),
    )
    entry = log_update(r)["decision_log"][0]
    _assert_plain(entry)
    assert entry["evidence"] == {"n": 42, "p": 0.03, "sig": True, "nested": [1]}
    assert type(entry["evidence"]["n"]) is int
    assert type(entry["evidence"]["p"]) is float
    assert type(entry["evidence"]["sig"]) is bool
    cand = entry["candidates"][0]["evidence"]
    assert (type(cand["ratio"]), type(cand["ok"]), type(cand["k"])) == (float, bool, int)
    chart = entry["chart_data"][0]
    assert [type(v) for v in chart["values"]] == [float, int]
    assert type(chart["flag"]) is bool


@pytest.mark.parametrize("raw, type_name", [
    (np.array([1, 2, 3]), "ndarray"),
    (pd.DataFrame({"x": [1, 2]}), "DataFrame"),
    (pd.Series([1, 2]), "Series"),
])
def test_raw_data_rejected_with_decision_point(raw, type_name):
    r = _record("stage1.clean.outliers", evidence={"values": raw})
    with pytest.raises(ValueError, match=rf"stage1\.clean\.outliers.*{type_name}"):
        log_update(r)


def test_raw_data_in_candidate_and_chart_data_rejected():
    r1 = _record("p.cand", candidates=(Candidate(option="o", eligible=True, reason="r",
                                                 evidence={"x": np.array([1.0])}),))
    with pytest.raises(ValueError, match=r"p\.cand.*ndarray"):
        log_update(r1)
    r2 = _record("p.chart", chart_data=({"rows": pd.Series([1, 2])},))
    with pytest.raises(ValueError, match=r"p\.chart.*Series"):
        log_update(r2)


def test_non_serializable_object_rejected_naming_type():
    class Opaque:
        pass

    r = _record("p.opaque", evidence={"thing": Opaque()})
    with pytest.raises(ValueError, match=r"p\.opaque.*Opaque"):
        log_update(r)


def test_non_decision_record_raises_type_error():
    with pytest.raises(TypeError, match="dict"):
        log_update(_record().to_dict())
    with pytest.raises(TypeError, match="Candidate"):
        log_update(_record(), Candidate(option="o", eligible=True, reason="r"))


def test_original_record_not_mutated():
    r = _record(evidence={"n": np.int64(5), "xs": [np.float64(1.0)]},
                chart_data=({"v": np.int64(3)},))
    before = r.to_dict()
    log_update(r)
    assert type(r.evidence["n"]) is np.int64
    assert type(r.evidence["xs"][0]) is np.float64
    assert type(r.chart_data[0]["v"]) is np.int64
    assert r.to_dict() == before


# --- graph merging --------------------------------------------------------

def _compile(builder):
    return builder.compile(checkpointer=InMemorySaver())


def test_two_nodes_append_in_order():
    r1, r2 = _record("p.first"), _record("p.second")
    b = StateGraph(AgentState)
    b.add_node("one", lambda s: log_update(r1))
    b.add_node("two", lambda s: log_update(r2))
    b.add_edge(START, "one")
    b.add_edge("one", "two")
    b.add_edge("two", END)
    out = _compile(b).invoke({"request_text": "q"}, {"configurable": {"thread_id": "t1"}})
    assert out["decision_log"] == [r1.to_dict(), r2.to_dict()]


def test_initial_state_without_decision_log():
    r = _record("p.only")
    b = StateGraph(AgentState)
    b.add_node("quiet", lambda s: {"report": "no decisions here"})
    b.add_node("logs", lambda s: log_update(r))
    b.add_edge(START, "quiet")
    b.add_edge("quiet", "logs")
    b.add_edge("logs", END)
    app = _compile(b)
    cfg = {"configurable": {"thread_id": "t2"}}
    state = {"request_text": "q"}
    assert "decision_log" not in state
    out = app.invoke(state, cfg)
    assert out["decision_log"] == [r.to_dict()]
    assert out["report"] == "no decisions here"


def test_interrupt_and_resume_does_not_duplicate():
    r1, r2 = _record("p.before_interrupt"), _record("p.after_resume")
    calls = {"first": 0, "ask": 0}

    def first(s):
        calls["first"] += 1
        return log_update(r1)

    def ask(s):
        calls["ask"] += 1
        answer = interrupt({"type": "plan_approval"})
        assert answer == "approve"
        return log_update(r2)

    b = StateGraph(AgentState)
    b.add_node("first", first)
    b.add_node("ask", ask)
    b.add_edge(START, "first")
    b.add_edge("first", "ask")
    b.add_edge("ask", END)
    app = _compile(b)
    cfg = {"configurable": {"thread_id": "t3"}}

    paused = app.invoke({"request_text": "q"}, cfg)
    assert "__interrupt__" in paused
    assert paused["decision_log"] == [r1.to_dict()]

    out = app.invoke(Command(resume="approve"), cfg)
    assert "__interrupt__" not in out
    assert out["decision_log"] == [r1.to_dict(), r2.to_dict()]
    assert calls == {"first": 1, "ask": 2}  # ask re-runs on resume; its record is still logged once
