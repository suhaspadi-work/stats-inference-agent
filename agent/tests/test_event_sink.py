import inspect

import pytest

from core.decision_log import log_update
from core.decision_record import DecisionRecord
from core.event_log import EventLog, events_path, read_decision_log
from core.event_sink import with_event_log

CONFIG = {"configurable": {"thread_id": "thread-1"}}
STATE = {"session_id": "session-abc", "request_text": "x"}


def _records(*charts):
    records = [DecisionRecord(decision_point="intake.question_type", chosen="two_group_comparison", decided_by="model"),
               DecisionRecord(decision_point="stage2.post_hoc", chosen="none", decided_by="code_rule",
                              chart_data=tuple(charts))]
    return log_update(*records)


def _node(update):
    def node(state):
        return update
    return node


def test_writes_each_record_with_meta_and_returns_the_update(tmp_path):
    update = {"plan": "unchanged", **_records()}
    wrapped = with_event_log("classify", _node(update), tmp_path)
    assert wrapped(STATE, CONFIG) is update

    path = events_path(tmp_path, "thread-1")
    entries = EventLog(log_path=path).read_all()
    assert [e["data"] for e in entries] == update["decision_log"]
    assert all(e["record_type"] == "decision" for e in entries)
    assert all(e["meta"] == {"node": "classify", "thread_id": "thread-1", "session_id": "session-abc"}
               for e in entries)
    assert read_decision_log(path) == update["decision_log"]


@pytest.mark.parametrize("update", [{"dataset": "x"}, {"decision_log": []}, None])
def test_no_records_writes_nothing_and_creates_no_directory(tmp_path, update):
    base = tmp_path / "events"
    assert with_event_log("profile", _node(update), base)(STATE, CONFIG) == update
    assert not base.exists()


def test_node_exception_propagates_and_writes_nothing(tmp_path):
    def failing(state):
        raise KeyError("column")
    base = tmp_path / "events"
    with pytest.raises(KeyError, match="column"):
        with_event_log("wrangle_review", failing, base)(STATE, CONFIG)
    assert not base.exists()


def test_disallowed_chart_raises_before_anything_is_written(tmp_path):
    base = tmp_path / "events"
    update = _records({"chart_type": "boxplot_by_group", "data": {"values": [1.0, 2.0]}})
    with pytest.raises(ValueError, match=r"stage2\.post_hoc.*boxplot_by_group"):
        with_event_log("execute", _node(update), base)(STATE, CONFIG)
    assert not base.exists()


def test_post_hoc_chart_is_accepted(tmp_path):
    update = _records({"chart_type": "post_hoc_comparison_chart", "data": {"comparisons": ["A-B"]}})
    with_event_log("execute", _node(update), tmp_path)(STATE, CONFIG)
    stored = read_decision_log(events_path(tmp_path, "thread-1"))
    assert stored[1]["chart_data"][0]["chart_type"] == "post_hoc_comparison_chart"


def test_write_failure_raises_runtime_error_naming_the_node(tmp_path):
    events_path(tmp_path, "thread-1").mkdir(parents=True)  # the log file's path is now a directory
    with pytest.raises(RuntimeError, match="node 'freeze'"):
        with_event_log("freeze", _node(_records()), tmp_path)(STATE, CONFIG)


@pytest.mark.parametrize("config", [{}, {"configurable": {}}, None])
def test_missing_thread_id_raises(tmp_path, config):
    with pytest.raises(ValueError, match="thread_id"):
        with_event_log("classify", _node(_records()), tmp_path)(STATE, config)
    assert list(tmp_path.iterdir()) == []


def test_wrapper_keeps_the_config_parameter_visible():
    def node(state):
        return {}
    assert list(inspect.signature(with_event_log("n", node, "/tmp/unused")).parameters) == ["state", "config"]
