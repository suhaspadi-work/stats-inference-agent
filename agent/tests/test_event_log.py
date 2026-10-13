import json
import math

import pytest
import core.event_log as event_log_module
from core.event_log import EventLog, events_path, read_decision_log
from core.dataset import Dataset
from core.operation import Operation, OperationImpact, ApprovalStatus
from core.plan import AnalysisPlan, QuestionType, CausalStatus, OutcomeType, MethodDecision


@pytest.fixture
def log(tmp_path):
    return EventLog(log_path=tmp_path / "session.jsonl")


def test_append_and_read_dataset(log):
    d = Dataset(
        session_id="session-abc", name="customers", version=1,
        storage_key="session-abc/customers_v1.csv",
        parent_version=None, created_from_operation=None,
    )
    log.append("dataset", d)

    entries = log.read_by_type("dataset")
    assert len(entries) == 1
    assert entries[0]["name"] == "customers"  # sanity: it's the dict form, and the field survived
    assert entries[0]["name"] == "customers"
    assert entries[0]["version"] == 1


def test_append_and_read_operation_preserves_enum_as_value(log):
    op = Operation(
        op_type="dedupe", session_id="session-abc",
        input_handles=("customers@v1",), params={"subset": ["customer_id"]},
        approval_status=ApprovalStatus.PENDING,
    )
    log.append("operation", op)

    entries = log.read_by_type("operation")
    assert len(entries) == 1
    # Enums must serialize to their plain string value, not a Python repr
    assert entries[0]["approval_status"] == "pending"


def test_append_and_read_plan_preserves_nested_method_decisions(log):
    decision = MethodDecision(
        stage="test_selection", chosen_method="welch_t_test",
        rationale="Roughly equal variances observed", alternatives=(),
        evidence={"shapiro_p": 0.34},
    )
    plan = AnalysisPlan(
        request_text="Did West spend more than East?", session_id="session-abc",
        question_type=QuestionType.TWO_GROUP_COMPARISON,
        causal_status=CausalStatus.OBSERVATIONAL,
        outcome_name="total_spend", outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="customer", data_sources=("customers@v2",),
    ).record_method_decision(decision)

    log.append("plan", plan)

    entries = log.read_by_type("plan")
    assert len(entries) == 1
    # The nested MethodDecision must survive as a real nested structure, not lost
    assert len(entries[0]["method_decisions"]) == 1
    assert entries[0]["method_decisions"][0]["chosen_method"] == "welch_t_test"
    assert entries[0]["question_type"] == "two_group_comparison"


def test_multiple_record_types_coexist_in_one_log(log):
    d = Dataset(
        session_id="session-abc", name="customers", version=1,
        storage_key="session-abc/customers_v1.csv",
        parent_version=None, created_from_operation=None,
    )
    op = Operation(
        op_type="dedupe", session_id="session-abc",
        input_handles=("customers@v1",), params={},
        approval_status=ApprovalStatus.PENDING,
    )
    log.append("dataset", d)
    log.append("operation", op)

    all_entries = log.read_all()
    assert len(all_entries) == 2
    assert [e["record_type"] for e in all_entries] == ["dataset", "operation"]

    # And filtering by type still isolates correctly
    assert len(log.read_by_type("dataset")) == 1
    assert len(log.read_by_type("operation")) == 1


def test_log_persists_across_separate_instances(tmp_path):
    """The whole point: a new EventLog instance pointed at the same file sees prior entries."""
    log_path = tmp_path / "session.jsonl"

    log1 = EventLog(log_path=log_path)
    d = Dataset(
        session_id="session-abc", name="customers", version=1,
        storage_key="session-abc/customers_v1.csv",
        parent_version=None, created_from_operation=None,
    )
    log1.append("dataset", d)

    # Simulate a new process/session reopening the same log
    log2 = EventLog(log_path=log_path)
    entries = log2.read_by_type("dataset")
    assert len(entries) == 1
    assert entries[0]["name"] == "customers"


# --- meta, non-finite floats, decision logs and paths (M0.2 step 13) -------------------

def _strict_lines(path):
    def reject(constant):
        raise ValueError(f"non-strict JSON constant {constant}")
    return [json.loads(line, parse_constant=reject) for line in path.read_text().splitlines() if line.strip()]


def test_meta_is_stored_only_when_given(log):
    log.append("decision", {"id": "a"})
    log.append("decision", {"id": "b"}, meta={"node": "classify", "thread_id": "t1"})
    raw = _strict_lines(log.log_path)
    assert raw[0] == {"record_type": "decision", "data": {"id": "a"}}
    assert raw[1] == {"record_type": "decision", "data": {"id": "b"}, "meta": {"node": "classify", "thread_id": "t1"}}


def test_non_finite_floats_are_tagged_strict_and_restored(tmp_path):
    path = tmp_path / "events.jsonl"
    EventLog(log_path=path).append("decision", {
        "ci": [float("nan"), 1.5, None], "bounds": {"hi": float("inf"), "lo": float("-inf")},
        "nested": [{"x": [float("nan")]}],
    })
    (raw,) = _strict_lines(path)
    assert raw["data"]["ci"] == [{"$float": "nan"}, 1.5, None]
    assert raw["data"]["bounds"] == {"hi": {"$float": "inf"}, "lo": {"$float": "-inf"}}
    assert raw["data"]["nested"] == [{"x": [{"$float": "nan"}]}]

    (data,) = EventLog(log_path=path).read_by_type("decision")
    assert math.isnan(data["ci"][0]) and data["ci"][1] == 1.5 and data["ci"][2] is None
    assert math.isinf(data["bounds"]["hi"]) and data["bounds"]["hi"] > 0
    assert math.isinf(data["bounds"]["lo"]) and data["bounds"]["lo"] < 0
    assert math.isnan(data["nested"][0]["x"][0])
    (entry,) = EventLog(log_path=path).read_all()
    assert math.isnan(entry["data"]["ci"][0])


def test_a_non_finite_float_that_bypasses_tagging_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(event_log_module, "_serialize", lambda obj: obj)
    with pytest.raises(ValueError):
        EventLog(log_path=tmp_path / "events.jsonl").append("decision", {"x": float("nan")})


def test_read_decision_log(tmp_path):
    path = tmp_path / "events.jsonl"
    log = EventLog(log_path=path)
    log.append("decision", {"id": "d1"}, meta={"node": "a"})
    log.append("dataset", {"name": "customers"})
    log.append("decision", {"id": "d2", "p": float("nan")})
    decisions = read_decision_log(path)
    assert [d["id"] for d in decisions] == ["d1", "d2"]
    assert math.isnan(decisions[1]["p"])
    assert read_decision_log(tmp_path / "missing" / "events.jsonl") == []
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("thread_id, folder", [
    ("../evil", ".._evil"), ("a/b", "a_b"), (".", "_"), ("..", "_"), ("run 1", "run_1"),
    ("café", "caf_"), ("ok-1_A.b", "ok-1_A.b"),
])
def test_events_path_sanitises_and_stays_inside_base(tmp_path, thread_id, folder):
    base = tmp_path / "events"
    path = events_path(base, thread_id)
    assert path == base / folder / "events.jsonl"
    assert path.resolve().is_relative_to(base.resolve())
    assert not base.exists()


@pytest.mark.parametrize("thread_id", ["", None, 7])
def test_events_path_rejects_bad_thread_ids(tmp_path, thread_id):
    with pytest.raises(ValueError):
        events_path(tmp_path, thread_id)
