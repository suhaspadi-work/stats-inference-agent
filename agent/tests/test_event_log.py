import pytest
from core.event_log import EventLog
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