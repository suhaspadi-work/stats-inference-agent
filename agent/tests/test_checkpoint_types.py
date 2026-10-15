"""
M0.4 (G4): the checkpoint serializer's allowlist of state types.

With an explicit allowed_msgpack_modules, langgraph 1.2.12 BLOCKS an unlisted type silently: it comes
back as its raw payload (a dict or a plain value) with a "Blocked deserialization" log record and no
exception. So these tests check that state objects come back as real instances, not only that no
warning appears.
"""
import importlib
import logging
import typing
from dataclasses import asdict, fields, is_dataclass
from enum import Enum

import numpy as np
import pandas as pd
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde import jsonplus
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

from core.agent_state import AgentState
from core.checkpoint_types import ALLOWED_MSGPACK_TYPES, make_checkpointer, make_serde, state_types
from core.dataset import Dataset, LocalDiskStore
from core.detection import CandidateIssue
from core.graph import build_agent_graph
from core.model_safe_view import ColumnSummary, ModelSafeView
from core.operation import (
    AlternativeApproachConsidered, ApprovalStatus, Operation, OperationImpact, WranglingRationale,
)
from core.plan import (
    AlternativeConsidered, AnalysisPlan, CausalStatus, ClassificationDecision, MethodDecision, OutcomeType,
    QuestionType, UnaddressedIssue,
)
from tests.test_decision_point_guard import (
    _dataset, _two_group_df, _with_duplicate, confident, fake_classify, fake_draft, fake_propose, must_not_propose,
)

SERDE_LOGGER = "langgraph.checkpoint.serde.jsonplus"
BAD_MESSAGES = ("Blocked deserialization", "unregistered type")
STOP = ["execute", "eda"]

EXPECTED = [
    ("core.dataset", "Dataset"),
    ("core.detection", "CandidateIssue"),
    ("core.model_safe_view", "ColumnSummary"), ("core.model_safe_view", "ModelSafeView"),
    ("core.operation", "AlternativeApproachConsidered"), ("core.operation", "ApprovalStatus"),
    ("core.operation", "Operation"), ("core.operation", "OperationImpact"), ("core.operation", "WranglingRationale"),
    ("core.plan", "AlternativeConsidered"), ("core.plan", "AnalysisPlan"), ("core.plan", "CausalStatus"),
    ("core.plan", "ClassificationDecision"), ("core.plan", "Deviation"), ("core.plan", "MethodDecision"),
    ("core.plan", "OutcomeType"), ("core.plan", "PlanStatus"), ("core.plan", "QuestionType"),
    ("core.plan", "UnaddressedIssue"),
]


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())

    def bad(self):
        return [m for m in self.messages if any(b in m for b in BAD_MESSAGES)]


@pytest.fixture
def serde_log(monkeypatch):
    """Capture the serializer's warnings. LangGraph warns once per type per process, so reset those sets."""
    monkeypatch.setattr(jsonplus, "_warned_blocked_types", set())
    monkeypatch.setattr(jsonplus, "_warned_unregistered_types", set())
    handler = _Capture()
    logger = logging.getLogger(SERDE_LOGGER)
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    yield handler
    logger.removeHandler(handler)
    logger.setLevel(previous)


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _serde_without_dataset():
    return JsonPlusSerializer(
        allowed_msgpack_modules=[t for t in ALLOWED_MSGPACK_TYPES if t != ("core.dataset", "Dataset")])


# --- 1. the list ---------------------------------------------------------------------

def test_state_types_are_exactly_the_expected_19():
    assert state_types() == EXPECTED
    assert len(EXPECTED) == 19 and len(set(EXPECTED)) == 19 and EXPECTED == sorted(EXPECTED)
    assert ALLOWED_MSGPACK_TYPES == tuple(state_types())


@pytest.mark.parametrize("module_name, class_name", EXPECTED)
def test_each_listed_type_imports_and_is_a_dataclass_or_enum(module_name, class_name):
    cls = getattr(importlib.import_module(module_name), class_name)
    assert is_dataclass(cls) or issubclass(cls, Enum)


# --- 2. static coverage of AgentState ----------------------------------------------------

def _classes_in(hint, seen):
    """Every class named in a type hint, unwrapping Optional, Annotated and generics, and dataclass fields."""
    origin = typing.get_origin(hint)
    if origin is typing.Annotated:
        _classes_in(typing.get_args(hint)[0], seen)
        return
    if origin is not None:
        for arg in typing.get_args(hint):
            if arg is not Ellipsis:
                _classes_in(arg, seen)
        return
    if isinstance(hint, type) and hint not in seen:
        seen.add(hint)
        if is_dataclass(hint):
            for name, field_hint in typing.get_type_hints(hint, include_extras=True).items():
                if name in {f.name for f in fields(hint)}:
                    _classes_in(field_hint, seen)


def test_every_core_type_reachable_from_agent_state_is_allowed():
    seen = set()
    for hint in typing.get_type_hints(AgentState, include_extras=True).values():
        _classes_in(hint, seen)
    core_types = {(c.__module__, c.__name__) for c in seen
                  if c.__module__.startswith("core.") and (is_dataclass(c) or issubclass(c, Enum))}
    assert ("core.plan", "AnalysisPlan") in core_types and ("core.plan", "MethodDecision") in core_types
    missing = sorted(core_types - set(ALLOWED_MSGPACK_TYPES))
    assert not missing, f"state types missing from ALLOWED_MSGPACK_TYPES: {missing}"


# --- 3. round trip ---------------------------------------------------------------------

def _sample_objects():
    plan = AnalysisPlan(
        request_text="x", session_id="s", question_type=QuestionType.TWO_GROUP_COMPARISON,
        causal_status=CausalStatus.OBSERVATIONAL, outcome_name="total_spend", outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="customer", data_sources=("d@v1",), candidate_predictors=("region",),
    ).record_method_decision(MethodDecision(
        stage="test_selection", chosen_method="welch_t_test", rationale="r",
        alternatives=(AlternativeConsidered(method="mann_whitney_u", rejected_because="r"),), evidence={"p": 0.2},
    )).record_unaddressed_issue(UnaddressedIssue(
        issue_type="duplicate_rows", description="2 duplicates", proposed_operation="dedupe",
        rejection_reason="real repeat orders", affected_columns=("id",),
    ))
    operation = Operation(
        op_type="dedupe", session_id="s", input_handles=("d@v1",), params={"subset": None},
        approval_status=ApprovalStatus.APPROVED,
        rationale=WranglingRationale(rationale="r", alternatives=(
            AlternativeApproachConsidered(approach="keep", rejected_because="r"),)),
    ).with_result(OperationImpact(rows_before=4, rows_after=3, columns_before=2, columns_after=2), "d@v2")
    return [
        plan,
        Dataset(session_id="s", name="d", version=1, storage_key="k", parent_version=None, created_from_operation=None),
        ModelSafeView(dataset_handle="d@v1", row_count=4, column_count=1, columns=(
            ColumnSummary(name="region", dtype="object", missing_count=0, missing_pct=0.0, distinct_count=2),)),
        operation,
        CandidateIssue(issue_type="duplicate_rows", description="d", affected_columns=("id",),
                       evidence={"duplicate_row_count": 1}),
        QuestionType.MULTI_GROUP_COMPARISON,
        ApprovalStatus.FAILED,
    ]


def _lists(value):
    """Tuples become lists in a checkpoint round trip (a known LangGraph behaviour), so compare them as lists."""
    if isinstance(value, dict):
        return {k: _lists(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_lists(v) for v in value]
    return value


@pytest.mark.parametrize("obj", _sample_objects(), ids=lambda o: type(o).__name__)
def test_round_trip_returns_real_instances(obj, serde_log):
    serde = make_serde()
    result = serde.loads_typed(serde.dumps_typed(obj))
    assert type(result) is type(obj)
    if isinstance(obj, AnalysisPlan):
        # checkpointed tuples come back as lists, so compare the parts that matter
        assert type(result.method_decisions[0]) is MethodDecision
        assert type(result.method_decisions[0].alternatives[0]) is AlternativeConsidered
        assert type(result.unaddressed_issues[0]) is UnaddressedIssue
        assert result.question_type is QuestionType.TWO_GROUP_COMPARISON
        assert result.id == obj.id
    elif isinstance(obj, Operation):
        assert type(result.impact) is OperationImpact
        assert type(result.rationale) is WranglingRationale
        assert result.approval_status is ApprovalStatus.APPROVED
        assert result.id == obj.id
    elif isinstance(obj, ModelSafeView):
        assert type(result.columns[0]) is ColumnSummary
        assert _lists(asdict(result)) == _lists(asdict(obj))
    elif is_dataclass(obj):
        assert _lists(asdict(result)) == _lists(asdict(obj))
    else:
        assert result is obj  # enum members come back as the same member
    assert serde_log.bad() == []


# --- 4. negative: a missing type is caught ----------------------------------------------

def test_a_missing_type_is_blocked_and_logged(serde_log):
    serde = _serde_without_dataset()
    ds = _sample_objects()[1]
    result = serde.loads_typed(serde.dumps_typed(ds))
    assert not isinstance(result, Dataset)
    assert isinstance(result, dict) and result["storage_key"] == "k"
    assert any("Blocked deserialization of core.dataset.Dataset" in m for m in serde_log.messages)


# --- 5. scenarios through the real graph ---------------------------------------------------

def _assert_real_types(values, where):
    assert type(values["dataset"]) is Dataset, where
    if "profile" in values:
        assert type(values["profile"]) is ModelSafeView, where
    if values.get("classification") is not None:
        assert type(values["classification"]) is ClassificationDecision, where
    pending = values.get("pending_wrangling")
    if pending:
        assert all(type(o) is Operation for o in pending["operations"]), where
        assert all(type(i) is CandidateIssue for i in pending["issues"]), where
    for issue in values.get("pending_unaddressed_issues") or ():
        assert type(issue) is UnaddressedIssue, where
    plan = values.get("plan")
    if plan is not None:
        assert type(plan) is AnalysisPlan, where
        assert type(plan.question_type) is QuestionType, where
        assert all(type(d) is MethodDecision for d in plan.method_decisions), where
        assert all(type(i) is UnaddressedIssue for i in plan.unaddressed_issues), where


def _drive(app, state, thread, resumes):
    config = {"configurable": {"thread_id": thread}}
    queue, pauses = list(resumes), []
    result = app.invoke(state, config, interrupt_after=STOP)
    while True:
        values = app.get_state(config).values
        _assert_real_types(values, f"{thread} after {len(pauses)} pause(s)")
        if "__interrupt__" not in result:
            break
        pauses.append(result["__interrupt__"][0].value["type"])
        result = app.invoke(Command(resume=queue.pop(0)), config, interrupt_after=STOP)
    assert queue == []
    snapshot = app.get_state(config)
    return snapshot.values, pauses, snapshot.next


def _messy(rng):
    df = _two_group_df(rng)
    df["notes"] = [None if i % 6 == 0 else "ok" for i in range(len(df))]
    return _with_duplicate(df)


def _start(ds, **extra):
    return {"request_text": "x", "session_id": "session-abc", "dataset": ds, **extra}


def test_scenario_runs_keep_real_types(tmp_path, store, serde_log):
    rng = np.random.default_rng(7)
    results = {}

    # (a) clarification, both cleaning operations approved, plan approved
    app = build_agent_graph(store, classify_fn=fake_classify(QuestionType.TWO_GROUP_COMPARISON),
                            draft_fn=fake_draft(["region"]), propose_fn=fake_propose)
    results["a"] = _drive(app, _start(_dataset(_messy(rng), tmp_path, store, "a")), "a", [
        "the regions", {"action": "approve"},  # M1.3a: the stage-plan gate after the clarification
        {"decisions": [{"index": 0, "action": "approve"}, {"index": 1, "action": "approve"}]}, {"action": "approve"}])

    # (b) the cleaning filter rejected, so an UnaddressedIssue reaches the plan
    app = build_agent_graph(store, classify_fn=confident(QuestionType.TWO_GROUP_COMPARISON),
                            draft_fn=fake_draft(["region"]), propose_fn=fake_propose)
    results["b"] = _drive(app, _start(_dataset(_messy(rng), tmp_path, store, "b")), "b", [
        {"action": "approve"},  # M1.3a: the stage-plan gate
        {"decisions": [{"index": 0, "action": "approve"}, {"index": 1, "action": "reject", "reason": "keep them"}]},
        {"action": "approve"}])

    # (c) multi-group
    mg = pd.DataFrame({"region": [g for g in ("East", "North", "West") for _ in range(60)],
                       "total_spend": np.concatenate([rng.normal(m, 5, 60) for m in (50, 55, 65)])})
    app = build_agent_graph(store, classify_fn=confident(QuestionType.MULTI_GROUP_COMPARISON),
                            draft_fn=fake_draft(["region"]), propose_fn=must_not_propose)
    results["c"] = _drive(app, _start(_dataset(mg, tmp_path, store, "c")), "c", [{"action": "approve"}] * 2)  # M1.3a: gate first

    # (d) regression with a text predictor
    n = 200
    tier = rng.choice(["basic", "plus", "pro"], n)
    tenure = rng.normal(24, 8, n)
    reg = pd.DataFrame({"tenure": tenure, "tier": tier,
                        "total_spend": 20 + 2 * tenure + np.array([{"basic": 0, "plus": 5, "pro": 12}[t] for t in tier])
                        + rng.normal(0, 5, n)})
    app = build_agent_graph(store, classify_fn=confident(QuestionType.DRIVER_ANALYSIS),
                            draft_fn=fake_draft(["tenure", "tier"]), propose_fn=must_not_propose)
    results["d"] = _drive(app, _start(_dataset(reg, tmp_path, store, "d")), "d", [{"action": "approve"}] * 2)  # M1.3a: gate first

    # (e) unsupported association
    app = build_agent_graph(store, classify_fn=confident(QuestionType.ASSOCIATION),
                            draft_fn=fake_draft(["region", "channel"]), propose_fn=must_not_propose)
    results["e"] = _drive(app, _start(_dataset(_two_group_df(rng), tmp_path, store, "e")), "e", [])

    # (f) descriptive
    app = build_agent_graph(store, classify_fn=confident(QuestionType.DESCRIPTIVE), draft_fn=fake_draft([]),
                            propose_fn=must_not_propose)
    results["f"] = _drive(app, _start(_dataset(_two_group_df(rng), tmp_path, store, "f")), "f", [])

    pauses = {k: v[1] for k, v in results.items()}
    gate = "stage_plan_approval"  # M1.3a
    assert pauses == {"a": ["clarification_needed", gate, "wrangling_approval", "plan_approval"],
                      "b": [gate, "wrangling_approval", "plan_approval"], "c": [gate, "plan_approval"],
                      "d": [gate, "plan_approval"], "e": [], "f": []}
    assert {k: v[2] for k, v in results.items()} == {
        "a": ("report",), "b": ("report",), "c": ("report",), "d": ("report",), "e": (), "f": ("report",)}
    for k in ("a", "b", "c", "d"):
        assert results[k][0]["plan"].status.value == "executed", k
    assert results["f"][0]["plan"].status.value == "executed"
    assert results["c"][0]["test_result"]["result_type"] == "multi_group"
    assert results["d"][0]["test_result"]["result_type"] == "regression"
    assert results["e"][0]["stage_plan"]["status"] == "unsupported"  # M1.1: association ends at plan_stages, so no route
    b_issues = results["b"][0]["plan"].unaddressed_issues
    assert len(b_issues) == 1 and type(b_issues[0]) is UnaddressedIssue
    assert serde_log.bad() == []


# --- 6. negative scenario: what a missing type does to a real run ------------------------------

def test_a_missing_type_breaks_the_first_resume(tmp_path, store, serde_log):
    rng = np.random.default_rng(7)
    app = build_agent_graph(store, classify_fn=fake_classify(QuestionType.TWO_GROUP_COMPARISON),
                            draft_fn=fake_draft(["region"]), propose_fn=fake_propose,
                            checkpointer=InMemorySaver(serde=_serde_without_dataset()))
    config = {"configurable": {"thread_id": "neg"}}
    paused = app.invoke(_start(_dataset(_messy(rng), tmp_path, store, "neg")), config, interrupt_after=STOP)
    assert "__interrupt__" in paused
    assert type(app.get_state(config).values["dataset"]) is dict  # already a raw payload at the pause
    with pytest.raises(AttributeError, match="'dict' object has no attribute 'storage_key'"):
        app.invoke(Command(resume="the regions"), config, interrupt_after=STOP)
    assert any("Blocked deserialization of core.dataset.Dataset" in m for m in serde_log.messages)


# --- 7. the checkpointer argument ----------------------------------------------------------

def test_build_agent_graph_uses_the_given_checkpointer(store):
    custom = make_checkpointer()
    assert build_agent_graph(store, checkpointer=custom).checkpointer is custom
    default = build_agent_graph(store).checkpointer
    assert isinstance(default, InMemorySaver)
    assert default.serde is not None
