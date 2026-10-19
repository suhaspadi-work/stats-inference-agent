import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from core.agent_state import AgentState
from core.classifier import _DraftPlanOut
from core.decision_log import log_update
from core.draft_plan_records import draft_plan_records
from core.model_safe_view import ColumnSummary, ModelSafeView
from core.dataset import LocalDiskStore
from core.ingestion import load_dataset
from core.nodes import make_freeze_node
from core.plan import AnalysisPlan, MethodDecision, AlternativeConsidered, QuestionType, CausalStatus, OutcomeType
from core.selection_records import two_group_selection_record


def make_test_plan():
    plan = AnalysisPlan(
        request_text="Did West spend more than East?", session_id="session-abc",
        question_type=QuestionType.TWO_GROUP_COMPARISON, causal_status=CausalStatus.OBSERVATIONAL,
        outcome_name="total_spend", outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="customer", data_sources=("customers@v1",),
        candidate_predictors=("region",),
    )
    decision = MethodDecision(
        stage="test_selection", chosen_method="welch_t_test",
        rationale="test", alternatives=(AlternativeConsidered(method="mann_whitney_u", rejected_because="test"),),
    )
    return plan.record_method_decision(decision)


# M1.3c: freeze is make_freeze_node(store), and an edit re-profiles the real dataset, so these tests carry a
# small dataset with the same columns as the hand-made profile (region and plan_type with 2 levels each).
_TMP = Path(tempfile.mkdtemp())
STORE = LocalDiskStore(base_dir=_TMP / "store")
_rng = np.random.default_rng(5)
pd.DataFrame({"region": ["East", "West"] * 50, "plan_type": ["basic", "pro"] * 25 + ["pro", "basic"] * 25,
              "total_spend": _rng.normal(50, 10, 100)}).to_csv(_TMP / "freeze.csv", index=False)
DATASET = load_dataset(_TMP / "freeze.csv", session_id="session-abc", name="freeze", store=STORE)


def build_freeze_graph():
    graph = StateGraph(AgentState)
    graph.add_node("freeze", make_freeze_node(STORE))
    graph.add_edge(START, "freeze")
    graph.add_edge("freeze", END)
    return graph.compile(checkpointer=InMemorySaver())


def test_freeze_interrupts_and_waits_for_human():
    app = build_freeze_graph()
    config = {"configurable": {"thread_id": "t1"}}
    result = app.invoke({"plan": make_test_plan(), "request_text": "x", "session_id": "session-abc"}, config)

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["type"] == "plan_approval"
    assert payload["candidate_predictors"] == ["region"]
    assert payload["allowed_decisions"] == ["approve", "edit", "reject"]
    # Critically: the plan must NOT be frozen while waiting -- freeze_node
    # paused before ever calling plan.freeze(), so status is still DRAFT
    assert result["plan"].status.value == "draft"


def test_approve_freezes_the_plan():
    app = build_freeze_graph()
    config = {"configurable": {"thread_id": "t2"}}
    app.invoke({"plan": make_test_plan(), "request_text": "x", "session_id": "session-abc"}, config)

    result = app.invoke(Command(resume={"action": "approve"}), config)
    assert "__interrupt__" not in result
    assert result["plan"].status.value == "frozen"
    # Note: candidate_predictors round-trips through the checkpointer as a
    # list, not a tuple -- JSON has no native tuple type. This is a real,
    # documented behavior of checkpointed state, not a bug in freeze_node.
    assert list(result["plan"].candidate_predictors) == ["region"]


def test_edit_changes_predictors_before_freezing():
    from core.model_safe_view import ModelSafeView, ColumnSummary

    app = build_freeze_graph()
    config = {"configurable": {"thread_id": "t3"}}
    profile = ModelSafeView(
        dataset_handle="test@v1", row_count=100, column_count=3,
        columns=(
            ColumnSummary(name="region", dtype="object", missing_pct=0.0, missing_count=0, distinct_count=2),
            ColumnSummary(name="plan_type", dtype="object", missing_pct=0.0, missing_count=0, distinct_count=2),
            ColumnSummary(name="total_spend", dtype="float64", missing_pct=0.0, missing_count=0, distinct_count=100),
        ),
    )
    app.invoke(
        {"plan": make_test_plan(), "request_text": "x", "session_id": "session-abc", "profile": profile,
         "dataset": DATASET}, config,
    )

    # M1.3c: an accepted edit is not frozen; the plan is refined, re-routed and goes back through select_method
    # (two grouping columns would now be refused by the re-route, so the edit names the one other column)
    result = app.invoke(
        Command(resume={"action": "edit", "candidate_predictors": ["plan_type"]}), config,
    )
    assert result["plan"].status.value == "refined"
    assert list(result["plan"].candidate_predictors) == ["plan_type"]
    assert result["plan"].method_decisions == [] or result["plan"].method_decisions == ()
    assert result["route"]["capability"] == "two_group" and result["reselect_after_edit"] is not None


def test_reject_never_freezes_and_signals_clarification_needed():
    app = build_freeze_graph()
    config = {"configurable": {"thread_id": "t4"}}
    app.invoke({"plan": make_test_plan(), "request_text": "x", "session_id": "session-abc"}, config)

    result = app.invoke(Command(resume={"action": "reject", "reason": "Wrong predictors"}), config)

    assert result["plan"].status.value != "frozen"
    assert "clarification_needed" in result
    assert "rejected" in result["clarification_needed"].lower()
    assert "Wrong predictors" in result["clarification_needed"]

def test_edit_rejects_a_nonexistent_predictor_column():
    """
    Regression test: editing the plan's predictors at freeze to a column
    that doesn't exist in the dataset must fail cleanly, not silently
    proceed and crash later deep inside select_method_node/execute_node.
    """
    from core.model_safe_view import ModelSafeView, ColumnSummary

    app = build_freeze_graph()
    config = {"configurable": {"thread_id": "t5"}}
    profile = ModelSafeView(
        dataset_handle="test@v1", row_count=100, column_count=2,
        columns=(
            ColumnSummary(name="region", dtype="object", missing_pct=0.0, missing_count=0, distinct_count=2),
            ColumnSummary(name="total_spend", dtype="float64", missing_pct=0.0, missing_count=0, distinct_count=100),
        ),
    )
    app.invoke(
        {"plan": make_test_plan(), "request_text": "x", "session_id": "session-abc", "profile": profile,
         "dataset": DATASET}, config,
    )

    # M1.3c: the edit is refused and recorded (no exception); the plan is not frozen
    result = app.invoke(
        Command(resume={"action": "edit", "candidate_predictors": ["nonexistent_column_xyz"]}), config,
    )
    assert [v["rule"] for v in result["plan_approval_refusal"]["refusals"]] == ["unknown_column"]
    assert result["decision_log"][-1]["chosen"] == "refused"
    assert result["plan"].status.value != "frozen"


# --- plan approval decision records (M0.2 step 9) ---------------------------------

def _profile():
    return ModelSafeView(
        dataset_handle="test@v1", row_count=100, column_count=3,
        columns=(
            ColumnSummary(name="region", dtype="object", missing_pct=0.0, missing_count=0, distinct_count=2),
            ColumnSummary(name="plan_type", dtype="object", missing_pct=0.0, missing_count=0, distinct_count=2),
            ColumnSummary(name="total_spend", dtype="float64", missing_pct=0.0, missing_count=0, distinct_count=100),
        ),
    )


def _seed_log(with_selection=True):
    """A starting decision_log built with the real builders: the draft's predictors record and the test selection."""
    fields = _DraftPlanOut(outcome_name="total_spend", outcome_type=OutcomeType.CONTINUOUS,
                           unit_of_analysis="customer", candidate_predictors=["region"], rationale="draft")
    predictors = draft_plan_records(fields, _profile())[2]
    records = [predictors]
    if with_selection:
        selection = two_group_selection_record(MethodDecision(
            stage="test_selection", chosen_method="welch_t_test", rationale="selector", alternatives=(),
            evidence={"shapiro_p_group_a": 0.3, "shapiro_p_group_b": 0.4, "levene_p": 0.01,
                      "n_group_a": 50, "n_group_b": 50, "group_a_label": "East", "group_b_label": "West",
                      "group_a_passes_normality": True, "group_b_passes_normality": True,
                      "variances_equal": False},
        ))
        records.append(selection)
    return log_update(*records)["decision_log"]


def _pause(thread, with_selection=True):
    app = build_freeze_graph()
    config = {"configurable": {"thread_id": thread}}
    seed = _seed_log(with_selection)
    paused = app.invoke({"plan": make_test_plan(), "request_text": "x", "session_id": "session-abc",
                         "profile": _profile(), "decision_log": seed, "dataset": DATASET}, config)  # M1.3c: dataset
    return app, config, seed, paused


def _new_records(result, seed):
    return result["decision_log"][len(seed):]


def test_approve_logs_one_plan_approval_record():
    app, config, seed, _ = _pause("rec-approve")
    result = app.invoke(Command(resume={"action": "approve"}), config)
    new = _new_records(result, seed)
    assert [r["decision_point"] for r in new] == ["planner.plan_approval"]
    assert new[0]["chosen"] == "approve" and new[0]["decided_by"] == "human"
    assert new[0]["evidence"]["selection_record_id"] == seed[1]["id"]
    assert new[0]["evidence"]["method_at_approval"] == result["plan"].primary_method == "welch_t_test"
    assert result["plan"].status.value == "frozen"


def test_edit_logs_predictors_then_approval():
    app, config, seed, _ = _pause("rec-edit")
    # M1.3c: a single grouping column (two would be refused by the re-route)
    result = app.invoke(Command(resume={"action": "edit", "candidate_predictors": ["plan_type"]}), config)
    predictors, approval = _new_records(result, seed)
    assert (predictors["decision_point"], approval["decision_point"]) == ("planner.predictors", "planner.plan_approval")
    assert predictors["supersedes"] == seed[0]["id"]
    assert predictors["decided_by"] == "human" and predictors["overridden_by_human"] is True
    assert predictors["original_choice"] == "region"
    assert predictors["chosen"] == "plan_type"
    assert predictors["evidence"]["predictors_before"] == ["region"]
    assert predictors["evidence"]["predictors_after"] == ["plan_type"]
    assert predictors["evidence"]["selection_record_id"] == seed[1]["id"]
    assert approval["chosen"] == "edit"
    assert approval["evidence"]["predictors_record_id"] == predictors["id"]
    assert approval["original_choice"] == "approve as drafted: predictors [region]"
    assert list(result["plan"].candidate_predictors) == ["plan_type"]
    assert (approval["evidence"]["frozen"], approval["evidence"]["reselects"]) == (False, True)  # M1.3c


def test_reject_logs_one_record_and_does_not_freeze():
    app, config, seed, _ = _pause("rec-reject")
    result = app.invoke(Command(resume={"action": "reject", "reason": "Wrong predictors"}), config)
    new = _new_records(result, seed)
    assert [(r["decision_point"], r["chosen"]) for r in new] == [("planner.plan_approval", "reject")]
    assert new[0]["rationale"] == "Wrong predictors"
    assert new[0]["evidence"]["frozen"] is False
    assert result["plan"].status.value != "frozen"
    assert result["clarification_needed"] == "Plan rejected: Wrong predictors"


def test_unrecognized_action_freezes_by_code_rule():
    app, config, seed, _ = _pause("rec-g36-maybe")
    result = app.invoke(Command(resume={"action": "maybe"}), config)
    (approval,) = _new_records(result, seed)
    assert (approval["chosen"], approval["decided_by"]) == ("approve", "code_rule")
    assert approval["evidence"]["received_action"] == "maybe"
    assert result["plan"].status.value == "frozen"


def test_empty_resume_does_not_resume_the_node():
    """
    Pins LangGraph's behaviour: Command(resume={}) does not resume the paused
    node, so freeze_node never sees an empty decision. The run stays paused,
    nothing is logged and the plan is not frozen. (The builder's handling of
    an empty decision is covered in tests/test_freeze_records.py.)
    """
    app, config, seed, _ = _pause("rec-g36-empty")
    result = app.invoke(Command(resume={}), config)
    assert "__interrupt__" in result
    assert result["decision_log"] == seed
    assert result["plan"].status.value == "draft"
    assert app.get_state(config).next == ("freeze",)


def test_invalid_edit_is_refused_and_recorded():
    # M1.3c: a refused edit is a recorded human decision, not an exception, and it is saved at once
    app, config, seed, _ = _pause("rec-invalid")
    app.invoke(Command(resume={"action": "edit", "candidate_predictors": ["nonexistent_column_xyz"]}), config)
    new = app.get_state(config).values["decision_log"][len(seed):]
    assert [(r["decision_point"], r["chosen"], r["decided_by"]) for r in new] == [
        ("planner.plan_approval", "refused", "human")]


def test_no_earlier_selection_record_gives_none():
    app, config, seed, _ = _pause("rec-no-selection", with_selection=False)
    result = app.invoke(Command(resume={"action": "approve"}), config)
    (approval,) = _new_records(result, seed)
    assert approval["evidence"]["selection_record_id"] is None


def test_record_ids_unique_across_pause_and_resume():
    app, config, seed, paused = _pause("rec-ids")
    assert paused["decision_log"] == seed  # nothing is logged at the pause
    result = app.invoke(Command(resume={"action": "edit", "candidate_predictors": ["plan_type"]}), config)
    ids = [r["id"] for r in result["decision_log"]]
    assert len(ids) == len(set(ids)) == len(seed) + 2


def test_pause_payload_keys_unchanged():
    _, _, _, paused = _pause("rec-payload")
    payload = paused["__interrupt__"][0].value
    assert set(payload) == {
        "type", "question", "request_text", "question_type", "causal_status", "outcome_name",
        "outcome_type", "candidate_predictors", "classification_rationale", "allowed_decisions",
        "refused", "method", "reselected",  # M1.3c: the previous refusal, the method shown, and whether re-selected
        "stage1_report",  # M1.4b: None here, since this state has no stage_reports
        "unaddressed_issues",  # M2.3a: every unaddressed issue of the plan, plain JSON
    }
    assert payload["stage1_report"] is None  # M1.4b
    assert (payload["refused"], payload["method"], payload["reselected"]) == (None, "welch_t_test", False)
    assert payload["type"] == "plan_approval"
    assert payload["question"] == "Review this analysis plan before it is frozen and executed."
