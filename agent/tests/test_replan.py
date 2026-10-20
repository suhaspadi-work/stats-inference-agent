"""
M2.4a: the stage plan recomputed after the Stage 1 loop (core.stage_plan.replan_stage_plan and the replan node). Only
the modeling step of a comparison and the describe step's counts depend on the table; a step counts as changed only
when its state or run flag changes; a planner.stage_replan record (never another planner.stage_plan) is written for a
change, a loop of more than one round, or a finished or round-limit end (G77).
"""
import copy

import numpy as np
import pandas as pd
import pytest

from core import thresholds
from core.dataset import LocalDiskStore
from core.graph import build_agent_graph
from core.ingestion import profile_dataset
from core.nodes import _aggregated_outcome_issues, _imputed_outcome_issues, _question_log
from core.plan import QuestionType
from core.stage_plan import build_stage_plan, plan_steps, replan_stage_plan
from core.stage_plan_gate import apply_gate_decision
from core.stage_reports import current_question_records, round_records
from tests.test_decision_point_guard import _dataset, choose, confident, fake_draft
from tests.test_join_run import _bundle, _state
from tests.test_prompt_canaries import fake_model  # noqa: F401 (fixture)
from tests.test_stage1_loop import _app, _approve_all, _conversion, _dedupe_only, _input, _run, _wranglings
from tests.test_stage1_report import _drive

OFFERED = "a regression could adjust the comparison for other columns; exploratory, so you choose"
NOT_APPLICABLE = "fewer than 3 columns or no numeric column, so no adjusted model is possible"


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _profile(df, tmp_path, store, name):
    return profile_dataset(_dataset(df, tmp_path, store, name), store=store)


def _two(n=20):
    return pd.DataFrame({"region": ["East", "West"] * (n // 2), "total_spend": np.arange(float(n))})


def _four(n=20):
    return _two(n).assign(tier=["a", "b"] * (n // 2), visits=np.arange(n))


def _step(plan, name):
    return next(s for s in plan_steps(plan) if s["step"] == name)


def _plan(qt, profile):
    return build_stage_plan(confident(qt)("q"), profile, [])


# --- the pure function ---------------------------------------------------------------------------------------------

def test_a_comparison_both_ways_and_the_input_is_not_mutated(tmp_path, store):
    small, wide = _profile(_two(), tmp_path, store, "two"), _profile(_four(), tmp_path, store, "four")
    plan = _plan(QuestionType.TWO_GROUP_COMPARISON, small)
    before = copy.deepcopy(plan)
    assert (_step(plan, "modeling")["state"], _step(plan, "modeling")["reason"]) == ("not_applicable", NOT_APPLICABLE)
    new = replan_stage_plan(plan, wide)
    assert plan == before  # pure
    assert (_step(new, "modeling")["state"], _step(new, "modeling")["reason"]) == ("offered", OFFERED)
    assert _step(new, "modeling")["evidence"]["column_count"] == 4
    back = replan_stage_plan(new, small)
    assert (_step(back, "modeling")["state"], _step(back, "modeling")["reason"]) == ("not_applicable", NOT_APPLICABLE)
    for name in ("clean", "inference"):
        assert _step(new, name) == _step(plan, name)
    assert {k: v for k, v in new.items() if k != "stages"} == {k: v for k, v in plan.items() if k != "stages"}
    assert _step(new, "describe")["evidence"] == {"row_count": 20, "column_count": 4}  # the counts refresh


def test_driver_and_descriptive_keep_modeling(tmp_path, store):
    small, wide = _profile(_two(), tmp_path, store, "two"), _profile(_four(), tmp_path, store, "four")
    for qt in (QuestionType.DRIVER_ANALYSIS, QuestionType.DESCRIPTIVE):
        plan = _plan(qt, wide)
        new = replan_stage_plan(plan, small)
        assert _step(new, "modeling") == _step(plan, "modeling"), qt
        assert replan_stage_plan(plan, wide) == plan, qt  # the same table: untouched
    assert _step(_plan(QuestionType.DRIVER_ANALYSIS, wide), "modeling")["state"] == "planned"


def test_the_users_run_flags_are_kept(tmp_path, store):
    small, wide = _profile(_two(), tmp_path, store, "two"), _profile(_four(), tmp_path, store, "four")
    plan = apply_gate_decision(_plan(QuestionType.TWO_GROUP_COMPARISON, small), {"describe": False},
                               "approved_with_changes")
    new = replan_stage_plan(plan, wide)
    assert _step(new, "describe")["run"] is False and new["approval"] == plan["approval"]


# --- the node --------------------------------------------------------------------------------------------------------

def _replans(values):
    return [e for e in values["decision_log"] if e["decision_point"] == "planner.stage_replan"]


def test_a_clean_question_runs_no_replan(fake_model, tmp_path, store):
    df = pd.DataFrame({"region": ["East", "West"] * 20, "total_spend": np.arange(40.0), "channel": ["a", "b"] * 20})
    values, _, nodes = _drive(_app(store), _input(_dataset(df, tmp_path, store, "clean")), "clean")
    assert "replan" not in nodes and _replans(values) == []
    stage1 = next(e for e in values["stage_reports"] if e["stage"] == "stage1")
    assert "Rounds" not in [s["heading"] for s in stage1["sections"]]


def test_a_dedupe_refreshes_describe_silently(fake_model, tmp_path, store):
    values, _, nodes = _drive(_app(store), _input(_dedupe_only(tmp_path, store)), "dedupe")
    assert "replan" in nodes and _replans(values) == []
    rows = len(store.read(values["dataset"].storage_key))
    assert rows == 60 and _step(values["stage_plan"], "describe")["evidence"]["row_count"] == rows


def _join_tables(tmp_path, store):
    rng = np.random.default_rng(14)
    orders = pd.DataFrame({"customer_id": rng.integers(1, 41, 120), "total_spend": rng.normal(50, 5, 120)})
    customers = pd.DataFrame({"customer_id": range(1, 41), "region": ["East", "West"] * 20,
                              "tier": ["basic", "basic", "plus", "plus"] * 10, "age": rng.normal(40, 8, 40).round(1)})
    return orders, customers, _bundle(tmp_path, store, {"orders": orders, "customers": customers})


def test_a_join_offers_modeling_and_reaches_the_stage_gate(fake_model, tmp_path, store):
    orders, customers, tables = _join_tables(tmp_path, store)
    app = _app(store)
    config = {"configurable": {"thread_id": "join"}}
    values, pauses, nodes = _drive(app, _state(tables), "join", stage_gate={"action": "run"})
    (record,) = _replans(values)
    joined = orders.merge(customers, on="customer_id", how="left")
    assert (record["chosen"], record["decided_by"]) == ("changed", "code_rule")
    assert record["evidence"] == {
        "rounds": 1, "ended_by": "no_new_issues",
        "changed_steps": [{"step": "modeling", "before_state": "not_applicable", "after_state": "offered",
                           "reason": OFFERED}],
        "column_count": len(joined.columns),
        "numeric_column_count": len(joined.select_dtypes("number").columns)}
    assert _step(values["stage_plan"], "modeling")["state"] == "offered"
    assert "stage_gate" in [p["type"] for p in pauses] and len(values["rounds"]) == 2  # G77: the exploratory round
    assert any(e["stage"] == "analysis" and e["round"] == 2 for e in values["stage_reports"])
    assert [e["decision_point"] for e in values["decision_log"]].count("planner.stage_plan") == 1
    # the plan before replan (in the checkpoint history) was not mutated
    before = next(s for s in app.get_state_history(config) if s.next == ("replan",))
    assert _step(before.values["stage_plan"], "modeling")["state"] == "not_applicable"


@pytest.mark.parametrize("end", ["finished", "round_limit"])
def test_a_one_round_finished_or_limited_end_writes_a_record(end, fake_model, tmp_path, store, monkeypatch):
    _, ds = _conversion(tmp_path, store)
    if end == "round_limit":
        monkeypatch.setattr(thresholds, "MAX_STAGE1_ROUNDS", 1)
        values, pauses, _ = _run(_app(store), _input(ds), end)
    else:
        values, pauses, _ = _run(_app(store), _input(ds), end,
                                 wrangling_approval=lambda p: _approve_all(p, finish=True))
    (record,) = _replans(values)
    assert len(_wranglings(pauses)) == 1
    assert (record["chosen"], record["evidence"]["rounds"], record["evidence"]["ended_by"],
            record["evidence"]["changed_steps"]) == ("unchanged", 1, end, [])


def test_the_question_windows_ignore_the_replan_record(fake_model, tmp_path, store):
    _, ds = _conversion(tmp_path, store)
    values, _, _ = _run(_app(store), _input(ds), "windows")
    log = values["decision_log"]
    (record,) = _replans(values)
    without = [e for e in log if e is not record]

    def ids(records):
        return [e["id"] for e in records if e["id"] != record["id"]]
    assert ids(current_question_records(log)) == ids(current_question_records(without))
    assert ids(round_records(log)) == ids(round_records(without))
    assert ids(_question_log({"decision_log": log})) == ids(_question_log({"decision_log": without}))
    for helper in (_imputed_outcome_issues, _aggregated_outcome_issues):
        assert helper({"decision_log": log}, "total_spend") == helper({"decision_log": without}, "total_spend")


def test_unsupported_and_descriptive_questions(fake_model, tmp_path, store):
    df = pd.DataFrame({"region": ["East", "West"] * 20, "total_spend": np.arange(40.0), "channel": ["a", "b"] * 20})
    def app(qt):
        return build_agent_graph(store, classify_fn=confident(qt), draft_fn=fake_draft(["region", "channel"]),
                                 propose_fn=choose({}))
    _, _, nodes = _drive(app(QuestionType.ASSOCIATION), _input(_dataset(df, tmp_path, store, "assoc")), "assoc")
    assert "replan" not in nodes
    _, _, nodes = _drive(app(QuestionType.DESCRIPTIVE), _input(_dataset(df, tmp_path, store, "desc")), "desc")
    assert "replan" not in nodes  # a clean descriptive question
    # a descriptive question with something to clean passes replan, which writes nothing (modeling never changes)
    values, _, nodes = _drive(app(QuestionType.DESCRIPTIVE), _input(_dedupe_only(tmp_path, store)), "desc-dirty")
    assert "replan" in nodes and _replans(values) == []
