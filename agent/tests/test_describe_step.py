"""
M1.2: the stage plan starts to act. The describe node is the last step of stage 1: it runs after cleaning
and before the analysis plan is drafted, when the stage plan's describe step says run and the question is
not descriptive (a descriptive question is described by eda). Every graph node carries a stage tag.
"""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from core.capabilities import DESCRIPTIVE
from core.dataset import LocalDiskStore
from core.decision_points import DECISION_NODES, NODE_STAGES, NON_CHOOSING_NODES, STAGES
from core.describe_records import DESCRIBE_EXECUTION_POINT, describe_execution_record
from core.graph import build_agent_graph
from core.ingestion import profile_dataset
from core.model_safe_view import assert_model_safe
from core.nodes import make_describe_node, route_after_cleaning, route_after_wrangle_propose
from core.plan import QuestionType
from core.stage_plan import build_stage_plan, plan_steps
from tests.test_decision_point_guard import (
    _dataset, _drive, _two_group_df, _with_duplicate, confident, fake_draft, fake_propose, must_not_propose,
)
from tests.test_prompt_canaries import (
    NAME_CANARIES, _base_df, _clean, _drive as _drive_to_end, canary_hits, fake_model,  # noqa: F401 (fixture)
)
from tests.test_stage_plan import FOUR, classification

EXPECTED_STAGES = {
    "classify": "intake", "clarify": "intake", "reclassify": "intake", "profile": "intake", "plan_stages": "intake",
    "stage_plan_gate": "intake",  # M1.3a
    "stage_gate": "analysis", "start_round": "analysis",  # M1.3b
    "wrangle_propose": "stage1", "wrangle_review": "stage1", "describe": "stage1", "eda": "stage1",
    "stage1_report": "stage1",  # M1.4b
    "draft_plan": "analysis", "select_method": "analysis", "freeze": "analysis", "execute": "analysis",
    "report": "report",
    "summary": "report",  # M1.4b step 2b
    "diagnose": "stage1", "rediagnose": "stage1",  # M2.2: the quality snapshots
}
RUNNABLE = (QuestionType.TWO_GROUP_COMPARISON, QuestionType.AB_TEST, QuestionType.MULTI_GROUP_COMPARISON,
            QuestionType.DRIVER_ANALYSIS)


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _has_key(value, key):
    if isinstance(value, dict):
        return key in value or any(_has_key(v, key) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_key(v, key) for v in value)
    return False


def _describe_step(plan):
    return next(s for s in plan_steps(plan) if s["step"] == "describe")


def _points(values):
    return [e["decision_point"] for e in values["decision_log"]]


# --- stage tags ------------------------------------------------------------------------------------

def test_every_graph_node_has_a_stage(tmp_path):
    nodes = set(build_agent_graph(LocalDiskStore(base_dir=tmp_path)).get_graph().nodes) - {"__start__", "__end__"}
    assert set(NODE_STAGES) == nodes
    assert set(NODE_STAGES.values()) <= set(STAGES)
    assert NODE_STAGES == EXPECTED_STAGES
    assert set(DECISION_NODES) | set(NON_CHOOSING_NODES) <= set(NODE_STAGES)


# --- the record ------------------------------------------------------------------------------------

def _canary_result(tmp_path, store):
    return DESCRIPTIVE.compute(_dataset(_clean(_base_df()), tmp_path, store, "canary"), store)


def test_describe_execution_record(tmp_path, store):
    result = _canary_result(tmp_path, store)
    step = _describe_step(build_stage_plan(classification(QuestionType.TWO_GROUP_COMPARISON), FOUR, []))
    r = describe_execution_record(result, step)
    assert (r.decision_point, r.chosen, r.decided_by) == (DESCRIBE_EXECUTION_POINT, "ran", "code_rule")
    assert r.rationale == "the finished dataset is described as the last step of stage 1"
    assert r.evidence["row_count"] == 360 and r.evidence["column_count"] == 5
    assert r.evidence["chart_count"] == len(result["chart_data"]) > 0
    assert r.evidence["withheld_text_columns"] == ["customer_name"]
    assert (r.evidence["step_state"], r.evidence["exploratory"]) == ("offered", False)
    ran, skipped = r.candidates
    assert (ran.option, ran.eligible, ran.reason) == (
        "ran", True, "chosen: the stage plan's describe step says run (offered)")
    assert (skipped.option, skipped.eligible, skipped.reason) == (
        "skipped", True, "not chosen: describe is part of stage 1 and runs by default")
    d = r.to_dict()
    json.dumps(d)
    assert_model_safe(d, "record")
    assert d["chart_data"] == [] or d["chart_data"] == ()
    assert not _has_key(d["evidence"], "chart_data") and not _has_key(d["candidates"], "chart_data")


def test_skipped_is_not_eligible_for_a_planned_step(tmp_path, store):
    step = _describe_step(build_stage_plan(classification(QuestionType.DESCRIPTIVE), FOUR, []))
    assert step["state"] == "planned"
    r = describe_execution_record(_canary_result(tmp_path, store), step)
    assert r.candidates[1].eligible is False
    assert r.candidates[0].reason == "chosen: the stage plan's describe step says run (planned)"


# --- the node --------------------------------------------------------------------------------------

def test_describe_node(tmp_path, store):
    ds = _dataset(_clean(_base_df()), tmp_path, store, "node")
    profile = profile_dataset(ds, store=store)
    c = classification(QuestionType.TWO_GROUP_COMPARISON)
    state = {"dataset": ds, "profile": profile, "classification": c, "plan": None,
             "stage_plan": build_stage_plan(c, profile, [])}
    before = copy.deepcopy({k: state[k] for k in ("dataset", "profile", "classification", "plan", "stage_plan")})

    update = make_describe_node(store)(state)
    assert set(update) == {"descriptive_result", "decision_log"}
    assert repr(update["descriptive_result"]) == repr(DESCRIPTIVE.compute(ds, store))
    assert update["descriptive_result"]["chart_data"]
    assert [e["decision_point"] for e in update["decision_log"]] == [DESCRIBE_EXECUTION_POINT]
    assert {k: state[k] for k in before} == before


# --- routing ---------------------------------------------------------------------------------------

def _state(question_type, run=None, pending=None):
    plan = build_stage_plan(classification(question_type), FOUR, [])
    if run is not None:
        _describe_step(plan)["run"] = run
    return {"stage_plan": plan, "pending_wrangling": pending}


@pytest.mark.parametrize("question_type", RUNNABLE)
def test_route_after_cleaning_goes_to_describe(question_type):
    assert route_after_cleaning(_state(question_type)) == "describe"


@pytest.mark.parametrize("state", [
    _state(QuestionType.TWO_GROUP_COMPARISON, run=False),
    _state(QuestionType.DRIVER_ANALYSIS, run=False),
    _state(QuestionType.DESCRIPTIVE),
    {},
], ids=["run_false", "driver_run_false", "descriptive", "no_stage_plan"])
def test_route_after_cleaning_goes_to_draft_plan(state):
    assert route_after_cleaning(state) == "stage1_report"  # M1.4b: the Stage 1 report runs before draft_plan


def test_route_after_wrangle_propose():
    assert route_after_wrangle_propose(_state(QuestionType.TWO_GROUP_COMPARISON, pending={"x": 1})) == "wrangle_review"
    assert route_after_wrangle_propose(_state(QuestionType.TWO_GROUP_COMPARISON)) == "describe"
    assert route_after_wrangle_propose(_state(QuestionType.DESCRIPTIVE)) == "stage1_report"  # M1.4b: was draft_plan


def test_graph_edges(tmp_path):
    edges = {(e.source, e.target): e.conditional
             for e in build_agent_graph(LocalDiskStore(base_dir=tmp_path)).get_graph().edges}
    for target in ("wrangle_review", "describe", "stage1_report"):  # M1.4b: stage1_report replaces draft_plan
        assert edges[("wrangle_propose", target)] is True, target
    # M2.2: wrangle_review -> rediagnose (plain), and rediagnose routes on with route_after_cleaning
    assert edges[("wrangle_review", "rediagnose")] is False
    assert edges[("rediagnose", "describe")] is True
    assert edges[("rediagnose", "stage1_report")] is True  # M1.4b
    assert edges[("describe", "stage1_report")] is False  # M1.4b: describe -> stage1_report -> draft_plan


# --- full runs through the real graph -----------------------------------------------------------------

def _run(store, tmp_path, question_type, df, predictors, resumes, propose_fn=must_not_propose, name="run"):
    app = build_agent_graph(store, classify_fn=confident(question_type), draft_fn=fake_draft(predictors),
                            propose_fn=propose_fn)
    state = {"request_text": "x", "session_id": "session-abc", "dataset": _dataset(df, tmp_path, store, name)}
    emitted = {}
    values, pauses, next_nodes = _drive(app, state, name, resumes, emitted)
    return values, pauses, next_nodes, emitted


def test_two_group_without_cleaning(tmp_path, store):
    values, pauses, _, _ = _run(store, tmp_path, QuestionType.TWO_GROUP_COMPARISON,
                                _two_group_df(np.random.default_rng(31)), ["region"], [{"action": "approve"}])
    points = _points(values)
    i = points.index(DESCRIBE_EXECUTION_POINT)
    assert points.index("planner.step.modeling") < i < points.index("planner.outcome")
    assert pauses == ["stage_plan_approval", "plan_approval"]  # M1.3a: gate first
    assert values["descriptive_result"]["chart_data"] and values["test_result"]


def test_two_group_with_cleaning_describes_the_cleaned_data(tmp_path, store):
    df = _two_group_df(np.random.default_rng(32))
    values, pauses, _, _ = _run(store, tmp_path, QuestionType.TWO_GROUP_COMPARISON, _with_duplicate(df), ["region"],
                                [{"decisions": [{"index": 0, "action": "approve"}]}, {"action": "approve"}],
                                propose_fn=fake_propose)
    points = _points(values)
    i = points.index(DESCRIBE_EXECUTION_POINT)
    assert points.index("stage1.clean.execution") < i < points.index("planner.outcome")
    assert pauses == ["stage_plan_approval", "wrangling_approval", "plan_approval"]  # M1.3a: gate first
    assert values["descriptive_result"]["row_count"] == len(df) == len(_with_duplicate(df)) - 1


def test_descriptive_question_has_no_describe_step(tmp_path, store):
    values, pauses, next_nodes, emitted = _run(store, tmp_path, QuestionType.DESCRIPTIVE,
                                               _two_group_df(np.random.default_rng(33)), [], [])
    assert DESCRIBE_EXECUTION_POINT not in _points(values)
    assert "describe" not in emitted
    assert (pauses, next_nodes) == ([], ("report",))
    assert "planner.draft_plan" in _points(values) and values["descriptive_result"]  # set by eda, as before


def test_driver_analysis_runs_describe(tmp_path, store):
    rng = np.random.default_rng(34)
    n = 120
    tier = rng.choice(["basic", "plus", "pro"], n)
    tenure = rng.normal(24, 8, n)
    df = pd.DataFrame({"tenure": tenure, "tier": tier, "total_spend": 20 + 2 * tenure + rng.normal(0, 5, n)})
    values, _, _, _ = _run(store, tmp_path, QuestionType.DRIVER_ANALYSIS, df, ["tenure", "tier"],
                           [{"action": "approve"}])
    assert DESCRIBE_EXECUTION_POINT in _points(values)
    assert values["test_result"]["result_type"] == "regression"


def test_association_has_no_describe_step(tmp_path, store):
    values, pauses, next_nodes, _ = _run(store, tmp_path, QuestionType.ASSOCIATION,
                                         _two_group_df(np.random.default_rng(35)), ["region"], [])
    assert DESCRIBE_EXECUTION_POINT not in _points(values) and "descriptive_result" not in values
    assert _points(values)[-1] == "planner.stage_plan" and (pauses, next_nodes) == ([], ())


def test_multi_group_runs_describe_and_keeps_the_post_hoc(tmp_path, store):
    rng = np.random.default_rng(36)
    df = pd.DataFrame({"region": [g for g in ("East", "North", "West") for _ in range(60)],
                       "total_spend": np.concatenate([rng.normal(m, 5, 60) for m in (50, 55, 65)])})
    values, _, _, _ = _run(store, tmp_path, QuestionType.MULTI_GROUP_COMPARISON, df, ["region"],
                           [{"action": "approve"}])
    points = _points(values)
    assert DESCRIBE_EXECUTION_POINT in points and "stage2.post_hoc" in points


# --- privacy -----------------------------------------------------------------------------------------

def test_canary_two_group_run(fake_model, tmp_path, store):
    prompts, config = fake_model
    config.update(question_type=QuestionType.TWO_GROUP_COMPARISON, predictors=["region"])
    ds = _dataset(_clean(_base_df()), tmp_path, store, "canary_run")
    values, _ = _drive_to_end(build_agent_graph(store), {"request_text": "Did West spend more?",
                                                         "session_id": "s", "dataset": ds}, "canary_run")
    assert DESCRIBE_EXECUTION_POINT in _points(values) and values["report"]
    assert canary_hits(prompts) == []
    text = json.dumps(values["descriptive_result"], default=str)
    assert [c for c in NAME_CANARIES if c in text] == []
