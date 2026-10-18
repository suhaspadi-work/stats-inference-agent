"""
M1.1: the read-only stage planner. build_stage_plan decides by code which of the four steps (clean,
describe, inference, modeling) a run takes; plan_stages records the plan and one record per step, and
ends the run for an unsupported question before any cleaning.
"""
import copy
import json

import numpy as np
import pandas as pd
import pytest
from langgraph.graph import END

from core.dataset import LocalDiskStore
from core.decision_log import log_update
from core.detection import CandidateIssue
from core.graph import build_agent_graph, route_after_plan_stages
from core.ingestion import load_dataset, profile_dataset
from core.nodes import make_plan_stages_node
from core.plan import CausalStatus, ClassificationDecision, QuestionType
from core.stage_plan import UNSUPPORTED_MESSAGE, build_stage_plan, plan_steps
from core.stage_plan_records import STAGE_PLAN_POINT, STEP_POINTS, stage_plan_record, step_records
from tests.test_decision_point_guard import (
    _dataset, _drive, _two_group_df, _with_duplicate, confident, fake_draft, fake_propose,
)
from tests.test_routing_records import make_profile

FOUR = make_profile({"region": ("object", 2), "tier": ("object", 3), "tenure": ("float64", 80),
                     "total_spend": ("float64", 100)})
TWO_COLUMNS = make_profile({"region": ("object", 2), "total_spend": ("float64", 100)})
NO_NUMERIC = make_profile({"a": ("object", 2), "b": ("object", 3), "c": ("object", 4)})

DUPES = CandidateIssue(issue_type="duplicate_rows", description="4 duplicates",
                       affected_columns=("region", "total_spend"), evidence={"duplicate_row_count": 4})
MISSING = CandidateIssue(issue_type="high_missingness", description="notes 12% missing",
                         affected_columns=("notes",), evidence={"missing_pct": 0.12})

COMPARISONS = (QuestionType.TWO_GROUP_COMPARISON, QuestionType.AB_TEST, QuestionType.MULTI_GROUP_COMPARISON)
OFFERED_DESCRIBE = "describing the finished dataset is part of stage 1; the question does not need it, so you choose"


def classification(question_type, confidence=0.9):
    return ClassificationDecision(question_type=question_type, causal_status=CausalStatus.OBSERVATIONAL,
                                  rationale="fake", alternatives=(), confidence=confidence, evidence={})


def steps(plan):
    return {s["step"]: s for s in plan_steps(plan)}


def summary(plan):
    """step -> (state, run, exploratory, reason)"""
    return {n: (s["state"], s["run"], s["exploratory"], s["reason"]) for n, s in steps(plan).items()}


# --- the rules for every question type ------------------------------------------------------------

def test_descriptive_without_issues():
    plan = build_stage_plan(classification(QuestionType.DESCRIPTIVE), FOUR, [])
    assert (plan["status"], plan["message"]) == ("ok", None)
    assert summary(plan) == {
        "clean": ("not_applicable", False, False, "no data-quality issues found by the detectors"),
        "describe": ("planned", True, False, "the question asks for a description of the data"),
        "inference": ("not_applicable", False, False, "a descriptive question has no hypothesis to test"),
        "modeling": ("not_applicable", False, False, "a descriptive question has no model to fit"),
    }
    assert steps(plan)["describe"]["evidence"] == {"row_count": 100, "column_count": 4}
    assert steps(plan)["inference"]["evidence"] == {"question_type": "descriptive"}


def test_descriptive_with_issues():
    plan = build_stage_plan(classification(QuestionType.DESCRIPTIVE), FOUR, [DUPES])
    assert summary(plan)["clean"][:3] == ("planned", True, False)
    assert summary(plan)["describe"][:2] == ("planned", True)


@pytest.mark.parametrize("question_type", COMPARISONS)
def test_comparison_with_room_for_a_model(question_type):
    plan = build_stage_plan(classification(question_type), FOUR, [])
    assert plan["status"] == "ok"
    assert summary(plan) == {
        "clean": ("not_applicable", False, False, "no data-quality issues found by the detectors"),
        "describe": ("offered", True, False, OFFERED_DESCRIBE),
        "inference": ("planned", True, False, "the question compares groups"),
        "modeling": ("offered", False, True,
                     "a regression could adjust the comparison for other columns; exploratory, so you choose"),
    }
    assert steps(plan)["modeling"]["evidence"] == {"column_count": 4, "numeric_column_count": 2,
                                                   "rule": "3 or more columns with at least 1 numeric",
                                                   "provisional": True}
    assert steps(plan)["inference"]["evidence"] == {"question_type": question_type.value}


@pytest.mark.parametrize("question_type", COMPARISONS)
@pytest.mark.parametrize("profile, columns, numeric", [(TWO_COLUMNS, 2, 1), (NO_NUMERIC, 3, 0)],
                         ids=["two_columns", "no_numeric"])
def test_comparison_without_room_for_a_model(question_type, profile, columns, numeric):
    modeling = steps(build_stage_plan(classification(question_type), profile, []))["modeling"]
    assert (modeling["state"], modeling["run"], modeling["exploratory"], modeling["reason"]) == (
        "not_applicable", False, False, "fewer than 3 columns or no numeric column, so no adjusted model is possible")
    assert (modeling["evidence"]["column_count"], modeling["evidence"]["numeric_column_count"]) == (columns, numeric)


def test_driver_analysis():
    plan = build_stage_plan(classification(QuestionType.DRIVER_ANALYSIS), FOUR, [])
    assert summary(plan) == {
        "clean": ("not_applicable", False, False, "no data-quality issues found by the detectors"),
        "describe": ("offered", True, False, OFFERED_DESCRIBE),
        "inference": ("unavailable", False, False, "per-predictor association tests are not built yet (M3.3)"),
        "modeling": ("planned", True, False, "the question asks which variables drive the outcome"),
    }
    assert steps(plan)["inference"]["evidence"] == {"available_from": "M3.3"}


def test_association_is_unsupported():
    plan = build_stage_plan(classification(QuestionType.ASSOCIATION), FOUR, [DUPES])
    assert (plan["status"], plan["message"]) == ("unsupported", UNSUPPORTED_MESSAGE)
    assert plan["message"] == ("Association questions are not supported yet. To see which variables relate to a "
                               "column, ask what drives that column.")
    assert {n: s[:3] for n, s in summary(plan).items()} == {n: ("not_applicable", False, False) for n in
                                                            ("clean", "describe", "inference", "modeling")}
    assert {s[3] for s in summary(plan).values()} == {"the question is not supported yet (G1)"}


# --- the clean step --------------------------------------------------------------------------------

def test_clean_with_one_issue_type():
    clean = steps(build_stage_plan(classification(QuestionType.TWO_GROUP_COMPARISON), FOUR, [DUPES]))["clean"]
    assert clean["reason"] == ("1 data-quality issue(s) found (duplicate_rows); "
                               "cleaning proposals will be shown for approval")
    assert clean["evidence"] == {"issue_count": 1, "issue_types": ["duplicate_rows"],
                                 "affected_columns": ["region", "total_spend"]}


def test_clean_with_two_issue_types():
    clean = steps(build_stage_plan(classification(QuestionType.TWO_GROUP_COMPARISON), FOUR, [MISSING, DUPES]))["clean"]
    assert clean["reason"] == ("2 data-quality issue(s) found (duplicate_rows, high_missingness); "
                               "cleaning proposals will be shown for approval")
    assert clean["evidence"] == {"issue_count": 2, "issue_types": ["duplicate_rows", "high_missingness"],
                                 "affected_columns": ["notes", "region", "total_spend"]}


def test_clean_without_issues_has_empty_evidence():
    clean = steps(build_stage_plan(classification(QuestionType.DESCRIPTIVE), FOUR, []))["clean"]
    assert clean["evidence"] == {"issue_count": 0, "issue_types": [], "affected_columns": []}


# --- shape -----------------------------------------------------------------------------------------

ALL_TYPES = list(QuestionType)


@pytest.mark.parametrize("question_type", ALL_TYPES)
def test_shape(question_type):
    issues = [DUPES]
    before = (copy.deepcopy(FOUR), copy.deepcopy(issues))
    plan = build_stage_plan(classification(question_type), FOUR, issues)
    assert set(plan) == {"schema_version", "question_type", "status", "message", "stages"}
    assert (plan["schema_version"], plan["question_type"]) == (1, question_type.value)
    assert [(s["stage"], s["name"]) for s in plan["stages"]] == [
        (1, "prepare and describe"), (2, "statistical inference"), (3, "data modeling")]
    assert all(set(s) == {"stage", "name", "steps"} for s in plan["stages"])
    assert [s["step"] for s in plan_steps(plan)] == ["clean", "describe", "inference", "modeling"]
    for s in plan_steps(plan):
        assert set(s) == {"step", "state", "run", "exploratory", "reason", "evidence"}
        assert s["state"] in ("planned", "offered", "unavailable", "not_applicable")
        assert isinstance(s["run"], bool) and isinstance(s["exploratory"], bool) and s["reason"]
        if s["exploratory"]:
            assert s["state"] == "offered" and s["step"] != "describe"
    json.dumps(plan)
    assert (FOUR, issues) == before


def test_unknown_question_type_raises():
    fake = type("Fake", (), {"question_type": type("QT", (), {"value": "forecasting"})(), "confidence": 0.9})()
    with pytest.raises(ValueError, match="forecasting"):
        build_stage_plan(fake, FOUR, [])


# --- records -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("question_type, profile, issues, chosen", [
    (QuestionType.TWO_GROUP_COMPARISON, FOUR, [DUPES], "clean, describe, inference"),
    (QuestionType.DESCRIPTIVE, FOUR, [], "describe"),
    (QuestionType.DRIVER_ANALYSIS, FOUR, [DUPES, MISSING], "clean, describe, modeling"),
    (QuestionType.MULTI_GROUP_COMPARISON, TWO_COLUMNS, [], "describe, inference"),
    (QuestionType.ASSOCIATION, FOUR, [DUPES], "unsupported"),
])
def test_stage_plan_record(question_type, profile, issues, chosen):
    c = classification(question_type, confidence=0.75)
    plan = build_stage_plan(c, profile, issues)
    r = stage_plan_record(plan, c, issues)
    assert (r.decision_point, r.chosen, r.decided_by) == (STAGE_PLAN_POINT, chosen, "code_rule")
    assert [x.option for x in r.candidates] == ["clean", "describe", "inference", "modeling"]
    for cand, s in zip(r.candidates, plan_steps(plan)):
        assert cand.eligible is (s["state"] in ("planned", "offered"))
        assert cand.reason == f"{s['state']}: {s['reason']}" and cand.evidence == s["evidence"]
    assert r.rationale == f"question type '{question_type.value}': " + ", ".join(
        f"{s['step']} {s['state']}" for s in plan_steps(plan))
    assert r.evidence == {"question_type": question_type.value, "status": plan["status"],
                          "classifier_confidence": 0.75, "issue_count": len(issues),
                          "issue_types": sorted({i.issue_type for i in issues}), "message": plan["message"],
                          "tables": []}  # M2.1: the tables of the upload (none passed here)
    json.dumps(log_update(r))


def test_stage_plan_record_when_no_step_runs():
    c = classification(QuestionType.DESCRIPTIVE)
    plan = build_stage_plan(c, FOUR, [])
    for s in plan_steps(plan):
        s["run"] = False
    assert stage_plan_record(plan, c, []).chosen == "none"
    association = classification(QuestionType.ASSOCIATION)
    assert stage_plan_record(build_stage_plan(association, FOUR, []), association, []).chosen == "unsupported"


def test_step_records():
    plan = build_stage_plan(classification(QuestionType.TWO_GROUP_COMPARISON), FOUR, [DUPES])
    records = step_records(plan)
    assert [r.decision_point for r in records] == [STEP_POINTS[s] for s in ("clean", "describe", "inference", "modeling")]
    assert [r.chosen for r in records] == ["run", "run", "run", "skip"]
    assert {r.decided_by for r in records} == {"code_rule"}
    stage = {"clean": 1, "describe": 1, "inference": 2, "modeling": 3}
    for r, s in zip(records, plan_steps(plan)):
        assert r.rationale == s["reason"]
        assert r.evidence == {"stage": stage[s["step"]], "step": s["step"], "state": s["state"], "run": s["run"],
                              "exploratory": s["exploratory"], **s["evidence"]}
        run, skip = r.candidates
        assert (run.option, run.eligible) == ("run", s["state"] in ("planned", "offered"))
        assert (skip.option, skip.eligible) == ("skip", s["state"] != "planned")
        detail = f"{s['state']}: {s['reason']}"
        if s["run"]:
            assert (run.reason, skip.reason) == ("chosen: " + detail, "not chosen: " + detail)
        else:
            assert (run.reason, skip.reason) == ("not chosen: " + detail, "chosen: " + detail)
    ids = [r.id for r in (stage_plan_record(plan, classification(QuestionType.TWO_GROUP_COMPARISON), [DUPES]), *records)]
    assert len(set(ids)) == 5
    json.dumps(log_update(*records))


def test_no_step_records_for_an_unsupported_plan():
    assert step_records(build_stage_plan(classification(QuestionType.ASSOCIATION), FOUR, [])) == ()


# --- the node ----------------------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _messy(tmp_path, store):
    rng = np.random.default_rng(21)
    n = 100
    df = pd.DataFrame({"region": rng.choice(["East", "West"], n), "total_spend": rng.normal(50, 10, n),
                       "notes": [None if i < 12 else "ok" for i in range(n)]})
    df = pd.concat([df, df.iloc[[50, 51, 52, 53]]], ignore_index=True)  # 4 duplicates, notes 12/104 missing
    path = tmp_path / "messy.csv"
    df.to_csv(path, index=False)
    return load_dataset(path, session_id="session-abc", name="messy", store=store)


def _node_state(tmp_path, store, question_type):
    ds = _messy(tmp_path, store)
    return {"dataset": ds, "profile": profile_dataset(ds, store=store), "classification": classification(question_type),
            "request_text": "x", "session_id": "session-abc"}


def test_plan_stages_node(tmp_path, store):
    state = _node_state(tmp_path, store, QuestionType.TWO_GROUP_COMPARISON)
    before = copy.deepcopy({k: state[k] for k in ("dataset", "profile", "classification")})
    stored = store.read(state["dataset"].storage_key)

    update = make_plan_stages_node(store)(state)
    clean = steps(update["stage_plan"])["clean"]
    assert (clean["state"], clean["evidence"]["issue_count"]) == ("planned", 2)
    assert [e["decision_point"] for e in update["decision_log"]] == [
        "planner.stage_plan", "planner.step.clean", "planner.step.describe", "planner.step.inference",
        "planner.step.modeling"]
    assert "clarification_needed" not in update
    assert set(update) == {"stage_plan", "decision_log", "exploratory_round", "next_round",
                           "stage_gate_refusal", "stage_plan_refusal", "plan_approval_refusal",
                           "reselect_after_edit"}  # M1.3b: the round flags are reset (step 2b, M1.3c: the refusals too)
    assert (update["exploratory_round"], update["next_round"], update["stage_gate_refusal"]) == (False, None, None)
    assert (update["stage_plan_refusal"], update["plan_approval_refusal"], update["reselect_after_edit"]) == (
        None, None, None)
    assert {k: state[k] for k in ("dataset", "profile", "classification")} == before
    assert store.read(state["dataset"].storage_key).equals(stored)


def test_plan_stages_node_for_an_unsupported_question(tmp_path, store):
    update = make_plan_stages_node(store)(_node_state(tmp_path, store, QuestionType.ASSOCIATION))
    assert update["clarification_needed"] == update["stage_plan"]["message"] == UNSUPPORTED_MESSAGE
    assert [e["decision_point"] for e in update["decision_log"]] == ["planner.stage_plan"]


# --- the graph ---------------------------------------------------------------------------------------

def test_graph_edges(tmp_path):
    edges = {(e.source, e.target): e.conditional
             for e in build_agent_graph(LocalDiskStore(base_dir=tmp_path)).get_graph().edges}
    assert edges[("profile", "plan_stages")] is False
    assert edges[("plan_stages", "wrangle_propose")] is True
    assert edges[("plan_stages", "__end__")] is True
    assert ("profile", "wrangle_propose") not in edges


def test_route_after_plan_stages():
    ok = build_stage_plan(classification(QuestionType.DESCRIPTIVE), FOUR, [])
    unsupported = build_stage_plan(classification(QuestionType.ASSOCIATION), FOUR, [])
    assert route_after_plan_stages({"stage_plan": ok}) == "wrangle_propose"
    assert route_after_plan_stages({"stage_plan": unsupported}) == END


# --- full runs through the real graph ------------------------------------------------------------------

def _refuse(issues):
    raise AssertionError("the proposer must not be called")


def test_full_run_association_ends_at_the_stage_plan(tmp_path, store):
    df = _with_duplicate(_two_group_df(np.random.default_rng(22)))  # has an issue, but cleaning must not start
    app = build_agent_graph(store, classify_fn=confident(QuestionType.ASSOCIATION),
                            draft_fn=fake_draft(["region"]), propose_fn=_refuse)
    state = {"request_text": "Is region related to spend?", "session_id": "session-abc",
             "dataset": _dataset(df, tmp_path, store, "assoc")}
    values, pauses, next_nodes = _drive(app, state, "assoc", [], {})
    assert (pauses, next_nodes) == ([], ())
    points = [e["decision_point"] for e in values["decision_log"]]
    assert points[-1] == "planner.stage_plan"
    assert "planner.route" not in points and not any(p.startswith("stage1.clean") for p in points)
    assert values["clarification_needed"] == UNSUPPORTED_MESSAGE


def test_full_run_two_group_with_cleaning(tmp_path, store):
    df = _two_group_df(np.random.default_rng(23))
    df["notes"] = [None if i % 6 == 0 else "ok" for i in range(len(df))]
    app = build_agent_graph(store, classify_fn=confident(QuestionType.TWO_GROUP_COMPARISON),
                            draft_fn=fake_draft(["region"]), propose_fn=fake_propose)
    state = {"request_text": "Did West spend more?", "session_id": "session-abc",
             "dataset": _dataset(_with_duplicate(df), tmp_path, store, "tg")}
    values, pauses, _ = _drive(app, state, "tg", [
        {"decisions": [{"index": 0, "action": "approve"}, {"index": 1, "action": "approve"}]},
        {"action": "approve"},
    ], {})
    assert pauses == ["stage_plan_approval", "wrangling_approval", "plan_approval"]  # M1.3a: gate first
    points = [e["decision_point"] for e in values["decision_log"]]
    start = points.index("planner.stage_plan")
    assert points[start:start + 5] == ["planner.stage_plan", "planner.step.clean", "planner.step.describe",
                                       "planner.step.inference", "planner.step.modeling"]
    assert start + 6 == points.index("stage1.clean.scan")  # M1.3a: the gate's record comes between
    assert points[start + 5] == "planner.stage_plan_approval"
    assert values["stage_plan"]["status"] == "ok"
