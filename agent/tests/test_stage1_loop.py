"""
M2.4a: the Stage 1 loop through the real graph (guard fakes, fake chat model). After a round of fixes the diagnose step
runs again; issues that no earlier round of the question proposed start another round (the same wrangling_approval
pause, with "round" and "since_last_round"); the loop stops on no new issues, a round that applied nothing, the
user's finish, or the round limit. Expected values are computed independently with pandas.
"""
import json

import numpy as np
import pandas as pd
import pytest
from langgraph.types import Command

import core.nodes as nodes_module
from core import thresholds
from core.dataset import LocalDiskStore
from core.detection import CandidateIssue, issue_key
from core.graph import build_agent_graph
from core.plan import QuestionType
from tests.test_decision_point_guard import _dataset, choose, confident, fake_draft
from tests.test_join_lineage_skip import REGIONS, _again, _proposed
from tests.test_join_run import (
    _bundle, _customers, _drive_checking_mirror, _lineage_problems, _orders, _sections, _state, _store_frame,
)
from tests.test_prompt_canaries import fake_model  # noqa: F401 (fixture)
from tests.test_stage1_report import _answer

GATE = "stage_plan_approval"
OLD_KEYS = {"type", "question", "proposed_operations", "allowed_decisions"}


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


class Counting:
    """A fake proposer (choose) that counts its calls."""
    def __init__(self, mapping=None):
        self.calls, self.inner = 0, choose(mapping or {})

    def __call__(self, issues, candidates, request_text=None):
        self.calls += 1
        return self.inner(issues, candidates, request_text)


def _app(store, proposer=None):
    return build_agent_graph(store, classify_fn=confident(QuestionType.TWO_GROUP_COMPARISON),
                             draft_fn=fake_draft(["region"]), propose_fn=proposer or choose({}))


def _run(app, graph_input, thread, **overrides):
    """Run to the end; returns the final values, the pause payloads and every (node, update) in order."""
    config = {"configurable": {"thread_id": thread}}
    pauses, updates = [], []
    while True:
        for event in app.stream(graph_input, config, stream_mode="updates"):
            updates.extend((node, update) for node, update in event.items() if node != "__interrupt__")
        snapshot = app.get_state(config)
        if not snapshot.interrupts:
            return snapshot.values, pauses, updates
        payload = snapshot.interrupts[0].value
        pauses.append(payload)
        graph_input = Command(resume=_answer(payload, overrides))


def _wranglings(pauses):
    return [p for p in pauses if p["type"] == "wrangling_approval"]


def _points(values):
    return [e["decision_point"] for e in values["decision_log"]]


def _input(ds, **extra):
    return {"request_text": "Did West spend more?", "session_id": "s", "dataset": ds, **extra}


def _conversion_frame(duplicate=False):
    """The G72 case: score is text; 4 of 100 values are missing and 4 more are "unknown" (95.8% of the non-missing
    values are numbers, so it is flagged); converting makes 8 missing (8%), above the 5% threshold."""
    rng = np.random.default_rng(5)
    score = [f"{v:.1f}" for v in rng.normal(10, 2, 100)]
    for i in range(4):
        score[10 + i], score[50 + i] = "unknown", None
    df = pd.DataFrame({"region": ["East", "West"] * 50, "channel": ["web", "store", "app", "web"] * 25,
                       "score": score, "total_spend": rng.normal(50, 5, 100)})
    return pd.concat([df, df.iloc[[0]]], ignore_index=True) if duplicate else df


def _conversion(tmp_path, store, name="g72", duplicate=False):
    df = _conversion_frame(duplicate)
    converted = pd.to_numeric(df["score"], errors="coerce")
    assert df["score"].isna().mean() <= 0.05 < converted.isna().mean()  # not flagged before, flagged after
    return df, _dataset(df, tmp_path, store, name)


def _approve_all(payload, **extra):
    return {"decisions": [{"index": o["index"], "action": "approve"} for o in payload["proposed_operations"]], **extra}


# --- (a) a conversion that creates a missing-value issue --------------------------------------------------------

def test_a_conversion_starts_a_second_round(fake_model, tmp_path, store):
    df, ds = _conversion(tmp_path, store)
    values, pauses, _ = _run(_app(store), _input(ds), "a")
    first, second = _wranglings(pauses)
    assert [o["candidate_id"] for o in first["proposed_operations"]] == ["convert_to_number"]
    assert first["round"] == 1 and "since_last_round" not in first
    assert set(first) == OLD_KEYS | {"round"}  # (g) the old keys plus round
    assert second["round"] == 2 and [o["candidate_id"] for o in second["proposed_operations"]] == ["delete_rows"]
    assert second["proposed_operations"][0]["params"]["column"] == "score"
    converted = df.assign(score=pd.to_numeric(df["score"], errors="coerce"))
    assert second["since_last_round"] == {
        "round": 1, "applied": ["Convert score to numbers (4 value(s) that are not numbers become missing)."],
        "quality": {"rows": len(converted), "duplicate_rows": int(converted.duplicated().sum()),
                    "columns_with_missing": int(converted.isna().any().sum())}}
    assert (values["stage1_round"], values["stage1_end_reason"]) == (2, "no_new_issues")


# --- (b) a normalization that creates exact duplicates ----------------------------------------------------------

def test_a_normalization_starts_a_dedupe_round(fake_model, tmp_path, store):
    rng = np.random.default_rng(8)
    df = pd.DataFrame({"region": ["East", "West"] * 30, "channel": ["web", "store", "app"] * 20,
                       "total_spend": rng.normal(50, 5, 60).round(3)})
    copies = df.iloc[[0, 2, 4]].assign(region="east ")
    messy = pd.concat([df, copies], ignore_index=True)
    assert int(messy.duplicated().sum()) == 0
    normalized = messy.assign(region=messy["region"].str.strip().str.lower())
    duplicates = int(normalized.duplicated().sum())
    assert duplicates == 3
    ds = _dataset(messy, tmp_path, store, "norm")
    values, pauses, _ = _run(_app(store, choose({"inconsistent_text": "trim_and_lowercase"})), _input(ds), "b")
    first, second = _wranglings(pauses)
    assert [o["candidate_id"] for o in first["proposed_operations"]] == ["trim_and_lowercase"]
    (dedupe,) = second["proposed_operations"]
    assert dedupe["candidate_id"] == "dedupe_all" and dedupe["impact_preview"]["rows_changed"] == duplicates
    assert len(_store_frame(store, values["dataset"])) == len(messy) - duplicates


# --- (c) the stop rules -------------------------------------------------------------------------------------------

def _dedupe_only(tmp_path, store):
    rng = np.random.default_rng(9)
    df = pd.DataFrame({"region": ["East", "West"] * 30, "total_spend": rng.normal(50, 5, 60), "channel": ["a", "b", "c"] * 20})
    return _dataset(pd.concat([df, df.iloc[[1]]], ignore_index=True), tmp_path, store, "dedupe")


def test_nothing_new_ends_after_one_round(fake_model, tmp_path, store):
    values, pauses, _ = _run(_app(store), _input(_dedupe_only(tmp_path, store)), "c1")
    assert len(_wranglings(pauses)) == 1
    assert (values["stage1_round"], values["stage1_end_reason"]) == (1, "no_new_issues")


def test_declining_everything_ends_the_loop(fake_model, tmp_path, store):
    _, ds = _conversion(tmp_path, store)
    reject = lambda p: {"decisions": [{"index": o["index"], "action": "reject"} for o in p["proposed_operations"]]}
    values, pauses, _ = _run(_app(store), _input(ds), "c2", wrangling_approval=reject)
    assert len(_wranglings(pauses)) == 1 and values["stage1_end_reason"] == "declined_all"


def test_a_failed_operation_is_not_applied(fake_model, tmp_path, store):
    _, ds = _conversion(tmp_path, store)
    bad = lambda p: {"decisions": [{"index": o["index"], "action": "edit", "params": {"column": "nope", "to_type": "numeric"}}
                                   for o in p["proposed_operations"]]}
    values, pauses, _ = _run(_app(store), _input(ds), "c3", wrangling_approval=bad)
    (execution,) = [e for e in values["decision_log"] if e["decision_point"] == "stage1.clean.execution"]
    assert execution["chosen"] == "failed"
    assert len(_wranglings(pauses)) == 1 and values["stage1_end_reason"] == "declined_all"


def test_finish_ends_the_loop(fake_model, tmp_path, store):
    _, ds = _conversion(tmp_path, store)
    values, pauses, _ = _run(_app(store), _input(ds), "c4", wrangling_approval=lambda p: _approve_all(p, finish=True))
    assert len(_wranglings(pauses)) == 1 and values["stage1_end_reason"] == "finished"
    assert "Stage 1 ended after 1 round(s): you finished Stage 1." in _sections(values)["Rounds"]


def test_the_round_limit_is_read_at_call_time(fake_model, tmp_path, store, monkeypatch):
    monkeypatch.setattr(thresholds, "MAX_STAGE1_ROUNDS", 1)
    _, ds = _conversion(tmp_path, store)
    values, pauses, _ = _run(_app(store), _input(ds), "c5")
    assert len(_wranglings(pauses)) == 1 and values["stage1_end_reason"] == "round_limit"


def _with_synthetic(monkeypatch, marker=None):
    """detect_all_issues plus a duplicate_rows issue whose key is new for every dataset version (marker None) or the
    same every time (a fixed marker); its dedupe changes nothing but makes a new version."""
    real = nodes_module.detect_all_issues

    def wrapper(dataset, profile, store, **kwargs):
        column = marker or f"marker {dataset.handle}"
        return real(dataset, profile, store, **kwargs) + [CandidateIssue(
            "duplicate_rows", "synthetic", (column,), {"duplicate_row_count": 0, "total_rows": profile.row_count})]
    monkeypatch.setattr(nodes_module, "detect_all_issues", wrapper)


def test_the_default_limit_stops_after_three_rounds(fake_model, tmp_path, store, monkeypatch):
    _with_synthetic(monkeypatch)
    rng = np.random.default_rng(10)
    df = pd.DataFrame({"region": ["East", "West"] * 30, "total_spend": rng.normal(50, 5, 60), "channel": ["a", "b", "c"] * 20})
    proposer = Counting()
    values, pauses, _ = _run(_app(store, proposer), _input(_dataset(df, tmp_path, store, "chain")), "c6")
    assert [p["round"] for p in _wranglings(pauses)] == [1, 2, 3] and proposer.calls == 3  # the 4th never proposed
    assert (values["stage1_round"], values["stage1_end_reason"]) == (3, "round_limit")


# --- (d) an issue is proposed at most once per question ----------------------------------------------------------

def test_an_issue_is_proposed_once_per_question(fake_model, tmp_path, store, monkeypatch):
    _with_synthetic(monkeypatch, marker="always")
    _, ds = _conversion(tmp_path, store)
    app = _app(store)
    values, pauses, _ = _run(app, _input(ds), "d")
    rounds = [[o["params"].get("subset", "x") for o in p["proposed_operations"] if o["op_type"] == "dedupe"]
              for p in _wranglings(pauses)]
    assert rounds == [[None], []]  # the synthetic dedupe in round 1 only
    key = ["duplicate_rows", ["always"], None]
    assert values["stage1_seen_keys"].count(key) == 1
    second, pauses, updates = _run(app, _input(values["dataset"]), "d")
    (reset,) = [u for node, u in updates if node == "plan_stages"]
    assert [reset[k] for k in ("stage1_round", "stage1_seen_keys", "stage1_finish", "stage1_continue",
                               "stage1_end_reason")] == [0, [], False, False, None]
    assert [o["op_type"] for o in _wranglings(pauses)[0]["proposed_operations"]] == ["dedupe"]  # proposed again
    assert second["stage1_seen_keys"] == [key]


# --- (e) a round costs one proposer call ------------------------------------------------------------------------

def test_one_proposer_call_per_round(fake_model, tmp_path, store):
    proposer = Counting()
    _run(_app(store, proposer), _input(_conversion(tmp_path, store)[1]), "e1")
    assert proposer.calls == 2
    proposer = Counting()
    _run(_app(store, proposer), _input(_dedupe_only(tmp_path, store)), "e2")
    assert proposer.calls == 1
    proposer = Counting()
    clean = pd.DataFrame({"region": ["East", "West"] * 20, "total_spend": np.arange(40.0), "channel": ["a", "b"] * 20})
    _run(_app(store, proposer), _input(_dataset(clean, tmp_path, store, "clean")), "e3")
    assert proposer.calls == 0


def test_one_model_call_per_round(fake_model, tmp_path, store):
    prompts, config = fake_model
    config.update(question_type=QuestionType.TWO_GROUP_COMPARISON, predictors=["region"])
    counts = []
    for thread, ds in (("m1", _conversion(tmp_path, store)[1]), ("m2", _dedupe_only(tmp_path, store))):
        before = len(prompts)
        _run(build_agent_graph(store), _input(ds), thread)
        counts.append(sum(p["stage"] == "_WranglingProposalsOut" for p in prompts[before:]))
    assert counts == [2, 1]


# --- (f) pending_unaddressed_issues --------------------------------------------------------------------------------

def test_rejections_accumulate_and_are_not_inherited(fake_model, tmp_path, store):
    _, ds = _conversion(tmp_path, store, duplicate=True)

    def answer(payload):  # round 1: reject the dedupe, apply the conversion; round 2: reject the missing values
        return {"decisions": [{"index": o["index"], "action": "approve"} if o["op_type"] == "cast"
                              else {"index": o["index"], "action": "reject", "reason": "keep"}
                              for o in payload["proposed_operations"]]}
    app = _app(store)
    values, pauses, _ = _run(app, _input(ds), "f", wrangling_approval=answer)
    assert [p["round"] for p in _wranglings(pauses)] == [1, 2]
    assert [u.issue_type for u in values["plan"].unaddressed_issues] == ["duplicate_rows", "high_missingness"]
    second, _, _ = _run(app, _input(values["dataset"]), "f")  # every decision approve
    assert list(second["plan"].unaddressed_issues) == []  # the first question's caveats are not inherited


# --- (h) stakeholder mode -----------------------------------------------------------------------------------------

def test_stakeholder_round_one_auto_round_two_interrupts(fake_model, tmp_path, store):
    rng = np.random.default_rng(11)
    df = pd.DataFrame({"region": ["East", "West"] * 47 + ["East"], "total_spend": rng.normal(50, 5, 95),
                       "score": rng.normal(10, 2, 95)})
    df.loc[[3, 9, 15, 21, 27], "score"] = np.nan
    messy = pd.concat([df, df.iloc[40:50]], ignore_index=True)  # 10 duplicates of complete rows
    assert messy["score"].isna().mean() < 0.05 < messy.drop_duplicates()["score"].isna().mean()
    ds = _dataset(messy, tmp_path, store, "stake")
    values, pauses, _ = _run(_app(store), _input(ds, autonomy_mode="stakeholder"), "h")
    # no pause in round 1 (the dedupe is auto-approved); the stage gate offers the exploratory round at the end
    assert [p["type"] for p in pauses] == [GATE, "wrangling_approval", "plan_approval", "stage_gate"]
    assert _wranglings(pauses)[0]["round"] == 2
    approvals = [e["chosen"] for e in values["decision_log"] if e["decision_point"] == "stage1.clean.approval"]
    assert approvals == ["auto_approve", "approve"]


# --- (i) chained joins ---------------------------------------------------------------------------------------------

def test_chained_joins(fake_model, tmp_path, store):
    customers = _customers()
    tables = _bundle(tmp_path, store, {"orders": _orders(customers), "customers": customers, "regions": REGIONS})
    app = _app(store)
    values, pauses, _ = _run(app, _state(tables), "i")
    assert [(p["round"], _proposed([p])) for p in _wranglings(pauses)] == [(1, ["customers"]), (2, ["regions"])]
    final = _store_frame(store, values["dataset"])
    assert set(final.columns) == {"customer_id", "total_spend", "region", "tier", "manager"}
    assert _lineage_problems(values) == []
    for thread_question in (1, 2):  # a second and a third question
        values, pauses, _ = _run(app, _again(values), "i")
        assert _proposed(pauses) == []


# --- (j) the mirror ------------------------------------------------------------------------------------------------

def test_the_mirror_holds_in_a_two_round_run(fake_model, tmp_path, store):
    _, ds = _conversion(tmp_path, store)
    values, checked = _drive_checking_mirror(_app(store), _input(ds), "j")
    assert values["stage1_round"] == 2 and checked.count("rediagnose") == 2 and checked.count("wrangle_review") == 2


# --- (k) strict JSON and the decision-point order ------------------------------------------------------------------

def test_strict_json_and_the_order(fake_model, tmp_path, store):
    _, ds = _conversion(tmp_path, store)
    values, _, _ = _run(_app(store), _input(ds), "k")
    json.dumps(values["decision_log"], allow_nan=False)
    points = _points(values)
    start = points.index("stage1.clean.scan")
    assert points[:start] == ["intake.question_type", "intake.causal_status", "intake.clarification_gate",
                              "planner.stage_plan", "planner.step.clean", "planner.step.describe",
                              "planner.step.inference", "planner.step.modeling", "planner.stage_plan_approval",
                              "stage1.quality.snapshot"]
    one_round = ["stage1.clean.scan", "stage1.clean.fix", "stage1.clean.approval", "stage1.clean.execution",
                 "stage1.quality.snapshot"]
    assert points[start:start + 11] == one_round + one_round + ["planner.stage_replan"]
    scans = [e for e in values["decision_log"] if e["decision_point"] == "stage1.clean.scan"]
    assert [s["evidence"]["round"] for s in scans] == [1, 2]
    assert [issue_key(CandidateIssue("high_missingness", "", ("score",), {}))] == [
        k for k in values["stage1_seen_keys"] if k[0] == "high_missingness"]
