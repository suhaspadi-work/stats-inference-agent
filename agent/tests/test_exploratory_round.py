"""
M1.3b step 2: the boundary gate and the exploratory round. After a comparison whose stage plan offered
modeling, stage_gate offers an exploratory regression (same outcome, the grouping column plus up to 5 other
usable columns); run builds a new plan by code (start_round) and the existing select_method, freeze, execute
and report run it, labelled exploratory. Skip, reject or nothing to offer ends the run after round one.
"""
import copy
import json

import numpy as np
import pandas as pd
import pytest
from langgraph.types import Command

import core.report as report_module
from core.dataset import LocalDiskStore
from core.decision_log import log_update
from core.decision_points import DECISION_NODES, NODE_STAGES, registered_points
from core.detection import CandidateIssue
from core.event_log import events_path, read_decision_log
from core.graph import build_agent_graph
from core.ingestion import profile_dataset
from core.nodes import make_plan_stages_node, route_after_report, route_after_stage_gate
from core.plan import AnalysisPlan, PlanStatus, QuestionType, UnaddressedIssue
from core.round_records import ROUND_START_POINT, STAGE_GATE_POINT, round_start_record, stage_gate_record
from core.round_two import (
    MAX_OTHER_PREDICTORS, OFFER_NOTE, build_exploratory_plan, exploratory_offer, propose_exploratory_predictors,
    stage_gate_payload,
)
from core.rounds import EXPLORATORY_NOTE
from core.stage_plan import build_stage_plan, plan_steps
from eval.check_live_capabilities import filter_runs
from tests.test_decision_point_guard import _dataset, _drive, confident, fake_draft, must_not_propose
from tests.test_prompt_canaries import fake_model  # noqa: F401 (fixture)
from tests.test_routing_records import make_plan, make_profile
from tests.test_stage_plan import classification

GATE = "stage_plan_approval"
APPROVE = {"action": "approve"}


def _plan(question_type=QuestionType.TWO_GROUP_COMPARISON, predictors=("region",)):
    return make_plan(question_type, predictors)


PROFILE = make_profile({"region": ("object", 2), "tier": ("object", 3), "tenure": ("float64", 80),
                        "total_spend": ("float64", 100)})


# --- propose_exploratory_predictors --------------------------------------------------------------------

def test_verdicts():
    profile = make_profile({
        "region": ("object", 2), "total_spend": ("float64", 100), "flag": ("object", 1),
        "customer_id": ("int64", 100), "score": ("float64", 100), "city": ("object", 11), "state": ("object", 10),
    })
    before = copy.deepcopy(profile)
    p = propose_exploratory_predictors(profile, "region", "total_spend")
    assert p["predictors"] == ["region", "score", "state"]
    assert p["reasons"] == {
        "region": "the grouping column of the main analysis",
        "total_spend": "the outcome cannot also be a predictor",
        "flag": "constant or empty",
        "customer_id": "identifier-like: every value is distinct",
        "score": "numeric column",
        "city": "text column with more than 10 distinct values",
        "state": "text column with 10 levels",
    }
    assert p["included"] == ["region", "score", "state"]
    assert p["excluded"] == ["total_spend", "flag", "customer_id", "city"]
    assert set(p["reasons"]) == {c.name for c in profile.columns}
    assert set(p["included"]) | set(p["excluded"]) == set(p["reasons"])
    assert not set(p["included"]) & set(p["excluded"])
    assert profile == before
    json.dumps(p)


def test_limit_of_five_other_predictors():
    columns = {"region": ("object", 2), "total_spend": ("float64", 100)}
    columns.update({f"x{i}": ("float64", 50) for i in range(8)})
    p = propose_exploratory_predictors(make_profile(columns), "region", "total_spend")
    assert MAX_OTHER_PREDICTORS == 5
    assert p["predictors"] == ["region", "x0", "x1", "x2", "x3", "x4"]
    assert [p["reasons"][f"x{i}"] for i in (5, 6, 7)] == ["beyond the limit of 5 other predictors"] * 3


def test_identifier_rule_counts_missing_values():
    from core.model_safe_view import ColumnSummary, ModelSafeView
    profile = ModelSafeView(dataset_handle="x@v1", row_count=100, column_count=3, columns=(
        ColumnSummary(name="region", dtype="object", missing_count=0, missing_pct=0.0, distinct_count=2),
        ColumnSummary(name="total_spend", dtype="float64", missing_count=0, missing_pct=0.0, distinct_count=100),
        ColumnSummary(name="code", dtype="int64", missing_count=4, missing_pct=0.04, distinct_count=96),
    ))
    p = propose_exploratory_predictors(profile, "region", "total_spend")
    assert p["reasons"]["code"] == "identifier-like: every value is distinct"


# --- exploratory_offer -------------------------------------------------------------------------------------

def _stage_plan(question_type, profile=PROFILE):
    return build_stage_plan(classification(question_type), profile, [])


@pytest.mark.parametrize("question_type", [QuestionType.TWO_GROUP_COMPARISON, QuestionType.AB_TEST,
                                           QuestionType.MULTI_GROUP_COMPARISON])
def test_offer_for_comparisons(question_type):
    proposal = propose_exploratory_predictors(PROFILE, "region", "total_spend")
    assert exploratory_offer(_stage_plan(question_type), _plan(question_type), proposal) is True


def test_no_offer():
    proposal = propose_exploratory_predictors(PROFILE, "region", "total_spend")
    driver = QuestionType.DRIVER_ANALYSIS
    assert exploratory_offer(_stage_plan(driver), _plan(driver, ("region", "tenure")), proposal) is False
    small = make_profile({"region": ("object", 2), "total_spend": ("float64", 100)})
    two_group = QuestionType.TWO_GROUP_COMPARISON
    assert step(_stage_plan(two_group, small), "modeling")["state"] == "not_applicable"
    assert exploratory_offer(_stage_plan(two_group, small), _plan(), proposal) is False
    none = {"predictors": ["region"], "reasons": {}, "included": ["region"], "excluded": []}
    assert exploratory_offer(_stage_plan(two_group), _plan(), none) is False
    descriptive = QuestionType.DESCRIPTIVE
    assert exploratory_offer(_stage_plan(descriptive), _plan(descriptive, ()), proposal) is False


def step(plan, name):
    return next(s for s in plan_steps(plan) if s["step"] == name)


# --- build_exploratory_plan ------------------------------------------------------------------------------

def test_build_exploratory_plan():
    issue = UnaddressedIssue(issue_type="high_missingness", description="d", proposed_operation="filter",
                             rejection_reason="r")
    plan = _plan().record_unaddressed_issue(issue)
    before = copy.deepcopy(plan)
    new = build_exploratory_plan(plan, "data@v2", ["region", "tier", "tenure"])
    assert plan == before
    assert new.id != plan.id
    assert (new.question_type, new.status) == (QuestionType.DRIVER_ANALYSIS, PlanStatus.REFINED)
    for name in ("request_text", "session_id", "causal_status", "outcome_name", "outcome_type", "unit_of_analysis",
                 "alpha", "allowed_claims", "classification_decision"):
        assert getattr(new, name) == getattr(plan, name), name
    assert new.candidate_predictors == ("region", "tier", "tenure")
    assert new.data_sources == ("data@v2",)
    assert new.method_decisions == () and new.deviations == ()
    assert new.unaddressed_issues == (issue,)


# --- records ---------------------------------------------------------------------------------------------

PROPOSAL = propose_exploratory_predictors(PROFILE, "region", "total_spend")
EVIDENCE_KEYS = {"finished_round", "capability", "question_type", "method", "outcome", "group_column",
                 "proposed_predictors", "excluded_columns", "exploratory", "received_action",
                 "chosen_predictors", "refused_attempts"}  # M1.3b step 2b: two new evidence keys


@pytest.mark.parametrize("decision, choice, offered, decided_by, rationale", [
    ({"action": "run"}, "run", True, "human", "the user chose to run the exploratory regression"),
    ({"action": "skip"}, "skip", True, "human", "the user chose to end the run after the main analysis"),
    (None, "not_offered", False, "code_rule", "no other usable predictor in the data"),
    ({"action": "maybe"}, "skip", True, "code_rule",
     "action 'maybe' is not one of run or skip; the node skips the exploratory round"),
])
def test_stage_gate_record(decision, choice, offered, decided_by, rationale):
    r = stage_gate_record(_plan(), PROPOSAL, decision, choice, offered, "two_group")
    assert (r.decision_point, r.chosen, r.decided_by, r.rationale) == (STAGE_GATE_POINT, choice, decided_by, rationale)
    assert r.overridden_by_human is False
    run, skip = r.candidates
    assert (run.option, run.eligible) == ("run", offered)
    assert run.reason == ("run the exploratory regression: region, tier, tenure" if offered
                          else "no usable extra predictor")
    assert (skip.option, skip.eligible, skip.reason) == ("skip", True, "end the run after the main analysis")
    assert set(r.evidence) == EVIDENCE_KEYS
    assert r.evidence["proposed_predictors"] == ["region", "tier", "tenure"]
    assert r.evidence["excluded_columns"] == ["total_spend"]
    assert (r.evidence["capability"], r.evidence["group_column"], r.evidence["exploratory"]) == ("two_group", "region", True)
    assert r.evidence["received_action"] == (decision or {}).get("action")


def test_round_start_record():
    plan2 = build_exploratory_plan(_plan(), "x@v1", PROPOSAL["predictors"])
    r = round_start_record(plan2, PROFILE, PROPOSAL, from_round=1)
    assert (r.decision_point, r.chosen, r.decided_by) == (ROUND_START_POINT, "region, tier, tenure", "code_rule")
    assert r.rationale == "an exploratory regression adjusting the comparison for other columns"
    assert [(c.option, c.eligible, c.reason) for c in r.candidates] == [
        ("region", True, "the grouping column of the main analysis"),
        ("tier", True, "text column with 3 levels"),
        ("tenure", True, "numeric column"),
        ("total_spend", False, "the outcome cannot also be a predictor"),
    ]
    assert all(set(c.evidence) == {"dtype", "distinct_count", "missing_pct"} for c in r.candidates)
    assert r.evidence == {"round": 2, "from_round": 1, "exploratory": True, "capability": "regression",
                          "question_type": "driver_analysis", "outcome": "total_spend", "group_column": "region",
                          "max_other_predictors": 5, "predictors": ["region", "tier", "tenure"],
                          "excluded": ["total_spend"], "chosen_by": "code_rule"}  # M1.3b step 2b: chosen_by


def test_records_serialize_and_are_registered():
    plan2 = build_exploratory_plan(_plan(), "x@v1", PROPOSAL["predictors"])
    records = [stage_gate_record(_plan(), PROPOSAL, {"action": "run"}, "run", True, "two_group"),
               stage_gate_record(_plan(), PROPOSAL, None, "not_offered", False, "two_group"),
               round_start_record(plan2, PROFILE, PROPOSAL, from_round=1)]
    assert len({r.id for r in records}) == 3
    json.dumps(log_update(*records))
    assert {STAGE_GATE_POINT, ROUND_START_POINT} <= registered_points()


# --- payload -----------------------------------------------------------------------------------------------

def test_stage_gate_payload():
    plan, stage_plan = _plan(), _stage_plan(QuestionType.TWO_GROUP_COMPARISON)
    payload = stage_gate_payload(plan, {"p_value": 0.03}, "two_group", 1, stage_plan, PROPOSAL)
    assert set(payload) == {"type", "question", "request_text", "finished", "offer", "allowed_decisions",
                            "refused"}  # M1.3b step 2b: the previous refusal, None here
    assert (payload["type"], payload["question"], payload["request_text"]) == (
        "stage_gate", "The main analysis is finished. Run an exploratory follow-up?", "x")
    assert payload["finished"] == {"round": 1, "capability": "two_group", "question_type": "two_group_comparison",
                                   "method": None, "outcome": "total_spend", "predictors": ["region"],
                                   "p_value": 0.03}
    assert payload["offer"] == {"step": "modeling", "capability": "regression", "exploratory": True,
                                "reason": step(stage_plan, "modeling")["reason"], "outcome": "total_spend",
                                "proposed_predictors": ["region", "tier", "tenure"],
                                "excluded": {"total_spend": "the outcome cannot also be a predictor"},
                                "note": OFFER_NOTE,
                                # M1.3b step 2b: the columns you can choose, and the unsuitable ones with why
                                "selectable": [{"column": "tier", "proposed": True, "reason": "text column with 3 levels"},
                                               {"column": "tenure", "proposed": True, "reason": "numeric column"}],
                                "unsuitable": {"total_spend": {
                                    "reason": "the outcome cannot also be a predictor",
                                    "explanation": "it is the outcome being explained, so it cannot also be used to "
                                                   "explain itself"}}}
    assert OFFER_NOTE == "labelled exploratory; its results are not corrected for multiple testing"
    assert payload["allowed_decisions"] == ["run", "skip"]
    json.dumps(payload)
    assert stage_gate_payload(plan, {}, "two_group", 1, stage_plan, PROPOSAL)["finished"]["p_value"] is None
    assert stage_gate_payload(plan, None, "two_group", 1, stage_plan, PROPOSAL)["finished"]["p_value"] is None


# --- routing, edges, registry --------------------------------------------------------------------------------

def test_route_after_report():
    comparison = {"stage_plan": _stage_plan(QuestionType.TWO_GROUP_COMPARISON), "plan": _plan()}
    assert route_after_report(comparison) == "stage_gate"
    # M1.4b step 2b: where the router used to end the run it now goes to the summary
    assert route_after_report({**comparison, "exploratory_round": True}) == "summary"
    driver = QuestionType.DRIVER_ANALYSIS
    assert route_after_report({"stage_plan": _stage_plan(driver), "plan": _plan(driver, ("region",))}) == "summary"
    small = make_profile({"region": ("object", 2), "total_spend": ("float64", 100)})
    assert route_after_report({"stage_plan": _stage_plan(QuestionType.TWO_GROUP_COMPARISON, small),
                               "plan": _plan()}) == "summary"


def test_route_after_stage_gate():
    assert route_after_stage_gate({"next_round": {"run": True, "predictors": ["region"]}}) == "start_round"
    assert route_after_stage_gate({"next_round": {"run": False, "predictors": []}}) == "summary"  # M1.4b 2b: was END


def test_graph_edges(tmp_path):
    edges = {(e.source, e.target): e.conditional
             for e in build_agent_graph(LocalDiskStore(base_dir=tmp_path)).get_graph().edges}
    # M1.4b step 2b: report and stage_gate go to the summary instead of __end__; the summary ends the run
    assert edges[("report", "stage_gate")] is True and edges[("report", "summary")] is True
    assert edges[("stage_gate", "start_round")] is True and edges[("stage_gate", "summary")] is True
    assert ("report", "__end__") not in edges and ("stage_gate", "__end__") not in edges
    assert edges[("summary", "__end__")] is False
    assert edges[("start_round", "select_method")] is False
    assert edges[("stage_gate", "stage_gate")] is True  # M1.3b step 2b: a refused choice re-opens the gate


def test_registry():
    assert len(registered_points()) == 31 and len(DECISION_NODES) == 17  # M2.2: diagnose and rediagnose
    assert DECISION_NODES["stage_gate"] == (STAGE_GATE_POINT,)
    assert DECISION_NODES["start_round"] == (ROUND_START_POINT,)
    assert NODE_STAGES["stage_gate"] == NODE_STAGES["start_round"] == "analysis"


# --- plan_stages resets the round flags ------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _frame(seed=51, n=120, extra=True):
    rng = np.random.default_rng(seed)
    region = rng.choice(["East", "West"], n)
    df = pd.DataFrame({"region": region,
                       "total_spend": 50 + np.where(region == "West", 6.0, 0.0) + rng.normal(0, 5, n)})
    if extra:
        df["tier"] = rng.choice(["basic", "plus", "pro"], n)
        df["tenure"] = rng.normal(24, 8, n)
    return df


def test_plan_stages_resets_a_stale_exploratory_flag(tmp_path, store):
    ds = _dataset(_frame(), tmp_path, store, "stale")
    state = {"dataset": ds, "profile": profile_dataset(ds, store=store), "request_text": "x", "session_id": "s",
             "classification": classification(QuestionType.TWO_GROUP_COMPARISON),
             "exploratory_round": True, "next_round": {"run": True, "predictors": ["region"]}}
    update = make_plan_stages_node(store)(state)
    assert update["exploratory_round"] is False and update["next_round"] is None


# --- full runs through the real graph -----------------------------------------------------------------------------

def _run(store, tmp_path, df, resumes, question_type=QuestionType.TWO_GROUP_COMPARISON, predictors=("region",),
         stage_gate=None, name="run"):
    app = build_agent_graph(store, classify_fn=confident(question_type), draft_fn=fake_draft(list(predictors)),
                            propose_fn=must_not_propose)
    state = {"request_text": "Did West spend more?", "session_id": "session-abc",
             "dataset": _dataset(df, tmp_path, store, name)}
    values, pauses, next_nodes = _drive(app, state, name, resumes, {}, stage_gate=stage_gate, stop=())
    return app, values, pauses, next_nodes


def _points(values):
    return [e["decision_point"] for e in values["decision_log"]]


def _records(values, point):
    return [e for e in values["decision_log"] if e["decision_point"] == point]


def test_skip(fake_model, tmp_path, store):
    _, values, pauses, next_nodes = _run(store, tmp_path, _frame(), [APPROVE], stage_gate={"action": "skip"})
    assert pauses == [GATE, "plan_approval", "stage_gate"] and next_nodes == ()
    assert len(values["rounds"]) == 1
    (gate,) = _records(values, STAGE_GATE_POINT)
    assert (gate["chosen"], gate["decided_by"]) == ("skip", "human")
    assert ROUND_START_POINT not in _points(values)


def test_run(fake_model, tmp_path, store):
    app, values, pauses, next_nodes = _run(store, tmp_path, _frame(), [APPROVE, APPROVE], stage_gate={"action": "run"},
                                           name="explore")
    assert pauses == [GATE, "plan_approval", "stage_gate", "plan_approval"] and next_nodes == ()
    first, second = values["rounds"]
    assert (first["round"], first["capability"], first["exploratory"]) == (1, "two_group", False)
    assert (second["round"], second["capability"], second["exploratory"]) == (2, "regression", True)
    assert second["report"].startswith(EXPLORATORY_NOTE) and values["report"] == second["report"]
    assert not first["report"].startswith(EXPLORATORY_NOTE)

    plan = values["plan"]
    assert plan.status == PlanStatus.EXECUTED and plan.question_type == QuestionType.DRIVER_ANALYSIS
    assert plan.candidate_predictors[0] == "region" and {"tier", "tenure"} <= set(plan.candidate_predictors)
    assert values["test_result"]["result_type"] == "regression"
    assert repr(second["result"]) == repr(values["test_result"])
    assert first["result"].get("result_type", "two_group") == "two_group" and first["result"]["p_value"] is not None
    assert first["plan"].question_type == QuestionType.TWO_GROUP_COMPARISON and first["plan"].id != plan.id

    points = _points(values)
    gate_at = points.index(STAGE_GATE_POINT)
    order = [STAGE_GATE_POINT, ROUND_START_POINT, "stage3.regression_method", "stage3.encoding", "planner.plan_approval"]
    positions = [gate_at] + [points.index(p, gate_at) for p in order[1:]]
    assert positions == sorted(positions)
    assert _records(values, STAGE_GATE_POINT)[0]["chosen"] == "run"
    assert len(_records(values, "planner.plan_approval")) == 2
    assert sum(p == "planner.stage_plan" or p.startswith("planner.step.") for p in points) == 5

    stored = app.get_state({"configurable": {"thread_id": "explore"}}).values["rounds"]
    assert len(stored) == 2
    for entry in stored:
        assert type(entry["plan"]) is AnalysisPlan and type(entry["plan"].question_type) is QuestionType
    assert all(isinstance(p, str) for p in stored[1]["plan"].candidate_predictors)


def test_run_then_reject_round_two(fake_model, tmp_path, store):
    _, values, pauses, next_nodes = _run(store, tmp_path, _frame(), [APPROVE, {"action": "reject", "reason": "no"}],
                                         stage_gate={"action": "run"})
    assert next_nodes == () and len(values["rounds"]) == 1
    assert values["clarification_needed"].startswith("Plan rejected")
    assert _records(values, "planner.plan_approval")[1]["chosen"] == "reject"


def test_run_then_edit_round_two_predictors(fake_model, tmp_path, store):
    """Edit that keeps the same text predictors (region and tier), so the selected encodings still match."""
    edit = {"action": "edit", "candidate_predictors": ["region", "tier"]}
    # M1.3c: the accepted edit re-selects the method, so round two pauses at plan approval a second time
    _, values, pauses, _ = _run(store, tmp_path, _frame(), [APPROVE, edit, APPROVE], stage_gate={"action": "run"})
    assert pauses == [GATE, "plan_approval", "stage_gate", "plan_approval", "plan_approval"]
    assert list(values["plan"].candidate_predictors) == ["region", "tier"]
    edits = [e for e in _records(values, "planner.predictors") if e["decided_by"] == "human"]
    assert len(edits) == 1 and edits[0]["chosen"] == "region, tier"
    assert len(values["rounds"]) == 2 and values["rounds"][1]["plan"].candidate_predictors[-1] == "tier"


def test_run_then_edit_round_two_predictors_removing_a_text_column(fake_model, tmp_path, store):
    """G53 (fixed in M1.3c): the accepted edit re-selects the method, so the encodings match the new predictors."""
    edit = {"action": "edit", "candidate_predictors": ["region", "tenure"]}
    _, values, _, next_nodes = _run(store, tmp_path, _frame(), [APPROVE, edit, APPROVE], stage_gate={"action": "run"})
    assert next_nodes == () and len(values["rounds"]) == 2
    assert list(values["plan"].candidate_predictors) == ["region", "tenure"]
    round_two = _records(values, "stage3.regression_method")
    assert len(round_two) == 2 and round_two[1]["supersedes"] == round_two[0]["id"]


def test_not_offered_without_a_usable_extra_predictor(fake_model, tmp_path, store):
    df = _frame(extra=False)
    df["customer_id"] = range(1, len(df) + 1)  # 3 columns, so the stage plan offers modeling; the id is not usable
    _, values, pauses, next_nodes = _run(store, tmp_path, df, [APPROVE])
    assert pauses == [GATE, "plan_approval"] and next_nodes == ()
    (gate,) = _records(values, STAGE_GATE_POINT)
    assert (gate["chosen"], gate["decided_by"]) == ("not_offered", "code_rule")
    assert len(values["rounds"]) == 1


def test_driver_analysis_has_no_stage_gate(fake_model, tmp_path, store):
    _, values, _, next_nodes = _run(store, tmp_path, _frame(), [APPROVE], question_type=QuestionType.DRIVER_ANALYSIS,
                                    predictors=("region", "tenure"))
    assert STAGE_GATE_POINT not in _points(values) and next_nodes == () and len(values["rounds"]) == 1


def test_descriptive_has_no_stage_gate(fake_model, tmp_path, store):
    _, values, pauses, _ = _run(store, tmp_path, _frame(), [], question_type=QuestionType.DESCRIPTIVE, predictors=())
    assert STAGE_GATE_POINT not in _points(values) and pauses == []


def test_an_unrecognised_answer_skips(fake_model, tmp_path, store):
    _, values, _, next_nodes = _run(store, tmp_path, _frame(), [APPROVE], stage_gate={"action": "maybe"})
    (gate,) = _records(values, STAGE_GATE_POINT)
    assert (gate["chosen"], gate["decided_by"]) == ("skip", "code_rule")
    assert next_nodes == () and len(values["rounds"]) == 1


# --- the causal-language guard runs on round two's model text --------------------------------------------------------

class _ChatByPrompt:
    """Writes 'causes' only in the regression report, so round one passes and round two trips the guard."""
    def invoke(self, messages):
        text = "\n".join(content for _, content in messages)
        content = "Tenure causes higher spend." if "multiple linear regression" in text else "West spends more."
        return type("Response", (), {"content": content})()


def test_causal_check_runs_on_round_two(monkeypatch, tmp_path, store):
    # M1.4a (G52): the guard still runs on round two's model text, but a violation no longer fails the run:
    # after one retry round two's report is written from the computed facts
    monkeypatch.setattr(report_module, "init_chat_model", lambda *a, **k: _ChatByPrompt())
    _, values, _, next_nodes = _run(store, tmp_path, _frame(), [APPROVE, APPROVE], stage_gate={"action": "run"})
    assert next_nodes == () and len(values["rounds"]) == 2
    assert [r["chosen"] for r in _records(values, "report.narrative")] == ["model", "facts_fallback"]
    assert "causes" not in values["rounds"][1]["report"]


# --- the live script's LIVE_ONLY filter --------------------------------------------------------------------------------

def test_filter_runs():
    runs = [("a", None, "q1"), ("b", None, "q2"), ("c", None, "q3")]
    assert filter_runs(runs, []) == runs
    assert filter_runs(runs, ["c", "a"]) == [runs[0], runs[2]]
    with pytest.raises(ValueError, match=r"unknown run\(s\) \['zz'\]"):
        filter_runs(runs, ["a", "zz"])


# === M1.3b step 2b: choosing the predictors at the boundary gate ========================================================


def test_payload_new_keys():
    stage_plan = _stage_plan(QuestionType.TWO_GROUP_COMPARISON)
    profile = make_profile({"region": ("object", 2), "tier": ("object", 3), "customer_id": ("int64", 100),
                            "total_spend": ("float64", 100),
                            **{f"x{i}": ("float64", 50) for i in range(5)}})
    proposal = propose_exploratory_predictors(profile, "region", "total_spend")
    payload = stage_gate_payload(_plan(), {}, "two_group", 1, stage_plan, proposal)
    assert payload["refused"] is None
    assert payload["offer"]["selectable"] == (
        [{"column": "tier", "proposed": True, "reason": "text column with 3 levels"}]
        + [{"column": f"x{i}", "proposed": i < 4, "reason": "numeric column" if i < 4 else
            "beyond the limit of 5 other predictors"} for i in range(5)])
    assert set(payload["offer"]["unsuitable"]) == {"customer_id", "total_spend"}  # x4 is out only by the cap
    refusal = {"submitted": ["region", "customer_id"], "refusals": [{"column": "customer_id"}], "attempt": 1}
    assert stage_gate_payload(_plan(), {}, "two_group", 1, stage_plan, proposal, refused=refusal)["refused"] == refusal


REFUSAL = {"submitted": ["region", "customer_id"], "attempt": 2, "refusals": [
    {"column": "customer_id", "rule": "identifier_like", "reason": "identifier-like: every value is distinct",
     "explanation": "x"},
    {"column": None, "rule": "none_chosen", "reason": "no other predictor", "explanation": "y"}]}


def test_refused_record():
    r = stage_gate_record(_plan(), PROPOSAL, {"action": "run", "predictors": ["region", "customer_id"]}, "refused",
                          True, "two_group", refused_attempts=1, refusal=REFUSAL)
    assert (r.chosen, r.decided_by, r.overridden_by_human) == ("refused", "human", False)
    text = "customer_id: identifier-like: every value is distinct; none_chosen: no other predictor"
    assert r.rationale == "the chosen predictors do not fit the rules: " + text
    assert [(c.option, c.eligible) for c in r.candidates] == [("run", True), ("run with your predictors", False),
                                                              ("skip", True)]
    assert r.candidates[1].reason == text
    assert set(r.evidence) == EVIDENCE_KEYS | {"submitted_predictors", "refusals", "attempt"}
    assert (r.evidence["submitted_predictors"], r.evidence["attempt"], r.evidence["refused_attempts"]) == (
        ["region", "customer_id"], 2, 1)
    assert r.evidence["refusals"] == REFUSAL["refusals"] and r.evidence["chosen_predictors"] == []


def test_accepted_override_record():
    r = stage_gate_record(_plan(), PROPOSAL, {"action": "run", "predictors": ["region", "tenure"]}, "run", True,
                          "two_group", chosen_predictors=["region", "tenure"], refused_attempts=1)
    assert (r.chosen, r.decided_by, r.overridden_by_human) == ("run", "human", True)
    assert r.original_choice == "run with the proposed predictors: region, tier, tenure"
    assert r.rationale == "the user chose their own predictors (they fit the rules)"
    assert (r.evidence["chosen_predictors"], r.evidence["refused_attempts"]) == (["region", "tenure"], 1)


def test_proposal_record_is_unchanged_apart_from_the_new_keys():
    r = stage_gate_record(_plan(), PROPOSAL, {"action": "run"}, "run", True, "two_group",
                          chosen_predictors=PROPOSAL["predictors"])
    assert (r.overridden_by_human, r.original_choice) == (False, None)
    assert r.rationale == "the user chose to run the exploratory regression"
    assert (r.evidence["chosen_predictors"], r.evidence["refused_attempts"]) == (["region", "tier", "tenure"], 0)
    skip = stage_gate_record(_plan(), PROPOSAL, {"action": "skip"}, "skip", True, "two_group")
    assert skip.evidence["chosen_predictors"] == [] and skip.evidence["refused_attempts"] == 0


def test_round_start_record_with_a_user_choice():
    profile = make_profile({"region": ("object", 2), "tier": ("object", 3), "tenure": ("float64", 80),
                            "total_spend": ("float64", 100), **{f"x{i}": ("float64", 50) for i in range(5)}})
    proposal = propose_exploratory_predictors(profile, "region", "total_spend")
    assert "x3" in proposal["excluded"]  # beyond the cap in the proposal
    plan2 = build_exploratory_plan(_plan(), "x@v1", ["region", "tenure", "x3"])
    r = round_start_record(plan2, profile, proposal, from_round=1)
    reasons = {c.option: (c.eligible, c.reason) for c in r.candidates}
    assert reasons["x3"] == (True, "chosen by you; it fits the rules")
    assert reasons["tenure"] == (True, "numeric column")
    assert reasons["tier"] == (False, "text column with 3 levels")
    assert r.chosen == "region, tenure, x3" and r.evidence["chosen_by"] == "user"
    assert "x3" not in r.evidence["excluded"] and r.evidence["predictors"] == ["region", "tenure", "x3"]
    default = round_start_record(build_exploratory_plan(_plan(), "x@v1", proposal["predictors"]), profile, proposal, 1)
    assert default.evidence["chosen_by"] == "code_rule"
    json.dumps(log_update(r, default))


# --- through the real graph ----------------------------------------------------------------------------------------

def _gate_frame(seed=61, n=120):
    df = _frame(seed=seed, n=n)
    df["customer_id"] = range(1, n + 1)
    return df


def _drive_all(app, state, thread, resumes):
    """Answer every pause, in order, with the next value; stop (with the pause still open) when none are left."""
    config = {"configurable": {"thread_id": thread}}
    queue, payloads = list(resumes), []
    app.invoke(state, config)
    snapshot = app.get_state(config)
    while snapshot.interrupts:
        payloads.append(snapshot.interrupts[0].value)
        if not queue:
            break
        app.invoke(Command(resume=queue.pop(0)), config)
        snapshot = app.get_state(config)
    return snapshot.values, payloads, snapshot.next


def _gate_app(store, event_log_dir=None):
    return build_agent_graph(store, classify_fn=confident(QuestionType.TWO_GROUP_COMPARISON),
                             draft_fn=fake_draft(["region"]), propose_fn=must_not_propose, event_log_dir=event_log_dir)


def _start_state(tmp_path, store, df, name):
    return {"request_text": "Did West spend more?", "session_id": "session-abc",
            "dataset": _dataset(df, tmp_path, store, name)}


def _types(payloads):
    return [p["type"] for p in payloads]


def _gate_records(values):
    return _records(values, STAGE_GATE_POINT)


def test_run_with_chosen_predictors(fake_model, tmp_path, store):
    run = {"action": "run", "predictors": ["region", "tenure"]}
    values, payloads, nxt = _drive_all(_gate_app(store), _start_state(tmp_path, store, _gate_frame(), "a"), "a",
                                       [APPROVE, APPROVE, run, APPROVE])
    assert nxt == () and _types(payloads) == [GATE, "plan_approval", "stage_gate", "plan_approval"]
    assert list(values["plan"].candidate_predictors) == ["region", "tenure"]
    (gate,) = _gate_records(values)
    assert (gate["chosen"], gate["overridden_by_human"]) == ("run", True)
    assert _records(values, ROUND_START_POINT)[0]["evidence"]["chosen_by"] == "user"
    assert values["test_result"]["result_type"] == "regression" and len(values["rounds"]) == 2


def test_refusal_then_success(fake_model, tmp_path, store):
    app = _gate_app(store)
    config = {"configurable": {"thread_id": "b"}}
    bad = {"action": "run", "predictors": ["region", "customer_id", "total_spend"]}
    values, payloads, nxt = _drive_all(app, _start_state(tmp_path, store, _gate_frame(), "b"), "b",
                                       [APPROVE, APPROVE, bad])
    assert nxt == ("stage_gate",) and _types(payloads) == [GATE, "plan_approval", "stage_gate", "stage_gate"]
    (refused,) = _gate_records(values)  # saved before the second answer
    assert (refused["chosen"], refused["evidence"]["attempt"]) == ("refused", 1)
    assert [(v["column"], v["rule"]) for v in refused["evidence"]["refusals"]] == [
        ("customer_id", "identifier_like"), ("total_spend", "outcome")]
    assert payloads[-1]["refused"]["refusals"] == refused["evidence"]["refusals"]
    assert payloads[-1]["refused"]["attempt"] == 1

    app.invoke(Command(resume={"action": "run", "predictors": ["region", "tier"]}), config)
    app.invoke(Command(resume=APPROVE), config)
    values = app.get_state(config).values
    assert app.get_state(config).next == ()
    refused_again, accepted = _gate_records(values)
    assert refused_again["chosen"] == "refused"
    assert (accepted["chosen"], accepted["evidence"]["refused_attempts"]) == ("run", 1)
    assert values["stage_gate_refusal"] is None and list(values["plan"].candidate_predictors) == ["region", "tier"]


def test_pause_list_for_a_refusal_then_success(fake_model, tmp_path, store):
    bad = {"action": "run", "predictors": ["region", "customer_id"]}
    good = {"action": "run", "predictors": ["region", "tier"]}
    _, payloads, nxt = _drive_all(_gate_app(store), _start_state(tmp_path, store, _gate_frame(), "p"), "p",
                                  [APPROVE, APPROVE, bad, good, APPROVE])
    assert nxt == () and _types(payloads) == [GATE, "plan_approval", "stage_gate", "stage_gate", "plan_approval"]


def test_two_refusals_count_their_attempts(fake_model, tmp_path, store):
    bad = {"action": "run", "predictors": ["region", "customer_id"]}
    values, payloads, _ = _drive_all(_gate_app(store), _start_state(tmp_path, store, _gate_frame(), "c"), "c",
                                     [APPROVE, APPROVE, bad, bad])
    assert [r["evidence"]["attempt"] for r in _gate_records(values)] == [1, 2]
    assert payloads[-1]["refused"]["attempt"] == 2


def test_refusal_then_skip(fake_model, tmp_path, store):
    bad = {"action": "run", "predictors": ["region", "total_spend"]}
    values, _, nxt = _drive_all(_gate_app(store), _start_state(tmp_path, store, _gate_frame(), "d"), "d",
                                [APPROVE, APPROVE, bad, {"action": "skip"}])
    assert nxt == () and len(values["rounds"]) == 1
    assert [r["chosen"] for r in _gate_records(values)] == ["refused", "skip"]
    assert ROUND_START_POINT not in _points(values) and values["plan"].question_type == QuestionType.TWO_GROUP_COMPARISON


def test_a_refusal_is_saved_even_if_no_answer_follows(fake_model, tmp_path, store):
    events = tmp_path / "events"
    bad = {"action": "run", "predictors": ["region", "customer_id"]}
    values, _, nxt = _drive_all(_gate_app(store, event_log_dir=events), _start_state(tmp_path, store, _gate_frame(), "e"),
                                "e", [APPROVE, APPROVE, bad])
    assert nxt == ("stage_gate",)
    assert [r["chosen"] for r in _gate_records(values)] == ["refused"]
    in_file = [e for e in read_decision_log(events_path(events, "e")) if e["decision_point"] == STAGE_GATE_POINT]
    assert [e["chosen"] for e in in_file] == ["refused"]


def test_omitted_predictors_run_the_proposal(fake_model, tmp_path, store):
    values, _, _ = _drive_all(_gate_app(store), _start_state(tmp_path, store, _gate_frame(), "f"), "f",
                              [APPROVE, APPROVE, {"action": "run"}, APPROVE])
    (gate,) = _gate_records(values)
    assert (gate["chosen"], gate["overridden_by_human"]) == ("run", False)
    assert list(values["plan"].candidate_predictors) == ["region", "tier", "tenure"]
    assert _records(values, ROUND_START_POINT)[0]["evidence"]["chosen_by"] == "code_rule"


@pytest.mark.parametrize("predictors, rule", [("region", "not_a_list"), (["tier", "tenure"], "group_missing")])
def test_invalid_shapes_are_refused(predictors, rule, fake_model, tmp_path, store):
    values, _, nxt = _drive_all(_gate_app(store), _start_state(tmp_path, store, _gate_frame(), rule), rule,
                                [APPROVE, APPROVE, {"action": "run", "predictors": predictors}])
    assert nxt == ("stage_gate",)
    (refused,) = _gate_records(values)
    assert [v["rule"] for v in refused["evidence"]["refusals"]] == [rule]


def _wide_frame(seed=62, n=120):
    df = _frame(seed=seed, n=n, extra=False)
    rng = np.random.default_rng(seed)
    for i in range(8):
        df[f"x{i}"] = rng.normal(0, 1, n)
    return df


def test_the_cap(fake_model, tmp_path, store):
    seven = {"action": "run", "predictors": ["region", *[f"x{i}" for i in range(7)]]}
    values, _, _ = _drive_all(_gate_app(store), _start_state(tmp_path, store, _wide_frame(), "cap"), "cap",
                              [APPROVE, APPROVE, seven])
    (refused,) = _gate_records(values)
    assert refused["evidence"]["refusals"] == [{"column": None, "rule": "cap",
                                                "reason": "beyond the limit of 5 other predictors",
                                                "explanation": "you chose 7 other predictors; the limit is 5; remove 2",
                                                "evidence": {"chosen": 7, "limit": 5}}]  # M1.3c: evidence


def test_a_column_left_out_by_the_cap_can_be_swapped_in(fake_model, tmp_path, store):
    swap = {"action": "run", "predictors": ["region", "x0", "x1", "x2", "x3", "x7"]}
    values, _, nxt = _drive_all(_gate_app(store), _start_state(tmp_path, store, _wide_frame(), "swap"), "swap",
                                [APPROVE, APPROVE, swap, APPROVE])
    assert nxt == ()
    (gate,) = _gate_records(values)
    assert gate["evidence"]["proposed_predictors"] == ["region", "x0", "x1", "x2", "x3", "x4"]
    assert (gate["chosen"], gate["overridden_by_human"]) == ("run", True)
    assert list(values["plan"].candidate_predictors) == ["region", "x0", "x1", "x2", "x3", "x7"]


def test_plan_stages_clears_a_stale_refusal(tmp_path, store):
    ds = _dataset(_frame(), tmp_path, store, "stale_refusal")
    state = {"dataset": ds, "profile": profile_dataset(ds, store=store), "request_text": "x", "session_id": "s",
             "classification": classification(QuestionType.TWO_GROUP_COMPARISON), "stage_gate_refusal": REFUSAL}
    assert make_plan_stages_node(store)(state)["stage_gate_refusal"] is None
