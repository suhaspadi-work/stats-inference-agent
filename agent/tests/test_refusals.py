"""
M1.3c: refusals at the stage-plan gate and at the plan-approval edit, and re-selection after an accepted edit
(fixes G53 and G5). A choice that breaks a rule is a recorded human decision and the same pause re-opens; an
accepted edit is not frozen: the plan is re-routed, its method decisions are cleared, select_method runs again
(new selection records supersede the old ones) and the plan-approval pause shows the recomputed method.
"""
import copy
import json

import numpy as np
import pytest
from langgraph.types import Command

from core.dataset import LocalDiskStore
from core.decision_log import log_update
from core.decision_points import DECISION_NODES, registered_points
from core.detection import CandidateIssue
from core.event_log import events_path, read_decision_log
from core.freeze_records import PLAN_APPROVAL_POINT, plan_approval_record, refused_plan_approval_record
from core.graph import build_agent_graph
from core.ingestion import profile_dataset
from core.model_safe_view import ColumnSummary, ModelSafeView
from core.nodes import make_plan_stages_node, make_start_round_node
from core.plan import PlanStatus, QuestionType
from core.predictor_rules import check_predictor, validate_edit, validate_predictor_choice
from core.routing import route_for_plan
from core.stage_plan import build_stage_plan, plan_steps
from core.stage_plan_gate import check_run_overrides
from core.stage_plan_records import STAGE_PLAN_APPROVAL_POINT, stage_plan_approval_record
from tests.test_decision_point_guard import (
    _dataset, _two_group_df, _with_duplicate, confident, fake_draft, fake_propose, must_not_propose,
)
from tests.test_exploratory_round import _drive_all, _frame, _gate_frame
from tests.test_prompt_canaries import fake_model  # noqa: F401 (fixture)
from tests.test_routing_records import make_plan
from tests.test_stage_plan import FOUR, classification

APPROVE = {"action": "approve"}
SKIP = {"action": "skip"}
SPA = "stage_plan_approval"


def _profile(*columns, row_count=100):
    """columns: (name, dtype, distinct_count, missing_count)"""
    return ModelSafeView(dataset_handle="x@v1", row_count=row_count, column_count=len(columns), columns=tuple(
        ColumnSummary(name=n, dtype=d, missing_count=m, missing_pct=m / row_count, distinct_count=k)
        for n, d, k, m in columns))


PROFILE = _profile(("region", "object", 2, 0), ("tier", "object", 3, 0), ("tenure", "float64", 80, 0),
                   ("customer_id", "int64", 96, 4), ("flag", "float64", 1, 0), ("city", "object", 11, 0),
                   ("total_spend", "float64", 100, 0))
TWO_GROUP = make_plan(QuestionType.TWO_GROUP_COMPARISON, ["region"])


def _rules(verdicts):
    return [(v.get("column", v.get("step")), v["rule"]) for v in verdicts]


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _records(values, point):
    return [e for e in values["decision_log"] if e["decision_point"] == point]


def _points(values):
    return [e["decision_point"] for e in values["decision_log"]]


def _types(payloads):
    return [p["type"] for p in payloads]


# --- verdict evidence ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("name, rule, evidence", [
    ("colour", "unknown_column", {"available_columns": sorted(c.name for c in PROFILE.columns)}),
    ("total_spend", "outcome", {"dtype": "float64", "distinct_count": 100, "row_count": 100, "missing_count": 0}),
    ("flag", "constant", {"dtype": "float64", "distinct_count": 1, "row_count": 100, "missing_count": 0}),
    ("customer_id", "identifier_like", {"dtype": "int64", "distinct_count": 96, "row_count": 100, "missing_count": 4}),
    ("city", "too_many_levels", {"dtype": "object", "distinct_count": 11, "row_count": 100, "missing_count": 0}),
])
def test_check_predictor_evidence(name, rule, evidence):
    v = check_predictor(PROFILE, name, "total_spend")
    assert (v["rule"], v["evidence"]) == (rule, evidence)


def test_choice_and_edit_evidence():
    choice = validate_predictor_choice(PROFILE, ["tenure", "tenure"], "total_spend", "region", 5)["refusals"]
    assert [(v["rule"], v["evidence"]) for v in choice] == [("group_missing", {"group_column": "region"}),
                                                            ("duplicate", {"count": 2})]
    wide = _profile(("region", "object", 2, 0), ("total_spend", "float64", 100, 0),
                    *[(f"x{i}", "float64", 50, 0) for i in range(7)])
    (cap,) = validate_predictor_choice(wide, ["region", *[f"x{i}" for i in range(7)]], "total_spend", "region",
                                       5)["refusals"]
    assert cap["evidence"] == {"chosen": 7, "limit": 5}
    assert validate_predictor_choice(PROFILE, ["region"], "total_spend", "region", 5)["refusals"][0]["evidence"] == {}
    assert validate_edit(PROFILE, TWO_GROUP, "tier", "total_spend")["refusals"][0]["evidence"] == {}
    assert validate_edit(PROFILE, TWO_GROUP, ["tier", "tier"], "total_spend")["refusals"][0]["evidence"] == {"count": 2}
    json.dumps(choice)


# --- validate_edit -----------------------------------------------------------------------------------------------

def test_validate_edit_ok_returns_the_route():
    result = validate_edit(PROFILE, TWO_GROUP, ["region"], "total_spend")
    assert (result["ok"], result["predictors"], result["refusals"]) == (True, ["region"], [])
    assert result["route"]["status"] == "ok" and result["route"]["capability"] == "two_group"


@pytest.mark.parametrize("name, rule", [("colour", "unknown_column"), ("total_spend", "outcome"), ("flag", "constant"),
                                        ("customer_id", "identifier_like"), ("city", "too_many_levels")])
def test_validate_edit_refuses_each_rule(name, rule):
    result = validate_edit(PROFILE, TWO_GROUP, [name], "total_spend")
    assert result == {"ok": False, "predictors": [], "refusals": [check_predictor(PROFILE, name, "total_spend")]}
    assert result["refusals"][0]["rule"] == rule


def test_validate_edit_duplicates_and_empty():
    assert _rules(validate_edit(PROFILE, TWO_GROUP, ["region", "region", "region"], "total_spend")["refusals"]) == [
        ("region", "duplicate")]
    assert _rules(validate_edit(PROFILE, TWO_GROUP, [], "total_spend")["refusals"]) == [(None, "none_chosen")]


def test_two_predictors_on_a_comparison_is_route_unsupported():
    result = validate_edit(PROFILE, TWO_GROUP, ["region", "tier"], "total_spend")
    (v,) = result["refusals"]
    expected = route_for_plan(TWO_GROUP.refine(candidate_predictors=("region", "tier")), PROFILE)
    assert (v["column"], v["rule"], v["reason"]) == (None, "route_unsupported", expected["reason"])
    assert v["explanation"] == expected["message"] and "exactly one grouping column" in v["explanation"]
    assert v["evidence"] == {"question_type": "two_group_comparison", "predictors": ["region", "tier"]}


def test_the_route_is_checked_only_without_an_earlier_refusal():
    result = validate_edit(PROFILE, TWO_GROUP, ["region", "customer_id"], "total_spend")
    assert _rules(result["refusals"]) == [("customer_id", "identifier_like")]


def test_a_three_level_column_on_a_two_group_plan_is_reconciled():
    before = copy.deepcopy(TWO_GROUP)
    result = validate_edit(PROFILE, TWO_GROUP, ["tier"], "total_spend")
    assert result["ok"] is True
    assert (result["route"]["status"], result["route"]["capability"], result["route"]["question_type"]) == (
        "reconciled", "multi_group", "multi_group_comparison")
    assert TWO_GROUP == before


# --- check_run_overrides -------------------------------------------------------------------------------------------

def _issue():
    return CandidateIssue(issue_type="duplicate_rows", description="d", affected_columns=("region",), evidence={})


def _step(plan, name):
    return next(s for s in plan_steps(plan) if s["step"] == name)


def _evidence(plan, name):
    s = _step(plan, name)
    return {"state": s["state"], "run": s["run"], **s["evidence"]}


def test_every_override_rule():
    comparison = build_stage_plan(classification(QuestionType.TWO_GROUP_COMPARISON), FOUR, [_issue()])
    driver = build_stage_plan(classification(QuestionType.DRIVER_ANALYSIS), FOUR, [])
    descriptive = build_stage_plan(classification(QuestionType.DESCRIPTIVE), FOUR, [])

    def one(plan, overrides):
        (v,) = check_run_overrides(plan, overrides)["refusals"]
        return v

    assert one(comparison, {"clean": False}) == {
        "step": "clean", "rule": "planned_step", "reason": "needed by your question",
        "explanation": "'clean' is needed to answer your question, so it cannot be switched off",
        "evidence": _evidence(comparison, "clean")}
    assert one(driver, {"inference": True}) == {
        "step": "inference", "rule": "unavailable", "reason": "not built yet (M3.3)",
        "explanation": "'inference' would apply to your question but is not built yet; it arrives with M3.3",
        "evidence": _evidence(driver, "inference")}
    assert one(descriptive, {"inference": True}) == {
        "step": "inference", "rule": "not_applicable", "reason": "does not apply to this question",
        "explanation": _step(descriptive, "inference")["reason"], "evidence": _evidence(descriptive, "inference")}
    assert one(comparison, {"modeling": True}) == {
        "step": "modeling", "rule": "chosen_later", "reason": "chosen after the main analysis",
        "explanation": "you will be asked about this step after the main analysis, if the data allows it",
        "evidence": _evidence(comparison, "modeling")}
    assert one(comparison, {"describe": 1}) == {
        "step": "describe", "rule": "not_boolean", "reason": "must be true or false",
        "explanation": "the choice for 'describe' must be true or false", "evidence": _evidence(comparison, "describe")}
    assert one(comparison, {"forecast": True}) == {
        "step": "forecast", "rule": "unknown_step", "reason": "no such step",
        "explanation": "'forecast' is not a step of this plan; the steps are clean, describe, inference, modeling",
        "evidence": {}}
    assert one(comparison, ["describe"]) == {
        "step": None, "rule": "not_a_dict", "reason": 'run choices must be an object like {"describe": true}',
        "explanation": 'run choices must be an object like {"describe": true}', "evidence": {}}


def test_every_override_problem_is_collected():
    plan = build_stage_plan(classification(QuestionType.TWO_GROUP_COMPARISON), FOUR, [_issue()])
    checked = check_run_overrides(plan, {"forecast": True, "clean": False, "describe": "no"})
    assert checked["ok"] is False
    assert _rules(checked["refusals"]) == [("forecast", "unknown_step"), ("clean", "planned_step"),
                                           ("describe", "not_boolean")]
    assert check_run_overrides(plan, {"describe": False}) == {"ok": True, "overrides": {"describe": False},
                                                               "refusals": []}


# --- records ---------------------------------------------------------------------------------------------------------

STEP_REFUSAL = {"submitted": {"clean": False}, "attempt": 2, "refusals": [
    {"step": "clean", "rule": "planned_step", "reason": "needed by your question", "explanation": "x", "evidence": {}}]}
EDIT_REFUSAL = {"submitted": ["customer_id"], "attempt": 1, "refusals": [
    {"column": "customer_id", "rule": "identifier_like", "reason": "identifier-like: every value is distinct",
     "explanation": "x", "evidence": {}}]}


def test_refused_stage_plan_approval_record():
    plan = build_stage_plan(classification(QuestionType.TWO_GROUP_COMPARISON), FOUR, [_issue()])
    r = stage_plan_approval_record(plan, {"action": "approve", "run": {"clean": False}}, {}, "refused",
                                   refusal=STEP_REFUSAL)
    assert (r.decision_point, r.chosen, r.decided_by) == (STAGE_PLAN_APPROVAL_POINT, "refused", "human")
    assert [(c.option, c.eligible) for c in r.candidates] == [
        ("approve", True), ("approve_with_changes", True), ("reject", True), ("approve with your choices", False)]
    assert r.candidates[-1].reason == "clean: needed by your question"
    assert r.rationale == "the chosen run flags do not fit the rules: clean: needed by your question"
    assert (r.evidence["submitted"], r.evidence["refusals"], r.evidence["attempt"]) == (
        {"clean": False}, STEP_REFUSAL["refusals"], 2)
    assert r.evidence["run_after"] == r.evidence["run_before"]


def test_refused_plan_approval_record():
    r = refused_plan_approval_record(TWO_GROUP, {"action": "edit", "candidate_predictors": ["customer_id"]},
                                     EDIT_REFUSAL, "sel-1")
    assert (r.decision_point, r.chosen, r.decided_by) == (PLAN_APPROVAL_POINT, "refused", "human")
    assert [(c.option, c.eligible) for c in r.candidates][-1] == ("edit with your predictors", False)
    assert r.rationale == "the edited predictors do not fit the rules: customer_id: identifier-like: every value is distinct"
    assert (r.evidence["submitted_predictors"], r.evidence["attempt"], r.evidence["frozen"], r.evidence["reselects"]) == (
        ["customer_id"], 1, False, False)
    assert r.evidence["refusals"] == EDIT_REFUSAL["refusals"] and r.evidence["selection_record_id"] == "sel-1"


def test_plan_approval_record_frozen_and_reselects():
    edit = plan_approval_record(TWO_GROUP, {"action": "edit", "candidate_predictors": ["region"]}, "sel-1")
    approve = plan_approval_record(TWO_GROUP, APPROVE, "sel-1")
    assert (edit.evidence["frozen"], edit.evidence["reselects"]) == (False, True)
    assert (approve.evidence["frozen"], approve.evidence["reselects"]) == (True, False)
    records = [edit, approve, refused_plan_approval_record(TWO_GROUP, {"action": "edit"}, EDIT_REFUSAL, None)]
    assert len({r.id for r in records}) == 3
    json.dumps(log_update(*records))


# --- the stage-plan gate through the real graph ----------------------------------------------------------------------

def _messy(seed):
    df = _two_group_df(np.random.default_rng(seed))
    df["notes"] = [None if i % 6 == 0 else "ok" for i in range(len(df))]
    return _with_duplicate(df)


def _gate_app(store, propose_fn=fake_propose, question_type=QuestionType.TWO_GROUP_COMPARISON, predictors=("region",),
              event_log_dir=None):
    return build_agent_graph(store, classify_fn=confident(question_type), draft_fn=fake_draft(list(predictors)),
                             propose_fn=propose_fn, event_log_dir=event_log_dir)


def _start(tmp_path, store, df, name):
    return {"request_text": "x", "session_id": "session-abc", "dataset": _dataset(df, tmp_path, store, name)}


BAD_RUN = {"action": "approve", "run": {"clean": False}}


def test_stage_plan_refusal_reopens_the_gate(tmp_path, store):
    values, payloads, nxt = _drive_all(_gate_app(store), _start(tmp_path, store, _messy(71), "sa"), "sa", [BAD_RUN])
    assert nxt == ("stage_plan_gate",) and _types(payloads) == [SPA, SPA]
    (refused,) = _records(values, STAGE_PLAN_APPROVAL_POINT)
    assert (refused["chosen"], refused["decided_by"]) == ("refused", "human")
    assert [v["rule"] for v in payloads[-1]["refused"]["refusals"]] == ["planned_step"]
    assert not any(p.startswith("stage1.clean") for p in _points(values))


def test_stage_plan_refusal_then_acceptance_continues(tmp_path, store):
    good = {"action": "approve", "run": {"describe": False}}
    values, payloads, nxt = _drive_all(_gate_app(store), _start(tmp_path, store, _messy(72), "sb"), "sb", [BAD_RUN, good])
    assert _types(payloads) == [SPA, SPA, "wrangling_approval"] and nxt == ("wrangle_review",)
    refused, accepted = _records(values, STAGE_PLAN_APPROVAL_POINT)
    assert (refused["chosen"], accepted["chosen"]) == ("refused", "approve_with_changes")
    assert values["stage_plan"]["approval"]["status"] == "approved_with_changes"
    assert values["stage_plan_refusal"] is None


def test_three_problems_in_one_refusal(tmp_path, store):
    bad = {"action": "approve", "run": {"forecast": True, "clean": False, "describe": "no"}}
    values, _, _ = _drive_all(_gate_app(store), _start(tmp_path, store, _messy(73), "sc"), "sc", [bad])
    (refused,) = _records(values, STAGE_PLAN_APPROVAL_POINT)
    assert [v["rule"] for v in refused["evidence"]["refusals"]] == ["unknown_step", "planned_step", "not_boolean"]


def test_stage_plan_refusal_attempts(tmp_path, store):
    values, payloads, _ = _drive_all(_gate_app(store), _start(tmp_path, store, _messy(74), "sd"), "sd", [BAD_RUN, BAD_RUN])
    assert [r["evidence"]["attempt"] for r in _records(values, STAGE_PLAN_APPROVAL_POINT)] == [1, 2]
    assert payloads[-1]["refused"]["attempt"] == 2


def test_stage_plan_refusal_then_reject(tmp_path, store):
    values, _, nxt = _drive_all(_gate_app(store), _start(tmp_path, store, _messy(75), "se"), "se",
                                [BAD_RUN, {"action": "reject", "reason": "no"}])
    assert nxt == () and [r["chosen"] for r in _records(values, STAGE_PLAN_APPROVAL_POINT)] == ["refused", "reject"]


def test_stage_plan_refusal_is_in_the_event_log(tmp_path, store):
    events = tmp_path / "events"
    _drive_all(_gate_app(store, event_log_dir=events), _start(tmp_path, store, _messy(76), "sf"), "sf", [BAD_RUN])
    in_file = [e for e in read_decision_log(events_path(events, "sf")) if e["decision_point"] == STAGE_PLAN_APPROVAL_POINT]
    assert [e["chosen"] for e in in_file] == ["refused"]


def test_plan_stages_and_start_round_clear_the_refusal_keys(tmp_path, store):
    ds = _dataset(_frame(), tmp_path, store, "clear")
    stale = {"stage_plan_refusal": STEP_REFUSAL, "plan_approval_refusal": EDIT_REFUSAL,
             "reselect_after_edit": {"supersedes": "x"}}
    state = {"dataset": ds, "profile": profile_dataset(ds, store=store), "request_text": "x", "session_id": "s",
             "classification": classification(QuestionType.TWO_GROUP_COMPARISON), **stale}
    update = make_plan_stages_node(store)(state)
    assert [update[k] for k in stale] == [None, None, None]
    round_state = {"plan": TWO_GROUP, "dataset": ds, "rounds": [{}], "decision_log": [],
                   "next_round": {"run": True, "predictors": ["region", "tenure"]}, **stale}
    update = make_start_round_node(store)(round_state)
    assert (update["plan_approval_refusal"], update["reselect_after_edit"]) == (None, None)


# --- the plan-approval edit through the real graph ----------------------------------------------------------------------

def _edit(predictors):
    return {"action": "edit", "candidate_predictors": predictors}


def test_a_refused_edit_reopens_plan_approval(fake_model, tmp_path, store):
    values, payloads, nxt = _drive_all(_gate_app(store, must_not_propose), _start(tmp_path, store, _gate_frame(), "pa"),
                                       "pa", [APPROVE, _edit(["customer_id"])])
    assert nxt == ("freeze",) and _types(payloads) == [SPA, "plan_approval", "plan_approval"]
    refused = _records(values, PLAN_APPROVAL_POINT)[-1]
    assert (refused["chosen"], refused["decided_by"]) == ("refused", "human")
    assert [v["rule"] for v in payloads[-1]["refused"]["refusals"]] == ["identifier_like"]
    assert not [e for e in _records(values, "planner.predictors") if e["decided_by"] == "human"]
    assert values["plan"].status != PlanStatus.FROZEN and "test_result" not in values


def test_a_refused_edit_then_an_accepted_one(fake_model, tmp_path, store):
    values, payloads, nxt = _drive_all(_gate_app(store, must_not_propose), _start(tmp_path, store, _gate_frame(), "pb"),
                                       "pb", [APPROVE, _edit(["customer_id"]), _edit(["region"])])
    assert nxt == ("freeze",) and _types(payloads) == [SPA, "plan_approval", "plan_approval", "plan_approval"]
    assert [r["chosen"] for r in _records(values, PLAN_APPROVAL_POINT)] == ["refused", "edit"]
    assert payloads[-1]["refused"] is None and payloads[-1]["reselected"] is True
    assert values["plan_approval_refusal"] is None


def test_two_predictors_on_a_comparison_are_refused_by_route(fake_model, tmp_path, store):
    values, payloads, _ = _drive_all(_gate_app(store, must_not_propose), _start(tmp_path, store, _gate_frame(), "pc"),
                                     "pc", [APPROVE, _edit(["region", "tier"])])
    (v,) = payloads[-1]["refused"]["refusals"]
    assert v["rule"] == "route_unsupported" and "exactly one grouping column" in v["explanation"]


def test_driver_edit_reselects_and_executes(fake_model, tmp_path, store):
    """G53: removing a text predictor at plan approval re-selects the method, so the encodings match."""
    app = _gate_app(store, must_not_propose, QuestionType.DRIVER_ANALYSIS, ("region", "tier", "tenure"))
    values, payloads, nxt = _drive_all(app, _start(tmp_path, store, _frame(), "pd"), "pd",
                                       [APPROVE, _edit(["region", "tenure"])])
    assert nxt == ("freeze",) and _types(payloads) == [SPA, "plan_approval", "plan_approval"]
    plan = values["plan"]
    assert plan.status == PlanStatus.REFINED and list(plan.candidate_predictors) == ["region", "tenure"]
    assert len(plan.method_decisions) == 1 and plan.primary_method is not None
    assert payloads[-1]["reselected"] is True and payloads[-1]["method"] == plan.primary_method
    methods = _records(values, "stage3.regression_method")
    assert len(methods) == 2 and methods[1]["supersedes"] == methods[0]["id"]
    encodings = _records(values, "stage3.encoding")
    first_round = [e for e in encodings if e["supersedes"] is None]
    assert len(first_round) == 2 and len(encodings) == 3  # region and tier at first, then region alone
    assert encodings[-1]["supersedes"] == first_round[-1]["id"]

    config = {"configurable": {"thread_id": "pd"}}
    app.invoke(Command(resume=APPROVE), config)
    final = app.get_state(config)
    assert final.next == () and final.values["test_result"]["result_type"] == "regression"
    assert final.values["plan"].status == PlanStatus.EXECUTED


def test_a_three_level_edit_reconciles_to_multi_group(fake_model, tmp_path, store):
    values, payloads, _ = _drive_all(_gate_app(store, must_not_propose), _start(tmp_path, store, _gate_frame(), "pe"),
                                     "pe", [APPROVE, _edit(["tier"])])
    assert values["plan"].question_type == QuestionType.MULTI_GROUP_COMPARISON
    assert values["route"]["capability"] == "multi_group"
    selections = _records(values, "stage2.test_selection")
    assert len(selections) == 2 and selections[1]["supersedes"] == selections[0]["id"]
    assert selections[1]["chosen"] in ("one_way_anova", "kruskal_wallis")
    assert payloads[-1]["question_type"] == "multi_group_comparison"


def test_a_refused_edit_then_reject(fake_model, tmp_path, store):
    values, _, nxt = _drive_all(_gate_app(store, must_not_propose), _start(tmp_path, store, _gate_frame(), "pf"), "pf",
                                [APPROVE, _edit(["customer_id"]), {"action": "reject", "reason": "no"}])
    assert nxt == () and [r["chosen"] for r in _records(values, PLAN_APPROVAL_POINT)] == ["refused", "reject"]
    assert values["clarification_needed"] == "Plan rejected: no"


def test_attempts_restart_in_the_exploratory_round(fake_model, tmp_path, store):
    answers = [APPROVE, _edit(["customer_id"]), APPROVE, {"action": "run"}, _edit(["region", "customer_id"]), APPROVE]
    values, _, nxt = _drive_all(_gate_app(store, must_not_propose), _start(tmp_path, store, _gate_frame(), "pg"), "pg",
                                answers)
    assert nxt == () and len(values["rounds"]) == 2
    refused = [r for r in _records(values, PLAN_APPROVAL_POINT) if r["chosen"] == "refused"]
    assert [r["evidence"]["attempt"] for r in refused] == [1, 1]


def test_a_refused_edit_is_saved_even_if_no_answer_follows(fake_model, tmp_path, store):
    events = tmp_path / "events"
    values, _, nxt = _drive_all(_gate_app(store, must_not_propose, event_log_dir=events),
                                _start(tmp_path, store, _gate_frame(), "ph"), "ph", [APPROVE, _edit(["customer_id"])])
    assert nxt == ("freeze",) and _records(values, PLAN_APPROVAL_POINT)[-1]["chosen"] == "refused"
    in_file = [e for e in read_decision_log(events_path(events, "ph")) if e["decision_point"] == PLAN_APPROVAL_POINT]
    assert [e["chosen"] for e in in_file] == ["refused"]


def test_an_unrecognised_action_approves_and_freezes(fake_model, tmp_path, store):
    values, payloads, _ = _drive_all(_gate_app(store, must_not_propose), _start(tmp_path, store, _gate_frame(), "pi"),
                                     "pi", [APPROVE, {"action": "maybe"}, SKIP])
    approval = _records(values, PLAN_APPROVAL_POINT)[0]
    assert (approval["chosen"], approval["decided_by"]) == ("approve", "code_rule")
    assert values["plan"].status == PlanStatus.EXECUTED and values["test_result"]
    assert _types(payloads) == [SPA, "plan_approval", "stage_gate"]


# --- edge cases ------------------------------------------------------------------------------------------------------------

def test_an_edit_with_the_same_predictors_reselects_once(fake_model, tmp_path, store):
    values, payloads, nxt = _drive_all(_gate_app(store, must_not_propose), _start(tmp_path, store, _gate_frame(), "ea"),
                                       "ea", [APPROVE, _edit(["region"]), APPROVE, SKIP])
    assert nxt == () and _types(payloads) == [SPA, "plan_approval", "plan_approval", "stage_gate"]
    assert [r["chosen"] for r in _records(values, PLAN_APPROVAL_POINT)] == ["edit", "approve"]
    assert len(_records(values, "stage2.test_selection")) == 2
    assert values["plan"].status == PlanStatus.EXECUTED


@pytest.mark.parametrize("answers, rounds", [
    ([APPROVE, APPROVE, SKIP], 1),
    ([APPROVE, APPROVE, {"action": "run"}, APPROVE], 2),
    ([APPROVE, APPROVE, {"action": "run", "predictors": ["region", "customer_id"]}, {"action": "run"}, APPROVE], 2),
], ids=["skip", "run", "refusal_then_run"])
def test_boundary_gate_flows_still_work(answers, rounds, fake_model, tmp_path, store):
    values, _, nxt = _drive_all(_gate_app(store, must_not_propose), _start(tmp_path, store, _gate_frame(), "bg"), "bg",
                                answers)
    assert nxt == () and len(values["rounds"]) == rounds


# --- graph edges and registry ------------------------------------------------------------------------------------------------

def test_graph_edges_and_registry(tmp_path):
    edges = {(e.source, e.target): e.conditional
             for e in build_agent_graph(LocalDiskStore(base_dir=tmp_path)).get_graph().edges}
    assert edges[("stage_plan_gate", "stage_plan_gate")] is True
    assert edges[("freeze", "freeze")] is True and edges[("freeze", "select_method")] is True
    assert len(registered_points()) == 32 and len(DECISION_NODES) == 18  # M2.4a: replan (planner.stage_replan)
