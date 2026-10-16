"""
M1.4b step 2b: "how this was chosen" per analysis round and the overall summary. report_node adds one
stage_reports entry per analysis round, composed by code from that round's records (core.stage_reports
analysis_report); the summary node adds one "summary" entry after the last report. No model call, no new pause,
no new decision point. Expected names, numbers and predictors come from the run's own plan, result and records;
the fixed sentences of the composers are pinned as written.
"""
import json
import logging

import numpy as np
import pandas as pd
import pytest

import core.nodes as nodes_module
import core.report as report_module
from core.dataset import LocalDiskStore
from core.decision_log import log_update
from core.decision_points import DECISION_NODES, NODE_STAGES, NON_CHOOSING_NODES, registered_points
from core.graph import build_agent_graph
from core.method_glossary import GLOSSARY, render_glossary
from core.nodes import make_summary_node, route_after_report, route_after_stage_gate
from core.plan import QuestionType
from core.report_records import report_narrative_record
from core.stage_reports import analysis_report, make_stage_report_entry, method_name, round_records
from tests.test_decision_point_guard import confident, fake_draft, fake_propose, must_not_propose
from tests.test_exploratory_round import _frame
from tests.test_prompt_canaries import (
    _base_df, _with_duplicates, canary_hits, fake_model,  # noqa: F401 (fixture)
)
from tests.test_stage1_report import _drive, _messy_frame, _reject_filter, _start

ENTRY_KEYS = {"stage", "round", "title", "text", "sections", "fallback", "exploratory", "source_records"}
ANALYSIS_KEYS = ENTRY_KEYS | {"explain_options"}  # M1.4b step 3: analysis entries also list explain_options
GLOSSARY_DETAIL = ["When the agent uses it", "What it assumes", "How to read the result", "Other options"]
TWO_GROUP_HEADINGS = ["About the test", *GLOSSARY_DETAIL, "Why this test", "Your plan decision", "Narrative"]
SUMMARY_HEADINGS = ["Question", "Stage 1", "Results", "Caveats"]
BAD = "Region causes higher spend."
CLEAN = "West spends more on average; region is associated with spend."
NARRATIVE = {
    "model": "The narrative was written by the model.",
    "facts_fallback": "The model's text used causal language twice, so the narrative was written from the computed facts.",
    "minimal_fallback": "The narrative could not be written safely, so a minimal text was used.",
}


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


@pytest.fixture
def report_texts(monkeypatch):
    """The report model answers with the given texts in order (the last one repeats)."""
    texts = []

    class Chat:
        def invoke(self, messages):
            text = texts.pop(0) if len(texts) > 1 else texts[0]
            return type("Response", (), {"content": text})()
    monkeypatch.setattr(report_module, "init_chat_model", lambda *a, **k: Chat())
    return texts


def _app(store, question_type=QuestionType.TWO_GROUP_COMPARISON, predictors=("region",), propose_fn=must_not_propose,
         classify_fn=None):
    return build_agent_graph(store, classify_fn=classify_fn or confident(question_type),
                             draft_fn=fake_draft(list(predictors)), propose_fn=propose_fn)


def _run(store, tmp_path, df, name, question_type=QuestionType.TWO_GROUP_COMPARISON, predictors=("region",),
         propose_fn=must_not_propose, **overrides):
    return _drive(_app(store, question_type, predictors, propose_fn), _start(df, tmp_path, store, name), name,
                  **overrides)


def _of(values, stage):
    return [e for e in values.get("stage_reports") or [] if e["stage"] == stage]


def _analysis(values):
    entries = _of(values, "analysis")
    assert entries, "a finished analysis round must have its entry"
    return entries


def _summary(values):
    (entry,) = _of(values, "summary")
    return entry


def _sections(entry):
    return {s["heading"]: s["text"] for s in entry["sections"]}


def _lines(entry, heading):
    return _sections(entry)[heading].split("\n")


def _records(values, point):
    return [r for r in values["decision_log"] if r["decision_point"] == point]


def _with_id(n=120, seed=51):
    df = _frame(seed=seed, n=n)
    df.insert(0, "customer_id", range(1, n + 1))
    return df


def _three_groups(means, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"region": [g for g in ("East", "North", "West") for _ in range(60)],
                         "total_spend": np.concatenate([rng.normal(m, 10, 60) for m in means])})


# --- the entry shape ---------------------------------------------------------------------------------------------------

def test_the_entry_shape(fake_model, tmp_path, store):
    values, _, _ = _run(store, tmp_path, _frame(), "shape")
    (entry,) = _analysis(values)
    method = values["plan"].primary_method
    assert set(entry) == ANALYSIS_KEYS
    assert (entry["stage"], entry["round"], entry["exploratory"], entry["fallback"]) == ("analysis", 1, False, False)
    assert entry["title"] == f"Analysis (round 1): {GLOSSARY[method]['name']}"
    assert [s["heading"] for s in entry["sections"]] == TWO_GROUP_HEADINGS
    assert [s.get("collapsed") for s in entry["sections"]] == [None, True, True, True, True, None, None, None]
    assert all("collapsed" not in s for s in entry["sections"] if s["heading"] not in GLOSSARY_DETAIL)
    g = GLOSSARY[method]
    assert _sections(entry)["About the test"] == f"{g['name']}. {g['what']} It answers: {g['answers']}"
    rendered = {s["heading"]: s["text"] for s in render_glossary(method)["sections"]}
    assert all(_sections(entry)[h] == rendered[h] for h in GLOSSARY_DETAIL)
    ids = [r["id"] for r in values["decision_log"]]
    assert entry["source_records"] and set(entry["source_records"]) <= set(ids)
    assert entry["source_records"] == [i for i in ids if i in entry["source_records"]]  # log order
    assert entry["text"].startswith(entry["title"] + "\n\nAbout the test\n")
    json.dumps(entry)


# --- Why this test -------------------------------------------------------------------------------------------------------

def test_why_this_test_for_two_groups(fake_model, tmp_path, store):
    values, _, _ = _run(store, tmp_path, _frame(), "why")
    (selection,) = _records(values, "stage2.test_selection")
    lines = _lines(_analysis(values)[0], "Why this test")
    assert lines[0] == selection["rationale"]
    others = [c for c in selection["candidates"] if c["option"] != selection["chosen"]]
    assert len(others) == 2
    assert lines[1:] == [f"Not used: {GLOSSARY[c['option']]['name']}: {c['reason']}" for c in others]


@pytest.mark.parametrize("means, seed, follow_up", [((50, 60, 70), 81, True), ((50, 50, 50), 82, False)],
                         ids=["significant", "not_significant"])
def test_why_this_test_for_multi_group(means, seed, follow_up, fake_model, tmp_path, store):
    values, _, _ = _run(store, tmp_path, _three_groups(means, seed), "multi", QuestionType.MULTI_GROUP_COMPARISON)
    (post_hoc,) = _records(values, "stage2.post_hoc")
    lines = _lines(_analysis(values)[0], "Why this test")
    if follow_up:
        assert values["test_result"]["p_value"] < 0.05 and post_hoc["chosen"] in ("tukey_hsd", "dunn")
        assert f"Follow-up: {GLOSSARY[post_hoc['chosen']]['name']}: {post_hoc['rationale']}" in lines
        assert GLOSSARY[post_hoc["chosen"]]["name"] in ("Tukey's HSD", "Dunn's test")
    else:
        assert values["test_result"]["p_value"] >= 0.05 and post_hoc["chosen"] == "none"
        assert f"No pairwise follow-up ran: {post_hoc['rationale']}" in lines
        assert not any(line.startswith("Follow-up:") for line in lines)


def test_why_this_test_for_a_regression_with_a_text_predictor(fake_model, tmp_path, store):
    values, _, _ = _run(store, tmp_path, _frame(), "reg", QuestionType.DRIVER_ANALYSIS, ("tenure", "tier"))
    entry = _analysis(values)[0]
    (method,) = _records(values, "stage3.regression_method")
    encodings = _records(values, "stage3.encoding")
    assert [r["evidence"]["column"] for r in encodings] == ["tier"]
    lines = _lines(entry, "Why this test")
    assert lines[0] == method["rationale"]
    assert [line for line in lines if line.startswith("tier: ")] == [f"tier: {encodings[0]['rationale']}"]
    about = next(s for s in entry["sections"] if s["heading"] == "About Treatment coding")
    t = GLOSSARY["treatment"]
    assert about == {"heading": "About Treatment coding", "text": f"{t['what']} How to read it: {t['reading']}",
                     "collapsed": True}
    headings = [s["heading"] for s in entry["sections"]]
    assert headings.index("Why this test") < headings.index("About Treatment coding") < headings.index("Your plan decision")


def test_why_this_test_for_a_numeric_only_regression(fake_model, tmp_path, store):
    values, _, _ = _run(store, tmp_path, _frame(), "numeric", QuestionType.DRIVER_ANALYSIS, ("tenure",))
    entry = _analysis(values)[0]
    assert _records(values, "stage3.encoding") == []
    lines = _lines(entry, "Why this test")
    assert lines[0] == _records(values, "stage3.regression_method")[0]["rationale"]
    assert not any(s["heading"].startswith("About ") and s["heading"] != "About the test" for s in entry["sections"])
    assert not any(line.startswith("tenure: ") for line in lines)


# --- missing records ---------------------------------------------------------------------------------------------------

def _narrative(chosen="model", round_index=1, attempts=None):
    attempts = attempts or [{"attempt": 1, "passed": True, "violation": None}]
    return log_update(report_narrative_record(round_index, "two_group", False, attempts, chosen))["decision_log"][0]


def test_no_selection_and_no_plan_decision():
    composed = analysis_report([_narrative()], False)
    sections = {s["heading"]: s["text"] for s in composed["sections"]}
    assert list(sections) == ["Why this test", "Your plan decision", "Narrative"]
    assert sections["Why this test"] == "No method selection was recorded."
    assert sections["Your plan decision"] == "No plan decision was recorded."
    assert sections["Narrative"] == NARRATIVE["model"]


def test_the_last_record_must_be_the_narrative():
    with pytest.raises(ValueError, match="report.narrative"):
        analysis_report([_narrative(), {"decision_point": "planner.outcome", "id": "x"}], False)


# --- Your plan decision -------------------------------------------------------------------------------------------------

def test_approve(fake_model, tmp_path, store):
    values, _, _ = _run(store, tmp_path, _frame(), "approve")
    predictors = ", ".join(values["plan"].candidate_predictors)
    assert _lines(_analysis(values)[0], "Your plan decision") == ["Approved as drafted.", f"Predictors used: {predictors}."]


def _answers(*answers):
    queue = list(answers)
    return lambda payload: queue.pop(0) if len(queue) > 1 else queue[0]


def test_an_edit_then_approve(fake_model, tmp_path, store):
    drafted, edited = ["tenure", "tier"], ["tenure"]
    values, pauses, _ = _run(store, tmp_path, _frame(), "edit", QuestionType.DRIVER_ANALYSIS, drafted,
                             plan_approval=_answers({"action": "edit", "candidate_predictors": edited},
                                                    {"action": "approve"}))
    assert [p["type"] for p in pauses].count("plan_approval") == 2
    entry = _analysis(values)[0]
    assert _lines(entry, "Your plan decision") == [
        f"Edited the predictors from {', '.join(drafted)} to {', '.join(edited)}; the method was selected again.",
        "Approved as drafted.", f"Predictors used: {', '.join(values['plan'].candidate_predictors)}."]
    assert list(values["plan"].candidate_predictors) == edited
    assert _lines(entry, "Why this test")[-1] == "The method was selected again after you edited the predictors."


def test_a_refused_edit_then_an_accepted_one(fake_model, tmp_path, store):
    drafted = ["tenure", "tier"]
    values, _, _ = _run(store, tmp_path, _with_id(), "refused", QuestionType.DRIVER_ANALYSIS, drafted,
                        plan_approval=_answers({"action": "edit", "candidate_predictors": ["customer_id"]},
                                               {"action": "edit", "candidate_predictors": ["tenure"]},
                                               {"action": "approve"}))
    refused = next(r for r in _records(values, "planner.plan_approval") if r["chosen"] == "refused")
    reasons = "; ".join(f"{v['column'] or v['rule']}: {v['reason']}" for v in refused["evidence"]["refusals"])
    lines = _lines(_analysis(values)[0], "Your plan decision")
    assert lines == [f"Refused: {reasons}.",
                     f"Edited the predictors from {', '.join(drafted)} to tenure; the method was selected again.",
                     "Approved as drafted.", "Predictors used: tenure."]
    assert "customer_id" in reasons


# --- the narrative section ---------------------------------------------------------------------------------------------

def test_narrative_model(fake_model, tmp_path, store):
    values, _, _ = _run(store, tmp_path, _frame(), "model")
    assert _sections(_analysis(values)[0])["Narrative"] == NARRATIVE["model"]


def test_narrative_after_one_correction(report_texts, tmp_path, store):
    report_texts += [BAD, CLEAN]
    values, _, _ = _run(store, tmp_path, _frame(), "retry")
    (record,) = _records(values, "report.narrative")
    matched = record["evidence"]["attempts"][0]["violation"]["matched"]
    entry = _analysis(values)[0]
    assert record["chosen"] == "model_retry" and entry["fallback"] is False
    assert _sections(entry)["Narrative"] == (
        f"The narrative was written by the model after one correction (its first text used '{matched}').")
    assert matched == "causes"


def test_narrative_facts_fallback(report_texts, tmp_path, store):
    report_texts += [BAD]
    values, _, _ = _run(store, tmp_path, _frame(), "facts")
    entry = _analysis(values)[0]
    assert _sections(entry)["Narrative"] == NARRATIVE["facts_fallback"] and entry["fallback"] is True
    assert "The narrative of round 1 was written from the computed facts." in _lines(_summary(values), "Caveats")


def test_narrative_minimal_fallback(report_texts, tmp_path, store, monkeypatch):
    report_texts += [BAD]
    monkeypatch.setattr(nodes_module, "facts_fallback_report", lambda plan, result, name: BAD)
    values, _, _ = _run(store, tmp_path, _frame(), "minimal")
    entry = _analysis(values)[0]
    assert _sections(entry)["Narrative"] == NARRATIVE["minimal_fallback"] and entry["fallback"] is True


# --- the exploratory round ----------------------------------------------------------------------------------------------

def test_the_exploratory_round(fake_model, tmp_path, store):
    values, _, _ = _run(store, tmp_path, _with_id(), "explore", stage_gate={"action": "run"})
    first, second = _analysis(values)
    assert (first["round"], first["exploratory"], second["round"], second["exploratory"]) == (1, False, 2, True)
    assert second["title"].startswith("Exploratory analysis (round 2): ")
    assert second["title"] == f"Exploratory analysis (round 2): {GLOSSARY[values['plan'].primary_method]['name']}"
    headings = [s["heading"] for s in second["sections"]]
    i = headings.index("Your plan decision")
    assert headings[i + 1:] == ["Why these predictors", "Your choice at the boundary", "Narrative"]
    assert "Why these predictors" not in _sections(first) and "Your choice at the boundary" not in _sections(first)

    (start,) = _records(values, "planner.round_start")
    lines = _lines(second, "Why these predictors")
    assert lines[:2] == [f"Chosen: {', '.join(values['plan'].candidate_predictors)}.", "Chosen by: the agent's rules."]
    left = next(c for c in start["candidates"] if c["option"] == "customer_id")
    assert f"Left out: customer_id: {left['reason']}" in lines and left["reason"].startswith("identifier-like")
    assert _lines(second, "Your choice at the boundary") == ["You chose to run the exploratory regression."]
    (regression,) = _records(values, "stage3.regression_method")
    assert _lines(second, "Why this test")[0] == regression["rationale"]
    assert _lines(first, "Why this test")[0] == _records(values, "stage2.test_selection")[0]["rationale"]


def test_the_users_own_predictors(fake_model, tmp_path, store):
    values, _, _ = _run(store, tmp_path, _with_id(), "own", stage_gate={"action": "run", "predictors": ["region", "tier"]})
    second = _analysis(values)[1]
    assert _lines(second, "Why these predictors")[:2] == ["Chosen: region, tier.", "Chosen by: you."]
    assert _lines(second, "Your choice at the boundary") == [
        "You chose to run the exploratory regression with your own predictors."]


def test_a_refusal_at_the_boundary(fake_model, tmp_path, store):
    values, _, _ = _run(store, tmp_path, _with_id(), "gate_refused",
                        stage_gate=_answers({"action": "run", "predictors": ["region", "customer_id"]},
                                            {"action": "run", "predictors": ["region", "tier"]}))
    (refused,) = [g for g in _records(values, "planner.stage_gate") if g["chosen"] == "refused"]
    assert _lines(_analysis(values)[1], "Your choice at the boundary") == [
        f"Refused: {refused['rationale']}", "You chose to run the exploratory regression with your own predictors."]


# --- round_records -----------------------------------------------------------------------------------------------------

def _r(point, i):
    return {"decision_point": point, "id": i}


SP, N = "planner.stage_plan", "report.narrative"


def _ids(log):
    return [r["id"] for r in round_records(log)]


def test_round_records_one_question_one_report():
    assert _ids([_r(SP, "sp"), _r("a", "a"), _r(N, "n1")]) == ["sp", "a", "n1"]


def test_round_records_second_round_starts_after_the_first_narrative():
    log = [_r(SP, "sp"), _r("a", "a"), _r(N, "n1"), _r("gate", "g"), _r("b", "b"), _r(N, "n2")]
    assert _ids(log) == ["g", "b", "n2"]


def test_round_records_second_question_starts_at_its_stage_plan():
    log = [_r(SP, "sp1"), _r("a", "a"), _r(N, "n1"), _r(SP, "sp2"), _r("c", "c"), _r(N, "n2")]
    assert _ids(log) == ["sp2", "c", "n2"]


def test_round_records_without_a_stage_plan():
    assert _ids([_r("a", "a"), _r(N, "n1")]) == ["a", "n1"]
    assert _ids([_r("a", "a"), _r(N, "n1"), _r("b", "b"), _r(N, "n2")]) == ["b", "n2"]


# --- the summary --------------------------------------------------------------------------------------------------------

def test_the_summary_node(fake_model, tmp_path, store):
    assert make_summary_node()({"rounds": [], "decision_log": []}) == {}
    assert make_summary_node()({}) == {}
    values, _, _ = _run(store, tmp_path, _frame(), "node")
    update = make_summary_node()(values)
    assert set(update) == {"stage_reports"}
    (entry,) = update["stage_reports"]
    assert (entry["stage"], entry["round"], entry["title"]) == ("summary", None, "Summary")
    assert entry["sections"] == _summary(values)["sections"]


def _two_group_line(values):
    result = values["test_result"]
    return (f"Main analysis: {method_name(values['plan'].primary_method)} - p = {result['p_value']:.3g} "
            f"(n = {result['group_a_n']} and {result['group_b_n']}).")


def test_the_summary_of_a_two_group_run(fake_model, tmp_path, store):
    df = _frame()
    values, _, _ = _run(store, tmp_path, df, "summary")
    entry = _summary(values)
    assert [s["heading"] for s in entry["sections"]] == SUMMARY_HEADINGS
    assert _lines(entry, "Question") == [values["request_text"], "Run as: two_group_comparison"]
    assert _lines(entry, "Stage 1") == [f"{len(df)} rows and {df.shape[1]} columns checked.",
                                        "0 operation(s) proposed: 0 approved, 0 left as is.", f"{len(df)} rows analysed."]
    assert _lines(entry, "Results") == [_two_group_line(values)]
    assert _sections(entry)["Caveats"] == "None recorded."
    assert "decision_log" not in make_summary_node()(values)


def test_the_summary_of_a_messy_run(fake_model, tmp_path, store):
    df = _messy_frame()
    values, _, _ = _run(store, tmp_path, df, "messy", propose_fn=fake_propose)
    final = len(store.read(values["dataset"].storage_key))
    assert _lines(_summary(values), "Stage 1") == [f"{len(df)} rows and {df.shape[1]} columns checked.",
                                                   "2 operation(s) proposed: 2 approved, 0 left as is.",
                                                   f"{final} rows analysed."]


def test_the_summary_with_the_filter_rejected(fake_model, tmp_path, store):
    values, _, _ = _run(store, tmp_path, _messy_frame(), "rejected", propose_fn=fake_propose,
                        wrangling_approval=_reject_filter)
    entry = _summary(values)
    assert _lines(entry, "Stage 1")[1] == "2 operation(s) proposed: 1 approved, 1 left as is."
    (issue,) = values["plan"].unaddressed_issues
    assert _lines(entry, "Caveats") == [
        f"{issue.issue_type} ({issue.description}) was left unaddressed because: {issue.rejection_reason}"]


def test_the_summary_after_an_exploratory_run(fake_model, tmp_path, store):
    values, _, _ = _run(store, tmp_path, _frame(), "explore_summary", stage_gate={"action": "run"})
    first, second = values["rounds"]
    lines = _lines(_summary(values), "Results")
    r1, r2 = first["result"], second["result"]
    assert lines[0] == (f"Main analysis: {method_name(first['method'])} - p = {r1['p_value']:.3g} "
                        f"(n = {r1['group_a_n']} and {r1['group_b_n']}).")
    assert lines[1] == (f"Exploratory analysis (not corrected for multiple testing): {method_name(second['method'])} - "
                        f"R-squared = {r2['r_squared']:.2f}, overall F-test p = {r2['f_p_value']:.3g}, "
                        f"n = {r2['n_observations']}.")
    assert _lines(_summary(values), "Caveats") == ["Exploratory results are not corrected for multiple testing."]


def test_a_descriptive_question(fake_model, tmp_path, store):
    values, _, _ = _run(store, tmp_path, _frame(), "describe", QuestionType.DESCRIPTIVE, ())
    assert _of(values, "analysis") == []
    assert _lines(_summary(values), "Results") == ["Main analysis: descriptive - the data were described (see the report)."]


def test_a_second_question_summarizes_only_its_own_records(fake_model, tmp_path, store):
    app = _app(store, propose_fn=fake_propose)
    first_df, second_df = _messy_frame(), _frame(seed=62, n=100)
    _drive(app, _start(first_df, tmp_path, store, "q1"), "thread")
    values, _, _ = _drive(app, _start(second_df, tmp_path, store, "q2"), "thread")
    first, second = _of(values, "summary")
    assert _lines(first, "Stage 1")[0] == f"{len(first_df)} rows and 4 columns checked."
    assert _lines(second, "Stage 1") == [f"{len(second_df)} rows and 4 columns checked.",
                                         "0 operation(s) proposed: 0 approved, 0 left as is.",
                                         f"{len(second_df)} rows analysed."]
    assert _lines(second, "Results") == [_two_group_line(values)]


def test_a_descriptive_question_after_a_comparison(fake_model, tmp_path, store):
    def classify(request_text):
        kind = QuestionType.DESCRIPTIVE if request_text.startswith("Describe") else QuestionType.TWO_GROUP_COMPARISON
        return confident(kind)(request_text)
    app = _app(store, classify_fn=classify)
    values, _, _ = _drive(app, _start(_frame(), tmp_path, store, "c1"), "mixed")
    p_line = _two_group_line(values)
    p_text = p_line.split(" - ")[1].split(" (")[0]
    state = {**_start(_frame(seed=63), tmp_path, store, "c2"), "request_text": "Describe the data."}
    values, _, _ = _drive(app, state, "mixed")
    first, second = _of(values, "summary")
    assert _lines(first, "Results") == [p_line]
    assert _lines(second, "Results") == ["Main analysis: descriptive - the data were described (see the report)."]
    assert p_text not in second["text"]
    assert _lines(second, "Question")[0] == "Describe the data."


# --- runs that get no summary -------------------------------------------------------------------------------------------

def test_a_rejected_plan_gets_no_summary(fake_model, tmp_path, store):
    values, _, nodes = _run(store, tmp_path, _frame(), "reject", plan_approval={"action": "reject", "reason": "no"})
    assert values["clarification_needed"].startswith("Plan rejected")
    assert _of(values, "analysis") == [] and _of(values, "summary") == [] and "summary" not in nodes


def test_an_unsupported_question_gets_no_summary(fake_model, tmp_path, store):
    values, _, nodes = _run(store, tmp_path, _frame(), "unsupported", QuestionType.ASSOCIATION)
    assert (values.get("stage_reports") or []) == [] and "summary" not in nodes


def test_a_stage_plan_rejection_gets_no_summary(fake_model, tmp_path, store):
    values, _, nodes = _run(store, tmp_path, _frame(), "gate_reject",
                            stage_plan_approval={"action": "reject", "reason": "not now"})
    assert (values.get("stage_reports") or []) == [] and "summary" not in nodes


# --- failure safety ------------------------------------------------------------------------------------------------------

def test_a_failing_composer_never_loses_the_report(fake_model, tmp_path, store, monkeypatch, caplog):
    def broken(records, exploratory):
        raise RuntimeError("unexpected record shape")
    monkeypatch.setattr(nodes_module, "analysis_report", broken)
    with caplog.at_level(logging.WARNING, logger="core.nodes"):
        values, _, nodes = _run(store, tmp_path, _frame(), "broken")
    assert values["report"] and len(values["rounds"]) == 1 and _records(values, "report.narrative")
    assert [e["stage"] for e in values["stage_reports"]] == ["stage1", "summary"]
    assert "stage report composition failed for round 1" in caplog.text
    assert nodes[-1] == "summary"


def test_a_normal_run_has_its_analysis_entry(fake_model, tmp_path, store, caplog):
    with caplog.at_level(logging.WARNING, logger="core.nodes"):
        values, _, _ = _run(store, tmp_path, _frame(), "normal")
    assert [e["stage"] for e in values["stage_reports"]] == ["stage1", "analysis", "summary"]
    assert "stage report composition failed" not in caplog.text


# --- privacy ----------------------------------------------------------------------------------------------------------------

def test_no_canary_reaches_an_analysis_entry_or_the_summary(fake_model, tmp_path, store):
    _, config = fake_model
    config.update(question_type=QuestionType.TWO_GROUP_COMPARISON, predictors=["region"])
    ds = _start(_with_duplicates(_base_df()), tmp_path, store, "canary")
    values, pauses, _ = _drive(build_agent_graph(store), ds, "canary", stage_gate={"action": "run"})
    assert "wrangling_approval" in [p["type"] for p in pauses]
    entries = _analysis(values) + [_summary(values)]
    assert len(entries) == 3
    for e in entries:
        text = e["text"] + "\n" + json.dumps(e["sections"])
        assert canary_hits([{"stage": e["stage"], "text": text}]) == [], e["title"]


# --- graph and registry --------------------------------------------------------------------------------------------------

def test_graph_edges_and_registry(tmp_path):
    edges = {(e.source, e.target): e.conditional
             for e in build_agent_graph(LocalDiskStore(base_dir=tmp_path)).get_graph().edges}
    assert edges[("report", "stage_gate")] is True and edges[("report", "summary")] is True
    for target in ("start_round", "stage_gate", "summary"):
        assert edges[("stage_gate", target)] is True, target
    assert edges[("summary", "__end__")] is False
    assert ("report", "__end__") not in edges and ("stage_gate", "__end__") not in edges
    assert "summary" in NON_CHOOSING_NODES and "summary" not in DECISION_NODES and NODE_STAGES["summary"] == "report"
    assert (len(registered_points()), len(DECISION_NODES)) == (29, 15)


def test_the_routers_go_to_the_summary():
    assert route_after_report({"exploratory_round": True}) == "summary"
    assert route_after_report({}) == "summary"
    assert route_after_stage_gate({"next_round": {"run": False, "predictors": []}}) == "summary"
    assert route_after_stage_gate({}) == "summary"


# --- Stage 1 entries unchanged --------------------------------------------------------------------------------------------

def test_entries_without_collapsed_sections_are_unchanged():
    composed = {"sections": [{"heading": "A", "text": "a"}, {"heading": "B", "text": "b"}], "source_records": []}
    assert make_stage_report_entry("stage1", None, "T", composed)["sections"] == composed["sections"]


def test_a_collapsed_section_keeps_only_its_key():
    composed = {"sections": [{"heading": "A", "text": "a"}, {"heading": "B", "text": "b", "collapsed": True},
                             {"heading": "C", "text": "c", "collapsed": False}], "source_records": []}
    entry = make_stage_report_entry("analysis", 1, "T", composed)
    assert entry["sections"] == [{"heading": "A", "text": "a"}, {"heading": "B", "text": "b", "collapsed": True},
                                 {"heading": "C", "text": "c"}]
    assert entry["text"] == "T\n\nA\na\n\nB\nb\n\nC\nc"
