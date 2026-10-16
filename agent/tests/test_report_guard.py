"""
M1.4a (G52): the report guard. The causal-language check uses whole-word negation markers within the same
sentence; a violating report is retried once with the violation fed back, then replaced by a report written by
code from the computed facts, so a report can no longer fail the run. The choice is a report.narrative record.
"""
import dataclasses
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import core.nodes as nodes_module
import core.report as report_module
from core.capabilities import MULTI_GROUP, REGRESSION, TWO_GROUP
from core.dataset import LocalDiskStore
from core.decision_log import log_update
from core.decision_points import DECISION_NODES, NON_CHOOSING_NODES, registered_points
from core.graph import build_agent_graph
from core.model_boundary import prepare_for_model
from core.nodes import make_report_node
from core.plan import QuestionType, UnaddressedIssue
from core.report import (
    CausalLanguageViolation, check_for_causal_language, find_causal_violation, generate_descriptive_report,
    generate_multi_group_report, generate_regression_report, generate_report, violation_feedback,
)
from core.report_facts import multi_group_facts, regression_facts, two_group_facts
from core.report_fallback import MINIMAL_FALLBACK_TEXT, facts_fallback_report
from core.report_records import REPORT_NARRATIVE_POINT, report_narrative_record
from core.rounds import EXPLORATORY_NOTE
from tests.test_chart_guard import _dataset, _real_plan
from tests.test_decision_point_guard import _drive, _two_group_df, confident, fake_draft, must_not_propose
from tests.test_prompt_canaries import FakeChat

OBS = SimpleNamespace(allowed_claims="associations only")
EXPERIMENT = SimpleNamespace(allowed_claims="causal claims allowed")
HEADER = "Summary of the computed results (written from the computed facts, without narrative text):"
ASSOCIATIONS = "These results describe associations, not effects."


# --- the guard ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("word", ["cause", "causes", "caused", "causing", "drive", "drives", "driving", "leads to",
                                  "led to", "results in", "resulted in"])
def test_each_causal_word_is_flagged(word):
    text = f"Region {word} higher spend."
    assert find_causal_violation(text) == {"matched": word, "sentence": text}
    with pytest.raises(CausalLanguageViolation) as caught:
        check_for_causal_language(text, OBS)
    assert (caught.value.matched, caught.value.sentence) == (word, text)


@pytest.mark.parametrize("text", ["Region does not cause spend.", "Region cannot cause spend.",
                                  "Region never leads to spend.", "We saw this without driving it.",
                                  "Region neither causes nor explains spend.",
                                  "There is no evidence that it causes spend.", "It isn't what causes it.",
                                  "An association rather than causes."])
def test_each_negation_excuses_the_causal_word(text):
    assert find_causal_violation(text) is None
    check_for_causal_language(text, OBS)


@pytest.mark.parametrize("text, why", [
    ("Another factor causes spend.", "'another' contains 'not' but is not a marker"),
    ("Nothing causes spend.", "'nothing' is not a marker"),
    ("It knows that it causes spend.", "'knows' is not a marker"),
])
def test_markers_are_whole_words(text, why):
    assert find_causal_violation(text)["matched"] == "causes", why


@pytest.mark.parametrize("straight, curly", [
    ("Region doesn't cause spend.", "Region doesn\u2019t cause spend."),
    ("It isn't what causes it.", "It isn\u2019t what causes it."),
    ("We can't say that it causes spend.", "We can\u2019t say that it causes spend."),
])
def test_curly_apostrophes_are_excused_like_straight_ones(straight, curly):
    assert find_causal_violation(straight) is None
    assert find_causal_violation(curly) is None
    check_for_causal_language(curly, OBS)


def test_a_curly_apostrophe_elsewhere_does_not_excuse_and_is_kept():
    text = "Another\u2019s factor causes spend."
    assert find_causal_violation(text) == {"matched": "causes", "sentence": text}
    with pytest.raises(CausalLanguageViolation) as caught:
        check_for_causal_language(text, OBS)
    assert caught.value.sentence == text and "\u2019" in caught.value.sentence


def test_a_negation_in_the_previous_sentence_does_not_excuse():
    assert find_causal_violation("Not significant. Tenure causes spend.") == {
        "matched": "causes", "sentence": "Tenure causes spend."}


def test_a_negation_more_than_40_characters_before_does_not_excuse():
    sentence = "This is not, as the long and winding explanation here shows clearly, what causes spend."
    assert sentence.index("causes") - (sentence.index("not") + 3) > 40
    assert find_causal_violation(sentence)["matched"] == "causes"


def test_the_first_unexcused_word_is_reported():
    text = "Region does not cause spend. Tenure drives visits and causes spend."
    assert find_causal_violation(text) == {"matched": "drives", "sentence": "Tenure drives visits and causes spend."}


def test_other_claims_are_never_checked():
    check_for_causal_language("Region causes spend.", EXPERIMENT)


def test_the_message_and_feedback():
    with pytest.raises(CausalLanguageViolation) as caught:
        check_for_causal_language("Region causes spend.", OBS)
    message = str(caught.value)
    assert "('causes')" in message and '"Region causes spend."' in message
    assert "Rewrite using association language" in message
    expected = ("Correction needed: your previous text used causal language ('causes') in this sentence: "
                "\"Region causes spend.\". Rewrite the whole summary using association language only, such as 'is "
                "associated with'. Do not use the words cause, drive, lead to or result in.")
    assert violation_feedback(caught.value) == expected
    assert violation_feedback({"matched": "causes", "sentence": "Region causes spend."}) == expected


# --- real plans and results -----------------------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


@pytest.fixture
def real(tmp_path, store):
    rng = np.random.default_rng(81)
    n = 60
    mg = pd.DataFrame({"region": [g for g in ("East", "North", "West") for _ in range(n)],
                       "tenure": rng.normal(24, 8, 3 * n),
                       "total_spend": np.concatenate([rng.normal(m, 5, n) for m in (50, 55, 65)])})
    ds3 = _dataset(mg, tmp_path, store, "mg")
    ds2 = _dataset(mg[mg.region != "North"].reset_index(drop=True), tmp_path, store, "tg")
    out = SimpleNamespace()
    for name, ds, qt, predictors, cap in (
            ("two_group", ds2, QuestionType.TWO_GROUP_COMPARISON, ["region"], TWO_GROUP),
            ("multi_group", ds3, QuestionType.MULTI_GROUP_COMPARISON, ["region"], MULTI_GROUP),
            ("regression", ds3, QuestionType.DRIVER_ANALYSIS, ["tenure", "region"], REGRESSION)):
        plan = _real_plan(ds, qt, predictors)
        plan = plan.record_method_decision(cap.select(ds, store, plan)).freeze()
        setattr(out, name, (plan, cap.execute(ds, store, plan)))
    return out


FACTS = {"two_group": lambda p, r: two_group_facts(r, p.outcome_name, p.candidate_predictors[0]),
         "multi_group": lambda p, r: multi_group_facts(r, p.outcome_name, p.candidate_predictors[0]),
         "regression": lambda p, r: regression_facts(r, p.outcome_name)}


# --- the generators accept feedback ------------------------------------------------------------------------------------

@pytest.fixture
def recorded(monkeypatch):
    prompts = []
    monkeypatch.setattr(report_module, "init_chat_model", lambda *a, **k: FakeChat(prompts, {}))
    return prompts


@pytest.mark.parametrize("name, generate", [("two_group", generate_report), ("multi_group", generate_multi_group_report),
                                            ("regression", generate_regression_report)])
def test_feedback_is_appended_to_the_human_prompt(name, generate, real, recorded):
    plan, result = getattr(real, name)
    generate(plan, result)
    generate(plan, result, feedback="Correction needed: X.")
    without, with_feedback = recorded
    assert not without["text"].endswith("Correction needed: X.")
    assert with_feedback["text"].endswith("\n\nCorrection needed: X.")
    assert with_feedback["text"][:-len("\n\nCorrection needed: X.")] == without["text"]


def test_the_descriptive_generator_has_no_feedback_parameter():
    assert "feedback" not in inspect.signature(generate_descriptive_report).parameters


# --- facts_fallback_report -----------------------------------------------------------------------------------------------

ISSUE = UnaddressedIssue(issue_type="high_missingness", description="notes 12% missing", proposed_operation="filter",
                         rejection_reason="kept on purpose")


@pytest.mark.parametrize("name", ["two_group", "multi_group", "regression"])
def test_facts_fallback_report(name, real):
    plan, result = getattr(real, name)
    plan = dataclasses.replace(plan, unaddressed_issues=(ISSUE,))
    text = facts_fallback_report(plan, result, name)
    assert text == "\n".join([HEADER + "\n" + FACTS[name](plan, result),
                              "Caveat: high_missingness (notes 12% missing) was identified but left unaddressed "
                              "because: kept on purpose. Interpret the results with this in mind.",
                              ASSOCIATIONS])
    check_for_causal_language(text, plan)


def test_facts_fallback_without_the_associations_line(real):
    plan, result = real.two_group
    text = facts_fallback_report(dataclasses.replace(plan, allowed_claims="causal claims allowed"), result, "two_group")
    assert ASSOCIATIONS not in text and text.startswith(HEADER)


def test_facts_fallback_unknown_capability(real):
    plan, result = real.two_group
    with pytest.raises(ValueError, match="'descriptive'"):
        facts_fallback_report(plan, result, "descriptive")


def test_the_minimal_text_passes_the_guard():
    check_for_causal_language(MINIMAL_FALLBACK_TEXT, OBS)
    assert MINIMAL_FALLBACK_TEXT == ("The narrative could not be written safely for this result. The computed results "
                                     "are recorded with the analysis.")


# --- report_narrative_record ------------------------------------------------------------------------------------------------

PASS1 = {"attempt": 1, "passed": True, "violation": None}
FAIL1 = {"attempt": 1, "passed": False, "violation": {"matched": "causes", "sentence": "Region causes spend."}}
PASS2 = {"attempt": 2, "passed": True, "violation": None}
FAIL2 = {"attempt": 2, "passed": False, "violation": {"matched": "drives", "sentence": "Tenure drives spend."}}


@pytest.mark.parametrize("attempts, chosen, eligible, model_reason, retry_reason, rationale", [
    ([PASS1], "model", [True, False, True, False], "the model's text passed the causal-language check",
     "not needed: the first text passed", "the model's text passed the check"),
    ([FAIL1, PASS2], "model_retry", [False, True, True, False], "the model's text used 'causes' in: Region causes spend.",
     "the retried text passed the check", "the retried text passed the check after one correction"),
    ([FAIL1, FAIL2], "facts_fallback", [False, False, True, False],
     "the model's text used 'causes' in: Region causes spend.", "the retried text used 'drives' in: Tenure drives spend.",
     "both model texts used causal language, so the report was written from the computed facts"),
    ([FAIL1, FAIL2], "minimal_fallback", [False, False, True, True],
     "the model's text used 'causes' in: Region causes spend.", "the retried text used 'drives' in: Tenure drives spend.",
     "the fact-based text tripped the guard as well, so the minimal text was used"),
])
def test_report_narrative_record(attempts, chosen, eligible, model_reason, retry_reason, rationale):
    r = report_narrative_record(2, "regression", True, attempts, chosen)
    assert (r.decision_point, r.chosen, r.decided_by, r.rationale) == (REPORT_NARRATIVE_POINT, chosen, "code_rule",
                                                                       rationale)
    assert [c.option for c in r.candidates] == ["model", "model_retry", "facts_fallback", "minimal_fallback"]
    assert [c.eligible for c in r.candidates] == eligible
    assert [c.reason for c in r.candidates] == [
        model_reason, retry_reason, "always available: written by code from the computed facts",
        "last resort: used only when the fact-based fallback itself trips the guard"]
    assert r.evidence == {"round": 2, "capability": "regression", "exploratory": True, "attempts": attempts}


def test_records_serialize_and_the_point_is_registered():
    records = [report_narrative_record(1, "two_group", False, [PASS1], "model"),
               report_narrative_record(1, "two_group", False, [FAIL1, FAIL2], "facts_fallback")]
    assert len({r.id for r in records}) == 2
    json.dumps(log_update(*records))
    assert REPORT_NARRATIVE_POINT in registered_points()
    with pytest.raises(ValueError, match="'maybe'"):
        report_narrative_record(1, "two_group", False, [PASS1], "maybe")


# --- report_node with stub generators -------------------------------------------------------------------------------------------

class _Stub:
    """Returns (or raises) the given items in turn and records every call's feedback."""
    def __init__(self, *items):
        self.items, self.calls = list(items), []

    def __call__(self, plan, result, feedback=""):
        self.calls.append(feedback)
        item = self.items[min(len(self.calls), len(self.items)) - 1]
        if isinstance(item, Exception):
            raise item
        return item


def _node_state(plan, result, **extra):
    return {"plan": plan, "test_result": result, "request_text": "x", "session_id": "s", **extra}


def _run_node(stub, plan, result, **extra):
    update = make_report_node(report_fn=stub)(_node_state(plan, result, **extra))
    (record,) = update["decision_log"]
    return update, record


CLEAN = "West spends more on average; region is associated with spend."
BAD = "Region causes higher spend."


def test_the_first_text_passes(real):
    stub = _Stub(CLEAN)
    update, record = _run_node(stub, *real.two_group)
    assert stub.calls == [""] and record["chosen"] == "model" and update["report"] == CLEAN
    assert record["evidence"]["attempts"] == [PASS1]
    # M1.4b step 2b: the analysis branch also returns its "how this was chosen" entry
    assert set(update) == {"report", "rounds", "decision_log", "stage_reports"} and update["rounds"][0]["report"] == CLEAN
    assert [e["stage"] for e in update["stage_reports"]] == ["analysis"]


def test_one_retry_with_the_violation_fed_back(real):
    stub = _Stub(BAD, CLEAN)
    update, record = _run_node(stub, *real.two_group)
    assert record["chosen"] == "model_retry" and update["report"] == CLEAN
    assert stub.calls[0] == "" and stub.calls[1] == violation_feedback({"matched": "causes", "sentence": BAD})
    assert BAD in stub.calls[1]
    assert record["evidence"]["attempts"] == [
        {"attempt": 1, "passed": False, "violation": {"matched": "causes", "sentence": BAD}}, PASS2]


def test_two_violations_fall_back_to_the_facts(real):
    plan, result = real.two_group
    update, record = _run_node(_Stub(BAD, BAD), plan, result)
    _, safe_result = prepare_for_model(plan, result)
    assert update["report"] == facts_fallback_report(plan, safe_result, "two_group")
    assert record["chosen"] == "facts_fallback" and update["rounds"][0]["report"] == update["report"]


def test_the_minimal_text_when_the_facts_trip_the_guard(real):
    plan, result = real.two_group
    plan = dataclasses.replace(plan, outcome_name="what causes spend")
    assert find_causal_violation(facts_fallback_report(plan, result, "two_group")) is not None  # precondition
    update, record = _run_node(_Stub(BAD, BAD), plan, result)
    assert update["report"] == MINIMAL_FALLBACK_TEXT and record["chosen"] == "minimal_fallback"


@pytest.mark.parametrize("items, chosen", [((CLEAN,), "model"), ((BAD, CLEAN), "model_retry"),
                                           ((BAD, BAD), "facts_fallback")])
def test_the_exploratory_note_goes_in_front_of_whichever_text(items, chosen, real):
    update, record = _run_node(_Stub(*items), *real.two_group, exploratory_round=True)
    assert record["chosen"] == chosen and record["evidence"]["exploratory"] is True
    assert update["report"].startswith(EXPLORATORY_NOTE + "\n\n")
    body = update["report"][len(EXPLORATORY_NOTE) + 2:]
    if chosen == "facts_fallback":
        assert body.startswith(HEADER)
    else:
        assert body == CLEAN
    # the check ran on the model's own texts, never on the note
    assert all(a["violation"] is None or a["violation"]["sentence"] == BAD for a in record["evidence"]["attempts"])


def test_other_errors_propagate(real):
    with pytest.raises(RuntimeError, match="model down"):
        make_report_node(report_fn=_Stub(RuntimeError("model down")))(_node_state(*real.two_group))


def test_a_violation_raised_by_the_generator_is_handled_the_same(real):
    stub = _Stub(CausalLanguageViolation("drives", "Tenure drives spend."), CLEAN)
    update, record = _run_node(stub, *real.two_group)
    assert record["chosen"] == "model_retry" and update["report"] == CLEAN
    assert record["evidence"]["attempts"][0]["violation"] == {"matched": "drives", "sentence": "Tenure drives spend."}


def test_the_diagnostic_path_writes_no_record_and_never_retries(real, monkeypatch):
    stub = _Stub(BAD)
    monkeypatch.setattr(nodes_module, "_capability_for_state",
                        lambda state: SimpleNamespace(kind="diagnostic", result_key="descriptive_result", report=stub,
                                                      name="descriptive"))
    plan, _ = real.two_group
    update = make_report_node()({"plan": plan, "descriptive_result": {"row_count": 1, "chart_data": []},
                                 "request_text": "x", "session_id": "s"})
    assert set(update) == {"report", "rounds"} and update["report"] == BAD and stub.calls == [""]


def test_the_multi_group_path_uses_the_capability_generator(real, monkeypatch):
    stub = _Stub(BAD, BAD)
    monkeypatch.setattr(nodes_module, "analysis_for_result",
                        lambda result: SimpleNamespace(result_type="multi_group", report=stub))
    plan, result = real.multi_group
    update = make_report_node(report_fn=lambda p, r, **k: "must not be used")(_node_state(plan, result))
    assert update["decision_log"][0]["chosen"] == "facts_fallback" and len(stub.calls) == 2
    assert update["report"].startswith(HEADER)


# --- through the real graph ------------------------------------------------------------------------------------------------------

class _ScriptedChat:
    def __init__(self, texts):
        self.texts, self.calls = texts, 0

    def invoke(self, messages):
        self.calls += 1
        content = self.texts[min(self.calls, len(self.texts)) - 1]
        return type("Response", (), {"content": content})()


@pytest.mark.parametrize("texts, chosen", [([BAD, CLEAN], "model_retry"), ([BAD], "facts_fallback"),
                                           ([CLEAN], "model")])
def test_through_the_graph(texts, chosen, monkeypatch, tmp_path, store):
    chat = _ScriptedChat(texts)
    monkeypatch.setattr(report_module, "init_chat_model", lambda *a, **k: chat)
    app = build_agent_graph(store, classify_fn=confident(QuestionType.TWO_GROUP_COMPARISON),
                            draft_fn=fake_draft(["region"]), propose_fn=must_not_propose)
    state = {"request_text": "x", "session_id": "s",
             "dataset": _dataset(_two_group_df(np.random.default_rng(82)), tmp_path, store, "g")}
    values, _, next_nodes = _drive(app, state, "g", [{"action": "approve"}], {}, stop=())
    assert next_nodes == ()
    (record,) = [e for e in values["decision_log"] if e["decision_point"] == REPORT_NARRATIVE_POINT]
    assert record["chosen"] == chosen
    if chosen == "facts_fallback":
        assert values["report"].startswith("Summary of the computed results")
    else:
        assert values["report"] == CLEAN


# --- registry -----------------------------------------------------------------------------------------------------------------------

def test_registry():
    assert len(registered_points()) == 29 and len(DECISION_NODES) == 15
    assert DECISION_NODES["report"] == (REPORT_NARRATIVE_POINT,) and "report" not in NON_CHOOSING_NODES
