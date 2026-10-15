"""
M1.3b step 1: results per round. report_node appends one entry to state["rounds"] for every finished
round, holding the original result (charts included), the plan and the report; plan, test_result and
report keep meaning the latest round. An exploratory round's report gets a note, added after the
causal-language check has run on the model's own text.
"""
import copy
from types import SimpleNamespace

import pytest

import core.nodes as nodes_module
from core.dataset import LocalDiskStore
from core.graph import build_agent_graph
from core.ingestion import load_dataset
from core.nodes import make_report_node
from core.plan import AnalysisPlan, QuestionType
from core.report import CausalLanguageViolation
from core.rounds import EXPLORATORY_NOTE, make_round_entry, with_exploratory_note
from tests.test_chart_guard import BOX, _Recorder, _assert_received_safe, _plan, _state, real, store  # noqa: F401
from tests.test_prompt_canaries import REPORT_TEXT, _base_df, _clean, _drive as _drive_to_end, fake_model  # noqa: F401

ENTRY_KEYS = {"round", "exploratory", "capability", "question_type", "method", "plan", "result", "report"}


# --- pure helpers ----------------------------------------------------------------------------------

def test_exploratory_note():
    assert EXPLORATORY_NOTE == (
        "Exploratory analysis: this was not part of your original question, so it was chosen after seeing the "
        "main result. Its results are associations to explore, and they are not corrected for multiple testing.")
    assert with_exploratory_note("Body.") == EXPLORATORY_NOTE + "\n\nBody."


def test_make_round_entry(real):
    plan, result = real.two_group_plan, {"p_value": 0.01, "chart_data": [BOX]}
    before = copy.deepcopy(result)
    entry = make_round_entry(2, plan, "two_group", result, "text", True)
    assert set(entry) == ENTRY_KEYS
    assert entry == {"round": 2, "exploratory": True, "capability": "two_group",
                     "question_type": "two_group_comparison", "method": plan.primary_method, "plan": plan,
                     "result": result, "report": "text"}
    assert entry["plan"] is plan and entry["result"] is result and result == before
    assert entry["method"] is not None


def test_make_round_entry_without_a_method():
    plan = AnalysisPlan(request_text="x", session_id="s", question_type=QuestionType.DESCRIPTIVE,
                        causal_status=_plan().causal_status, outcome_name="(whole dataset)",
                        outcome_type=_plan().outcome_type, unit_of_analysis="row", data_sources=("d@v1",))
    entry = make_round_entry(1, plan, "descriptive", {}, "r", False)
    assert entry["method"] is None and entry["exploratory"] is False and entry["question_type"] == "descriptive"


# --- report_node -----------------------------------------------------------------------------------

def _check_entry(update, recorder, state, capability, result_key, round_index=1, exploratory=False):
    assert set(update) == {"report", "rounds"}
    (entry,) = update["rounds"]
    plan = state["plan"]
    assert set(entry) == ENTRY_KEYS
    assert (entry["round"], entry["exploratory"], entry["capability"]) == (round_index, exploratory, capability)
    assert (entry["question_type"], entry["method"]) == (plan.question_type.value, plan.primary_method)
    assert entry["plan"] is plan
    assert entry["report"] == update["report"]
    assert entry["result"] is state[result_key] and entry["result"]["chart_data"]  # the original, not the copy
    _assert_received_safe(recorder)
    return entry


def test_two_group_path(real):
    stub = _Recorder("two-group report")
    state = _state(real.two_group_plan, test_result={**real.two_group_result, "chart_data": [BOX]})
    update = make_report_node(report_fn=stub)(state)
    entry = _check_entry(update, stub, state, "two_group", "test_result")
    assert entry["report"] == "two-group report"


def test_multi_group_path(real, monkeypatch):
    stub = _Recorder("multi-group report")
    monkeypatch.setattr(nodes_module, "analysis_for_result",
                        lambda result: SimpleNamespace(result_type="multi_group", report=stub))
    state = _state(real.multi_group_plan, test_result=real.multi_group)
    update = make_report_node(report_fn=lambda p, r: "must not be used")(state)
    entry = _check_entry(update, stub, state, "multi_group", "test_result")
    assert entry["report"] == "multi-group report"


def test_diagnostic_path(real, monkeypatch):
    stub = _Recorder("descriptive report")
    monkeypatch.setattr(nodes_module, "_capability_for_state",
                        lambda state: SimpleNamespace(kind="diagnostic", result_key="descriptive_result", report=stub,
                                                      name="descriptive"))
    state = _state(real.two_group_plan, descriptive_result=real.descriptive)
    update = make_report_node()(state)
    entry = _check_entry(update, stub, state, "descriptive", "descriptive_result")
    assert entry["report"] == "descriptive report"


def test_numbering_follows_earlier_rounds(real):
    stub = _Recorder("third")
    state = _state(real.two_group_plan, test_result={**real.two_group_result, "chart_data": [BOX]},
                   rounds=[{"round": 1}, {"round": 2}])
    _check_entry(make_report_node(report_fn=stub)(state), stub, state, "two_group", "test_result", round_index=3)


# --- exploratory rounds ------------------------------------------------------------------------------------

def _two_group_state(real, **extra):
    return _state(real.two_group_plan, test_result={**real.two_group_result, "chart_data": [BOX]}, **extra)


def test_exploratory_round_gets_the_note(real):
    stub = _Recorder("Spend differs by region.")
    state = _two_group_state(real, exploratory_round=True)
    update = make_report_node(report_fn=stub)(state)
    assert update["report"] == EXPLORATORY_NOTE + "\n\nSpend differs by region."
    entry = _check_entry(update, stub, state, "two_group", "test_result", exploratory=True)
    assert entry["report"] == update["report"]


def test_causal_check_runs_on_the_model_text_before_the_note(real):
    assert real.two_group_plan.allowed_claims == "associations only"
    with pytest.raises(CausalLanguageViolation, match="causes"):
        make_report_node(report_fn=_Recorder("Region causes spend."))(_two_group_state(real, exploratory_round=True))


@pytest.mark.parametrize("extra", [{}, {"exploratory_round": False}], ids=["absent", "false"])
def test_no_note_outside_an_exploratory_round(real, extra):
    update = make_report_node(report_fn=_Recorder("plain"))(_two_group_state(real, **extra))
    assert update["report"] == "plain" and update["rounds"][0]["exploratory"] is False


# --- through the real graph -----------------------------------------------------------------------------------

@pytest.mark.parametrize("question_type, predictors", [(QuestionType.DESCRIPTIVE, []),
                                                       (QuestionType.TWO_GROUP_COMPARISON, ["region"])],
                         ids=["descriptive", "two_group"])
def test_full_run_records_one_round(question_type, predictors, fake_model, tmp_path):
    _, config = fake_model
    config.update(question_type=question_type, predictors=predictors)
    store = LocalDiskStore(base_dir=tmp_path / "store")
    path = tmp_path / "data.csv"
    _clean(_base_df()).to_csv(path, index=False)
    ds = load_dataset(path, session_id="s", name="data", store=store)
    app = build_agent_graph(store)
    values, _ = _drive_to_end(app, {"request_text": "x", "session_id": "s", "dataset": ds}, "rounds")

    (entry,) = values["rounds"]
    assert entry["report"] == values["report"] == REPORT_TEXT
    assert entry["round"] == 1 and entry["exploratory"] is False
    assert entry["plan"] == values["plan"]
    if question_type is QuestionType.TWO_GROUP_COMPARISON:
        assert repr(entry["result"]) == repr(values["test_result"]) and entry["capability"] == "two_group"
    else:
        assert repr(entry["result"]) == repr(values["descriptive_result"]) and entry["capability"] == "descriptive"

    stored = app.get_state({"configurable": {"thread_id": "rounds"}}).values["rounds"][0]
    assert type(stored["plan"]) is AnalysisPlan and type(stored["plan"].question_type) is QuestionType
