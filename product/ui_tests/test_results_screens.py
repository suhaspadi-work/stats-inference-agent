"""
The finished screen, the explain controls and the history screen (product/results_ui.py), and the overview on the
plan-approval screen, rendered headless with AppTest on REAL final states (the real graph with fake models).
"""
import copy
import json

from streamlit.testing.v1 import AppTest

import results_views as rv
from conftest import CANARY
from pause_types import widget_key
from run_state import explanation_key

SERIAL = 3
GLOSSARY_DETAIL = ["When the agent uses it", "What it assumes", "How to read the result", "Other options"]


def _finished_script():
    import streamlit as st
    from results_ui import render_finished
    st.session_state["request"] = render_finished(st.session_state["view"], st.session_state["serial"],
                                                  st.session_state["explanations"])


def _history_script():
    import streamlit as st
    from results_ui import render_history
    render_history(st.session_state["session_row"], st.session_state["messages"], st.session_state["explanation_rows"])


def _pause_script():
    import streamlit as st
    from pauses import render_pause
    st.session_state["answer"] = render_pause(st.session_state["payload"], st.session_state["values"], 1)


def _finished(values, explanations=None):
    at = AppTest.from_function(_finished_script)
    at.session_state["view"] = rv.finished_view(values)
    at.session_state["serial"] = SERIAL
    at.session_state["explanations"] = explanations
    at.run()
    assert not at.exception, at.exception
    return at


def _page_text(node) -> str:
    """Every text on the page or in a block, tables and JSON included."""
    parts = [e.value for kind in ("markdown", "caption", "subheader", "warning", "info", "error") for e in getattr(node, kind)]
    parts += [e.value for e in node.json]
    parts += [e.value.to_string() for e in node.dataframe] + [e.value.to_string() for e in node.table]
    parts += [b.label for b in node.button] + [e.label for e in node.expander]
    return "\n".join(str(p) for p in parts)


def _round_blocks(at):
    """The bordered block of each round, in page order."""
    blocks = []
    for child in at.main.children.values():
        subheaders = [s.value for s in getattr(child, "subheader", [])]
        if subheaders and subheaders[0].startswith("Round "):
            blocks.append(child)
    return blocks


def _explain_buttons(node):
    return [b for b in node.button if b.key and b.key.startswith(f"explain_btn-{SERIAL}-")]


def _has_chart_data(text) -> bool:
    return '"chart_data"' in text


# --- (i) the two-group finished screen ----------------------------------------------------------------------------------

def test_the_two_group_finished_screen(real_finished):
    values = real_finished["two_group"]
    view = rv.finished_view(values)
    at = _finished(values)
    subheaders = [s.value for s in at.subheader]
    assert [s for s in subheaders if s in ("Report", "Summary", "Your data")] == ["Report", "Summary", "Your data"]
    assert subheaders.index("Report") < subheaders.index("Summary") < subheaders.index("Your data")
    markdown = [m.value for m in at.markdown]
    assert all(f"**{s['heading']}**" in markdown for s in view["summary"]["sections"])

    overview = at.dataframe[0].value
    assert list(overview.columns) == ["Column", "Type", "Distinct values", "Summary"]
    assert list(overview["Column"]) == [c["name"] for c in values["descriptive_result"]["column_stats"]]

    labels = [e.label for e in at.expander]
    assert labels == [view["stage1"]["title"], *GLOSSARY_DETAIL, "See the underlying statistics"]

    (block,) = _round_blocks(at)
    (round_view,) = view["rounds"]
    assert block.subheader[0].value == f"Round 1: {round_view['method_label']}"
    assert [w.value for w in at.warning] == []
    block_markdown = [m.value for m in block.markdown]
    assert all(f"**{s['heading']}**" in block_markdown for s in round_view["visible_sections"])
    table = block.table[0].value
    p_row = next(r for r in round_view["statistics"] if r["label"] == "p-value")
    assert table[table["Statistic"] == "p-value"]["Value"].tolist() == [p_row["value"]]
    (details,) = block.json
    assert not _has_chart_data(details.value) and json.loads(details.value) == round_view["details"]

    expected = [f"Why not {o['name']}?" if o["role"] == "alternative" else f"Explain {o['name']}"
                for o in round_view["explain_options"]]
    assert [b.label for b in _explain_buttons(at)] == expected and expected[0].startswith("Explain ")
    assert CANARY not in _page_text(at)
    assert at.session_state["request"] is None


def test_the_full_statistics_hold_no_chart_data(real_finished):
    values = real_finished["multi_group"]
    assert '"chart_data"' in json.dumps(values["rounds"][-1]["result"], default=str)
    at = _finished(values)
    (details,) = at.json
    assert not _has_chart_data(details.value)


# --- (ii) explain requests ------------------------------------------------------------------------------------------------

def test_explain_requests(real_finished):
    values = real_finished["two_group"]
    options = rv.finished_view(values)["rounds"][0]["explain_options"]
    chosen, alternative = options[0], next(o for o in options if o["role"] == "alternative")

    at = _finished(values)
    _explain_buttons(at)[0].click().run()
    assert at.session_state["request"] == (1, chosen["method_id"], None)

    at = _finished(values)
    at.button(key=widget_key("explain_btn", SERIAL, 1, alternative["method_id"])).click().run()
    assert at.session_state["request"] == (1, alternative["method_id"], None)

    at = _finished(values)
    assert at.text_area(key=widget_key("explain_q", SERIAL, 1)).proto.max_chars == 500
    at.selectbox(key=widget_key("explain_pick", SERIAL, 1)).set_value(alternative["name"]).run()
    at.text_area(key=widget_key("explain_q", SERIAL, 1)).input("Why not this one?").run()
    at.button(key=widget_key("explain_ask", SERIAL, 1)).click().run()
    assert at.session_state["request"] == (1, alternative["method_id"], "Why not this one?")

    at = _finished(values)
    at.text_area(key=widget_key("explain_q", SERIAL, 1)).input("   ").run()
    at.button(key=widget_key("explain_ask", SERIAL, 1)).click().run()
    assert at.session_state["request"] is None
    assert "Type a question first." in [c.value for c in at.caption]


# --- (iii) cached explanations ----------------------------------------------------------------------------------------------

def test_cached_explanations_appear_under_their_round(real_finished):
    values = real_finished["exploratory"]
    view = rv.finished_view(values)
    first_method = view["rounds"][0]["explain_options"][0]["method_id"]
    second_method = view["rounds"][1]["explain_options"][0]["method_id"]
    explanations = {
        explanation_key(1, first_method, None): {"method_id": first_method, "round": 1, "question": None,
                                                 "answer": "ANSWER-ONE", "source": "model", "saved": True},
        explanation_key(1, first_method, "why?"): {"method_id": first_method, "round": 1, "question": "why?",
                                                   "answer": "ANSWER-UNSAVED", "source": "model_retry", "saved": False},
        explanation_key(2, second_method, None): {"method_id": second_method, "round": 2, "question": None,
                                                  "answer": "ANSWER-TWO", "source": "glossary_fallback"},
    }
    at = _finished(values, explanations)
    first, second = _round_blocks(at)
    one, two = _page_text(first), _page_text(second)
    assert "ANSWER-ONE" in one and "ANSWER-UNSAVED" in one and "ANSWER-TWO" not in one
    assert "ANSWER-TWO" in two and "ANSWER-ONE" not in two
    assert rv.explanation_caption("model") in one and rv.explanation_caption("model_retry") in one
    assert rv.explanation_caption("glossary_fallback") in two
    assert [w.value for w in first.warning] == ["This answer could not be saved to your history."]
    assert [w.value for w in second.warning] == []
    assert '"why?"' in one


# --- (iv) the exploratory run -------------------------------------------------------------------------------------------------

def test_the_exploratory_finished_screen(real_finished):
    at = _finished(real_finished["exploratory"])
    first, second = _round_blocks(at)
    assert first.subheader[0].value.startswith("Round 1: ") and not first.subheader[0].value.endswith("(exploratory)")
    assert second.subheader[0].value.startswith("Round 2: ") and second.subheader[0].value.endswith(" (exploratory)")
    assert "R-squared" in second.table[0].value["Statistic"].tolist()
    keys = [b.key for b in _explain_buttons(at)]
    assert len(keys) == len(set(keys)) and any(k.startswith(f"explain_btn-{SERIAL}-2-") for k in keys)


# --- (v) the fallback run -------------------------------------------------------------------------------------------------------

def test_the_fallback_warning(real_finished):
    at = _finished(real_finished["fallback"])
    (block,) = _round_blocks(at)
    assert [w.value for w in block.warning] == [rv.FALLBACK_NOTE]
    assert [w.value for w in at.warning] == [rv.FALLBACK_NOTE]


# --- (vi) the descriptive question -------------------------------------------------------------------------------------------------

def test_the_descriptive_finished_screen(real_finished):
    at = _finished(real_finished["descriptive"])
    assert "**Learn more about the methods**" not in [m.value for m in at.markdown]
    assert _explain_buttons(at) == [] and len(at.table) == 0 and len(at.json) == 0
    subheaders = [s.value for s in at.subheader]
    assert "Summary" in subheaders and "Your data" in subheaders and len(at.dataframe) == 1


# --- (vii) the overview on the plan-approval screen ---------------------------------------------------------------------------------

def _plan_pause(values, payload):
    at = AppTest.from_function(_pause_script)
    at.session_state["payload"] = payload
    at.session_state["values"] = values
    at.run()
    assert not at.exception, at.exception
    return at


def test_the_plan_approval_overview(real_pauses):
    pause = next(p for p in real_pauses if p["payload"]["type"] == "plan_approval" and p["payload"]["stage1_report"])
    at = _plan_pause(pause["values"], pause["payload"])
    (expander,) = at.expander
    assert "**Your data**" in [m.value for m in expander.markdown] and len(expander.dataframe) == 1

    stripped = copy.copy(pause["values"])
    stripped.pop("descriptive_result", None)
    stripped["decision_log"] = [r for r in stripped["decision_log"] if r["decision_point"] != "stage1.describe.execution"]
    at = _plan_pause(stripped, pause["payload"])
    (expander,) = at.expander
    assert "**Your data**" not in [m.value for m in expander.markdown] and len(at.dataframe) == 0


# --- (viii) the history screen ---------------------------------------------------------------------------------------------------------

def _message(values, message_id, report=True, stored=True):
    return {"id": message_id, "question_text": f"question {message_id}",
            "report_text": values["report"] if report else None,
            "test_result_json": json.dumps(values["test_result"], default=str) if stored else None,
            "stage_reports_json": json.dumps(values["stage_reports"], default=str) if stored else None,
            "created_at": "2026-10-16T10:00:00.000000"}


def _history(messages, explanation_rows):
    at = AppTest.from_function(_history_script)
    at.session_state["session_row"] = {"id": 1, "dataset_name": "spend", "created_at": "2026-10-16T09:00:00.000000"}
    at.session_state["messages"] = messages
    at.session_state["explanation_rows"] = explanation_rows
    at.run()
    assert not at.exception, at.exception
    return at


def test_the_history_screen(real_finished):
    values = real_finished["two_group"]
    rows = [{"message_id": 5, "round": 1, "method_id": "student_t_test", "question": None, "answer": "HISTORY-MINE",
             "source": "model", "created_at": "t"},
            {"message_id": 6, "round": 1, "method_id": "ols", "question": None, "answer": "HISTORY-OTHER",
             "source": "model", "created_at": "t"}]
    at = _history([_message(values, 5)], rows)
    assert at.subheader[0].value == "History: spend"
    text = _page_text(at)
    view = rv.history_view(_message(values, 5), rows)
    assert "**Q:** question 5" in text and f"**{view['analysis'][0]['title']}**" in text
    assert all(f"**{s['heading']}**" in text for s in view["summary"]["sections"])
    assert at.table[0].value["Value"].tolist() == [r["value"] for r in view["statistics"]]
    assert "HISTORY-MINE" in text and "HISTORY-OTHER" not in text
    assert CANARY not in text


def test_the_history_of_old_and_unfinished_messages(real_finished):
    values = real_finished["two_group"]
    at = _history([_message(values, 7, stored=False)], [])
    text = _page_text(at)
    assert "**Q:** question 7" in text and f"**A:** {values['report']}" in [m.value for m in at.markdown]
    assert len(at.table) == 0 and len(at.expander) == 0
    at = _history([_message(values, 8, report=False, stored=False)], [])
    assert "(This question did not produce a completed report.)" in [c.value for c in at.caption]
    at = _history([], [])
    assert "No questions were asked in this session." in [m.value for m in at.markdown]
