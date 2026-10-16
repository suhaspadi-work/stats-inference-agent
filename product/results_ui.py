"""
The finished screen, the explain controls and the history screen (M1.5 step P3b2). Every value shown comes from the
pure views of results_views.py; render_finished returns the explain request of a clicked control, (round,
method_id, question), or None, so app.py makes the model call. Widget keys carry the pause serial and the round.
No expander is ever placed inside another one.
"""
import streamlit as st

from pause_types import widget_key
from pauses import render_sections
from results_views import explanation_caption, explanation_items, history_view, overview_table

_SEE_THE_REPORT = [{"label": "Result", "value": "see the report"}]
UNSAVED_WARNING = "This answer could not be saved to your history."


def render_overview(rows) -> None:
    table = overview_table(rows)
    if table is None:
        return
    st.dataframe(table["rows"], hide_index=True, width="stretch")


def render_statistics(rows) -> None:
    if not rows or rows == _SEE_THE_REPORT:
        return
    st.table([{"Statistic": r["label"], "Value": r["value"]} for r in rows])


def render_text_sections(sections, folded=()) -> None:
    """Visible sections as bold headings with their text; folded sections each in their own expander."""
    render_sections({"sections": list(sections)})
    for section in folded:
        with st.expander(section["heading"]):
            st.markdown(section["text"].replace("\n", "  \n"))


def _button_label(option) -> str:
    return f"Why not {option['name']}?" if option["role"] == "alternative" else f"Explain {option['name']}"


def _render_explanations(items) -> None:
    for item in items:
        question = f' "{item["question"]}"' if item.get("question") else ""
        st.markdown(f"**{item['method_label']}**{question}")
        st.write(item["answer"])
        if item.get("caption"):
            st.caption(item["caption"])
        if item.get("saved") is False:
            st.warning(UNSAVED_WARNING)


def _explain_controls(round_view, serial):
    n, options = round_view["round"], round_view["explain_options"]
    st.markdown("**Learn more about the methods**")
    request = None
    for option in options:
        if st.button(_button_label(option), key=widget_key("explain_btn", serial, n, option["method_id"])):
            request = (n, option["method_id"], None)
    names = [o["name"] for o in options]
    picked = st.selectbox("Method", names, key=widget_key("explain_pick", serial, n))
    question = st.text_area("Your question (up to 500 characters)", max_chars=500,
                            key=widget_key("explain_q", serial, n))
    if st.button("Ask", key=widget_key("explain_ask", serial, n)):
        if (question or "").strip():
            method_id = options[names.index(picked)]["method_id"]
            request = (n, method_id, question)
        else:
            st.caption("Type a question first.")
    return request


def render_round(round_view, serial, explanations):
    """One analysis round; returns the explain request of a clicked control, or None."""
    request = None
    with st.container(border=True):
        title = f"Round {round_view['round']}: {round_view['method_label']}"
        st.subheader(title + (" (exploratory)" if round_view["exploratory"] else ""))
        if round_view["fallback_note"]:
            st.warning(round_view["fallback_note"])
        st.write(round_view["narrative"])
        render_text_sections(round_view["visible_sections"], round_view["folded_sections"])
        render_statistics(round_view["statistics"])
        if round_view["details"] is not None:
            with st.expander("See the underlying statistics"):
                st.json(round_view["details"])
        if round_view["explain_options"]:
            request = _explain_controls(round_view, serial)
        _render_explanations(explanation_items(explanations, round_view["round"]))
    return request


def render_finished(view, serial, explanations):
    """The finished screen of a question with a report; returns the first explain request, or None."""
    st.subheader("Report")
    st.write(f"**Question:** {view['question']}")
    if view["summary"] is not None:
        st.subheader("Summary")
        render_sections(view["summary"])
    if view["overview"] is not None:
        st.subheader("Your data")
        render_overview(view["overview"])
    if view["stage1"] is not None:
        with st.expander(view["stage1"].get("title") or "Stage 1 report"):
            render_sections(view["stage1"])
    request = None
    for round_view in view["rounds"]:
        clicked = render_round(round_view, serial, explanations)
        request = request or clicked
    return request


def render_history(session_row, messages, explanations) -> None:
    """A past session, read-only: each question with its summary, analysis entries, statistics and explanations."""
    st.subheader(f"History: {session_row['dataset_name']}")
    st.caption(f"Session started {session_row['created_at'][:19].replace('T', ' ')}")
    if not messages:
        st.write("No questions were asked in this session.")
    for message in messages:
        view = history_view(message, explanations)
        with st.container(border=True):
            st.write(f"**Q:** {view['question']}")
            if view["summary"] is not None:
                render_sections(view["summary"])
            for entry in view["analysis"]:
                st.markdown(f"**{entry['title']}**")
                render_text_sections(entry["visible_sections"], entry["folded_sections"])
            render_statistics(view["statistics"])
            if view["report"]:
                st.write(f"**A:** {view['report']}")
            else:
                st.caption("(This question did not produce a completed report.)")
            _render_explanations([{**e, "caption": explanation_caption(e["source"])} for e in view["explanations"]])
            st.caption((view["created_at"] or "")[:19].replace("T", " "))
