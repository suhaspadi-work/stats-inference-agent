"""
The Streamlit screens of every pause (M1.5 step P2b). render_pause draws the screen of an interrupt payload
through the pure views of pause_views.py and returns the resume value when the user clicks the pause's submit
button (None otherwise), so app.py has one resume handler. Every widget key carries the pause serial, so a
re-opened pause starts clean. Editing the predictors at plan approval is not offered yet (G38).
"""
import streamlit as st

import pause_views as pv
from pause_types import widget_key

GATE_OPTIONS = ("proposed", "own", "skip")


def render_refusal(view) -> None:
    """A refused choice, shown in red at the top of the screen: the header, then each reason as a list item."""
    lines = view.get("refusal") or []
    if lines:
        st.error(lines[0] + "\n\n" + "\n".join(f"- {line}" for line in lines[1:]))


def render_sections(report) -> None:
    """A stage report section by section: the heading in bold, the text with its line breaks kept."""
    for section in report.get("sections") or []:
        st.markdown(f"**{section['heading']}**")
        st.markdown(section["text"].replace("\n", "  \n"))


def _stage_plan(view, serial):
    st.subheader("Review which steps this run will take")
    render_refusal(view)
    st.write(f"**Question:** {view['request_text']}")
    toggles = {}
    for row in view["rows"]:
        st.markdown(f"**{row['step']}**: {row['state_label']}")
        st.caption(row["reason"])
        if row["note"]:
            st.caption(row["note"])
        if row["adjustable"]:
            toggles[row["step"]] = st.checkbox(f"Run the {row['step']} step", value=bool(row["run"]),
                                               key=widget_key("sp_toggle", serial, row["step"]))
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve", type="primary", key=widget_key("sp_approve", serial)):
            return pv.approve_stage_plan(view, toggles)
    with col2:
        reason = st.text_input("Reason for rejecting (optional)", key=widget_key("sp_reason", serial))
        if st.button("❌ Reject", key=widget_key("sp_reject", serial)):
            return pv.reject_stage_plan(reason)
    return None


def _plan_approval(view, serial):
    st.subheader("Review the analysis plan")
    render_refusal(view)
    st.write(f"**Question:** {view['request_text']}")
    st.write(f"**Outcome variable:** {view['outcome']} ({view['outcome_type']})")
    st.write(f"**Comparing on:** {view['predictors']}")
    st.write(f"**Data type:** {view['data_type']}")
    st.write(f"**Method:** {view['method_label']}")
    if view["reselected_note"]:
        st.info(view["reselected_note"])
    if view["rationale"]:
        st.caption(view["rationale"])
    report = view["stage1_report"]
    if report:
        with st.expander(report.get("title") or "Stage 1 report"):
            render_sections(report)
    st.caption("Editing the predictors here is not available yet.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve", type="primary", key=widget_key("plan_approve", serial)):
            return pv.approve_plan()
    with col2:
        reason = st.text_input("Reason for rejecting (optional)", key=widget_key("plan_reason", serial))
        if st.button("❌ Reject", key=widget_key("plan_reject", serial)):
            return pv.reject_plan(reason)
    return None


def _stage_gate(view, serial, values):
    st.subheader("The main analysis is finished")
    render_refusal(view)
    with st.container(border=True):
        st.markdown("**Main analysis**")
        for line in view["finished"]:
            st.write(line)
        if values.get("report"):
            st.write(values["report"])
    st.caption(view["note"])
    labels = {
        "proposed": f"Run the exploratory analysis with the proposed predictors: {', '.join(view['proposed'])}",
        "own": "Run it with my own predictors",
        "skip": "Skip the exploratory analysis",
    }
    label = st.radio("Next step", [labels[c] for c in GATE_OPTIONS], index=0, key=widget_key("gate_choice", serial))
    choice = next(c for c in GATE_OPTIONS if labels[c] == label)
    chosen = None
    if choice == "own":
        st.caption(f"{view['group_column']} always stays in the model")
        columns = [o["column"] for o in view["options"]]
        chosen = st.multiselect(
            "Other predictors", columns, default=[c for c in view["default_own"] if c in columns],
            help="\n".join(f"{o['column']}: {o['reason']}" for o in view["options"]),
            key=widget_key("gate_own", serial))
    if view["left_out"]:
        with st.expander("Columns left out, and why"):
            for item in view["left_out"]:
                st.write(f"**{item['column']}**: {item['explanation']}")
    if st.button("Submit", type="primary", key=widget_key("gate_submit", serial)):
        return pv.gate_answer(view, choice, chosen)
    return None


def _wrangling(view, serial):
    # the same screen as before M1.5, with serial-keyed widgets
    st.subheader("Review proposed data-cleaning steps")
    st.write("The agent found some data-quality issues and proposes fixing them before analysis:")
    if view["note"]:
        st.info(view["note"])
    decisions = []
    for op in view["operations"]:
        with st.container(border=True):
            st.write(f"**{op['op_type']}** — {op['rationale']}")
            st.caption(f"Alternative considered: {op['alternative']}")
            action = st.radio("Decision", ["approve", "edit", "reject"], horizontal=True,
                              key=widget_key("wrangle_action", serial, op["index"]))
            decision = {"index": op["index"], "action": action}
            if action == "reject":
                reason = st.text_input("Reason for rejecting", key=widget_key("wrangle_reason", serial, op["index"]))
                decision["reason"] = reason or "No reason given"
            elif action == "edit":
                st.caption(f"Current params: {op['params']}")
                st.caption("Editing params via UI isn't built yet in this version — approving with original params instead.")
                decision["action"] = "approve"
            decisions.append(decision)
    if st.button("Submit decisions", key=widget_key("wrangle_submit", serial)):
        return {"decisions": decisions}
    return None


def _clarification(view, serial):
    st.subheader("A quick clarification")
    st.write(view["question"])
    clarification = st.text_input("Your answer", key=widget_key("clarification_input", serial))
    if st.button("Submit answer", key=widget_key("clarification_submit", serial)) and clarification:
        return clarification
    return None


def render_pause(payload, values, serial):
    """The resume value when the user submitted this pause, otherwise None."""
    view = pv.view_for(payload)
    kind = payload.get("type")
    if view.get("unsupported"):
        st.warning(f"Interrupt type '{kind}' isn't wired up in the UI yet.")
        st.json(payload)
        return None
    if kind == "stage_plan_approval":
        return _stage_plan(view, serial)
    if kind == "plan_approval":
        return _plan_approval(view, serial)
    if kind == "stage_gate":
        return _stage_gate(view, serial, values or {})
    if kind == "wrangling_approval":
        return _wrangling(view, serial)
    return _clarification(view, serial)
