"""
Stats Inference Agent -- product layer, v1.
Step 3: auth + file upload + dataset loading + the agent graph itself
(question -> interrupts -> report), currently handling the plan_approval
interrupt; other interrupt types fall back to a raw JSON display until
wired up individually.
"""
import sys
import tempfile
from pathlib import Path
from datetime import datetime
import re
import time

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from dotenv import load_dotenv

# Make the backend's `core` package importable from this sibling folder.
# This MUST happen before any `from core...` import below.
sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))
load_dotenv(Path(__file__).parent.parent / "agent" / ".env")

from langgraph.types import Command
from core.dataset import LocalDiskStore
from core.ingestion import load_dataset
from core.graph import build_agent_graph
from core.explain import explain_method
from db import (
    init_db, create_session, record_message, list_sessions, list_messages, record_explanation, list_explanations,
)
from run_state import record_once, reset_run, store_result
from pauses import render_pause
from results_views import finished_view
from explain_flow import explain_once
from results_ui import render_finished, render_history

MAX_RETRIES = 4


def is_transient(msg: str) -> bool:
    markers = (
        "429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED", "UNAVAILABLE",
        "RateLimitError", "rate_limit_exceeded", "timed out", "Timeout", "timeout",
    )
    return any(m in msg for m in markers)


def invoke_with_retry(fn):
    """
    Wraps a single agent_app.invoke(...) call with retry logic for transient
    service issues (rate limits, timeouts) -- the same approach eval/runner.py
    uses, now applied to the live product path. Shows the user a waiting
    message rather than letting the app crash on a 429.
    """
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if not is_transient(msg):
                raise
            if attempt == MAX_RETRIES - 1:
                raise
            match = re.search(r"try again in ([\d.]+)s", msg)
            wait = float(match.group(1)) + 3 if match else 20 * (2 ** attempt)
            st.info(f"The model service is briefly busy — retrying in {wait:.0f} seconds...")
            time.sleep(wait)

class AgentRunFailed(Exception):
    """Raised when the agent run fails for a known, expected reason (a deliberately engineered ValueError from the backend) -- the message is already clean and safe to show directly to the user."""


def run_agent_step(fn):
    """
    The single entry point every call site should use. Wraps invoke_with_retry
    (which handles transient service issues) with a second layer that
    distinguishes the backend's OWN deliberate ValueErrors (non-numeric
    outcome, invalid predictor edit, zero variance, wrong group count --
    all designed to fail with a clear, user-safe message) from genuinely
    unexpected exceptions, which should never be shown raw to a user.
    """
    try:
        return invoke_with_retry(fn)
    except ValueError as e:
        raise AgentRunFailed(str(e))
    except Exception:
        raise AgentRunFailed(
            "Something unexpected went wrong processing this request. "
            "Please try a different question, or start a new session."
        )


def handle_explain(request):
    """One on-request explanation (M1.5 step P3b2): computed once per (round, method, question), recorded once
    against the finished question's message, then the screen is redrawn with it."""
    state = st.session_state.graph_result
    try:
        with st.spinner("Writing the explanation..."):
            explain_once(
                st.session_state, request,
                lambda method_id, round_index, question: run_agent_step(
                    lambda: explain_method(method_id, state, round_index, question)),
                lambda result: record_explanation(
                    st.session_state.db_session_id, st.session_state.get("message_id"), result["round"],
                    result["method_id"], result["question"], result["answer"], result["source"]),
            )
    except AgentRunFailed as e:
        st.error(str(e))
        return
    st.rerun()

def render_session_history(session_row, messages):
    """
    Renders one past session's Q&A history, read-only. Deliberately
    separated into its own function: once persistent checkpoints exist
    (a future SqliteSaver swap), a sibling function like
    resume_session(session_row) can be added alongside this one, and the
    sidebar selector extended with a 'Resume' button -- this function
    itself won't need to change when that happens.
    """
    render_history(session_row, messages, list_explanations(session_row["id"]))

st.set_page_config(page_title="Stats Inference Agent", page_icon="📊", layout="wide")

init_db()

# --- Load auth config and build the authenticator ---
with open(Path(__file__).parent / "config.yaml") as f:
    auth_config = yaml.load(f, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    auth_config["credentials"],
    auth_config["cookie"]["name"],
    auth_config["cookie"]["key"],
    auth_config["cookie"]["expiry_days"],
)

# --- Render the login widget ---
authenticator.login()

if st.session_state.get("authentication_status") is False:
    st.error("Username or password is incorrect.")
elif st.session_state.get("authentication_status") is None:
    st.warning("Please enter your username and password.")
elif st.session_state.get("authentication_status"):
    name = st.session_state.get("name")
    username = st.session_state.get("username")

    with st.sidebar:
        st.write(f"Welcome, **{name}**")
        authenticator.logout()

        st.divider()
        st.subheader("Past sessions")
        past_sessions = list_sessions(username)
        if past_sessions:
            session_labels = [f"{s['dataset_name']} ({s['created_at'][:10]})" for s in past_sessions]
            selected_label = st.selectbox(
                "View a past session's history", ["(none)"] + session_labels, key="history_selector",
            )
            if selected_label != "(none)":
                selected_idx = session_labels.index(selected_label)
                st.session_state.viewing_session_id = past_sessions[selected_idx]["id"]
            else:
                st.session_state.pop("viewing_session_id", None)
        else:
            st.caption("No past sessions yet.")

    st.title("📊 Stats Inference Agent")

    if st.session_state.get("viewing_session_id"):
        session_row = next(s for s in list_sessions(username) if s["id"] == st.session_state.viewing_session_id)
        messages = list_messages(st.session_state.viewing_session_id)
        render_session_history(session_row, messages)
        st.stop()

    # One LocalDiskStore per logged-in session, created once and reused.
    if "store" not in st.session_state:
        store_dir = Path(__file__).parent / "agent_data" / username
        st.session_state.store = LocalDiskStore(base_dir=store_dir)

    if "dataset" not in st.session_state:
        st.subheader("Upload a dataset to get started")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = Path(tmp.name)

            dataset_name = uploaded_file.name.rsplit(".", 1)[0]
            thread_id = f"{username}-{dataset_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

            dataset = load_dataset(
                tmp_path, session_id=username, name=dataset_name, store=st.session_state.store,
            )
            session_id = create_session(username, dataset_name, thread_id)

            st.session_state.dataset = dataset
            st.session_state.thread_id = thread_id
            st.session_state.db_session_id = session_id

            st.rerun()
    else:
        st.success(f"Dataset loaded: **{st.session_state.dataset.name}** (version {st.session_state.dataset.version})")

        if "agent_app" not in st.session_state:
            st.session_state.agent_app = build_agent_graph(st.session_state.store)

        graph_config = {"configurable": {"thread_id": st.session_state.thread_id}}

        # --- No run in progress: show the question input ---
        if "graph_result" not in st.session_state:
            question = st.text_input("What would you like to know about this data?")
            if st.button("Ask") and question:
                st.session_state.current_question = question
                try:
                    result = run_agent_step(lambda: st.session_state.agent_app.invoke(
                        {"request_text": question, "session_id": username, "dataset": st.session_state.dataset},
                        graph_config,
                    ))
                    store_result(st.session_state, result)
                    st.rerun()
                except AgentRunFailed as e:
                    st.error(str(e))

        # --- An interrupt is pending: render the right widget ---
        elif "__interrupt__" in st.session_state.graph_result:
            payload = st.session_state.graph_result["__interrupt__"][0].value
            answer = render_pause(payload, st.session_state.graph_result, st.session_state.get("pause_serial", 0))
            if answer is not None:
                try:
                    result = run_agent_step(lambda: st.session_state.agent_app.invoke(
                        Command(resume=answer), graph_config,
                    ))
                    store_result(st.session_state, result)
                    st.rerun()
                except AgentRunFailed as e:
                    st.error(str(e))

        # --- Run finished: show the report ---
        else:
            plan = st.session_state.graph_result.get("plan")
            report = st.session_state.graph_result.get("report")
            test_result = st.session_state.graph_result.get("test_result")

            view = finished_view(st.session_state.graph_result)
            if view["kind"] == "report":
                request = render_finished(view, st.session_state.get("pause_serial", 0),
                                          st.session_state.get("explanations"))

                record_once(st.session_state, lambda: st.session_state.__setitem__("message_id", record_message(
                    st.session_state.db_session_id, st.session_state.current_question, report, test_result,
                    stage_reports=st.session_state.graph_result.get("stage_reports"))))

                if request is not None:
                    handle_explain(request)

                if st.button("Ask another question"):
                    st.session_state.dataset = st.session_state.graph_result.get("dataset", st.session_state.dataset)
                    reset_run(st.session_state)
                    st.rerun()
            else:
                if st.session_state.graph_result.get("clarification_needed"):
                    st.warning(st.session_state.graph_result["clarification_needed"])
                    st.write("You can ask a new question below.")
                else:
                    st.warning("The run finished without producing a report.")
                if st.button("Try a new question"):
                    reset_run(st.session_state)
                    st.rerun()

        if st.button("Start a new session"):
            for key in ("dataset", "thread_id", "db_session_id", "agent_app", "graph_result", "current_question",
                        "recorded_run", "message_id", "explanations"):
                st.session_state.pop(key, None)
            st.rerun()