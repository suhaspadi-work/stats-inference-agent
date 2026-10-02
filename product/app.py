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
from db import init_db, create_session, record_message

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

    st.title("📊 Stats Inference Agent")

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
                result = st.session_state.agent_app.invoke(
                    {"request_text": question, "session_id": username, "dataset": st.session_state.dataset},
                    graph_config,
                )
                st.session_state.graph_result = result
                st.rerun()

        # --- An interrupt is pending: render the right widget ---
        elif "__interrupt__" in st.session_state.graph_result:
            payload = st.session_state.graph_result["__interrupt__"][0].value
            itype = payload.get("type")

            if itype == "plan_approval":
                st.subheader("Review the analysis plan")
                st.write(f"**Question:** {payload['request_text']}")
                st.write(f"**Outcome variable:** {payload['outcome_name']} ({payload['outcome_type']})")
                st.write(f"**Comparing on:** {', '.join(payload['candidate_predictors'])}")
                st.write(f"**Data type:** {payload['causal_status']}")
                if payload.get("classification_rationale"):
                    st.caption(payload["classification_rationale"])

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Approve"):
                        result = st.session_state.agent_app.invoke(
                            Command(resume={"action": "approve"}), graph_config,
                        )
                        st.session_state.graph_result = result
                        st.rerun()
                with col2:
                    reason = st.text_input("Reason for rejecting (optional)", key="reject_reason")
                    if st.button("❌ Reject"):
                        result = st.session_state.agent_app.invoke(
                            Command(resume={"action": "reject", "reason": reason or "No reason given"}), graph_config,
                        )
                        st.session_state.graph_result = result
                        st.rerun()
            elif itype == "wrangling_approval":
                st.subheader("Review proposed data-cleaning steps")
                st.write("The agent found some data-quality issues and proposes fixing them before analysis:")

                if payload.get("auto_approved_note"):
                    st.info(payload["auto_approved_note"])

                decisions = []
                for op in payload["proposed_operations"]:
                    with st.container(border=True):
                        st.write(f"**{op['op_type']}** — {op['rationale']}")
                        st.caption(f"Alternative considered: {op['alternative']}")

                        action = st.radio(
                            "Decision", ["approve", "edit", "reject"],
                            key=f"wrangle_action_{op['index']}", horizontal=True,
                        )
                        decision = {"index": op["index"], "action": action}

                        if action == "reject":
                            reason = st.text_input(
                                "Reason for rejecting", key=f"wrangle_reason_{op['index']}",
                            )
                            decision["reason"] = reason or "No reason given"
                        elif action == "edit":
                            st.caption(f"Current params: {op['params']}")
                            st.caption("Editing params via UI isn't built yet in this version — approving with original params instead.")
                            decision["action"] = "approve"

                        decisions.append(decision)

                if st.button("Submit decisions"):
                    result = st.session_state.agent_app.invoke(
                        Command(resume={"decisions": decisions}), graph_config,
                    )
                    st.session_state.graph_result = result
                    st.rerun()

            elif itype == "clarification_needed":
                st.subheader("A quick clarification")
                st.write(payload["question"])
                clarification = st.text_input("Your answer", key="clarification_input")
                if st.button("Submit answer") and clarification:
                    result = st.session_state.agent_app.invoke(
                        Command(resume=clarification), graph_config,
                    )
                    st.session_state.graph_result = result
                    st.rerun()

            else:
                st.warning(f"Interrupt type '{itype}' isn't wired up in the UI yet.")
                st.json(payload)

        # --- Run finished: show the report ---
        else:
            plan = st.session_state.graph_result.get("plan")
            report = st.session_state.graph_result.get("report")
            test_result = st.session_state.graph_result.get("test_result")

            if plan and plan.status.value == "executed" and report:
                st.subheader("Report")
                st.write(report)
                if test_result:
                    with st.expander("See the underlying statistics"):
                        st.json(test_result)

                record_message(st.session_state.db_session_id, st.session_state.current_question, report, test_result)

                if st.button("Ask another question"):
                    st.session_state.dataset = st.session_state.graph_result.get("dataset", st.session_state.dataset)
                    for key in ("graph_result", "current_question"):
                        st.session_state.pop(key, None)
                    st.rerun()
            else:
                st.warning("The run finished without producing a report.")
                st.json(st.session_state.graph_result)
                if st.button("Start over"):
                    for key in ("graph_result", "current_question"):
                        st.session_state.pop(key, None)
                    st.rerun()

        if st.button("Start a new session"):
            for key in ("dataset", "thread_id", "db_session_id", "agent_app", "graph_result", "current_question"):
                st.session_state.pop(key, None)
            st.rerun()