"""
Stats Inference Agent -- product layer, v1.
Step 3 (piece 1): auth + file upload + dataset loading, wired to the
backend's LocalDiskStore and the product's own SQLite session tracking.
The actual agent graph (question -> interrupts -> report) comes next.
"""
import sys
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# Make the backend's `core` package importable from this sibling folder.
sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))

from core.dataset import LocalDiskStore
from core.ingestion import load_dataset
from db import init_db, create_session

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
        st.write(f"Thread ID: `{st.session_state.thread_id}`")

        if st.button("Start a new session"):
            for key in ("dataset", "thread_id", "db_session_id"):
                st.session_state.pop(key, None)
            st.rerun()