"""
Stats Inference Agent -- product layer, v1.
Step 1: auth skeleton only -- login/logout working, nothing else yet.
"""
import sys
from pathlib import Path

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# Make the backend's `core` package importable from this sibling folder.
sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))

st.set_page_config(page_title="Stats Inference Agent", page_icon="📊", layout="wide")

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
    st.write(f"Logged in as `{username}`. This is where the real product will go next.")