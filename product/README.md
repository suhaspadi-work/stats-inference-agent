# Stats Inference Agent: product

The Streamlit app in front of the agent in `../agent`. It adds a login, file upload, the review screens for the
agent's pauses and a SQLite history of past sessions.

## Set up

From this folder:

```
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` includes the agent's own requirements (`../agent/requirements.txt`). The agent reads its model
key from `../agent/.env`, which is not in git.

`config.yaml` holds the login credentials and the cookie settings for streamlit-authenticator. It is not in git:
create it by hand in this folder (see the streamlit-authenticator documentation for its format, with hashed
passwords).

## Run

From this folder, with the virtual environment active:

```
python -m streamlit run app.py
```

The app creates `app_data.db` (the session history) and `agent_data/` (uploaded datasets) here; neither is in git.
