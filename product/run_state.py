"""
Run bookkeeping for the product (M1.5 step P1). Pure: works on any dict-like, so a plain dict in tests and
st.session_state in the app.
"""

# M1.5 step P3a: the finished question's message id and its cached explanations belong to the run too
RUN_KEYS = ("graph_result", "current_question", "recorded_run", "message_id", "explanations")


def next_pause_serial(state) -> int:
    """Increments state["pause_serial"] (1 when absent) and returns it."""
    state["pause_serial"] = state.get("pause_serial", 0) + 1
    return state["pause_serial"]


def store_result(state, result):
    """Stores a graph result; a result that pauses gets a new pause serial (returned), so the pause's widgets start
    clean. None when the result did not pause."""
    state["graph_result"] = result
    if "__interrupt__" in result:
        return next_pause_serial(state)
    return None


def record_once(state, record_fn) -> bool:
    """Calls record_fn once per finished run: False without calling it when this run is already recorded. When
    record_fn raises, the flag stays unset and the exception propagates, so a later rerun can record."""
    if state.get("recorded_run"):
        return False
    record_fn()
    state["recorded_run"] = True
    return True


def reset_run(state) -> None:
    """Forgets the current run; pause_serial is kept so widget keys never repeat."""
    for key in RUN_KEYS:
        state.pop(key, None)


def explanation_key(round_index, method_id, question) -> str:
    """The cache key of one explanation; no question and a blank one are the same key."""
    return f"{round_index}|{method_id}|{(question or '').strip()}"


def remember_explanation(state, key, result):
    state.setdefault("explanations", {})[key] = result
    return result


def get_explanation(state, key):
    return (state.get("explanations") or {}).get(key)
