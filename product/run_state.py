"""
Run bookkeeping for the product (M1.5 step P1). Pure: works on any dict-like, so a plain dict in tests and
st.session_state in the app.
"""

RUN_KEYS = ("graph_result", "current_question", "recorded_run")


def next_pause_serial(state) -> int:
    """Increments state["pause_serial"] (1 when absent) and returns it."""
    state["pause_serial"] = state.get("pause_serial", 0) + 1
    return state["pause_serial"]


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
