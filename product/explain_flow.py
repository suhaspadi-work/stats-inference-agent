"""
One on-request explanation (M1.5 step P3b1). Pure: no streamlit, no model. explain_once calls the model only for a
(round, method, question) it has not answered in this run, caches the answer and records it once. A failed record
never loses a computed answer: it is kept and marked saved False.
"""
from run_state import explanation_key, get_explanation, remember_explanation


def explain_once(state, request, explain_fn, record_fn=None):
    """(result, computed): the cached result and False, or the new result and True. explain_fn(method_id,
    round_index, question) returns the explain_method result; an error there propagates and nothing is kept."""
    round_index, method_id, question = request
    key = explanation_key(round_index, method_id, question)
    cached = get_explanation(state, key)
    if cached is not None:
        return cached, False
    result = remember_explanation(state, key, explain_fn(method_id, round_index, question))
    if record_fn is not None:
        try:
            record_fn(result)
        except Exception:
            result["saved"] = False
        else:
            result["saved"] = True
    return result, True
