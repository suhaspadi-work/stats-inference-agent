"""
The rules a predictor of the exploratory regression must fit (M1.3b step 2b). One function, check_predictor,
is used both by the agent's proposal (core.round_two) and by every choice the user makes at the boundary
gate, so the two can never disagree. Pure: no graph, no model call, no import of core.nodes, and the inputs
are never mutated.

A verdict is a dict {"column", "rule", "reason", "explanation"}: reason is the short text shown in a list,
explanation the sentence that says why.
"""
from __future__ import annotations

from core.capabilities import is_numeric_dtype_name
from core.model_safe_view import ModelSafeView
from core.regression import MAX_CATEGORICAL_LEVELS


def _verdict(column, rule, reason, explanation) -> dict:
    return {"column": column, "rule": rule, "reason": reason, "explanation": explanation}


def _is_float(dtype) -> bool:
    return str(dtype).startswith("float")


def check_predictor(profile: ModelSafeView, name: str, outcome_name: str):
    """The first rule the column breaks, as a verdict; None when the column fits."""
    columns = {c.name: c for c in profile.columns}
    if name not in columns:
        return _verdict(name, "unknown_column", "not a column in the dataset",
                        f"'{name}' is not a column in this dataset. Available columns: {sorted(columns)}.")
    col = columns[name]
    if name == outcome_name:
        return _verdict(name, "outcome", "the outcome cannot also be a predictor",
                        "it is the outcome being explained, so it cannot also be used to explain itself")
    if col.distinct_count < 2:
        return _verdict(name, "constant", "constant or empty",
                        "a column with a single value (or no values) cannot explain any difference in the outcome")
    if not _is_float(col.dtype) and col.distinct_count == profile.row_count - col.missing_count:
        return _verdict(name, "identifier_like", "identifier-like: every value is distinct",
                        "every row has its own value in this column, so a model would fit each row exactly and tell "
                        "you nothing that carries over to other data")
    if not is_numeric_dtype_name(col.dtype) and col.distinct_count > MAX_CATEGORICAL_LEVELS:
        return _verdict(name, "too_many_levels", f"text column with more than {MAX_CATEGORICAL_LEVELS} distinct values",
                        f"each of its {col.distinct_count} levels becomes its own coefficient; with more than "
                        f"{MAX_CATEGORICAL_LEVELS} levels the model becomes hard to read and unstable")
    return None


def validate_predictor_choice(profile: ModelSafeView, chosen, outcome_name: str, group_column: str,
                              max_others: int) -> dict:
    """
    Check a user's predictor list against every rule at once: {"ok", "predictors", "refusals"}. When ok,
    predictors is the list in the order given with the grouping column moved to the front.
    """
    if not isinstance(chosen, (list, tuple)) or not all(isinstance(n, str) for n in chosen):
        return {"ok": False, "predictors": [], "refusals": [
            _verdict(None, "not_a_list", "predictors must be a list of column names",
                     "predictors must be a list of column names")]}

    refusals = []
    if group_column not in chosen:
        refusals.append(_verdict(group_column, "group_missing", "the grouping column must stay",
                                 "the grouping column of the main analysis must stay in the model, because the "
                                 "exploratory model adjusts that comparison"))
    seen, repeated = set(), []
    for name in chosen:
        if name in seen and name not in repeated:
            repeated.append(name)
        seen.add(name)
    refusals += [_verdict(name, "duplicate", "listed twice", "it was listed more than once") for name in repeated]

    others = list(dict.fromkeys(n for n in chosen if n != group_column))
    for name in others:
        verdict = check_predictor(profile, name, outcome_name)
        if verdict is not None:
            refusals.append(verdict)
    if len(others) > max_others:
        n = len(others)
        refusals.append(_verdict(None, "cap", f"beyond the limit of {max_others} other predictors",
                                 f"you chose {n} other predictors; the limit is {max_others}; remove {n - max_others}"))
    if not others:
        refusals.append(_verdict(None, "none_chosen", "no other predictor",
                                 "choose at least one other predictor besides the grouping column"))

    if refusals:
        return {"ok": False, "predictors": [], "refusals": refusals}
    return {"ok": True, "predictors": [group_column, *others], "refusals": []}
