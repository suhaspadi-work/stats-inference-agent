"""
Decision records for choices made at run time (M0.2 step 11, G31). Pure
functions: no graph, no model calls, no data access, and no import of
core.nodes.

  stage2.post_hoc   whether pairwise comparisons ran after the omnibus test,
                    and which post-hoc test (Tukey HSD after ANOVA, Dunn's
                    test with Holm adjustment after Kruskal-Wallis, or none)
"""
from __future__ import annotations
import copy
from typing import Any, Optional

from core.decision_record import Candidate, DecisionRecord
from core.selection_records import _p
from core.thresholds import OMNIBUS_ALPHA

POST_HOC_POINT = "stage2.post_hoc"

# The omnibus cut-off of multi_group.execute_multi_group_test, by reference (core.thresholds, M1.4b step 2a).
ALPHA = OMNIBUS_ALPHA

OMNIBUS_METHODS = ("one_way_anova", "kruskal_wallis")
POST_HOC_OPTIONS = ("tukey_hsd", "dunn", "none")
_REQUIRED = ("method", "statistic", "p_value", "post_hoc_method", "pairwise_comparisons")
_POST_HOC_FOR = {"one_way_anova": "tukey_hsd", "kruskal_wallis": "dunn"}
_LABEL = {"one_way_anova": "ANOVA", "kruskal_wallis": "Kruskal-Wallis"}


def _plain(value: Any) -> Any:
    """Copy into plain dicts and lists (tuples become lists); NaN floats are kept as they are."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return copy.deepcopy(value)


def post_hoc_record(result: dict, selection_entry: Optional[dict]) -> DecisionRecord:
    missing = [k for k in _REQUIRED if k not in result]
    if missing:
        raise ValueError(f"post_hoc_record: result is missing key(s): {', '.join(missing)}")
    method, chosen = result["method"], result["post_hoc_method"]
    if method not in OMNIBUS_METHODS:
        raise ValueError(f"post_hoc_record: unknown omnibus method {method!r}; expected one of {OMNIBUS_METHODS}")
    if chosen not in POST_HOC_OPTIONS:
        raise ValueError(f"post_hoc_record: unknown post-hoc method {chosen!r}; expected one of {POST_HOC_OPTIONS}")

    statistic, p_value = float(result["statistic"]), float(result["p_value"])
    significant = p_value < ALPHA
    p_text = _p(p_value)
    omnibus = f"the omnibus test ({method}, statistic = {statistic:.4f}, {p_text})"
    not_significant = f"the omnibus test is not significant ({p_text}, at or above {ALPHA})"

    def follow_up(option: str, test_text: str):
        follows = _POST_HOC_FOR[method] == option
        if follows and significant:
            return True, f"chosen: {omnibus} is significant, below {ALPHA}, so {test_text} compares each pair of groups"
        if follows:
            return False, f"not eligible: {not_significant}"
        other = next(m for m, o in _POST_HOC_FOR.items() if o == option)
        return False, f"not eligible: {test_text.split(' with ')[0]} follows {_LABEL[other]}, but the omnibus test was {_LABEL[method]}"

    reasons = {
        "tukey_hsd": follow_up("tukey_hsd", "Tukey HSD"),
        "dunn": follow_up("dunn", "Dunn's test with Holm adjustment"),
        "none": ((True, f"chosen: {not_significant}, so no post-hoc comparisons are run") if not significant
                 else (False, f"not eligible: the omnibus test is significant ({p_text}, below {ALPHA}), "
                              f"so post-hoc comparisons run")),
    }
    # A candidate that is eligible but was not the one the result reports keeps a plain note.
    candidates = []
    for option in POST_HOC_OPTIONS:
        eligible, reason = reasons[option]
        if eligible and option != chosen:
            reason = "eligible; not chosen by the execution rule"
        candidates.append(Candidate(option=option, eligible=eligible, reason=reason))

    predicted = None
    if selection_entry is not None:
        predicted = (selection_entry.get("evidence") or {}).get("post_hoc_if_omnibus_significant") or None
    if predicted is None:
        consistent = None
    else:
        consistent = (chosen == predicted) if significant else (chosen == "none")

    return DecisionRecord(
        decision_point=POST_HOC_POINT, chosen=chosen, decided_by="code_rule",
        candidates=tuple(candidates), rationale=reasons[chosen][1],
        evidence={
            "omnibus_method": method,
            "omnibus_statistic": statistic,
            "omnibus_p_value": p_value,
            "alpha": ALPHA,
            "n_pairwise_comparisons": len(result["pairwise_comparisons"]),
            "selection_record_id": selection_entry.get("id") if selection_entry is not None else None,
            "post_hoc_if_significant_per_selection": predicted,
            "consistent_with_selection": consistent,
        },
        # These charts hold only summary numbers (pair labels, mean differences, intervals), no raw values.
        chart_data=tuple(_plain(result.get("chart_data") or ())),
    )


def multi_group_execution_records(result: dict, selection_entry: Optional[dict]) -> tuple[DecisionRecord, ...]:
    return (post_hoc_record(result, selection_entry),)
