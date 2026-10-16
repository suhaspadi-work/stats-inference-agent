"""
What the finished screen and the history show (M1.5 step P3a). Pure: no streamlit. finished_view turns the graph's
final state into the summary, one block per analysis round (narrative, "how this was chosen" with the collapsed
sections folded, a fallback note, the explain options, a statistics table) and a data overview; history_view does
the same from a stored message. Only the CURRENT question is shown: rounds, stage reports and the decision log
accumulate across questions on one thread (G42). core is imported only inside guarded helpers.
"""
import copy
import json
import math
from numbers import Integral, Real

from pause_views import method_label

FALLBACK_NOTE = ("This narrative was written from the computed facts, because the model's own text used causal "
                 "language twice.")
NO_REPORT_MESSAGE = "The run finished without producing a report."
WITHHELD_SUMMARY = "values are not shown (too many distinct values)"
MAX_PAIRS = 10
MAX_COEFFICIENTS = 12
_FALLBACKS = ("facts_fallback", "minimal_fallback")
_DESCRIBE_POINT = "stage1.describe.execution"
_ANALYSES = ("two_group", "multi_group", "regression")


def format_number(x) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, Integral):
        return str(int(x))
    if isinstance(x, Real):
        return "n/a" if math.isnan(x) else f"{float(x):.3g}"
    return str(x)


def strip_charts(value):
    """A deep copy without any "chart_data" key at any depth, with tuples as lists (for st.json)."""
    if isinstance(value, dict):
        return {k: strip_charts(v) for k, v in value.items() if k != "chart_data"}
    if isinstance(value, (list, tuple)):
        return [strip_charts(v) for v in value]
    return copy.deepcopy(value)


def _high_cardinality_threshold() -> int:
    try:
        from core.ingestion import HIGH_CARDINALITY_THRESHOLD
    except ImportError:
        return 20
    return HIGH_CARDINALITY_THRESHOLD


def _current_question_records(log) -> list:
    try:
        from core.stage_reports import current_question_records
    except ImportError:
        start = max((i for i, e in enumerate(log) if e.get("decision_point") == "planner.stage_plan"), default=0)
        return list(log[start:])
    return current_question_records(log)


# --- statistics ------------------------------------------------------------------------------------------------------

def _row(label, value) -> dict:
    return {"label": label, "value": value}


def _more(rows, items, cap) -> list:
    if len(items) > cap:
        rows.append(_row(f"and {len(items) - cap} more", ""))
    return rows


def statistics_rows(result) -> list:
    """Plain {"label", "value"} rows for the three analysis results; a missing key skips its row."""
    result = result or {}
    kind = result.get("result_type")
    rows = []
    if kind not in ("two_group", "multi_group", "regression"):
        return [_row("Result", "see the report")]
    if "method" in result:
        rows.append(_row("Method", method_label(result["method"])))
    if kind == "two_group":
        a = result.get("group_a_label", "the first group")
        b = result.get("group_b_label", "the second group")
        if "p_value" in result:
            rows.append(_row("p-value", format_number(result["p_value"])))
        if "effect_size" in result:
            # the measure and its sign follow the method (core/testing.py: both are B minus A)
            label = (f"Effect size (rank-biserial correlation, positive when {b} tends higher)"
                     if result.get("method") == "mann_whitney_u" else f"Effect size (Cohen's d, positive when {b} is higher)")
            rows.append(_row(label, format_number(result["effect_size"])))
        ci = result.get("confidence_interval")
        if ci is not None and len(ci) == 2:
            rows.append(_row(f"95% confidence interval for {b} minus {a}",
                             f"{format_number(ci[0])} to {format_number(ci[1])}"))
        for side in ("a", "b"):
            if f"group_{side}_mean" in result:
                label = result.get(f"group_{side}_label", f"Group {side.upper()}")
                n = result.get(f"group_{side}_n")
                rows.append(_row(str(label), f"mean {format_number(result[f'group_{side}_mean'])}"
                                             + (f" (n = {format_number(n)})" if n is not None else "")))
    elif kind == "multi_group":
        if "p_value" in result:
            rows.append(_row("p-value", format_number(result["p_value"])))
        ns = result.get("group_ns") or {}
        for group, mean in (result.get("group_means") or {}).items():
            n = ns.get(group)
            rows.append(_row(str(group), f"mean {format_number(mean)}" + (f" (n = {format_number(n)})" if n is not None else "")))
        pairs = result.get("pairwise_comparisons") or []
        for c in pairs[:MAX_PAIRS]:
            value = f"difference {format_number(c.get('mean_diff'))}, p = {format_number(c.get('p_value'))}"
            rows.append(_row(f"{c.get('group_a')} vs {c.get('group_b')}", value + (", significant" if c.get("significant") else "")))
        _more(rows, pairs, MAX_PAIRS)
    else:
        for key, label in (("r_squared", "R-squared"), ("adj_r_squared", "Adjusted R-squared"),
                           ("f_p_value", "Overall F-test p-value"), ("n_observations", "Observations")):
            if key in result:
                rows.append(_row(label, format_number(result[key])))
        coefficients = result.get("coefficients") or []
        for c in coefficients[:MAX_COEFFICIENTS]:
            rows.append(_row(str(c.get("name")), f"estimate {format_number(c.get('estimate'))}, "
                                                 f"p = {format_number(c.get('p_value'))}"))
        _more(rows, coefficients, MAX_COEFFICIENTS)
    return rows


# --- the data overview -------------------------------------------------------------------------------------------------

def _numeric_summary(c) -> str:
    parts = [f"{name} {format_number(c[key])}" for key, name in (("mean", "mean"), ("median", "median"), ("std", "std"))
             if c.get(key) is not None]
    if c.get("min") is not None and c.get("max") is not None:
        parts.append(f"range {format_number(c['min'])} to {format_number(c['max'])}")
    return ", ".join(parts)


def overview_rows(descriptive_result):
    """One row per column of the describe result, or None. The describe step holds no missing counts, so
    "missing" is "n/a"; a withheld text column shows no value at all."""
    stats = (descriptive_result or {}).get("column_stats") if isinstance(descriptive_result, dict) else None
    if not stats:
        return None
    threshold = _high_cardinality_threshold()
    rows = []
    for c in stats:
        numeric = bool(c.get("is_numeric"))
        distinct = c.get("distinct_count") or 0
        if numeric:
            summary = _numeric_summary(c)
        elif distinct > threshold:
            summary = WITHHELD_SUMMARY
        elif c.get("top_value_counts"):
            summary = "most common: " + ", ".join(f"{level} ({count})" for level, count in list(c["top_value_counts"])[:3])
        else:
            summary = f"text column with {distinct} distinct values"
        rows.append({"column": c.get("name"), "type": "numeric" if numeric else "text", "distinct": str(distinct),
                     "missing": "n/a", "summary": summary})
    return rows


# --- the finished screen -------------------------------------------------------------------------------------------------

def _split_sections(entry):
    sections = (entry or {}).get("sections") or []
    visible = [{"heading": s["heading"], "text": s["text"]} for s in sections if not s.get("collapsed")]
    folded = [{"heading": s["heading"], "text": s["text"]} for s in sections if s.get("collapsed")]
    return visible, folded


def _current_entries(stage_reports) -> list:
    entries = list(stage_reports or [])
    starts = [i for i, e in enumerate(entries) if e.get("stage") == "stage1"]
    return entries[starts[-1]:] if starts else entries


def _summary_of(entries):
    summary = next((e for e in reversed(entries) if e.get("stage") == "summary"), None)
    return {k: summary.get(k) for k in ("title", "text", "sections")} if summary is not None else None


def _round_view(entry, analysis) -> dict:
    visible, folded = _split_sections(analysis)
    fallback = bool((analysis or {}).get("fallback"))
    method = entry.get("method")
    return {
        "round": entry.get("round"),
        "exploratory": bool(entry.get("exploratory")),
        "capability": entry.get("capability"),
        "method_label": method_label(method) if method else str(entry.get("capability")),
        "narrative": entry.get("report"),
        "analysis": analysis,
        "fallback": fallback,
        "fallback_note": FALLBACK_NOTE if fallback else None,
        "visible_sections": visible,
        "folded_sections": folded,
        "explain_options": list((analysis or {}).get("explain_options") or []),
        "statistics": statistics_rows(entry.get("result")) if entry.get("result") is not None else [],
        # the full result for a "details" view, without chart data; a descriptive round has none
        "details": strip_charts(entry.get("result")) if entry.get("capability") in _ANALYSES else None,
    }


def current_overview(values):
    """The overview of the CURRENT question's describe or eda result, or None. A result left in state by an earlier
    question on the same thread is never shown."""
    values = values or {}
    plan = values.get("plan")
    records = _current_question_records(list(values.get("decision_log") or []))
    question_type = getattr(getattr(plan, "question_type", None), "value", None)
    described = any(r.get("decision_point") == _DESCRIBE_POINT for r in records) or question_type == "descriptive"
    return overview_rows(values.get("descriptive_result")) if described else None


_OVERVIEW_HEADINGS = (("column", "Column"), ("type", "Type"), ("distinct", "Distinct values"), ("missing", "Missing"),
                      ("summary", "Summary"))


def overview_table(rows):
    """{"columns", "rows"} for display, or None. A column that is "n/a" in every row is left out (the describe step
    holds no missing counts, G62)."""
    if not rows:
        return None
    kept = [(key, heading) for key, heading in _OVERVIEW_HEADINGS if any(r.get(key) != "n/a" for r in rows)]
    return {"columns": [heading for _, heading in kept],
            "rows": [{heading: r.get(key) for key, heading in kept} for r in rows]}


_EXPLANATION_CAPTIONS = {
    "model": "Written by the model from the computed facts and the recorded reasons.",
    "model_retry": "Written by the model after one correction.",
    "glossary_fallback": "The model's answer did not pass the checks, so this is the glossary description.",
}


def explanation_caption(source) -> str:
    return _EXPLANATION_CAPTIONS.get(source, "")


def explanation_items(explanations, round_index) -> list:
    """The explanations of one round, in the order they were asked, ready to show."""
    return [{"method_label": method_label(r.get("method_id")), "question": r.get("question"), "answer": r.get("answer"),
             "caption": explanation_caption(r.get("source")), "saved": r.get("saved", True)}
            for r in (explanations or {}).values() if r.get("round") == round_index]


def finished_view(values) -> dict:
    values = values or {}
    plan = values.get("plan")
    status = getattr(getattr(plan, "status", None), "value", None)
    if plan is None or status != "executed" or not values.get("report"):
        return {"kind": "no_report", "message": values.get("clarification_needed") or NO_REPORT_MESSAGE}

    entries = _current_entries(values.get("stage_reports"))
    analyses = [e for e in entries if e.get("stage") == "analysis"]
    by_round = {e.get("round"): e for e in analyses}
    all_rounds = list(values.get("rounds") or [])
    rounds = [r for r in all_rounds if r.get("round") in by_round] if analyses else all_rounds[-1:]

    round_views = [_round_view(r, by_round.get(r.get("round"))) for r in rounds]
    return {
        "kind": "report",
        "question": getattr(plan, "request_text", None),
        "stage1": next((e for e in entries if e.get("stage") == "stage1"), None),
        "summary": _summary_of(entries),
        "rounds": round_views,
        "overview": current_overview(values),
        "fallback_any": any(r["fallback"] for r in round_views),
    }


# --- the history --------------------------------------------------------------------------------------------------------

def _get(row, key):
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


def _parse(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def history_view(message_row, explanation_rows) -> dict:
    """A stored question: its report, summary, analysis entries, statistics and its own explanations."""
    reports = _parse(_get(message_row, "stage_reports_json"))
    reports = reports if isinstance(reports, list) else []
    entries = [e for e in reports if isinstance(e, dict)]
    analysis = []
    for e in entries:
        if e.get("stage") == "analysis":
            visible, folded = _split_sections(e)
            analysis.append({"title": e.get("title"), "round": e.get("round"), "visible_sections": visible,
                             "folded_sections": folded})
    result = _parse(_get(message_row, "test_result_json"))
    message_id = _get(message_row, "id")
    return {
        "question": _get(message_row, "question_text"),
        "report": _get(message_row, "report_text"),
        "created_at": _get(message_row, "created_at"),
        "summary": _summary_of(entries),
        "analysis": analysis,
        "statistics": statistics_rows(result) if isinstance(result, dict) else [],
        "explanations": [{"round": _get(r, "round"), "method_label": method_label(_get(r, "method_id")),
                          "question": _get(r, "question"), "answer": _get(r, "answer"), "source": _get(r, "source")}
                         for r in explanation_rows or [] if _get(r, "message_id") == message_id],
    }
