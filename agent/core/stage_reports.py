"""
Reports per stage (M1.4b), composed by code from the decision records of the current question. Pure: no model
call, no data access, and no import of core.nodes; inputs are never mutated. Every sentence restates a value a
record holds, so no number in a stage report is model-generated. Each entry names the records it was built from.

  stage1   prepare and describe: what was checked, what was changed and why, what was left as is, what was
           described
"""
from __future__ import annotations

import copy

from core.cleaning_records import APPROVAL_POINT, EXECUTION_POINT, FIX_POINT, SCAN_POINT
from core.describe_records import DESCRIBE_EXECUTION_POINT
from core.stage_plan_records import STAGE_PLAN_POINT

STAGE1_TITLE = "Stage 1: prepare and describe"


def current_question_records(log: list) -> list:
    """The records from the last planner.stage_plan record to the end (the whole log when there is none)."""
    start = 0
    for i, entry in enumerate(log):
        if entry.get("decision_point") == STAGE_PLAN_POINT:
            start = i
    return list(log[start:])


def make_stage_report_entry(stage: str, round_index, title: str, composed: dict, fallback: bool = False,
                            exploratory: bool = False) -> dict:
    """One stage_reports entry; text is the title line, then each section as "<heading>\\n<text>"."""
    sections = [{"heading": s["heading"], "text": s["text"]} for s in composed["sections"]]
    blocks = [title] + [f"{s['heading']}\n{s['text']}" for s in sections]
    return {"stage": stage, "round": round_index, "title": title, "text": "\n\n".join(blocks),
            "sections": sections, "fallback": bool(fallback), "exploratory": bool(exploratory),
            "source_records": list(composed["source_records"])}


def _bare(text) -> str:
    """Text without a trailing period, so a sentence built around it ends with exactly one."""
    return str(text).rstrip().rstrip(".")


def _latest(records: list, point: str):
    return next((r for r in reversed(records) if r.get("decision_point") == point), None)


def _checked(scan) -> str:
    if scan is None:
        return "No data-quality scan was recorded."
    ev = scan["evidence"]
    lines = [f"Checked {ev['rows']} rows and {ev['columns_scanned']} columns."]
    duplicates = ev.get("duplicate_row_count") or 0
    if duplicates > 0:
        lines.append(f"Found {duplicates} exact duplicate rows.")
    for col in ev.get("columns_above_missing_threshold") or []:
        lines.append(f"{col['column']}: {col['missing_pct']:.1%} missing (threshold {ev['missing_threshold']:.0%}).")
    if len(lines) == 1:
        lines.append("No data-quality issues found.")
    return "\n".join(lines)


# The filter conditions core.wrangling supports, in words
_FILTER_WORDS = {
    "equals": "equals {value}",
    "not_equals": "does not equal {value}",
    "greater_than": "is greater than {value}",
    "less_than": "is less than {value}",
    "is_null": "has no value",
    "not_null": "has a value",
}


def _what(op_type: str, params: dict) -> str:
    if op_type == "dedupe":
        subset = params.get("subset")
        return "Removed duplicate rows" + (f" (comparing {', '.join(str(c) for c in subset)})" if subset else "")
    if op_type == "filter":
        column, condition, value = params.get("column"), params.get("condition"), params.get("value")
        if condition in _FILTER_WORDS:
            return f"Kept rows where {column} " + _FILTER_WORDS[condition].format(value=value)
        text = f"Kept rows where {column} {condition}"  # a condition without wording keeps the stored text
        if value is not None:
            text += f" {value}"
        return text
    return f"{op_type} {params}"


def _alternative(fix: dict):
    """The model's suggested alternative: a model-generated candidate other than the chosen operation."""
    return next((c for c in fix.get("candidates") or []
                 if c.get("generated_by") == "model" and c.get("option") != fix["chosen"]), None)


def _decision_text(approval) -> str:
    if approval is None:
        return "none recorded"
    chosen = approval["chosen"]
    if chosen == "no_decision":
        text = "no decision received, so it was not run"
    elif chosen == "reject":
        text = f"rejected: {_bare(approval['rationale'])}"
    elif chosen == "edit":
        text = "approved with your edits"
    else:  # approve, auto_approve
        text = "approved"
    if approval["decided_by"] == "code_rule" and chosen != "no_decision":
        text += " automatically"
    return text


def _changed(records: list, used: set) -> tuple[str, list]:
    """The Changed section text and the Left as is lines."""
    fixes = [r for r in records if r.get("decision_point") == FIX_POINT]
    if not fixes:
        return "No changes were proposed.", []
    lines, left = [], []
    for k, fix in enumerate(fixes, start=1):
        used.add(fix["id"])
        approval = next((r for r in records if r.get("decision_point") == APPROVAL_POINT
                         and r["evidence"].get("fix_record_id") == fix["id"]), None)
        execution = None
        if approval is not None:
            used.add(approval["id"])
            execution = next((r for r in records if r.get("decision_point") == EXECUTION_POINT
                              and r["evidence"].get("approval_record_id") == approval["id"]), None)
        if execution is not None:
            used.add(execution["id"])
        # the params that ran (after an edit) when the operation ran, otherwise the proposed ones
        params = (execution or fix)["evidence"].get("params") or {}
        parts = [f"{k}. {_what(fix['chosen'], params)}.", f"Why: {_bare(fix['rationale'])}."]
        alt = _alternative(fix)
        if alt is not None:
            parts.append(f"Alternative the model suggested: {alt['option']} ({_bare(alt['reason'])}).")
        parts.append(f"Your decision: {_decision_text(approval)}.")
        if execution is not None and execution["chosen"] == "failed":
            parts.append(f"Failed: {_bare(execution['evidence'].get('failure_reason'))}.")
        elif execution is not None:
            parts.append(f"Effect: {execution['evidence']['rows_before']} to {execution['evidence']['rows_after']} rows.")
        lines.append(" ".join(parts))
        if approval is not None and approval["chosen"] == "reject":
            ev = fix["evidence"]
            about = ev.get("description") or ", ".join(ev.get("affected_columns") or [])
            left.append(f"{ev['issue_type']} ({about}) was left unaddressed because: {approval['rationale']}")
    return "\n".join(lines), left


def _described(record) -> str:
    if record is None:
        return "The description was not run (it was switched off or does not apply)."
    ev = record["evidence"]
    withheld = ", ".join(ev.get("withheld_text_columns") or []) or "none"
    return f"{ev['row_count']} rows and {ev['column_count']} columns described. Text values withheld for: {withheld}."


def stage1_report(records: list, stage_plan: dict) -> dict:
    """{"sections": [{"heading", "text"}], "source_records": [ids]} from the records of this question only."""
    records = copy.deepcopy(list(records))
    used = set()
    scan = _latest(records, SCAN_POINT)
    if scan is not None:
        used.add(scan["id"])
    sections = [{"heading": "Checked", "text": _checked(scan)}]
    changed, left = _changed(records, used)
    sections.append({"heading": "Changed", "text": changed})
    if left:
        sections.append({"heading": "Left as is", "text": "\n".join(left)})
    if (stage_plan or {}).get("question_type") != "descriptive":
        describe = _latest(records, DESCRIBE_EXECUTION_POINT)
        if describe is not None:
            used.add(describe["id"])
        sections.append({"heading": "Described", "text": _described(describe)})
    ordered = []
    for r in records:
        if r.get("id") in used and r["id"] not in ordered:
            ordered.append(r["id"])
    return {"sections": sections, "source_records": ordered}
