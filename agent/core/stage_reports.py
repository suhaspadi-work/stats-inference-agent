"""
Reports per stage (M1.4b), composed by code from the decision records of the current question. Pure: no model
call, no data access, and no import of core.nodes; inputs are never mutated. Every sentence restates a value a
record holds, so no number in a stage report is model-generated. Each entry names the records it was built from.

  stage1   prepare and describe: what was checked, what was changed and why, what was left as is, what was
           described
  analysis how this was chosen, one per analysis round (M1.4b step 2b): the test (from the method glossary), why
           this test, your plan decision, for an exploratory round the predictors and your choice at the boundary,
           and whose text the narrative is
  summary  the overall summary of the question: the question, Stage 1, the results of every round, the caveats
"""
from __future__ import annotations

import copy

from core.cleaning_records import APPROVAL_POINT, EXECUTION_POINT, FIX_POINT, SCAN_POINT
from core.describe_records import DESCRIBE_EXECUTION_POINT
from core.draft_plan_records import PREDICTORS_POINT
from core.execution_records import POST_HOC_POINT
from core.freeze_records import PLAN_APPROVAL_POINT
from core.method_glossary import GLOSSARY, render_glossary
from core.plan import unaddressed_text
from core.quality_records import LINKS_POINT, QUALITY_SNAPSHOT_POINT
from core.report_records import REPORT_NARRATIVE_POINT
from core.wrangling import describe_operation
from core.round_records import ROUND_START_POINT, STAGE_GATE_POINT
from core.selection_records import ENCODING_POINT, REGRESSION_METHOD_POINT, TEST_SELECTION_POINT
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
                            exploratory: bool = False, explain_options=None) -> dict:
    """One stage_reports entry; text is the title line, then each section as "<heading>\\n<text>". An analysis
    entry also gets "explain_options" (M1.4b step 3); every other entry is unchanged."""
    # a section keeps "collapsed" only when it is true, so entries without collapsed sections are unchanged
    sections = [{"heading": s["heading"], "text": s["text"], **({"collapsed": True} if s.get("collapsed") else {})}
                for s in composed["sections"]]
    blocks = [title] + [f"{s['heading']}\n{s['text']}" for s in sections]
    entry = {"stage": stage, "round": round_index, "title": title, "text": "\n\n".join(blocks),
             "sections": sections, "fallback": bool(fallback), "exploratory": bool(exploratory),
             "source_records": list(composed["source_records"])}
    if explain_options is not None:
        entry["explain_options"] = [dict(o) for o in explain_options]
    return entry


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
    for t in ev.get("text_issues") or []:  # M2.3a step 2: counts only, never a level
        lines.append(f"{t['column']}: {t['levels_before']} distinct values reduce to {t['levels_after']} after trimming "
                     "spaces and ignoring case." if t["levels_after"] < t["levels_before"] else
                     f"{t['column']}: {t['cells_changed_trim']} value(s) have extra spaces.")
    for t in ev.get("numeric_text_columns") or []:
        lines.append(f"{t['column']}: stored as text, but {t['parse_rate']:.1%} of its values are numbers.")
    for t in ev.get("joinable_tables") or []:  # M2.3b
        lines.append(f"{t['table']}: can be joined to the analysis table on {t['left_key']}.")
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


def _what(op_type: str, params: dict, columns_scanned=None) -> str:
    if op_type == "dedupe":
        subset = params.get("subset")
        if subset and columns_scanned is not None and len(subset) == columns_scanned:
            return "Removed duplicate rows (comparing all columns)"
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


def _effect(ev) -> str:
    """M2.3a step 2: the Effect line from the executed numbers; an operation without details keeps the row counts."""
    d, op = ev.get("details") or {}, ev.get("op_type")
    if op == "impute" and d:
        text = f"Effect: {d['cells_filled']} value(s) filled"
        if d.get("mean_before") is not None and d.get("mean_after") is not None:
            text += f"; the mean of {d['column']} moves from {d['mean_before']:.4g} to {d['mean_after']:.4g}"
            if d.get("std_before") is not None and d.get("std_after") is not None:
                text += f" and its spread from {d['std_before']:.4g} to {d['std_after']:.4g}"
        text += "."
        if d.get("fallback_cells"):
            text += f" {d['fallback_cells']} cell(s) used the overall value because their group had no observed value."
        return text
    if op == "drop_column" and d:
        return f"Effect: the column {d['column']} was removed."
    if op == "normalize_text" and d:
        return (f"Effect: {d['levels_before']} distinct values became {d['levels_after']}; "
                f"{d['cells_changed']} value(s) changed.")
    if op == "cast" and d.get("to_type") == "numeric":
        return (f"Effect: {d['became_missing']} value(s) that were not numbers became missing." if d["became_missing"]
                else "Effect: every value was converted.")
    if op == "join" and d:  # M2.3b
        added = len(d.get("columns_added") or [])
        if d.get("how") == "inner":
            return f"Effect: {ev['rows_before']} to {ev['rows_after']} rows; {added} column(s) added."
        text = f"Effect: {ev['rows_before']} rows stay {ev['rows_after']}; {added} column(s) added; "
        unmatched = d.get("unmatched_left_rows") or 0
        if unmatched:
            return text + f"{d.get('new_missing_cells', 0)} new missing value(s) from {unmatched} row(s) with no match."
        return text + "no new missing values."
    return f"Effect: {ev['rows_before']} to {ev['rows_after']} rows."


def _operation_words(fix, params, columns_scanned, execution=None) -> str:
    """M2.3a: the chosen candidate's plain words; an edited operation (other params) or an older record falls
    back to the wording built from its op_type and params. M2.3b: a join that ran is re-rendered from its executed
    details, so the sentence names the columns it added."""
    ev = fix.get("evidence") or {}
    if (ev.get("op_type") or fix["chosen"]) == "join":
        details = ((execution or {}).get("evidence") or {}).get("details") if execution is not None else None
        if details:
            return _bare(describe_operation("join", params, details))
        if ev.get("what_was_run") and params == (ev.get("params") or {}):
            return _bare(ev["what_was_run"])
        return _bare(describe_operation("join", params))
    if ev.get("what_was_run") and params == (ev.get("params") or {}):
        return _bare(ev["what_was_run"])
    return _what(ev.get("op_type") or fix["chosen"], params, columns_scanned)


def _why_label(fix) -> str:
    """M2.3a: the reason's label; it says when the model's choice needed its one correction."""
    retried = (fix.get("evidence") or {}).get("proposal_retried")
    return "Why (the model's reason, after one correction)" if retried else "Why (the model's reason)"


def _alternative_words(fix):
    """(what was run, because) of the model's alternative: from the record's evidence (M2.3a), else from an older
    record's model-generated candidate."""
    ev = fix.get("evidence") or {}
    alt_id = ev.get("alternative_candidate_id")
    if alt_id is not None:
        candidate = next((c for c in fix.get("candidates") or [] if c.get("option") == alt_id), None)
        words = ((candidate or {}).get("evidence") or {}).get("what_was_run") or alt_id
        return _bare(words), ev.get("alternative_rejected_because") or ""
    alt = _alternative(fix)
    return (alt["option"], alt["reason"]) if alt is not None else None


def _changed(records: list, used: set, columns_scanned=None) -> tuple[str, list]:
    """The Changed section text and the Left as is lines; columns_scanned (from the scan) names a full-row dedupe."""
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
        ran = execution if execution is not None and execution["chosen"] == "applied" else None
        parts = [f"{k}. {_operation_words(fix, params, columns_scanned, ran)}.",
                 f"{_why_label(fix)}: {_bare(fix['rationale'])}."]
        fix_ev = fix.get("evidence") or {}
        joined = fix_ev.get("issue_type") == "joinable_table"
        if (joined and fix_ev.get("recommendation") == "skip" and approval is not None
                and approval["chosen"] in ("approve", "edit", "auto_approve")):
            parts.append("The model recommended not joining this table.")  # M2.3b
        alt = _alternative_words(fix)
        if alt is not None:
            parts.append(f"Alternative the model suggested: {alt[0]} ({_bare(alt[1])}).")
        parts.append(f"Your decision: {_decision_text(approval)}.")
        if execution is not None and execution["chosen"] == "failed":
            parts.append(f"Failed: {_bare(execution['evidence'].get('failure_reason'))}.")
        elif execution is not None:
            parts.append(_effect(execution["evidence"]))
        lines.append(" ".join(parts))
        if approval is not None and approval["chosen"] == "reject" and joined:  # M2.3b: both reasons, in plain words
            advice = "recommended against joining" if fix_ev.get("recommendation") == "skip" else "recommended joining"
            reason = approval["rationale"]
            given = f": {_bare(reason)}" if reason and reason != "no reason given" else " without giving a reason"
            left.append(f"{fix_ev.get('right_table')} was not joined. The model {advice}: {_bare(fix['rationale'])}. "
                        f"You rejected the join{given}.")
        elif approval is not None and approval["chosen"] == "reject":
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


_MAX_TABLE_LINES = 10


def _applied_joins(records) -> list:
    """M2.3b: the evidence of every applied join of the current question, in order."""
    return [r["evidence"] for r in records or [] if r.get("decision_point") == EXECUTION_POINT
            and r.get("chosen") == "applied" and (r.get("evidence") or {}).get("op_type") == "join"]


def _tables_section(record, records=None):
    """M2.1: the Tables section, or None for one table or none; at most 10 lines, the rest counted. M2.3b: the
    roles come from the stage-plan record (taken before cleaning); an applied join rewords the lines of the tables
    it joined and adds a line for its result (a later join's left table is the previous result)."""
    tables = ((record or {}).get("evidence") or {}).get("tables") or []
    if len(tables) <= 1:
        return None
    roles, results = {}, []
    left = next((t["name"] for t in tables if t.get("role") == "analysis"), None)
    for ev in _applied_joins(records):
        params = ev.get("params") or {}
        right, result = params.get("right_table"), params.get("result_name")
        joined = f"joined with {right} into {result}"
        if results and results[-1][0] == left:
            results[-1][3] = f"main table after the join; {joined}"
        else:
            roles[left] = f"main table; {joined}"
        roles[right] = f"joined into {result}"
        results.append([result, ev.get("rows_after"), ev.get("columns_after"), "main table after the join"])
        left = result

    def line(t):
        role = "main table" if t.get("role") == "analysis" else "carried along; not joined"  # M2.3b: the wording
        return f"{t['name']}: {t['rows']} rows, {t['columns']} columns ({roles.get(t['name'], role)})"
    extra = [f"{name}: {rows} rows, {columns} columns ({role})" for name, rows, columns, role in results]
    main = [t for t in tables if t.get("role") == "analysis"]
    carried = [t for t in tables if t.get("role") != "analysis"]
    if len(tables) <= _MAX_TABLE_LINES:
        return "\n".join([*(line(t) for t in tables), *extra])
    shown = main + carried[:_MAX_TABLE_LINES - 1]
    rest = len(carried) - (_MAX_TABLE_LINES - 1)
    return "\n".join([*(line(t) for t in shown), f"and {rest} more carried tables", *extra])


_MAX_LISTED = 8
_CARDINALITY_WORDS = {"one_to_one": "one to one", "one_to_many": "one to many", "many_to_one": "many to one",
                      "many_to_many": "many to many"}


def _listed(items) -> str:
    """At most 8 items joined by "; ", then " and N more"."""
    items = list(items)
    text = "; ".join(items[:_MAX_LISTED])
    return text + (f" and {len(items) - _MAX_LISTED} more" if len(items) > _MAX_LISTED else "")


def _missing_text(snapshot) -> str:
    missing = snapshot.get("missing") or []
    return _listed(f"{m['column']} {m['missing_pct']:.1%}" for m in missing) + "." if missing else "none."


def _outliers_text(snapshot) -> str:
    outliers = snapshot.get("outliers") or []
    return _listed(f"{o['column']}: {o['below']} below, {o['above']} above" for o in outliers) + "." if outliers \
        else "none."


def _quality_section(before, after, cleaned) -> str:
    """M2.2: the Data quality section from the before (and after) snapshot records."""
    b = before["evidence"]
    lines = [f"Before: {b['rows']} rows, {b['columns']} columns, {b['duplicate_rows']} exact duplicate rows.",
             f"Missing values: {_missing_text(b)}"]
    if b.get("constant_columns"):
        lines.append(f"Constant columns: {_listed(b['constant_columns'])}.")
    outliers = _outliers_text(b)
    lines.append(f"Outliers (beyond 1.5 times the interquartile range): {outliers}" if outliers != "none."
                 else "Outliers: none.")
    if b.get("skewed_columns"):
        lines.append("Skewed columns (skewness beyond 1 in either direction): "
                     f"{_listed(c['column'] for c in b['skewed_columns'])}.")
    if after is not None:
        a = after["evidence"]
        lines += [f"After cleaning: {a['rows']} rows, {a['duplicate_rows']} exact duplicate rows.",
                  f"Missing values after cleaning: {_missing_text(a)}",
                  f"Outliers after cleaning: {_outliers_text(a)}"]
    elif cleaned:
        lines.append("Cleaning did not change the data.")
    return "\n".join(lines)


def _links_section(record):
    """M2.2: the Links between tables section from the links scan record."""
    ev = record["evidence"]
    if record["chosen"] == "none" or not ev.get("candidates_total"):
        return "No shared key columns were found between the tables."
    lines = []
    for c in ev.get("computed") or []:
        line = (f"{c['table_a']}.{c['column_a']} with {c['table_b']}.{c['column_b']}: "
                f"{_CARDINALITY_WORDS.get(c['cardinality'], c['cardinality'])}; {c['matched_distinct']} of "
                f"{c['distinct_a']} keys in {c['table_a']} are found in {c['table_b']}; {c['orphan_rows_a']} rows of "
                f"{c['table_a']} and {c['orphan_rows_b']} rows of {c['table_b']} have no match; an inner join would "
                f"give {c['inner_join_rows']} rows.")
        if c.get("normalizing_helps"):
            line += (f" Matching would improve by {c['matched_distinct_normalized'] - c['matched_distinct']} keys "
                     "after trimming spaces and ignoring case.")
        lines.append(line)
    if ev.get("not_computed"):
        lines.append(f"and {ev['not_computed']} more candidate(s) not computed.")
    return "\n".join(lines)


def _quality_record(records, label):
    return next((r for r in reversed(records)
                 if r.get("decision_point") == QUALITY_SNAPSHOT_POINT and r.get("chosen") == label), None)


def stage1_report(records: list, stage_plan: dict) -> dict:
    """{"sections": [{"heading", "text"}], "source_records": [ids]} from the records of this question only."""
    records = copy.deepcopy(list(records))
    used = set()
    scan = _latest(records, SCAN_POINT)
    if scan is not None:
        used.add(scan["id"])
    sections = [{"heading": "Checked", "text": _checked(scan)}]
    plan_record = _latest(records, STAGE_PLAN_POINT)
    tables_text = _tables_section(plan_record, records)
    if tables_text is not None:
        sections.append({"heading": "Tables", "text": tables_text})
        used.add(plan_record["id"])
    before, after = _quality_record(records, "before"), _quality_record(records, "after")
    if after is not None and before is not None and after["evidence"].get("handle") == before["evidence"].get("handle"):
        after = None
    if before is not None:
        # "Cleaning did not change the data." only when the scan found something to clean (chosen propose_fixes)
        issues_found = scan is not None and scan.get("chosen") == "propose_fixes"
        sections.append({"heading": "Data quality", "text": _quality_section(before, after, issues_found)})
        used.update(r["id"] for r in (before, after) if r is not None)
    links = _latest(records, LINKS_POINT)
    several = len(((plan_record or {}).get("evidence") or {}).get("tables") or []) > 1
    if links is not None and (links["evidence"].get("candidates_total") or several):
        sections.append({"heading": "Links between tables", "text": _links_section(links)})
        used.add(links["id"])
    columns_scanned = scan["evidence"].get("columns_scanned") if scan is not None else None
    changed, left = _changed(records, used, columns_scanned)
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


# --- analysis rounds (M1.4b step 2b) ---------------------------------------------------------------------------------

_FALLBACKS = ("facts_fallback", "minimal_fallback")
_GLOSSARY_DETAIL = ("When the agent uses it", "What it assumes", "How to read the result", "Other options")


def round_records(log: list) -> list:
    """
    The records of the round being reported: from just after the previous round's report.narrative record (when
    there is one after the last planner.stage_plan) or from the last planner.stage_plan, to the end. The final
    record of the log is the current round's own report.narrative record.
    """
    s = 0
    for i, entry in enumerate(log):
        if entry.get("decision_point") == STAGE_PLAN_POINT:
            s = i
    p = None
    for i, entry in enumerate(log[:-1]):
        if entry.get("decision_point") == REPORT_NARRATIVE_POINT:
            p = i
    start = p + 1 if p is not None and p > s else s
    return list(log[start:])


def method_name(method_id) -> str:
    """The glossary's display name of a method, or the id itself when it has no entry."""
    entry = GLOSSARY.get(method_id) if method_id is not None else None
    return entry["name"] if entry is not None else str(method_id)


def selection_record(records: list):
    """The latest test-selection or regression-method record."""
    return next((r for r in reversed(records)
                 if r.get("decision_point") in (TEST_SELECTION_POINT, REGRESSION_METHOD_POINT)), None)


def _names_text(names) -> str:
    return ", ".join(str(n) for n in names) if names else "none"


def explain_options(records: list) -> list:
    """
    The methods of one round that can be explained on request (M1.4b step 3), as {"method_id", "name", "role"}:
    the chosen method, the other candidates of the selection record ("alternative"), the follow-up test
    ("follow_up") and each coding scheme of a text predictor ("encoding"). Only methods with a glossary entry;
    a method keeps its first role.
    """
    found = []
    selection = selection_record(records)
    if selection is not None:
        found.append((selection["chosen"], "chosen"))
        found += [(c["option"], "alternative") for c in selection.get("candidates") or []
                  if c["option"] != selection["chosen"]]
    post_hoc = _latest(records, POST_HOC_POINT)
    if post_hoc is not None and post_hoc["chosen"] != "none":
        found.append((post_hoc["chosen"], "follow_up"))
    found += [(r["chosen"], "encoding") for r in records if r.get("decision_point") == ENCODING_POINT]
    options, seen = [], set()
    for method_id, role in found:
        if method_id in GLOSSARY and method_id not in seen:
            seen.add(method_id)
            options.append({"method_id": method_id, "name": method_name(method_id), "role": role})
    return options


def _why_this_test(records, selection, use):
    lines, extra = [], []
    if selection is None:
        return "No method selection was recorded.", extra
    use(selection)
    chosen = selection["chosen"]
    lines.append(selection["rationale"])
    for c in selection.get("candidates") or []:
        if c["option"] != chosen:
            lines.append(f"Not used: {method_name(c['option'])}: {c['reason']}")
    post_hoc = _latest(records, POST_HOC_POINT)
    if post_hoc is not None:
        use(post_hoc)
        if post_hoc["chosen"] == "none":
            lines.append(f"No pairwise follow-up ran: {post_hoc['rationale']}")
        else:
            lines.append(f"Follow-up: {method_name(post_hoc['chosen'])}: {post_hoc['rationale']}")
    encodings = {}
    for r in records:
        if r.get("decision_point") == ENCODING_POINT:
            encodings[r["evidence"].get("column")] = r  # the latest per column
    schemes = []
    for column, r in encodings.items():
        use(r)
        lines.append(f"{column}: {r['rationale']}")
        if r["chosen"] not in schemes:
            schemes.append(r["chosen"])
    for scheme in schemes:
        entry = GLOSSARY.get(scheme)
        if entry is not None:
            extra.append({"heading": f"About {entry['name']}", "text": f"{entry['what']} How to read it: {entry['reading']}",
                          "collapsed": True})
    if selection.get("supersedes"):
        lines.append("The method was selected again after you edited the predictors.")
    return "\n".join(lines), extra


def _plan_decision(records, use) -> str:
    by_id = {r.get("id"): r for r in records}
    approvals = [r for r in records if r.get("decision_point") == PLAN_APPROVAL_POINT]
    lines = []
    for a in approvals:
        use(a)
        ev, chosen = a.get("evidence") or {}, a["chosen"]
        if chosen == "approve":
            lines.append("Approved as drafted.")
        elif chosen == "edit":
            linked = by_id.get(ev.get("predictors_record_id"))
            if linked is not None:
                use(linked)
                before = _names_text(linked["evidence"].get("predictors_before"))
                after = _names_text(linked["evidence"].get("predictors_after"))
            else:
                before, after = _names_text(ev.get("predictors_before")), "the new predictors"
            lines.append(f"Edited the predictors from {before} to {after}; the method was selected again.")
        elif chosen == "refused":
            reasons = "; ".join(f"{v.get('column') or v.get('rule')}: {v.get('reason')}" for v in ev.get("refusals") or [])
            lines.append(f"Refused: {reasons}.")
        elif chosen == "reject":
            lines.append("Rejected.")
        else:
            lines.append(f"{chosen}.")
    predictors = _latest(records, PREDICTORS_POINT)
    if predictors is not None:
        use(predictors)
        after = predictors.get("evidence", {}).get("predictors_after")
        lines.append(f"Predictors used: {_names_text(after) if after is not None else predictors['chosen']}.")
    elif approvals and approvals[-1].get("evidence", {}).get("predictors_before") is not None:
        lines.append(f"Predictors used: {_names_text(approvals[-1]['evidence']['predictors_before'])}.")
    return "\n".join(lines) if lines else "No plan decision was recorded."


def _exploratory_sections(records, use) -> list:
    sections = []
    start = _latest(records, ROUND_START_POINT)
    if start is not None:
        use(start)
        lines = [f"Chosen: {start['chosen']}.",
                 "Chosen by: you." if start["evidence"].get("chosen_by") == "user" else "Chosen by: the agent's rules."]
        lines += [f"Left out: {c['option']}: {c['reason']}" for c in start.get("candidates") or [] if c["eligible"] is False]
        sections.append({"heading": "Why these predictors", "text": "\n".join(lines)})
    gates = [r for r in records if r.get("decision_point") == STAGE_GATE_POINT]
    if gates:
        lines = []
        for g in gates:
            use(g)
            if g["chosen"] == "refused":
                lines.append(f"Refused: {g['rationale']}")
        run = [g for g in gates if g["chosen"] == "run"]
        if run:
            lines.append("You chose to run the exploratory regression"
                         + (" with your own predictors." if run[-1].get("overridden_by_human") else "."))
        sections.append({"heading": "Your choice at the boundary", "text": "\n".join(lines) or "No choice was recorded."})
    return sections


def _narrative_text(narrative) -> str:
    chosen = narrative["chosen"]
    if chosen == "model":
        return "The narrative was written by the model."
    if chosen == "model_retry":
        attempts = (narrative.get("evidence") or {}).get("attempts") or []
        matched = ((attempts[0].get("violation") or {}).get("matched") if attempts else None)
        return f"The narrative was written by the model after one correction (its first text used '{matched}')."
    if chosen == "facts_fallback":
        return "The model's text used causal language twice, so the narrative was written from the computed facts."
    if chosen == "minimal_fallback":
        return "The narrative could not be written safely, so a minimal text was used."
    return f"{chosen}."


def _ordered_ids(records, used) -> list:
    ordered = []
    for r in records:
        if r.get("id") in used and r["id"] not in ordered:
            ordered.append(r["id"])
    return ordered


def analysis_report(records: list, exploratory: bool) -> dict:
    """{"sections", "source_records", "fallback", "explain_options"} from the records of one round; the last
    record is its report.narrative record."""
    records = copy.deepcopy(list(records))
    used = set()

    def use(record):
        used.add(record["id"])

    selection = selection_record(records)
    chosen = selection["chosen"] if selection is not None else None
    sections = []
    entry = GLOSSARY.get(chosen) if chosen is not None else None
    if entry is not None:
        sections.append({"heading": "About the test",
                         "text": f"{entry['name']}. {entry['what']} It answers: {entry['answers']}"})
        sections += [{**s, "collapsed": True} for s in render_glossary(chosen)["sections"]
                     if s["heading"] in _GLOSSARY_DETAIL]
    why, encoding_sections = _why_this_test(records, selection, use)
    sections.append({"heading": "Why this test", "text": why})
    sections += encoding_sections
    sections.append({"heading": "Your plan decision", "text": _plan_decision(records, use)})
    if exploratory:
        sections += _exploratory_sections(records, use)
    narrative = records[-1] if records and records[-1].get("decision_point") == REPORT_NARRATIVE_POINT else None
    if narrative is None:
        raise ValueError("analysis_report: the last record must be the report.narrative record of this round.")
    use(narrative)
    sections.append({"heading": "Narrative", "text": _narrative_text(narrative)})
    return {"sections": sections, "source_records": _ordered_ids(records, used),
            "fallback": narrative["chosen"] in _FALLBACKS, "explain_options": explain_options(records)}


# --- the summary (M1.4b step 2b) ---------------------------------------------------------------------------------------

def _headline(entry) -> str:
    result, capability = entry.get("result") or {}, entry.get("capability")
    if capability in ("two_group", "multi_group"):
        text = f"p = {result['p_value']:.3g}" if result.get("p_value") is not None else "no p-value recorded"
        if capability == "two_group" and result.get("group_a_n") is not None and result.get("group_b_n") is not None:
            text += f" (n = {result['group_a_n']} and {result['group_b_n']})"
        return text
    if capability == "regression":
        parts = []
        if result.get("r_squared") is not None:
            parts.append(f"R-squared = {result['r_squared']:.2f}")
        if result.get("f_p_value") is not None:
            parts.append(f"overall F-test p = {result['f_p_value']:.3g}")
        if result.get("n_observations") is not None:
            parts.append(f"n = {result['n_observations']}")
        return ", ".join(parts) or "no fit statistics recorded"
    if capability == "descriptive":
        return "the data were described (see the report)"
    return "no headline recorded"


def summary_report(records: list, rounds: list, request_text: str) -> dict:
    """{"sections", "source_records"} from the records of the current question and its round entries."""
    records = copy.deepcopy(list(records))
    used = set()
    narratives = [r for r in records if r.get("decision_point") == REPORT_NARRATIVE_POINT]
    # a descriptive question writes no narrative record: its round is the last one (rounds accumulate across questions)
    question_rounds = list(rounds[-len(narratives):]) if narratives else list(rounds[-1:])

    question = [str(request_text)]
    if question_rounds:
        question.append(f"Run as: {question_rounds[0].get('question_type')}")
    sections = [{"heading": "Question", "text": "\n".join(question)}]

    scan = _latest(records, SCAN_POINT)
    if scan is None:
        stage1 = ["No Stage 1 scan was recorded."]
    else:
        used.add(scan["id"])
        ev = scan["evidence"]
        stage1 = [f"{ev['rows']} rows and {ev['columns_scanned']} columns checked."]
        fixes = [r for r in records if r.get("decision_point") == FIX_POINT]
        approved = left = 0
        for fix in fixes:
            used.add(fix["id"])
            approval = next((r for r in records if r.get("decision_point") == APPROVAL_POINT
                             and r["evidence"].get("fix_record_id") == fix["id"]), None)
            if approval is not None:
                used.add(approval["id"])
            if approval is not None and approval["chosen"] == "reject":
                left += 1
            else:
                approved += 1
        stage1.append(f"{len(fixes)} operation(s) proposed: {approved} approved, {left} left as is.")
    describe = _latest(records, DESCRIBE_EXECUTION_POINT)
    if describe is not None:
        used.add(describe["id"])
        stage1.append(f"{describe['evidence']['row_count']} rows analysed.")
    sections.append({"heading": "Stage 1", "text": "\n".join(stage1)})

    results = []
    for entry in question_rounds:
        label = "Exploratory analysis (not corrected for multiple testing)" if entry.get("exploratory") else "Main analysis"
        name = method_name(entry["method"]) if entry.get("method") else str(entry.get("capability"))
        results.append(f"{label}: {name} - {_headline(entry)}.")
    sections.append({"heading": "Results", "text": "\n".join(results) or "No results were recorded."})

    caveats = []
    last_plan = question_rounds[-1].get("plan") if question_rounds else None
    for issue in (getattr(last_plan, "unaddressed_issues", None) or ()):
        caveats.append(unaddressed_text(issue))  # M2.3a: a plain caveat (an imputed outcome) renders alone
    for r in narratives:
        used.add(r["id"])
        if r["chosen"] in _FALLBACKS:
            caveats.append(f"The narrative of round {r['evidence'].get('round')} was written from the computed facts.")
    if any(entry.get("exploratory") for entry in question_rounds):
        caveats.append("Exploratory results are not corrected for multiple testing.")
    sections.append({"heading": "Caveats", "text": "\n".join(caveats) or "None recorded."})
    return {"sections": sections, "source_records": _ordered_ids(records, used)}
