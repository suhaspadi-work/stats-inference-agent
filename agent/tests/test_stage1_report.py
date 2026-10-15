"""
M1.4b step 1: the Stage 1 report. core.stage_reports composes it by code from the decision records of the current
question (no model call), the stage1_report node appends it to state["stage_reports"] before draft_plan, and the
plan-approval pause shows it as stage1_report. The node chooses nothing and writes no record.
"""
import copy
import inspect
import json
import re

import numpy as np
import pandas as pd
import pytest
from langgraph.types import Command

from core import stage_reports, wrangling
from core.cleaning_records import approval_record, execution_record, fix_record, scan_record
from core.dataset import LocalDiskStore
from core.decision_log import log_update
from core.decision_points import DECISION_NODES, NODE_STAGES, NON_CHOOSING_NODES, registered_points
from core.describe_records import describe_execution_record
from core.detection import CandidateIssue
from core.graph import build_agent_graph
from core.ingestion import HIGH_CARDINALITY_THRESHOLD
from core.model_safe_view import ColumnSummary, ModelSafeView
from core.nodes import make_stage1_report_node
from core.operation import (
    AlternativeApproachConsidered, ApprovalStatus, Operation, OperationImpact, WranglingRationale,
)
from core.plan import QuestionType
from core.stage_reports import STAGE1_TITLE, current_question_records, make_stage_report_entry, stage1_report
from tests.test_decision_point_guard import _dataset, confident, fake_draft, fake_propose, must_not_propose
from tests.test_exploratory_round import _frame
from tests.test_freeze_node import _pause
from tests.test_prompt_canaries import (
    _base_df, _drive as _drive_canaries, _with_duplicates, canary_hits, fake_model,  # noqa: F401 (fixture)
)

ENTRY_KEYS = {"stage", "round", "title", "text", "sections", "fallback", "exploratory", "source_records"}
COMPARISON = {"question_type": "two_group_comparison"}
DESCRIPTIVE_PLAN = {"question_type": "descriptive"}
NOT_RUN = "The description was not run (it was switched off or does not apply)."


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


# --- record builders (the real ones from core.cleaning_records and core.describe_records) -------------------------

def _profile(rows=300, columns=4):
    cols = tuple(ColumnSummary(name=f"c{i}", dtype="float64", missing_count=0, missing_pct=0.0, distinct_count=10)
                 for i in range(columns))
    return ModelSafeView(dataset_handle="d@v1", row_count=rows, column_count=columns, columns=cols)


def _dup_issue(count=4, total=300):
    return CandidateIssue(issue_type="duplicate_rows", description=f"{count} exact duplicate rows",
                          affected_columns=(), evidence={"duplicate_row_count": count, "total_rows": total})


def _miss_issue(column="tenure", pct=0.1234):
    return CandidateIssue(issue_type="high_missingness", description=f"{column} is often missing",
                          affected_columns=(column,), evidence={"missing_pct": pct})


def _op(op_type, params, rationale="fix it", alternative=("leave the data as is", "it would bias the result")):
    alts = (AlternativeApproachConsidered(approach=alternative[0], rejected_because=alternative[1]),) if alternative else ()
    return Operation(op_type=op_type, session_id="s", input_handles=("d@v1",), params=params,
                     approval_status=ApprovalStatus.PENDING,
                     rationale=WranglingRationale(rationale=rationale, alternatives=alts))


DEDUPE = {"subset": None}
NOT_NULL = {"column": "tenure", "condition": "not_null", "value": None}


def _ran(op, before=300, after=296, params=None):
    if params is not None:
        op = Operation(op_type=op.op_type, session_id=op.session_id, input_handles=op.input_handles, params=params,
                       approval_status=ApprovalStatus.PENDING, rationale=op.rationale)
    return op.approve().with_result(OperationImpact(rows_before=before, rows_after=after, columns_before=4,
                                                    columns_after=4), "d@v2")


def _dicts(*records):
    return log_update(*records)["decision_log"]


def _describe(withheld=("customer_name",), rows=296, columns=4):
    stats = [{"name": f"n{i}", "is_numeric": True, "distinct_count": 10} for i in range(columns - len(withheld))]
    stats += [{"name": w, "is_numeric": False, "distinct_count": HIGH_CARDINALITY_THRESHOLD + 1} for w in withheld]
    result = {"row_count": rows, "column_stats": stats, "chart_data": []}
    return describe_execution_record(result, {"state": "offered", "exploratory": False})


def _cleaning(issue, op, decision, auto_rule=None, ran=None, edited=None, index=0):
    """fix, approval and (when ran is given) execution records for one operation."""
    fix = fix_record(issue, op, index)
    approval = approval_record(op, issue, decision, auto_rule, fix.id, edited_params=edited)
    records = [fix, approval]
    if ran is not None:
        records.append(execution_record(ran, approval.id))
    return records


def _section(composed, heading):
    return next((s["text"] for s in composed["sections"] if s["heading"] == heading), None)


def _headings(composed):
    return [s["heading"] for s in composed["sections"]]


APPROVE = {"index": 0, "action": "approve"}


# --- current_question_records ---------------------------------------------------------------------------------------

def test_current_question_records_starts_at_the_last_stage_plan():
    log = [{"decision_point": "planner.stage_plan", "id": "1"}, {"decision_point": "x", "id": "2"},
           {"decision_point": "planner.stage_plan", "id": "3"}, {"decision_point": "y", "id": "4"}]
    before = copy.deepcopy(log)
    assert [r["id"] for r in current_question_records(log)] == ["3", "4"]
    assert log == before


def test_current_question_records_without_a_stage_plan_is_everything():
    log = [{"decision_point": "x", "id": "1"}, {"decision_point": "y", "id": "2"}]
    before = copy.deepcopy(log)
    assert current_question_records(log) == log and current_question_records([]) == []
    assert log == before


# --- Checked ----------------------------------------------------------------------------------------------------------

def test_checked_duplicates_only():
    text = _section(stage1_report(_dicts(scan_record(_profile(), [_dup_issue()])), COMPARISON), "Checked")
    assert text == "Checked 300 rows and 4 columns.\nFound 4 exact duplicate rows."


def test_checked_missingness_only_formats_the_percentages():
    text = _section(stage1_report(_dicts(scan_record(_profile(), [_miss_issue(pct=0.1234)])), COMPARISON), "Checked")
    assert text == "Checked 300 rows and 4 columns.\ntenure: 12.3% missing (threshold 5%)."


def test_checked_both():
    scan = scan_record(_profile(), [_dup_issue(), _miss_issue(pct=0.0667), _miss_issue("notes", 0.5)])
    assert _section(stage1_report(_dicts(scan), COMPARISON), "Checked") == (
        "Checked 300 rows and 4 columns.\nFound 4 exact duplicate rows.\n"
        "tenure: 6.7% missing (threshold 5%).\nnotes: 50.0% missing (threshold 5%).")


def test_checked_neither():
    text = _section(stage1_report(_dicts(scan_record(_profile(120, 3), [])), COMPARISON), "Checked")
    assert text == "Checked 120 rows and 3 columns.\nNo data-quality issues found."


def test_checked_without_a_scan_record():
    composed = stage1_report([], COMPARISON)
    assert _section(composed, "Checked") == "No data-quality scan was recorded."
    assert _section(composed, "Changed") == "No changes were proposed."


def test_the_latest_scan_is_used():
    old, new = scan_record(_profile(10, 2), []), scan_record(_profile(), [_dup_issue()])
    composed = stage1_report(_dicts(old, new), COMPARISON)
    assert _section(composed, "Checked").startswith("Checked 300 rows") and composed["source_records"] == [new.id]


# --- Changed ----------------------------------------------------------------------------------------------------------

def test_approved_dedupe_with_its_effect():
    op = _op("dedupe", DEDUPE, rationale="duplicates inflate the counts")
    records = _cleaning(_dup_issue(), op, APPROVE, ran=_ran(op, 300, 296))
    assert _section(stage1_report(_dicts(*records), COMPARISON), "Changed") == (
        "1. Removed duplicate rows. Why (the model's reason): duplicates inflate the counts. Alternative the model "
        "suggested: leave the data as is (it would bias the result). Your decision: approved. Effect: 300 to 296 rows.")
    # M1.4b 2a: the model's reason is labelled


def test_dedupe_with_a_subset_names_the_columns():
    op = _op("dedupe", {"subset": ["region", "tenure"]})
    records = _cleaning(_dup_issue(), op, APPROVE, ran=_ran(op))
    assert "1. Removed duplicate rows (comparing region, tenure)." in _section(
        stage1_report(_dicts(*records), COMPARISON), "Changed")


def _dedupe_line(subset, columns_scanned=4, scan=True):
    op = _op("dedupe", {"subset": subset})
    records = [scan_record(_profile(300, columns_scanned), [_dup_issue()])] if scan else []
    records += _cleaning(_dup_issue(), op, APPROVE, ran=_ran(op))
    return _section(stage1_report(_dicts(*records), COMPARISON), "Changed").split(". Why")[0]


def test_dedupe_of_every_scanned_column_says_all_columns():
    assert _dedupe_line(["customer_id", "region", "tenure", "total_spend"]) == (
        "1. Removed duplicate rows (comparing all columns)")


def test_dedupe_of_fewer_columns_keeps_the_list():
    assert _dedupe_line(["region", "tenure"]) == "1. Removed duplicate rows (comparing region, tenure)"


def test_dedupe_without_a_scan_keeps_the_list():
    assert _dedupe_line(["a", "b", "c", "d"], scan=False) == "1. Removed duplicate rows (comparing a, b, c, d)"


@pytest.mark.parametrize("subset", [[], None], ids=["empty", "none"])
def test_dedupe_without_a_subset_is_plain(subset):
    assert _dedupe_line(subset) == "1. Removed duplicate rows"


def test_approved_filter():
    op = _op("filter", NOT_NULL)
    records = _cleaning(_miss_issue(), op, APPROVE, ran=_ran(op, 296, 260))
    text = _section(stage1_report(_dicts(*records), COMPARISON), "Changed")
    assert text.startswith("1. Kept rows where tenure has a value. Why (the model's reason): fix it.")  # M1.4b 2a: the model's reason is labelled
    assert text.endswith("Your decision: approved. Effect: 296 to 260 rows.")
    assert "not_null" not in text


def test_an_edited_operation_shows_the_params_that_ran():
    op = _op("filter", NOT_NULL)
    edited = {"column": "tenure", "condition": "greater_than", "value": 10}
    records = _cleaning(_miss_issue(), op, {"index": 0, "action": "edit", "params": edited},
                        ran=_ran(op, 296, 250, params=edited), edited=edited)
    text = _section(stage1_report(_dicts(*records), COMPARISON), "Changed")
    assert text.startswith("1. Kept rows where tenure is greater than 10.")
    assert "Your decision: approved with your edits. Effect: 296 to 250 rows." in text
    assert "has a value" not in text


def test_a_rejected_operation():
    op = _op("filter", NOT_NULL)
    records = _cleaning(_miss_issue(), op, {"index": 0, "action": "reject", "reason": "the gaps are meaningful"})
    composed = stage1_report(_dicts(*records), COMPARISON)
    text = _section(composed, "Changed")
    assert text.endswith("Your decision: rejected: the gaps are meaningful.")
    assert "Effect" not in text
    assert _section(composed, "Left as is") == (
        "high_missingness (tenure) was left unaddressed because: the gaps are meaningful")


def test_left_as_is_uses_the_description_when_the_fix_evidence_has_one():
    op = _op("filter", NOT_NULL)
    records = _dicts(*_cleaning(_miss_issue(), op, {"index": 0, "action": "reject", "reason": "keep them"}))
    records[0]["evidence"]["description"] = "tenure is often missing"
    assert _section(stage1_report(records, COMPARISON), "Left as is") == (
        "high_missingness (tenure is often missing) was left unaddressed because: keep them")


def test_an_automatically_approved_operation():
    op = _op("dedupe", DEDUPE)
    records = _cleaning(_dup_issue(), op, APPROVE, auto_rule="stakeholder mode auto-approves duplicate_rows only",
                        ran=_ran(op))
    assert records[1].decided_by == "code_rule"
    assert "Your decision: approved automatically. Effect: 300 to 296 rows." in _section(
        stage1_report(_dicts(*records), COMPARISON), "Changed")


def test_an_unknown_action_treated_as_approval_is_automatic():
    op = _op("dedupe", DEDUPE)
    records = _cleaning(_dup_issue(), op, {"index": 0, "action": "maybe"}, ran=_ran(op))
    assert "Your decision: approved automatically." in _section(stage1_report(_dicts(*records), COMPARISON), "Changed")


def test_a_failed_execution():
    op = _op("filter", {"column": "gone", "condition": "not_null", "value": None})
    failed = op.approve().mark_failed("Column not found: gone")
    records = _cleaning(_miss_issue("gone"), op, APPROVE, ran=failed)
    text = _section(stage1_report(_dicts(*records), COMPARISON), "Changed")
    assert text.endswith("Your decision: approved. Failed: Column not found: gone.")
    assert "Effect" not in text


def test_a_no_decision_approval():
    op = _op("dedupe", DEDUPE)
    records = _cleaning(_dup_issue(), op, None)
    text = _section(stage1_report(_dicts(*records), COMPARISON), "Changed")
    assert text.endswith("Your decision: no decision received, so it was not run.")
    assert "Effect" not in text and "automatically" not in text


def test_a_fix_with_no_approval_record():
    fix = fix_record(_dup_issue(), _op("dedupe", DEDUPE), 0)
    composed = stage1_report(_dicts(fix), COMPARISON)
    assert _section(composed, "Changed").endswith("Your decision: none recorded.")
    assert composed["source_records"] == [fix.id]


def test_a_fix_without_a_model_alternative():
    op = _op("dedupe", DEDUPE, alternative=None)
    text = _section(stage1_report(_dicts(*_cleaning(_dup_issue(), op, APPROVE, ran=_ran(op))), COMPARISON), "Changed")
    assert text == ("1. Removed duplicate rows. Why (the model's reason): fix it. Your decision: approved. "
                    "Effect: 300 to 296 rows.")  # M1.4b 2a: the model's reason is labelled
    assert "Alternative" not in text


def test_rationales_that_end_in_a_period_give_no_double_period():
    dup = _op("dedupe", DEDUPE, rationale="Duplicates inflate counts.", alternative=("keep them", "they bias it."))
    miss = _op("filter", NOT_NULL, rationale="Rows lack tenure.")
    records = [scan_record(_profile(), [_dup_issue(), _miss_issue()]),
               *_cleaning(_dup_issue(), dup, APPROVE, ran=_ran(dup)),
               *_cleaning(_miss_issue(), miss, {"index": 1, "action": "reject", "reason": "keep them."}, index=1),
               _describe()]
    entry = make_stage_report_entry("stage1", None, STAGE1_TITLE, stage1_report(_dicts(*records), COMPARISON))
    assert ".." not in entry["text"]
    assert ("Why (the model's reason): Duplicates inflate counts. Alternative the model suggested: keep them "
            "(they bias it).") in entry["text"]  # M1.4b 2a: the model's reason is labelled
    assert "Your decision: rejected: keep them." in entry["text"]


def test_operations_are_numbered_in_log_order():
    dup, miss = _op("dedupe", DEDUPE), _op("filter", NOT_NULL)
    records = [*_cleaning(_dup_issue(), dup, APPROVE, ran=_ran(dup, 300, 296)),
               *_cleaning(_miss_issue(), miss, {"index": 1, "action": "approve"}, ran=_ran(miss, 296, 260), index=1)]
    lines = _section(stage1_report(_dicts(*records), COMPARISON), "Changed").split("\n")
    assert [line[:3] for line in lines] == ["1. ", "2. "]
    assert lines[0].startswith("1. Removed duplicate rows") and lines[1].startswith("2. Kept rows where tenure")


def test_no_fix_records():
    assert _section(stage1_report(_dicts(scan_record(_profile(), [])), COMPARISON), "Changed") == (
        "No changes were proposed.")


@pytest.mark.parametrize("condition, value, words", [
    ("equals", "East", "Kept rows where tenure equals East"),
    ("not_equals", "East", "Kept rows where tenure does not equal East"),
    ("greater_than", 10, "Kept rows where tenure is greater than 10"),
    ("less_than", 2.5, "Kept rows where tenure is less than 2.5"),
    ("is_null", None, "Kept rows where tenure has no value"),
    ("not_null", None, "Kept rows where tenure has a value"),
    ("between", 3, "Kept rows where tenure between 3"),  # not in the mapping: the stored text is kept
    ("odd_rule", None, "Kept rows where tenure odd_rule"),
], ids=["equals", "not_equals", "greater_than", "less_than", "is_null", "not_null", "unmapped_with_value",
        "unmapped_without_value"])
def test_filter_conditions_in_words(condition, value, words):
    op = _op("filter", {"column": "tenure", "condition": condition, "value": value})
    text = _section(stage1_report(_dicts(*_cleaning(_miss_issue(), op, APPROVE, ran=_ran(op))), COMPARISON), "Changed")
    assert text.startswith(f"1. {words}. Why (the model's reason):")  # M1.4b 2a: the model's reason is labelled


def test_every_supported_filter_condition_is_mapped():
    supported = set(re.findall(r'"(\w+)"', re.search(r"allowed = \{([^}]*)\}",
                                                    inspect.getsource(wrangling.propose_filter)).group(1)))
    assert supported == set(stage_reports._FILTER_WORDS)


# --- Described --------------------------------------------------------------------------------------------------------

def test_described_with_withheld_columns():
    text = _section(stage1_report(_dicts(_describe(("customer_name", "email"), rows=296, columns=5)), COMPARISON),
                    "Described")
    assert text == "296 rows and 5 columns described. Text values withheld for: customer_name, email."


def test_described_with_none_withheld():
    text = _section(stage1_report(_dicts(_describe((), rows=120, columns=3)), COMPARISON), "Described")
    assert text == "120 rows and 3 columns described. Text values withheld for: none."


def test_described_is_absent_for_a_descriptive_question():
    composed = stage1_report(_dicts(scan_record(_profile(), []), _describe()), DESCRIPTIVE_PLAN)
    assert _headings(composed) == ["Checked", "Changed"]
    assert len(composed["source_records"]) == 1


def test_described_not_run():
    assert _section(stage1_report(_dicts(scan_record(_profile(), [])), COMPARISON), "Described") == NOT_RUN
    assert _section(stage1_report([], {}), "Described") == NOT_RUN


# --- order, sources, serialization ----------------------------------------------------------------------------------

def _full_log():
    dup, miss = _op("dedupe", DEDUPE), _op("filter", NOT_NULL)
    return _dicts(
        scan_record(_profile(), [_dup_issue(), _miss_issue()]),
        *_cleaning(_dup_issue(), dup, APPROVE, ran=_ran(dup)),
        *_cleaning(_miss_issue(), miss, {"index": 1, "action": "reject", "reason": "keep them"}, index=1),
        _describe(),
    )


def test_sections_come_in_order():
    assert _headings(stage1_report(_full_log(), COMPARISON)) == ["Checked", "Changed", "Left as is", "Described"]


def test_source_records_are_the_used_ids_in_log_order():
    log = _full_log()
    unrelated = {"decision_point": "planner.outcome", "id": "unrelated", "evidence": {}}
    mixed = [log[0], unrelated, *log[1:], copy.deepcopy(log[0])]  # the scan again at the end: the latest is used
    mixed[-1]["id"] = "later-scan"
    composed = stage1_report(mixed, COMPARISON)
    expected = [r["id"] for r in log[1:]] + ["later-scan"]
    assert composed["source_records"] == expected
    assert len(set(composed["source_records"])) == len(composed["source_records"])


def test_the_input_is_not_mutated_and_the_result_serializes():
    log = _full_log()
    before = copy.deepcopy(log)
    composed = stage1_report(log, COMPARISON)
    assert log == before
    json.dumps(composed)
    json.dumps(make_stage_report_entry("stage1", None, STAGE1_TITLE, composed))


# --- make_stage_report_entry ---------------------------------------------------------------------------------------

def test_make_stage_report_entry():
    composed = {"sections": [{"heading": "A", "text": "one\ntwo"}, {"heading": "B", "text": "three"}],
                "source_records": ["r1", "r2"]}
    before = copy.deepcopy(composed)
    entry = make_stage_report_entry("stage1", None, "Title", composed)
    assert set(entry) == ENTRY_KEYS
    assert entry["text"] == "Title\n\nA\none\ntwo\n\nB\nthree"
    assert (entry["stage"], entry["round"], entry["title"]) == ("stage1", None, "Title")
    assert (entry["fallback"], entry["exploratory"]) == (False, False)
    assert entry["sections"] == composed["sections"] and entry["source_records"] == ["r1", "r2"]
    entry["sections"][0]["text"] = "changed"
    entry["source_records"].append("r3")
    assert composed == before
    flagged = make_stage_report_entry("stage1", 2, "T", composed, fallback=True, exploratory=True)
    assert (flagged["round"], flagged["fallback"], flagged["exploratory"]) == (2, True, True)


# --- the node -------------------------------------------------------------------------------------------------------

def test_the_node_uses_only_the_current_question():
    old_scan = scan_record(_profile(10, 2), [_dup_issue(1, 10)])
    new_scan = scan_record(_profile(120, 3), [])
    log = [{"decision_point": "planner.stage_plan", "id": "plan-1"}, *_dicts(old_scan),
           {"decision_point": "planner.stage_plan", "id": "plan-2"}, *_dicts(new_scan)]
    state = {"decision_log": log, "stage_plan": COMPARISON}
    before = copy.deepcopy(state)
    update = make_stage1_report_node()(state)
    assert set(update) == {"stage_reports"}
    (entry,) = update["stage_reports"]
    assert entry["stage"] == "stage1" and entry["round"] is None and entry["title"] == STAGE1_TITLE
    assert entry["source_records"] == [new_scan.id]
    assert "Checked 120 rows and 3 columns." in entry["text"] and "Found 1" not in entry["text"]
    assert state == before


def test_the_node_with_an_empty_state():
    (entry,) = make_stage1_report_node()({})["stage_reports"]
    assert entry["source_records"] == [] and "No data-quality scan was recorded." in entry["text"]


# --- through the real graph -----------------------------------------------------------------------------------------

def _answer(payload, overrides):
    kind = payload["type"]
    if kind in overrides:
        value = overrides[kind]
        return value(payload) if callable(value) else value
    if kind == "wrangling_approval":
        return {"decisions": [{"index": op["index"], "action": "approve"} for op in payload["proposed_operations"]]}
    if kind in ("stage_plan_approval", "plan_approval"):
        return {"action": "approve"}
    if kind == "stage_gate":
        return {"action": "skip"}
    raise AssertionError(f"unexpected pause {kind}")


def _drive(app, graph_input, thread, **overrides):
    """Run to the end; returns the final values, the pauses (type, payload, stage_reports at the pause) and the
    nodes in the order they ran."""
    config = {"configurable": {"thread_id": thread}}
    pauses, nodes = [], []
    while True:
        for event in app.stream(graph_input, config, stream_mode="updates"):
            nodes.extend(node for node in event if node != "__interrupt__")
        snapshot = app.get_state(config)
        if not snapshot.interrupts:
            return snapshot.values, pauses, nodes
        payload = snapshot.interrupts[0].value
        pauses.append({"type": payload["type"], "payload": payload,
                       "stage_reports": list(snapshot.values.get("stage_reports") or [])})
        graph_input = Command(resume=_answer(payload, overrides))


def _messy_frame():
    df = _frame(seed=61)
    df.loc[10:21, "tenure"] = np.nan  # 12 of 122 rows after the duplicates: above the 5% threshold
    return pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)  # 2 exact duplicate rows


def _app(store, question_type=QuestionType.TWO_GROUP_COMPARISON, predictors=("region",), propose_fn=fake_propose):
    return build_agent_graph(store, classify_fn=confident(question_type), draft_fn=fake_draft(list(predictors)),
                             propose_fn=propose_fn)


def _start(df, tmp_path, store, name):
    return {"request_text": "Did West spend more?", "session_id": "session-abc",
            "dataset": _dataset(df, tmp_path, store, name)}


def _stage1(values):
    entries = [e for e in values["stage_reports"] if e["stage"] == "stage1"]
    assert len(entries) == 1
    return entries[0]


def _sections(entry):
    return {s["heading"]: s["text"] for s in entry["sections"]}


def _reject_filter(payload):
    return {"decisions": [{"index": op["index"], "action": "reject", "reason": "the gaps are meaningful"}
                          if op["op_type"] == "filter" else {"index": op["index"], "action": "approve"}
                          for op in payload["proposed_operations"]]}


def test_a_comparison_with_cleaning_approved(fake_model, tmp_path, store):
    df = _messy_frame()
    deduped = df.drop_duplicates()
    final = deduped[deduped["tenure"].notna()]
    values, pauses, _ = _drive(_app(store), _start(df, tmp_path, store, "approved"), "approved")
    assert [p["type"] for p in pauses] == ["stage_plan_approval", "wrangling_approval", "plan_approval", "stage_gate"]
    entry = _stage1(values)
    text = entry["text"]
    assert text.startswith(STAGE1_TITLE + "\n\nChecked\n")
    assert "Removed duplicate rows" in text and "Kept rows where tenure has a value" in text
    assert f"Effect: {len(df)} to {len(deduped)} rows." in text
    assert f"Effect: {len(deduped)} to {len(final)} rows." in text
    assert len(store.read(values["dataset"].storage_key)) == len(final)
    assert f"{len(final)} rows and 4 columns described." in text
    assert "Left as is" not in _sections(entry)
    ids = {r["id"] for r in values["decision_log"]}
    assert set(entry["source_records"]) <= ids


def test_a_comparison_with_the_filter_rejected(fake_model, tmp_path, store):
    values, _, _ = _drive(_app(store), _start(_messy_frame(), tmp_path, store, "rejected"), "rejected",
                          wrangling_approval=_reject_filter)
    sections = _sections(_stage1(values))
    assert list(sections) == ["Checked", "Changed", "Left as is", "Described"]
    assert sections["Left as is"] == "high_missingness (tenure) was left unaddressed because: the gaps are meaningful"
    assert "Your decision: rejected: the gaps are meaningful." in sections["Changed"]


def test_a_comparison_with_nothing_to_clean(fake_model, tmp_path, store):
    values, pauses, _ = _drive(_app(store, propose_fn=must_not_propose), _start(_frame(), tmp_path, store, "clean"),
                               "clean")
    sections = _sections(_stage1(values))
    assert sections["Checked"] == "Checked 120 rows and 4 columns.\nNo data-quality issues found."
    assert sections["Changed"] == "No changes were proposed."
    assert sections["Described"].startswith("120 rows and 4 columns described.")


def test_a_descriptive_question(fake_model, tmp_path, store):
    values, pauses, nodes = _drive(_app(store, QuestionType.DESCRIPTIVE, (), must_not_propose),
                                   _start(_frame(), tmp_path, store, "descriptive"), "descriptive")
    entry = _stage1(values)
    assert [s["heading"] for s in entry["sections"]] == ["Checked", "Changed"]
    assert nodes.index("stage1_report") < nodes.index("draft_plan") < nodes.index("eda")


def test_describe_switched_off_at_the_stage_plan_gate(fake_model, tmp_path, store):
    values, _, nodes = _drive(_app(store, propose_fn=must_not_propose), _start(_frame(), tmp_path, store, "off"), "off",
                              stage_plan_approval={"action": "approve", "run": {"describe": False}})
    assert "describe" not in nodes
    assert _sections(_stage1(values))["Described"] == NOT_RUN


def test_the_plan_approval_payload_carries_the_report(fake_model, tmp_path, store):
    values, pauses, _ = _drive(_app(store), _start(_messy_frame(), tmp_path, store, "payload"), "payload")
    entry = _stage1(values)
    (plan_pause,) = [p for p in pauses if p["type"] == "plan_approval"]
    assert plan_pause["payload"]["stage1_report"] == {k: entry[k] for k in ("title", "text", "sections")}


def test_the_plan_approval_payload_without_stage_reports():
    _, _, _, paused = _pause("no-stage-reports")
    assert paused["__interrupt__"][0].value["stage1_report"] is None


def test_the_exploratory_round_shows_the_same_report(fake_model, tmp_path, store):
    values, pauses, _ = _drive(_app(store, propose_fn=must_not_propose), _start(_frame(), tmp_path, store, "explore"),
                               "explore", stage_gate={"action": "run"})
    first, second = [p["payload"]["stage1_report"] for p in pauses if p["type"] == "plan_approval"]
    assert first is not None and first == second
    assert len(values["rounds"]) == 2 and len(values["stage_reports"]) == 1


def test_the_report_is_built_before_draft_plan(fake_model, tmp_path, store):
    _, pauses, nodes = _drive(_app(store), _start(_messy_frame(), tmp_path, store, "before"), "before")
    assert nodes.index("describe") < nodes.index("stage1_report") < nodes.index("draft_plan")
    plan_pause = next(p for p in pauses if p["type"] == "plan_approval")
    assert [e["stage"] for e in plan_pause["stage_reports"]] == ["stage1"]
    earlier = next(p for p in pauses if p["type"] == "wrangling_approval")
    assert earlier["stage_reports"] == []


def test_a_second_question_on_the_same_thread_gets_its_own_report(fake_model, tmp_path, store):
    app = _app(store)
    _drive(app, _start(_messy_frame(), tmp_path, store, "q1"), "thread")
    values, _, _ = _drive(app, _start(_frame(seed=62), tmp_path, store, "q2"), "thread")
    first, second = values["stage_reports"]
    assert (first["stage"], second["stage"]) == ("stage1", "stage1")
    assert "Removed duplicate rows" in first["text"]
    assert _sections(second)["Changed"] == "No changes were proposed."
    log = values["decision_log"]
    starts = [i for i, r in enumerate(log) if r["decision_point"] == "planner.stage_plan"]
    assert len(starts) == 2
    first_ids = {r["id"] for r in log[:starts[1]]}
    second_ids = {r["id"] for r in log[starts[1]:]}
    assert set(first["source_records"]) <= first_ids
    assert second["source_records"] and set(second["source_records"]) <= second_ids


# --- privacy --------------------------------------------------------------------------------------------------------

def test_no_canary_reaches_a_stage_report(fake_model, tmp_path, store):
    _, config = fake_model
    config.update(question_type=QuestionType.TWO_GROUP_COMPARISON, predictors=["region"])
    ds = _dataset(_with_duplicates(_base_df()), tmp_path, store, "canary")
    values, pauses = _drive_canaries(build_agent_graph(store), {"request_text": "Did West spend more?",
                                                                "session_id": "s", "dataset": ds}, "canary")
    assert "wrangling_approval" in pauses
    entry = _stage1(values)
    assert "Removed duplicate rows" in entry["text"] and "Kept rows where notes has a value" in entry["text"]
    text = entry["text"] + "\n" + json.dumps(entry["sections"])
    assert canary_hits([{"stage": "stage1_report", "text": text}]) == []


# --- graph and registry ---------------------------------------------------------------------------------------------

def test_graph_edges_and_registry(tmp_path):
    edges = {(e.source, e.target): e.conditional
             for e in build_agent_graph(LocalDiskStore(base_dir=tmp_path)).get_graph().edges}
    for target in ("wrangle_review", "describe", "stage1_report"):
        assert edges[("wrangle_propose", target)] is True, target
    assert edges[("wrangle_review", "describe")] is True and edges[("wrangle_review", "stage1_report")] is True
    assert edges[("describe", "stage1_report")] is False
    assert edges[("stage1_report", "draft_plan")] is False
    assert ("describe", "draft_plan") not in edges
    assert "stage1_report" in NON_CHOOSING_NODES and "stage1_report" not in DECISION_NODES
    assert NODE_STAGES["stage1_report"] == "stage1"
    assert (len(registered_points()), len(DECISION_NODES)) == (29, 15)
