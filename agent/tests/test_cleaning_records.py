import json

import pytest

from core.cleaning_records import approval_record, execution_record, fix_record, scan_record
from core.decision_log import log_update
from core.detection import CandidateIssue
from core.model_safe_view import ColumnSummary, ModelSafeView
from core.operation import (
    AlternativeApproachConsidered, ApprovalStatus, Operation, OperationImpact, WranglingRationale,
)


def _profile(rows=21, columns=3):
    cols = tuple(ColumnSummary(name=f"c{i}", dtype="object", missing_count=0, missing_pct=0.0, distinct_count=2)
                 for i in range(columns))
    return ModelSafeView(dataset_handle="d@v1", row_count=rows, column_count=columns, columns=cols)


def _dup_issue():
    return CandidateIssue(issue_type="duplicate_rows", description="1 exact duplicate row(s) detected.",
                          affected_columns=("id", "region", "spend"),
                          evidence={"duplicate_row_count": 1, "total_rows": 21})


def _missing_issue(column="region", pct=0.142857):
    return CandidateIssue(issue_type="high_missingness", description=f"Column '{column}' is missing.",
                          affected_columns=(column,), evidence={"missing_pct": pct, "missing_count": 3})


def _op(op_type="dedupe", params=None, approach="leave the data as is", because="it would bias the result",
        status=ApprovalStatus.PENDING, **extra):
    return Operation(
        op_type=op_type, session_id="s", input_handles=("d@v1",),
        params=params if params is not None else {"subset": None}, approval_status=status,
        rationale=WranglingRationale(
            rationale="remove exact duplicate rows",
            alternatives=(AlternativeApproachConsidered(approach=approach, rejected_because=because),),
        ),
        **extra,
    )


def _by_option(record):
    return {c.option: c for c in record.candidates}


def _snapshot(*objs):
    return [repr(o) for o in objs]


# --- scan ---------------------------------------------------------------------

def test_scan_no_issues_is_skipped():
    r = scan_record(_profile(), [])
    assert r.decision_point == "stage1.clean.scan"
    assert r.decided_by == "code_rule"
    assert r.chosen == "skipped"
    assert r.evidence == {"rows": 21, "columns_scanned": 3, "duplicate_row_count": 0,
                          "columns_above_missing_threshold": [], "missing_threshold": 0.05,
                          "text_issues": [], "numeric_text_columns": [],  # M2.3a step 2: the two new issue lists
                          "joinable_tables": []}  # M2.3b: the join issues (names only)
    expected = "no exact duplicate rows and no column above the 5% missing threshold (21 rows, 3 columns scanned)"
    assert r.rationale == expected
    c = _by_option(r)
    assert [x.option for x in r.candidates] == ["propose_fixes", "skipped"]
    assert c["skipped"].eligible is True and c["skipped"].reason == expected
    assert c["propose_fixes"].eligible is False and c["propose_fixes"].reason == f"not eligible: {expected}"


def test_scan_duplicates_only():
    r = scan_record(_profile(), [_dup_issue()])
    assert r.chosen == "propose_fixes"
    assert r.rationale == "1 issue(s) found: 1 exact duplicate row(s)"
    assert r.evidence["duplicate_row_count"] == 1
    assert r.evidence["columns_above_missing_threshold"] == []


def test_scan_missingness_only_formats_the_percentage():
    r = scan_record(_profile(), [_missing_issue()])
    assert r.rationale == "1 issue(s) found: column 'region' is 14.3% missing (above the 5% threshold)"
    assert r.evidence["duplicate_row_count"] == 0
    assert r.evidence["columns_above_missing_threshold"] == [{"column": "region", "missing_pct": 0.142857}]


def test_scan_both_issues_and_candidate_eligibility():
    r = scan_record(_profile(), [_dup_issue(), _missing_issue()])
    assert r.rationale == ("2 issue(s) found: 1 exact duplicate row(s); "
                           "column 'region' is 14.3% missing (above the 5% threshold)")
    c = _by_option(r)
    assert [x.option for x in r.candidates] == ["propose_fixes", "skipped"]
    assert c["propose_fixes"].eligible is True and c["propose_fixes"].reason == r.rationale
    assert c["skipped"].eligible is False and c["skipped"].reason == f"not eligible: {r.rationale}"
    assert all(x.generated_by == "code" for x in r.candidates)


# --- fix ----------------------------------------------------------------------

def test_fix_duplicates_with_dedupe():
    r = fix_record(_dup_issue(), _op(), 0)
    assert r.decision_point == "stage1.clean.fix"
    assert r.decided_by == "model"
    assert r.chosen == "dedupe"
    assert r.rationale == "remove exact duplicate rows"
    assert [(c.option, c.generated_by) for c in r.candidates] == [
        ("dedupe", "code"), ("leave_as_is", "code"), ("leave the data as is", "model")]
    c = _by_option(r)
    assert c["dedupe"].reason == "chosen: removes the 1 exact duplicate row(s) out of 21 rows"
    assert c["leave_as_is"].reason.startswith("eligible; not chosen: keeps all rows")
    assert c["dedupe"].eligible is True and c["leave_as_is"].eligible is True
    assert c["leave the data as is"].eligible is None
    assert c["leave the data as is"].reason == "it would bias the result"


def test_fix_missingness_with_filter():
    op = _op("filter", {"column": "region", "condition": "not_null", "value": None})
    r = fix_record(_missing_issue(), op, 1)
    c = _by_option(r)
    assert [x.option for x in r.candidates][:2] == ["filter", "leave_as_is"]
    assert c["filter"].reason.startswith("chosen: drops the rows with a missing value in 'region' (14.3% of its values)")
    assert "no imputation, G17" in c["filter"].reason
    assert r.evidence["issue_index"] == 1
    assert r.evidence["params"] == {"column": "region", "condition": "not_null", "value": None}


def test_fix_unmapped_op_adds_a_model_candidate():
    op = _op("filter", {"column": "id", "condition": "not_null", "value": None})
    r = fix_record(_dup_issue(), op, 0)
    c = _by_option(r)
    assert [x.option for x in r.candidates] == ["dedupe", "leave_as_is", "filter", "leave the data as is"]
    assert c["filter"].eligible is None and c["filter"].generated_by == "model"
    assert c["filter"].reason == ("proposed by the model; the code does not map this operation "
                                  "to a duplicate_rows issue")
    assert c["dedupe"].reason.startswith("eligible; not chosen: ")


def test_fix_evidence_is_plain_and_has_no_imputation_candidate():
    r = fix_record(_dup_issue(), _op(params={"subset": ("id",)}), 0)
    assert r.evidence["affected_columns"] == ["id", "region", "spend"]
    assert r.evidence["params"] == {"subset": ["id"]}
    assert r.evidence["issue_evidence"] == {"duplicate_row_count": 1, "total_rows": 21}
    json.dumps(r.to_dict())
    for record in (r, fix_record(_missing_issue(), _op("filter", {"column": "region"}), 0)):
        assert not any("imput" in c.option.lower() for c in record.candidates)


# --- approval -------------------------------------------------------------------

def test_approval_approve():
    r = approval_record(_op(), _dup_issue(), {"index": 0, "action": "approve"}, None, "fix-1")
    assert r.decision_point == "stage1.clean.approval"
    assert (r.chosen, r.decided_by, r.rationale) == ("approve", "human", "approved as proposed")
    assert r.evidence == {"op_type": "dedupe", "params": {"subset": None}, "issue_type": "duplicate_rows",
                          "received_action": "approve", "fix_record_id": "fix-1"}


def test_approval_edit_is_a_human_override():
    decision = {"index": 0, "action": "edit", "params": {"subset": ["region"]}}
    r = approval_record(_op(), _dup_issue(), decision, None, "fix-1")
    assert (r.chosen, r.decided_by) == ("edit", "human")
    assert r.overridden_by_human is True
    assert r.original_choice == "approve as proposed: dedupe({'subset': None})"
    assert r.rationale == "edited before running"
    assert r.evidence["params"] == {"subset": None}
    assert r.evidence["edited_params"] == {"subset": ["region"]}


def test_approval_reject():
    decision = {"index": 0, "action": "reject", "reason": "they are real repeat orders"}
    r = approval_record(_op(), _dup_issue(), decision, None, "fix-1")
    assert (r.chosen, r.decided_by, r.rationale) == ("reject", "human", "they are real repeat orders")
    assert r.evidence["carried_forward_as_unaddressed"] is True


def test_approval_reject_without_reason():
    r = approval_record(_op(), _dup_issue(), {"index": 0, "action": "reject"}, None, None)
    assert r.rationale == "no reason given"


def test_approval_auto():
    rule = "stakeholder mode auto-approves duplicate_rows only"
    r = approval_record(_op(), _dup_issue(), {"index": 0, "action": "approve"}, rule, "fix-1")
    assert (r.chosen, r.decided_by, r.rationale) == ("auto_approve", "code_rule", rule)


def test_approval_no_decision():
    r = approval_record(_op(), _dup_issue(), None, None, "fix-1")
    assert (r.chosen, r.decided_by) == ("no_decision", "code_rule")
    assert r.rationale == ("no decision was received for this operation, so it was not run "
                           "and not recorded as unaddressed")
    assert r.evidence["received_action"] is None


def test_approval_unrecognized_action_counts_as_approval():
    r = approval_record(_op(), _dup_issue(), {"index": 0, "action": "maybe"}, None, "fix-1")
    assert (r.chosen, r.decided_by) == ("approve", "code_rule")
    assert "G36" in r.rationale and "'maybe'" in r.rationale
    assert r.evidence["received_action"] == "maybe"


@pytest.mark.parametrize("decision, rule", [
    ({"action": "approve"}, None), ({"action": "edit", "params": {}}, None), ({"action": "reject"}, None),
    ({"action": "approve"}, "rule"), (None, None), ({"action": "maybe"}, None),
])
def test_approval_candidates_and_fix_id_always(decision, rule):
    r = approval_record(_op(), _dup_issue(), decision, rule, "fix-42")
    assert [(c.option, c.eligible, c.generated_by) for c in r.candidates] == [
        ("approve", True, "code"), ("edit", True, "code"), ("reject", True, "code")]
    assert r.evidence["fix_record_id"] == "fix-42"


# --- execution -----------------------------------------------------------------

def test_execution_applied():
    op = _op(status=ApprovalStatus.APPROVED).with_result(
        OperationImpact(rows_before=21, rows_after=20, columns_before=3, columns_after=3, warnings=("w",)),
        output_handle="d@v2")
    r = execution_record(op, "appr-1")
    assert r.decision_point == "stage1.clean.execution"
    assert (r.chosen, r.decided_by) == ("applied", "code_rule")
    assert r.rationale == "rows 21 -> 20, columns 3 -> 3"
    assert r.candidates == ()
    assert r.evidence == {
        "op_type": "dedupe", "params": {"subset": None}, "operation_id": op.id,
        "input_handles": ["d@v1"], "output_handle": "d@v2",
        "rows_before": 21, "rows_after": 20, "columns_before": 3, "columns_after": 3,
        "warnings": ["w"], "approval_record_id": "appr-1",
    }


def test_execution_failed():
    op = _op(status=ApprovalStatus.APPROVED).mark_failed("Cannot dedupe — columns not found: ['nope']")
    r = execution_record(op, "appr-1")
    assert (r.chosen, r.decided_by) == ("failed", "code_rule")
    assert r.rationale == "the operation failed: Cannot dedupe — columns not found: ['nope']"
    assert r.candidates == ()
    assert r.evidence == {"op_type": "dedupe", "params": {"subset": None}, "operation_id": op.id,
                          "failure_reason": "Cannot dedupe — columns not found: ['nope']",
                          "approval_record_id": "appr-1"}


# --- every record is plain, and inputs are not mutated ---------------------------

def test_all_records_pass_log_update_without_mutating_inputs():
    profile, dup, miss = _profile(), _dup_issue(), _missing_issue()
    op = _op(params={"subset": ("id",)})
    decision = {"index": 0, "action": "edit", "params": {"subset": ("region",)}}
    applied = _op(status=ApprovalStatus.APPROVED).with_result(
        OperationImpact(rows_before=4, rows_after=3, columns_before=2, columns_after=2), "d@v2")
    failed = _op(status=ApprovalStatus.APPROVED).mark_failed("boom")
    before = _snapshot(profile, dup, miss, op, decision, applied, failed)

    records = [
        scan_record(profile, [dup, miss]), fix_record(dup, op, 0),
        approval_record(op, dup, decision, None, "f"), approval_record(op, dup, None, None, "f"),
        approval_record(op, dup, {"action": "approve"}, "rule", "f"),
        execution_record(applied, "a"), execution_record(failed, "a"),
    ]
    update = log_update(*records)
    json.dumps(update)
    assert update["decision_log"][2]["evidence"]["edited_params"] == {"subset": ["region"]}
    assert _snapshot(profile, dup, miss, op, decision, applied, failed) == before
