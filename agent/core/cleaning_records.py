"""
Decision records for data cleaning (M0.2 step 8b, G25, G35). Pure functions:
no graph, no model calls, no data access, and no import of core.nodes.

  stage1.clean.scan       whether any issue was found (fixes proposed or skipped)
  stage1.clean.fix        the operation proposed for one issue
  stage1.clean.approval   what was decided about that operation
  stage1.clean.execution  what running it did, or why it failed

Candidate lists hold only operations the agent can run (dedupe, filter) plus
leaving the data as is; there is no imputation candidate (G17).
"""
from __future__ import annotations
import copy
from typing import Any, Optional

from core.decision_record import Candidate, DecisionRecord
from core.detection import MISSINGNESS_THRESHOLD
from core.model_safe_view import ModelSafeView
from core.operation import ApprovalStatus, Operation

SCAN_POINT = "stage1.clean.scan"
FIX_POINT = "stage1.clean.fix"
APPROVAL_POINT = "stage1.clean.approval"
EXECUTION_POINT = "stage1.clean.execution"

_THRESHOLD_TEXT = f"{MISSINGNESS_THRESHOLD:.0%}"


def _plain(value: Any) -> Any:
    """Copy into plain dicts and lists (tuples become lists)."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return copy.deepcopy(value)


def _op_text(op_type: str, params: dict) -> str:
    return f"{op_type}({params})"


# ------------------------------------------------------------------------- scan

def scan_record(profile: ModelSafeView, issues) -> DecisionRecord:
    duplicate_count = 0
    missing_columns = []
    parts = []
    for issue in issues:
        if issue.issue_type == "duplicate_rows":
            duplicate_count = int(issue.evidence.get("duplicate_row_count", 0))
            parts.append(f"{duplicate_count} exact duplicate row(s)")
        elif issue.issue_type == "high_missingness":
            column = str(issue.affected_columns[0])
            pct = float(issue.evidence.get("missing_pct", 0.0))
            missing_columns.append({"column": column, "missing_pct": pct})
            parts.append(f"column '{column}' is {pct:.1%} missing (above the {_THRESHOLD_TEXT} threshold)")
        else:
            parts.append(issue.description)

    rows, columns = int(profile.row_count), int(profile.column_count)
    if issues:
        fact = f"{len(issues)} issue(s) found: " + "; ".join(parts)
    else:
        fact = (f"no exact duplicate rows and no column above the {_THRESHOLD_TEXT} missing threshold "
                f"({rows} rows, {columns} columns scanned)")
    found = bool(issues)
    chosen = "propose_fixes" if found else "skipped"
    candidates = (
        Candidate(option="propose_fixes", eligible=found, reason=fact if found else f"not eligible: {fact}"),
        Candidate(option="skipped", eligible=not found, reason=f"not eligible: {fact}" if found else fact),
    )
    return DecisionRecord(
        decision_point=SCAN_POINT, chosen=chosen, decided_by="code_rule",
        candidates=candidates, rationale=fact,
        evidence={
            "rows": rows, "columns_scanned": columns,
            "duplicate_row_count": duplicate_count,
            "columns_above_missing_threshold": missing_columns,
            "missing_threshold": MISSINGNESS_THRESHOLD,
        },
    )


# -------------------------------------------------------------------------- fix

def _runnable_options(issue) -> list[tuple[str, str]]:
    """The (option, fact) pairs code can run for this issue type, then leaving it as is."""
    ev = issue.evidence
    if issue.issue_type == "duplicate_rows":
        return [
            ("dedupe", f"removes the {int(ev.get('duplicate_row_count', 0))} exact duplicate row(s) "
                       f"out of {int(ev.get('total_rows', 0))} rows"),
            ("leave_as_is", "keeps all rows; the duplicates stay in the analysis and the issue is "
                            "carried forward as unaddressed"),
        ]
    if issue.issue_type == "high_missingness":
        column = issue.affected_columns[0]
        pct = float(ev.get("missing_pct", 0.0))
        return [
            ("filter", f"drops the rows with a missing value in '{column}' ({pct:.1%} of its values); "
                       f"dropping rows is the only missing-value handling the agent can run today "
                       f"(no imputation, G17)"),
            ("leave_as_is", f"keeps all rows; the missing values in '{column}' stay and the issue is "
                            f"carried forward as unaddressed"),
        ]
    return [("leave_as_is", "keeps the data as is; the issue is carried forward as unaddressed")]


def fix_record(issue, operation: Operation, issue_index: int) -> DecisionRecord:
    chosen = operation.op_type
    candidates = []
    options = _runnable_options(issue)
    for option, fact in options:
        prefix = "chosen: " if option == chosen else "eligible; not chosen: "
        candidates.append(Candidate(option=option, eligible=True, reason=prefix + fact))
    if chosen not in {option for option, _ in options}:
        candidates.append(Candidate(
            option=chosen, eligible=None, generated_by="model",
            reason=f"proposed by the model; the code does not map this operation to a {issue.issue_type} issue",
        ))
    rationale = operation.rationale
    if rationale is not None and rationale.alternatives:
        alt = rationale.alternatives[0]
        candidates.append(Candidate(option=alt.approach, eligible=None, generated_by="model",
                                    reason=alt.rejected_because))
    return DecisionRecord(
        decision_point=FIX_POINT, chosen=chosen, decided_by="model",
        candidates=tuple(candidates),
        rationale=rationale.rationale if rationale is not None else "",
        evidence={
            "issue_index": int(issue_index), "issue_type": issue.issue_type,
            "affected_columns": [str(c) for c in issue.affected_columns],
            "params": _plain(operation.params), "issue_evidence": _plain(issue.evidence),
        },
    )


# --------------------------------------------------------------------- approval

_APPROVAL_CANDIDATES = (
    Candidate(option="approve", eligible=True, reason="run the operation as proposed"),
    Candidate(option="edit", eligible=True, reason="change its parameters, then run it"),
    Candidate(option="reject", eligible=True, reason="do not run it; the issue is carried forward as unaddressed"),
)


def approval_record(operation: Operation, issue, decision: Optional[dict], auto_rule: Optional[str],
                    fix_record_id: Optional[str], edited_params: Optional[dict] = None) -> DecisionRecord:
    """
    edited_params: for an "edit", the params that actually ran (after normalization);
    defaults to the params in the decision.
    """
    action = decision.get("action") if decision is not None else None
    evidence = {
        "op_type": operation.op_type, "params": _plain(operation.params), "issue_type": issue.issue_type,
        "received_action": action, "fix_record_id": fix_record_id,
    }
    overridden, original = False, None

    if auto_rule is not None:
        chosen, decided_by, rationale = "auto_approve", "code_rule", auto_rule
    elif decision is None:
        chosen, decided_by = "no_decision", "code_rule"
        rationale = "no decision was received for this operation, so it was not run and not recorded as unaddressed"
    elif action == "approve":
        chosen, decided_by, rationale = "approve", "human", "approved as proposed"
    elif action == "edit":
        chosen, decided_by, rationale = "edit", "human", "edited before running"
        overridden = True
        original = f"approve as proposed: {_op_text(operation.op_type, operation.params)}"
        ran = edited_params if edited_params is not None else decision.get("params", operation.params)
        evidence["edited_params"] = _plain(ran)
    elif action == "reject":
        chosen, decided_by = "reject", "human"
        rationale = decision.get("reason", "no reason given")
        evidence["carried_forward_as_unaddressed"] = True
    else:
        chosen, decided_by = "approve", "code_rule"
        rationale = f"action '{action}' is not one of approve, edit or reject; the node treats it as approval (G36)"

    return DecisionRecord(
        decision_point=APPROVAL_POINT, chosen=chosen, decided_by=decided_by,
        overridden_by_human=overridden, original_choice=original,
        candidates=_APPROVAL_CANDIDATES, rationale=str(rationale), evidence=evidence,
    )


# -------------------------------------------------------------------- execution

def execution_record(finished_operation: Operation, approval_record_id: Optional[str]) -> DecisionRecord:
    op = finished_operation
    if op.approval_status == ApprovalStatus.FAILED:
        return DecisionRecord(
            decision_point=EXECUTION_POINT, chosen="failed", decided_by="code_rule",
            rationale=f"the operation failed: {op.failure_reason}",
            evidence={
                "op_type": op.op_type, "params": _plain(op.params), "operation_id": op.id,
                "failure_reason": op.failure_reason, "approval_record_id": approval_record_id,
            },
        )
    impact = op.impact
    rows_before = int(impact.rows_before) if impact else None
    rows_after = int(impact.rows_after) if impact else None
    cols_before = int(impact.columns_before) if impact else None
    cols_after = int(impact.columns_after) if impact else None
    return DecisionRecord(
        decision_point=EXECUTION_POINT, chosen="applied", decided_by="code_rule",
        rationale=f"rows {rows_before} -> {rows_after}, columns {cols_before} -> {cols_after}",
        evidence={
            "op_type": op.op_type, "params": _plain(op.params), "operation_id": op.id,
            "input_handles": list(op.input_handles), "output_handle": op.output_handle,
            "rows_before": rows_before, "rows_after": rows_after,
            "columns_before": cols_before, "columns_after": cols_after,
            "warnings": list(impact.warnings) if impact else [],
            "approval_record_id": approval_record_id,
        },
    )
