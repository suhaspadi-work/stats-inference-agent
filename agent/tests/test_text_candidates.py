"""
M2.3a step 2: the code-enumerated candidates of the two text issues (core/operation_candidates.py): their ids,
eligibility and exact reasons; a dry-run preview that writes nothing and equals the executed impact; the same
preview keys as before; plain words that name counts but never a level or a value; JSON-clean candidates; the
sequential recomputation; and at most one read of the data per call.
"""
import json

import numpy as np
import pandas as pd
import pytest

from core.capabilities import operation_named
from core.detection import CandidateIssue
from core.operation import ApprovalStatus, Operation
from core.operation_candidates import dry_run, enumerate_candidates, recompute_candidates
from tests.test_operation_candidates import CountingStore, _dataset

PREVIEW_KEYS = {"rows_before", "rows_after", "columns_before", "columns_after", "rows_changed", "share_changed",
                "columns_affected", "cells_changed", "shift", "warnings"}
LEVEL_CANARIES = ("CANARY-East ", "CANARY-east", "CANARY-West")
NUMBER_CANARIES = ("31415.926", "27182.818")


@pytest.fixture
def store(tmp_path):
    return CountingStore(tmp_path / "store")


def _issue(issue_type, column):
    return CandidateIssue(issue_type=issue_type, description="d", affected_columns=(column,), evidence={})


def _frame(n=40):
    """region: three canary levels (one with a trailing space); tenure: numbers as text, two canary numbers and two
    tokens that are not numbers (so it stays text after the CSV round trip)."""
    rng = np.random.default_rng(4)
    tenure = [f"{v:.1f}" for v in rng.normal(24, 8, n)]
    tenure[0], tenure[1], tenure[5], tenure[9] = NUMBER_CANARIES[0], NUMBER_CANARIES[1], "unknown", "n/a"
    return pd.DataFrame({"region": [LEVEL_CANARIES[i % 3] for i in range(n)], "tenure": tenure,
                         "spend": rng.normal(50, 5, n)})


def _by_id(issue_type, column, df, tmp_path, store, name):
    ds = _dataset(df, tmp_path, store, name)
    return ds, {c["id"]: c for c in enumerate_candidates(_issue(issue_type, column), ds, None, store)}


# --- the candidates of an inconsistent_text issue --------------------------------------------------------------------

def test_the_text_candidates(tmp_path, store):
    _, c = _by_id("inconsistent_text", "region", _frame(), tmp_path, store, "text")
    assert list(c) == ["leave_as_is", "trim_text", "trim_and_lowercase"]
    assert (c["leave_as_is"]["eligible"], c["leave_as_is"]["reason"]) == (True, "keep the values as they are")
    assert (c["trim_text"]["eligible"], c["trim_text"]["reason"]) == (True, "removes extra spaces")
    assert (c["trim_and_lowercase"]["eligible"], c["trim_and_lowercase"]["reason"]) == (
        True, "removes extra spaces and treats upper and lower case as the same")
    assert (c["trim_text"]["op_type"], c["trim_text"]["params"]) == ("normalize_text", {"column": "region", "lowercase": False})
    assert c["trim_and_lowercase"]["params"] == {"column": "region", "lowercase": True}
    levels = _frame()["region"]
    lowered = levels.str.strip().str.lower()
    assert c["trim_and_lowercase"]["what_was_run"] == (
        f"Trim spaces and ignore case in region ({levels.nunique()} distinct values become {lowered.nunique()}).")
    assert c["trim_text"]["what_was_run"] == "Trim spaces in region."  # 3 levels stay 3 after trimming


def test_trim_text_is_ineligible_without_extra_spaces(tmp_path, store):
    _, c = _by_id("inconsistent_text", "region", pd.DataFrame({"region": ["East", "east", "West"] * 5, "y": range(15)}),
                  tmp_path, store, "case")
    assert (c["trim_text"]["eligible"], c["trim_text"]["reason"], c["trim_text"]["preview"]) == (
        False, "no value has extra spaces", None)
    assert c["trim_and_lowercase"]["eligible"]


def test_both_are_ineligible_when_nothing_changes(tmp_path, store):
    _, c = _by_id("inconsistent_text", "region", pd.DataFrame({"region": ["east", "west"] * 6, "y": range(12)}),
                  tmp_path, store, "clean")  # defensive: the detector would not flag this column
    assert (c["trim_text"]["eligible"], c["trim_text"]["reason"], c["trim_text"]["preview"]) == (
        False, "no value has extra spaces", None)
    assert (c["trim_and_lowercase"]["eligible"], c["trim_and_lowercase"]["reason"],
            c["trim_and_lowercase"]["preview"]) == (False, "no value changes", None)


# --- the candidates of a numeric_as_text issue -----------------------------------------------------------------------

def test_the_numeric_text_candidates(tmp_path, store):
    ds, c = _by_id("numeric_as_text", "tenure", _frame(), tmp_path, store, "numeric")
    df = store.read(ds.storage_key)  # the stored data: the CSV round trip reads "n/a" as missing
    assert list(c) == ["leave_as_is", "convert_to_number"]
    convert = c["convert_to_number"]
    assert (convert["eligible"], convert["reason"]) == (
        True, "the values that are numbers are converted; the others become missing")
    assert (convert["op_type"], convert["params"]) == ("cast", {"column": "tenure", "to_type": "numeric"})
    lost = int(pd.to_numeric(df["tenure"].dropna(), errors="coerce").isna().sum())
    assert lost == 1  # "unknown"
    assert convert["what_was_run"] == f"Convert tenure to numbers ({lost} value(s) that are not numbers become missing)."
    assert convert["preview"]["warnings"] == [f"{lost} value(s) that are not numbers become missing"]


def test_leading_zeros_make_the_conversion_ineligible(tmp_path, store):
    codes = pd.DataFrame({"zip": ["02134", "10001", "unknown"] + [str(10000 + i) for i in range(20)], "y": range(23)})
    _, c = _by_id("numeric_as_text", "zip", codes, tmp_path, store, "codes")
    assert c["convert_to_number"]["eligible"] is False
    assert c["convert_to_number"]["reason"] == "values with leading zeros look like codes, not numbers"
    assert c["convert_to_number"]["preview"] is None


# --- ids, previews and privacy ---------------------------------------------------------------------------------------

def _all_candidates(tmp_path, store):
    ds = _dataset(_frame(), tmp_path, store, "all")
    return ds, (enumerate_candidates(_issue("inconsistent_text", "region"), ds, None, store)
                + enumerate_candidates(_issue("numeric_as_text", "tenure"), ds, None, store))


def test_ids_have_no_punctuation_or_whitespace(tmp_path, store):
    _, candidates = _all_candidates(tmp_path, store)
    for c in candidates:
        assert c["id"].replace("_", "").isalnum(), c["id"]


def test_every_preview_equals_the_executed_impact(tmp_path, store):
    ds, candidates = _all_candidates(tmp_path, store)
    checked = 0
    for c in candidates:
        if not c["eligible"] or c["op_type"] is None:
            continue
        writes = len(store.writes)
        preview = dry_run(ds, store, c["op_type"], c["params"])
        assert len(store.writes) == writes  # a dry run writes nothing
        assert preview == c["preview"] and set(preview) == PREVIEW_KEYS, c["id"]
        op = Operation(op_type=c["op_type"], session_id="s", input_handles=(ds.handle,), params=c["params"],
                       approval_status=ApprovalStatus.APPROVED)
        _, finished = operation_named(c["op_type"]).execute(ds, store, op)
        impact = finished.impact
        assert (preview["rows_before"], preview["rows_after"], preview["columns_before"], preview["columns_after"]) == (
            impact.rows_before, impact.rows_after, impact.columns_before, impact.columns_after), c["id"]
        assert impact.rows_before == impact.rows_after, c["id"]
        if c["op_type"] == "normalize_text":  # the rows touched, as for an imputation
            assert preview["rows_changed"] == preview["cells_changed"] == impact.details["cells_changed"] > 0, c["id"]
        else:
            assert preview["rows_changed"] == 0 and preview["cells_changed"] == impact.details["non_missing_before"], c["id"]
        checked += 1
    assert checked == 3  # trim_text, trim_and_lowercase, convert_to_number


def test_no_level_or_value_appears_in_a_candidate(tmp_path, store):
    _, candidates = _all_candidates(tmp_path, store)
    for c in candidates:
        text = json.dumps(c, allow_nan=False)  # JSON-clean
        visible = [c["id"], c["reason"], c["what_was_run"], *((c["preview"] or {}).get("warnings") or []),
                   json.dumps(c["preview"])]
        for canary in ("CANARY", *NUMBER_CANARIES, "unknown", "n/a"):
            assert canary not in text, (c["id"], canary)
            assert not any(canary in v for v in visible), (c["id"], canary)


def test_recompute_candidates_for_the_new_operations(tmp_path, store):
    ds, candidates = _all_candidates(tmp_path, store)
    frame = store.read(ds.storage_key).iloc[:20]
    fresh = recompute_candidates(candidates, ds, frame, "after_earlier_operations")
    for old, new in zip(candidates, fresh):
        if old["eligible"]:
            assert new["preview"]["basis"] == "after_earlier_operations" and new["preview"]["rows_before"] == 20
            if old["op_type"] is not None:
                expected = dry_run(ds, frame, old["op_type"], old["params"])
                assert {k: v for k, v in new["preview"].items() if k != "basis"} == expected, old["id"]
        else:
            assert new == old


def test_one_read_per_call(tmp_path, store):
    ds = _dataset(_frame(), tmp_path, store, "reads")
    for issue in (_issue("inconsistent_text", "region"), _issue("numeric_as_text", "tenure")):
        store.reads.clear()
        enumerate_candidates(issue, ds, None, store)
        assert store.reads == [ds.storage_key]
        store.reads.clear()
        enumerate_candidates(issue, ds, None, store, pd.DataFrame(_frame()))
        assert store.reads == []
