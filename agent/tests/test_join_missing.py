"""
M2.4a: the missing values an AGGREGATE join makes for the keys that have no rows in the carried table (join_missing):
detection (names only), the three candidates (leave_as_is, delete_unmatched_rows, fill_empty_sums_with_zero) with
their previews, the fill_unmatched operation, and the end-to-end second round of the aggregate shape. Expected values
are computed independently with pandas.
"""
import json
import re

import numpy as np
import pandas as pd
import pytest

from core.capabilities import OPERATIONS, run_transform
from core.dataset import LocalDiskStore
from core.detection import CandidateIssue, detect_all_issues, detect_join_missing, issue_key
from core.ingestion import profile_dataset
from core.operation import ApprovalStatus, Operation
from core.operation_candidates import dry_run, enumerate_candidates, recompute_candidates
from core.wrangling import describe_operation, execute_fill_unmatched, execute_filter, transform_fill_unmatched, transform_join
from tests.test_aggregate_run import _app, _customers, _orders
from tests.test_decision_point_guard import _dataset, choose
from tests.test_join_run import _bundle, _records, _sections, _state, _store_frame
from tests.test_prompt_canaries import fake_model  # noqa: F401 (fixture)
from tests.test_stage1_report import _drive

PREVIEW_KEYS = {"rows_before", "rows_after", "columns_before", "columns_after", "rows_changed", "share_changed",
                "columns_affected", "cells_changed", "shift", "warnings"}
SUMMARY = {"count": True, "columns": ["amount", "units"], "stats": ["mean", "sum"]}


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _left(n=10):
    return pd.DataFrame({"customer_id": range(1, n + 1), "region": ["East", "West"] * (n // 2)})


def _right():
    """Orders for customers 1 to 7 (8, 9 and 10 have none); customer 7's amounts are all missing."""
    rows = [(k, float(10 * k + j), j + 1) for k in range(1, 7) for j in range(3)] + [(7, np.nan, 2), (7, np.nan, 1)]
    return pd.DataFrame(rows, columns=["customer_id", "amount", "units"])


def _params(aggregate=None, right="orders"):
    return {"right_table": right, "left_key": "customer_id", "right_key": "customer_id", "how": "left",
            "normalize_keys": False, "result_name": f"customers_{right}", "aggregate": aggregate or SUMMARY}


def _joined(aggregate=None, left=None, right=None, name="orders"):
    """(the joined frame, the evidence dict of the applied join execution)."""
    params = _params(aggregate, name)
    df, details, _ = transform_join(_left() if left is None else left, params, {name: _right() if right is None else right})
    return df, {"op_type": "join", "params": params, "details": details}


ADDED = ["amount_mean", "amount_sum", "units_mean", "units_sum"]


# --- detection ---------------------------------------------------------------------------------------------------

def test_one_issue_with_exact_evidence_and_description():
    df, ev = _joined()
    (issue,) = detect_join_missing(df, [ev])
    assert (issue.issue_type, issue.affected_columns) == ("join_missing", tuple(ADDED))
    assert issue.evidence == {"right_table": "orders", "count_column": "rows_in_orders", "columns": ADDED,
                              "sum_columns": ["amount_sum", "units_sum"], "result_table": "customers_orders"}
    assert issue.description == ("The columns amount_mean, amount_sum, units_mean, units_sum that the join of 'orders' "
                                 "added are missing for the rows that have no rows in 'orders'.")
    # independently: 3 unmatched customers have a count of 0
    assert int((df["rows_in_orders"] == 0).sum()) == 3


def test_the_fifteen_name_cap():
    names = [f"c{chr(97 + i)}_mean" for i in range(17)]
    df = pd.DataFrame({"customer_id": [1, 2], "rows_in_t": [0, 2], **{c: [np.nan, 1.0] for c in names}})
    ev = {"op_type": "join", "params": {"right_table": "t", "aggregate": {"count": True}, "result_name": "x_t"},
          "details": {"columns_added": ["rows_in_t", *names], "aggregate": {"count_column": "rows_in_t", "stats": ["mean"]}}}
    (issue,) = detect_join_missing(df, [ev])
    assert issue.description == (f"The columns {', '.join(names[:15])} and 2 more that the join of 't' added are missing "
                                 "for the rows that have no rows in 't'.")
    assert issue.evidence["columns"] == names and issue.evidence["sum_columns"] == []


def test_names_only():
    right = _right().assign(note=["CANARY-NOTE"] * 20)
    right.loc[0, "amount"] = 31415.926
    df, ev = _joined(right=right)
    (issue,) = detect_join_missing(df, [ev])
    text = json.dumps({"d": issue.description, "e": issue.evidence, "c": list(issue.affected_columns)})
    assert "CANARY" not in text and "31415" not in text and not re.search(r"\d", text)


def test_no_issue_cases():
    df, ev = _joined()
    plain = {"op_type": "join", "params": {**ev["params"], "aggregate": None}, "details": ev["details"]}
    assert detect_join_missing(df, [plain]) == []  # a plain join
    every = _left(6)  # customers 1 to 6 all have orders
    matched, ev_all = _joined(left=every)
    assert detect_join_missing(matched, [ev_all]) == []  # no unmatched row
    assert detect_join_missing(df.drop(columns=["rows_in_orders"]), [ev]) == []  # the count column is gone
    sums_only, ev_sums = _joined({"count": True, "columns": ["units"], "stats": ["sum"]})
    filled, _, _ = transform_fill_unmatched(sums_only, {"count_column": "rows_in_orders", "columns": ["units_sum"],
                                                        "value": 0, "right_table": "orders"})
    assert detect_join_missing(filled, [ev_sums]) == []  # every added column is complete
    assert detect_join_missing(df, []) == []


def test_sum_columns_and_two_joins():
    means, ev_means = _joined({"count": True, "columns": ["amount"], "stats": ["mean"]})
    (issue,) = detect_join_missing(means, [ev_means])
    assert issue.evidence["sum_columns"] == [] and issue.evidence["columns"] == ["amount_mean"]
    df, ev = _joined()
    visits = pd.DataFrame({"customer_id": [1, 1, 2], "minutes": [3.0, 4.0, 5.0]})
    params = {**_params({"count": True, "columns": ["minutes"], "stats": ["sum"]}, "visits"), "result_name": "cv"}
    both, details, _ = transform_join(df, params, {"visits": visits})
    issues = detect_join_missing(both, [ev, {"op_type": "join", "params": params, "details": details}])
    assert [i.evidence["right_table"] for i in issues] == ["orders", "visits"]
    assert issues[1].evidence["columns"] == issues[1].evidence["sum_columns"] == ["minutes_sum"]


def test_detect_all_issues_suppresses_only_the_structural_columns(tmp_path, store):
    df, ev = _joined()
    df["other"] = np.where(df["rows_in_orders"] == 0, np.nan, 1.0)  # the same share missing, another cause
    ds = _dataset(df, tmp_path, store, "joined")
    profile = profile_dataset(ds, store=store)
    today = detect_all_issues(ds, profile, store)
    assert today == detect_all_issues(ds, profile, store, join_records=None)  # None: exactly as before
    assert {i.affected_columns[0] for i in today if i.issue_type == "high_missingness"} == {*ADDED, "other"}
    issues = detect_all_issues(ds, profile, store, join_records=[ev])
    assert [i.affected_columns for i in issues if i.issue_type == "high_missingness"] == [("other",)]
    assert [i.issue_type for i in issues][-1] == "join_missing"


def test_issue_key_of_join_missing_is_its_table():
    # M2.4a: a join_missing issue is keyed by its carried table only, whatever columns are still missing
    key = issue_key(CandidateIssue("join_missing", "d", ("b", "a"), {"right_table": "orders"}))
    assert key == ["join_missing", [], "orders"] and json.loads(json.dumps(key)) == key
    assert issue_key(CandidateIssue("join_missing", "other", ("a_mean",), {"right_table": "orders"})) == key
    assert issue_key(CandidateIssue("join_missing", "d", ("b", "a"), {"right_table": "visits"})) != key
    df, ev = _joined()
    filled, _, _ = transform_fill_unmatched(df, FILL)
    (before,), (after,) = detect_join_missing(df, [ev]), detect_join_missing(filled, [ev])
    assert before.affected_columns != after.affected_columns and issue_key(before) == issue_key(after)


def test_every_other_issue_key_is_unchanged():
    ev = {"right_table": "orders", "left_key": "customer_id", "right_key": "customer_id"}
    cases = [(CandidateIssue("duplicate_rows", "d", ("b", "a"), {"duplicate_row_count": 1}), ["duplicate_rows", ["a", "b"], None]),
             (CandidateIssue("high_missingness", "d", ("tenure",), {}), ["high_missingness", ["tenure"], None]),
             (CandidateIssue("inconsistent_text", "d", ("region",), {}), ["inconsistent_text", ["region"], None]),
             (CandidateIssue("numeric_as_text", "d", ("score",), {}), ["numeric_as_text", ["score"], None]),
             (CandidateIssue("joinable_table", "d", ("customer_id",), ev), ["joinable_table", ["customer_id"], "orders"]),
             (CandidateIssue("aggregatable_table", "d", ("customer_id",), ev),
              ["aggregatable_table", ["customer_id"], "orders"])]
    for issue, key in cases:
        assert issue_key(issue) == key, issue.issue_type


# --- candidates --------------------------------------------------------------------------------------------------

def _issue_and_frame(tmp_path, store, aggregate=None):
    df, ev = _joined(aggregate)
    ds = _dataset(df, tmp_path, store, "customers_orders")
    df = store.read(ds.storage_key)  # as stored (the CSV round trip)
    (issue,) = detect_join_missing(df, [ev])
    return issue, ds, df


def test_the_candidates(tmp_path, store):
    issue, ds, df = _issue_and_frame(tmp_path, store)
    c = {x["id"]: x for x in enumerate_candidates(issue, ds, None, store, df)}
    assert list(c) == ["leave_as_is", "delete_unmatched_rows", "fill_empty_sums_with_zero"]
    assert all(re.fullmatch(r"[a-z_]+", cid) for cid in c)
    assert [(x["eligible"], x["reason"]) for x in c.values()] == [
        (True, "keep the missing values; the analysis leaves those rows out where it needs the column"),
        (True, "removes the rows that have no rows in orders"),
        (True, "fills the sums of those rows with 0 (an empty sum is 0); the means stay missing")]
    assert c["delete_unmatched_rows"]["params"] == {"column": "rows_in_orders", "condition": "greater_than", "value": 0,
                                                    "unmatched_of": "orders"}
    assert c["fill_empty_sums_with_zero"]["params"] == {"count_column": "rows_in_orders",
                                                        "columns": ["amount_sum", "units_sum"], "value": 0,
                                                        "right_table": "orders"}


def test_mean_only_cannot_fill(tmp_path, store):
    issue, ds, df = _issue_and_frame(tmp_path, store, {"count": True, "columns": ["amount"], "stats": ["mean"]})
    c = {x["id"]: x for x in enumerate_candidates(issue, ds, None, store, df)}
    assert (c["fill_empty_sums_with_zero"]["eligible"], c["fill_empty_sums_with_zero"]["reason"]) == (
        False, "none of the added columns is a sum")
    assert c["delete_unmatched_rows"]["eligible"] is True


def _executed(ds, store, op_type, params):
    op = Operation(op_type=op_type, session_id=ds.session_id, input_handles=(ds.handle,), params=params,
                   approval_status=ApprovalStatus.APPROVED)
    return (execute_filter if op_type == "filter" else execute_fill_unmatched)(ds, store, op)


def test_previews_equal_the_execution(tmp_path, store):
    issue, ds, df = _issue_and_frame(tmp_path, store)
    unmatched = int((df["rows_in_orders"] == 0).sum())
    for cand in enumerate_candidates(issue, ds, None, store, df):
        if cand["op_type"] is None:
            continue
        preview = cand["preview"]
        assert set(preview) == PREVIEW_KEYS, cand["id"]
        new, finished = _executed(ds, store, cand["op_type"], cand["params"])
        impact, after = finished.impact, store.read(new.storage_key)
        assert (preview["rows_before"], preview["rows_after"], preview["columns_before"], preview["columns_after"]) == (
            impact.rows_before, impact.rows_after, impact.columns_before, impact.columns_after) == (
            len(df), len(after), len(df.columns), len(after.columns))
        if cand["op_type"] == "filter":
            assert preview["rows_changed"] == unmatched == len(df) - len(after)
            assert preview["cells_changed"] == unmatched * len(df.columns)
        else:
            d = impact.details
            assert (preview["rows_changed"], preview["cells_changed"]) == (d["rows_filled"], d["cells_filled"]) == (
                unmatched, 2 * unmatched)
            assert preview["columns_affected"] == ["amount_sum", "units_sum"] and preview["shift"] is None


def test_option_words_hold_no_digit_and_no_column_name(tmp_path, store):
    issue, ds, df = _issue_and_frame(tmp_path, store)
    for cand in enumerate_candidates(issue, ds, None, store, df):
        if cand["op_type"] is None:
            continue
        words = describe_operation(cand["op_type"], cand["params"])  # what the proposer sees
        assert not re.search(r"\d", words), words
        assert not any(str(c) in words for c in df.columns), words


def test_recomputed_previews(tmp_path, store):
    issue, ds, df = _issue_and_frame(tmp_path, store)
    candidates = enumerate_candidates(issue, ds, None, store, df)
    other = df.iloc[5:]  # customers 6 to 10: 3 of them unmatched
    for old, new in zip(candidates, recompute_candidates(candidates, ds, other, "after_earlier_operations")):
        if old["op_type"] is not None:
            assert new["preview"]["rows_before"] == 5 and new["preview"]["basis"] == "after_earlier_operations"
            assert {k: v for k, v in new["preview"].items() if k != "basis"} == dry_run(ds, other, old["op_type"],
                                                                                        old["params"])


# --- transform_fill_unmatched ------------------------------------------------------------------------------------

FILL = {"count_column": "rows_in_orders", "columns": ["amount_sum", "units_sum"], "value": 0, "right_table": "orders"}


def test_the_fill_transform():
    df, _ = _joined()
    before = df.copy()
    new, details, warnings = transform_fill_unmatched(df, FILL)
    assert df.equals(before) and warnings == ()  # pure
    unmatched = df["rows_in_orders"] == 0
    for column in ("amount_sum", "units_sum"):
        assert (new.loc[unmatched, column] == 0).all()
        assert new.loc[~unmatched, column].equals(df.loc[~unmatched, column])  # a matched row is never touched
    seven = df["customer_id"] == 7  # matched, but every amount missing: its sum stays missing
    assert df.loc[seven, "amount_sum"].isna().all() and new.loc[seven, "amount_sum"].isna().all()
    assert new.loc[unmatched, ["amount_mean", "units_mean"]].isna().all().all()  # the means stay missing
    assert details == {"count_column": "rows_in_orders", "columns": ["amount_sum", "units_sum"], "value": 0,
                       "rows_filled": int(unmatched.sum()), "cells_filled": 2 * int(unmatched.sum())}


def test_the_fill_errors():
    df, _ = _joined()
    with pytest.raises(ValueError, match="^Cannot fill — column not found: nope$"):
        transform_fill_unmatched(df, {**FILL, "columns": ["nope"]})
    with pytest.raises(ValueError, match="^Cannot fill — region is not numeric.$"):
        transform_fill_unmatched(df, {**FILL, "columns": ["region"]})
    with pytest.raises(ValueError, match="^Cannot fill — column not found: gone$"):
        transform_fill_unmatched(df, {**FILL, "count_column": "gone"})


def test_the_words():
    assert describe_operation("fill_unmatched", FILL) == (
        "Fill the empty sums with zero for the rows that have no rows in orders.")
    assert describe_operation("fill_unmatched", FILL, {"columns": ["amount_sum", "units_sum"], "cells_filled": 6}) == (
        "Fill amount_sum, units_sum with zero for the rows that have no rows in orders (6 value(s) filled).")
    delete = {"column": "rows_in_orders", "condition": "greater_than", "value": 0, "unmatched_of": "orders"}
    assert describe_operation("filter", delete) == "Remove the rows that have no rows in orders."
    assert describe_operation("filter", delete, {"rows_removed": 3}) == (
        "Remove the rows that have no rows in orders (3 removed).")
    plain = {"column": "rows_in_orders", "condition": "greater_than", "value": 0}  # every other filter: unchanged
    assert describe_operation("filter", plain) == "Keep only rows where rows_in_orders is greater than 0."
    df, _ = _joined()
    assert run_transform("filter", df, delete)[0].equals(run_transform("filter", df, plain)[0])  # ignored


def test_the_executor_and_the_registry(tmp_path, store):
    entry = OPERATIONS["fill_unmatched"]
    assert (entry.multi_input, entry.transform, entry.execute) == (False, transform_fill_unmatched,
                                                                  execute_fill_unmatched)
    df, _ = _joined()
    ds = _dataset(df, tmp_path, store, "filled")
    new, finished = _executed(ds, store, "fill_unmatched", FILL)
    assert finished.approval_status != ApprovalStatus.FAILED and new.version == ds.version + 1
    assert (store.read(new.storage_key).loc[lambda d: d["rows_in_orders"] == 0, "units_sum"] == 0).all()
    _, failed = _executed(ds, store, "fill_unmatched", {**FILL, "columns": ["region"]})
    assert (failed.approval_status, failed.failure_reason) == (ApprovalStatus.FAILED, "Cannot fill — region is not numeric.")


# --- end to end: the aggregate shape ------------------------------------------------------------------------------

DESCRIPTION = ("The columns amount_mean, amount_sum, quantity_mean, quantity_sum that the join of 'orders' added are "
               "missing for the rows that have no rows in 'orders'.")


def _wranglings(pauses):
    return [p["payload"] for p in pauses if p["type"] == "wrangling_approval"]


def _setup(tmp_path, store):
    customers = _customers()
    tables = _bundle(tmp_path, store, {"customers": customers, "orders": _orders(customers)})
    orders = _store_frame(store, tables["orders"])
    unmatched = set(customers["customer_id"]) - set(orders["customer_id"])
    assert len(customers) == 40 and len(unmatched) == 8
    return customers, tables, unmatched


def _second_round(reason=None, choice=None):
    def answer(payload):
        if payload["round"] == 1:
            return {"decisions": [{"index": o["index"], "action": "approve"} for o in payload["proposed_operations"]]}
        if choice == "approve":
            return {"decisions": [{"index": o["index"], "action": "approve"} for o in payload["proposed_operations"]]}
        return {"decisions": [{"index": o["index"], "action": "reject", **({"reason": reason} if reason else {})}
                              for o in payload["proposed_operations"]]}
    return answer


@pytest.mark.parametrize("reason", ["the customers without orders matter", None], ids=["reason", "no_reason"])
def test_round_two_rejected(reason, fake_model, tmp_path, store):
    customers, tables, unmatched = _setup(tmp_path, store)
    values, pauses, _ = _drive(_app(store), _state(tables), f"r-{reason}", wrangling_approval=_second_round(reason))
    first, second = _wranglings(pauses)
    (op,) = second["proposed_operations"]
    issue = next(i for i in values["decision_log"] if i["decision_point"] == "stage1.clean.fix"
                 and i["evidence"]["issue_type"] == "join_missing")
    assert issue["evidence"]["issue_evidence"]["right_table"] == "orders"
    assert (op["candidate_id"], op["recommendation"]) == ("delete_unmatched_rows", "apply")
    assert [c["id"] for c in op["candidates"]] == ["leave_as_is", "delete_unmatched_rows", "fill_empty_sums_with_zero"]
    assert op["impact_preview"]["rows_changed"] == len(unmatched) and all(c["preview"] for c in op["candidates"])
    joined = _store_frame(store, values["dataset"])
    assert len(joined) == 40 and joined.loc[joined["rows_in_orders"] == 0, "amount_sum"].isna().all()
    given = f": {reason}." if reason else " without giving a reason."
    assert _sections(values)["Left as is"] == ("orders: the columns the join added stay missing for the rows that have "
                                               f"no rows in orders. You rejected the fix{given}")
    assert [u.issue_type for u in values["plan"].unaddressed_issues] == ["outcome_aggregated"]  # it survives


def test_the_round_two_issue(fake_model, tmp_path, store):
    _, tables, _ = _setup(tmp_path, store)
    values, pauses, _ = _drive(_app(store), _state(tables), "issue", wrangling_approval=_second_round("x"))
    fix = [r for r in _records(values, "stage1.clean.fix") if r["evidence"]["issue_type"] == "join_missing"]
    (fix,) = fix
    scan = [r for r in _records(values, "stage1.clean.scan")][1]
    assert scan["evidence"]["round"] == 2 and scan["evidence"]["join_missing_tables"] == [
        {"table": "orders", "columns": ["amount_mean", "amount_sum", "quantity_mean", "quantity_sum"]}]
    assert DESCRIPTION in scan["rationale"]


def test_delete_unmatched_rows(fake_model, tmp_path, store):
    customers, tables, unmatched = _setup(tmp_path, store)
    values, _, _ = _drive(_app(store), _state(tables), "delete")  # every decision approve
    joined = _store_frame(store, values["dataset"])
    assert len(joined) == 32 and not set(joined["customer_id"]) & unmatched
    line = _sections(values)["Changed"].split("\n")[1]
    assert line.startswith("2. (round 2) Remove the rows that have no rows in orders (8 removed). Why (the model's "
                           "reason): fix join_missing.") and line.endswith("Effect: 40 to 32 rows.")
    assert [u.issue_type for u in values["plan"].unaddressed_issues] == ["outcome_aggregated"]


def test_fill_empty_sums_with_zero(fake_model, tmp_path, store):
    customers, tables, unmatched = _setup(tmp_path, store)
    pick = choose({"join_missing": "fill_empty_sums_with_zero"})
    values, pauses, _ = _drive(_app(store, pick), _state(tables), "fill")
    joined = _store_frame(store, values["dataset"])
    no_orders = joined["customer_id"].isin(unmatched)
    assert len(joined) == 40 and (joined.loc[no_orders, ["amount_sum", "quantity_sum", "rows_in_orders"]] == 0).all().all()
    assert joined.loc[no_orders, ["amount_mean", "quantity_mean"]].isna().all().all()
    assert joined["amount_mean"].isna().mean() >= 0.2  # the 8 of 40 customers (20%) at least
    # the matched customers' summaries are unchanged: recomputed from the orders with pandas
    orders = _store_frame(store, tables["orders"])
    expected = orders.groupby("customer_id")["quantity"].sum()
    matched = joined.loc[~no_orders].set_index("customer_id")["quantity_sum"]
    assert matched.sort_index().astype(float).equals(expected.loc[matched.index].sort_index().astype(float))
    line = _sections(values)["Changed"].split("\n")[1]
    assert line.startswith("2. (round 2) Fill amount_sum, quantity_sum with zero for the rows that have no rows in "
                           "orders (16 value(s) filled).")
    assert line.endswith("Effect: 8 row(s) had their sums filled with 0 (16 value(s)).")
    # the structural rule: no third round proposes the mean columns
    assert [p["round"] for p in _wranglings(pauses)] == [1, 2]
    assert [u.issue_type for u in values["plan"].unaddressed_issues] == ["outcome_aggregated"]


def _join_missing_ops(payload):
    return [o for o in payload["proposed_operations"] if o["candidate_id"] in ("delete_unmatched_rows",
                                                                             "fill_empty_sums_with_zero")]


def test_a_rejected_join_missing_is_not_proposed_again(fake_model, tmp_path, store, monkeypatch):
    from tests.test_aggregate_run import reject_join_missing
    from tests.test_stage1_loop import _with_synthetic
    import core.nodes as nodes_module
    real_detect = nodes_module.detect_all_issues
    _with_synthetic(monkeypatch)  # a new issue every round, so the loop runs to its limit of 3
    _, tables, _ = _setup(tmp_path, store)
    app = _app(store)
    values, pauses, _ = _drive(app, _state(tables), "again", wrangling_approval=reject_join_missing)
    rounds = _wranglings(pauses)
    assert [p["round"] for p in rounds] == [1, 2, 3]
    assert [len(_join_missing_ops(p)) for p in rounds] == [0, 1, 0]  # rejected in round 2, not proposed in round 3
    # the next question on the same thread: the join belongs to the previous question, so its missing values come
    # back as plain high_missingness issues (join records are per question), each proposed again
    monkeypatch.setattr(nodes_module, "detect_all_issues", real_detect)  # the real detector (the fake chat stays)
    second, pauses, _ = _drive(app, {"request_text": "q", "session_id": "s-join", "dataset": values["dataset"]},
                               "again", wrangling_approval=reject_join_missing)
    (first,) = _wranglings(pauses)
    fixes = [r["evidence"] for r in _records(second, "stage1.clean.fix")[-len(first["proposed_operations"]):]]
    assert sorted(f["affected_columns"][0] for f in fixes if f["issue_type"] == "high_missingness") == [
        "amount_mean", "amount_sum", "quantity_mean", "quantity_sum"]


def test_two_aggregate_joins_raise_two_join_missing_issues(fake_model, tmp_path, store):
    customers = _customers()
    rng = np.random.default_rng(81)
    visits = pd.DataFrame({"customer_id": rng.integers(1, 31, 90), "minutes": rng.normal(5, 1, 90)})
    no_visits = set(customers["customer_id"]) - set(visits["customer_id"])
    tables = _bundle(tmp_path, store, {"customers": customers, "orders": _orders(customers), "visits": visits})
    pick = choose({"join_missing": "fill_empty_sums_with_zero"})
    values, pauses, _ = _drive(_app(store, pick), _state(tables), "two")
    rounds = _wranglings(pauses)
    assert [p["round"] for p in rounds] == [1, 2]  # handled once each: no third round
    assert [o["candidate_id"] for o in rounds[0]["proposed_operations"]] == ["aggregate_left_summary"] * 2
    ops = _join_missing_ops(rounds[1])
    assert [o["params"]["right_table"] for o in ops] == ["orders", "visits"]
    keys = values["stage1_seen_keys"]
    assert ["join_missing", [], "orders"] in keys and ["join_missing", [], "visits"] in keys
    assert sum(k[0] == "join_missing" for k in keys) == 2
    joined = _store_frame(store, values["dataset"])
    assert (joined.loc[joined["customer_id"].isin(no_visits), "minutes_sum"] == 0).all() and len(joined) == 40
