"""
M2.3b step 2: aggregate-then-join through the real graph (guard fakes, fake chat model). A carried table with several
rows per key is summarized per key by code and joined; the derived table becomes the analysis table, the originals stay,
the mirror holds after every node and the lineage is sound; an outcome made by the summary is a plain caveat; a declined
aggregate is not a caveat; the carried table's duplicates and missing shares are stated, never sent to a model.
"""
import json
import re

import numpy as np
import pandas as pd
import pytest

from core.dataset import LocalDiskStore
from core.decision_points import DECISION_NODES, registered_points
from core.graph import build_agent_graph
from core.plan import QuestionType
from tests.test_decision_point_guard import choose, confident, fake_draft, reject_joins
from tests.test_join_run import (
    ALTERNATIVE, QUESTION, _bundle, _drive_checking_mirror, _lineage_problems, _mirror, _records, _sections, _state,
    _store_frame, _wrangling,
)
from tests.test_prompt_canaries import fake_model  # noqa: F401 (fixture)
from tests.test_stage1_report import _drive

CAVEAT_REASON = "the column was made by summarizing a table in Stage 1, before the outcome was chosen"


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _app(store, propose_fn=None, outcome="amount_mean", predictors=("region",)):
    return build_agent_graph(store, classify_fn=confident(QuestionType.TWO_GROUP_COMPARISON),
                             draft_fn=fake_draft(list(predictors), outcome), propose_fn=propose_fn or choose({}))


def _customers(n=40, seed=51):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"customer_id": range(1, n + 1), "region": rng.choice(["East", "West"], n),
                         "age": rng.normal(40, 8, n).round(1), "tier": rng.choice(["basic", "plus"], n)})


def _orders(customers, n=120, seed=52, clean=False):
    rng = np.random.default_rng(seed)
    ids = rng.integers(1, 33, n)  # customers 33 to 40 have no orders
    west = customers.set_index("customer_id").loc[ids, "region"].to_numpy() == "West"
    df = pd.DataFrame({"customer_id": ids, "amount": 20 + np.where(west, 5.0, 0.0) + rng.normal(0, 2, n),
                       "quantity": rng.integers(1, 6, n), "channel": rng.choice(["web", "store"], n)})
    if clean:
        return df
    df.loc[df.index % 9 == 0, "amount"] = np.nan
    return pd.concat([df, df.iloc[[1, 2]]], ignore_index=True)  # 2 exact duplicate rows


def _counts(store, tables):
    """The numbers the preview and the report must state, computed independently from the stored frames."""
    customers, orders = _store_frame(store, tables["customers"]), _store_frame(store, tables["orders"])
    unmatched = int((~customers["customer_id"].isin(orders["customer_id"].dropna())).sum())
    return {"unmatched": unmatched, "duplicates": int(orders.duplicated().sum()),
            "share": float(orders["amount"].isna().mean()), "rows": len(customers)}


def _warnings(c):
    return [f"{c['unmatched']} row(s) have no match in orders",
            f"{c['duplicates']} exact duplicate row(s) in orders are counted again in the summary",
            f"amount is {c['share']:.1%} missing in orders: its mean and sum use only the rows that have a value"]


ADDED = ["rows_in_orders", "amount_mean", "amount_sum", "quantity_mean", "quantity_sum"]

JOIN_MISSING_IDS = ("delete_unmatched_rows", "fill_empty_sums_with_zero")
JOIN_MISSING_REASON = "the unmatched customers stay as they are"


def _is_join_missing_fix(op) -> bool:
    """M2.4a: an operation proposed for a join_missing issue (by its candidate id, or its op_type and params)."""
    return (op.get("candidate_id") in JOIN_MISSING_IDS or op["op_type"] == "fill_unmatched"
            or (op["op_type"] == "filter" and "unmatched_of" in (op.get("params") or {})))


def reject_join_missing(payload):
    """M2.4a: the wrangling answers that reject every fix of an aggregate join's unmatched rows (round 2) and approve
    the rest, so a test keeps the aggregate join's own result; a join-only run is answered as before (approve)."""
    return {"decisions": [{"index": op["index"], "action": "reject", "reason": JOIN_MISSING_REASON}
                          if _is_join_missing_fix(op) else {"index": op["index"], "action": "approve"}
                          for op in payload["proposed_operations"]]}


# --- (a) an approved aggregate-then-join -------------------------------------------------------------------------

def test_an_approved_aggregate_join(fake_model, tmp_path, store):
    customers = _customers()
    tables = _bundle(tmp_path, store, {"customers": customers, "orders": _orders(customers)})
    c = _counts(store, tables)
    assert (c["unmatched"], c["duplicates"]) == (8, 3)  # the 2 planted repeats and one chance repeat
    _, pauses, _ = _drive(_app(store), _state(tables), "a-pauses")
    (op,) = _wrangling(pauses)["proposed_operations"]
    assert (op["candidate_id"], op["op_type"], op["recommendation"]) == ("aggregate_left_summary", "join", "apply")
    assert op["what_was_run"] == ("Summarize orders per customer_id (a row count, and the mean and sum of amount and "
                                  "quantity), then join the summary to the analysis table, keeping every row. Adds 5 "
                                  f"column(s): {', '.join(ADDED)}.")
    assert op["impact_preview"]["warnings"] == _warnings(c)
    assert [x["id"] for x in op["candidates"]] == ["leave_as_is", "aggregate_left_summary", "aggregate_left_count",
                                                   "aggregate_left_normalized"]

    # M2.4a: round 2 proposes a fix of the 8 unmatched customers; rejecting it keeps the aggregate join's own result
    values, checked = _drive_checking_mirror(_app(store), _state(tables), "a", wrangling_approval=reject_join_missing)
    assert {"profile", "wrangle_review", "rediagnose"} <= set(checked)
    assert values["analysis_table"] == "customers_orders" and set(values["tables"]) == {"customers", "orders",
                                                                                        "customers_orders"}
    assert values["tables"]["customers"] == tables["customers"] and values["tables"]["orders"] == tables["orders"]
    assert _store_frame(store, values["tables"]["orders"]).equals(_store_frame(store, tables["orders"]))
    joined = _store_frame(store, values["dataset"])
    assert list(joined.columns) == [*customers.columns, *ADDED] and len(joined) == c["rows"]
    _mirror(values)
    assert list(values["plan"].data_sources) == [values["dataset"].handle] == ["customers_orders@v1"]
    assert values["dataset"].parents == ("customers@v1", "orders@v1") and _lineage_problems(values) == []
    assert (values["route"]["capability"], values["route"]["status"]) == ("two_group", "ok")
    json.dumps(values["decision_log"], allow_nan=False)


# --- (b) the outcome_aggregated caveat ---------------------------------------------------------------------------

def _caveat(outcome):
    return (f"The outcome '{outcome}' is a summary of the table 'orders' per customer_id, so each row stands for one "
            "value of customer_id, not for a row of orders.")


@pytest.mark.parametrize("outcome", ["amount_mean", "rows_in_orders"])
def test_the_outcome_aggregated_caveat(outcome, fake_model, tmp_path, store):
    prompts, _ = fake_model
    customers = _customers()
    tables = _bundle(tmp_path, store, {"customers": customers, "orders": _orders(customers)})
    c = _counts(store, tables)
    values, pauses, _ = _drive(_app(store, outcome=outcome), _state(tables), f"b-{outcome}")
    caveat = _caveat(outcome)
    assert not re.search(r"\d", caveat)
    (issue,) = values["plan"].unaddressed_issues
    assert (issue.issue_type, issue.description, list(issue.affected_columns), issue.rejection_reason) == (
        "outcome_aggregated", caveat, [outcome], CAVEAT_REASON)
    approval = next(p["payload"] for p in pauses if p["type"] == "plan_approval")
    assert approval["unaddressed_issues"] == [{"issue_type": "outcome_aggregated", "description": caveat,
                                               "affected_columns": [outcome], "reason": CAVEAT_REASON}]
    report_prompts = [p["text"] for p in prompts if p["stage"] == "report"]
    assert report_prompts and all(caveat in t for t in report_prompts)
    for text in report_prompts:
        for number in (f"{c['share']:.1%}", "counted again", "missing in orders", "rows_summarized"):
            assert number not in text, number
    summary = next(e for e in values["stage_reports"] if e["stage"] == "summary")
    lines = next(s["text"] for s in summary["sections"] if s["heading"] == "Caveats").split("\n")
    assert caveat in lines and not any("left unaddressed because" in line for line in lines)


def test_no_caveat_for_an_outcome_of_the_analysis_table(fake_model, tmp_path, store):
    customers = _customers()
    tables = _bundle(tmp_path, store, {"customers": customers, "orders": _orders(customers)})
    values, _, _ = _drive(_app(store, outcome="age"), _state(tables), "b-age")
    assert values["analysis_table"] == "customers_orders" and list(values["plan"].unaddressed_issues) == []


def test_the_exploratory_round_keeps_the_caveat(fake_model, tmp_path, store):
    prompts, _ = fake_model
    customers = _customers()
    tables = _bundle(tmp_path, store, {"customers": customers, "orders": _orders(customers)})
    values, pauses, _ = _drive(_app(store), _state(tables), "b-explore", stage_gate={"action": "run"})
    assert "stage_gate" in [p["type"] for p in pauses] and len(values["rounds"]) == 2
    second = values["rounds"][1]
    assert second["exploratory"] and [u.description for u in second["plan"].unaddressed_issues
                                      if u.issue_type == "outcome_aggregated"] == [_caveat("amount_mean")]
    report_prompts = [p["text"] for p in prompts if p["stage"] == "report"]
    assert len(report_prompts) == 2 and _caveat("amount_mean") in report_prompts[1]


# --- (c) a rejected aggregate, and the model's recommendation ----------------------------------------------------

def _reject(reason):
    def answer(payload):
        decision = {"index": payload["proposed_operations"][0]["index"], "action": "reject"}
        return {"decisions": [{**decision, **({"reason": reason} if reason is not None else {})}]}
    return answer


@pytest.mark.parametrize("recommendation, reason, line", [
    ("apply", "the question does not need this table",
     "orders was not summarized and joined. The model recommended summarizing and joining: fix aggregatable_table. "
     "You rejected it: the question does not need this table."),
    ("skip", "the question does not need this table",
     "orders was not summarized and joined. The model recommended against joining: fix aggregatable_table. You "
     "rejected it: the question does not need this table."),
    ("apply", None,
     "orders was not summarized and joined. The model recommended summarizing and joining: fix aggregatable_table. "
     "You rejected it without giving a reason."),
], ids=["reason", "skip", "no_reason"])
def test_a_rejected_aggregate(recommendation, reason, line, fake_model, tmp_path, store):
    customers = _customers()
    tables = _bundle(tmp_path, store, {"customers": customers, "orders": _orders(customers)})
    values, _, _ = _drive(_app(store, choose({}, recommendation), outcome="age"), _state(tables),
                          f"c-{recommendation}-{reason}", wrangling_approval=_reject(reason))
    assert values["tables"] == tables and values["analysis_table"] == "customers"
    assert list(values["plan"].unaddressed_issues) == []
    (approval,) = _records(values, "stage1.clean.approval")
    assert approval["evidence"]["carried_forward_as_unaddressed"] is False
    assert approval["rationale"] == (reason or "no reason given")
    assert _sections(values)["Left as is"] == line
    summary = next(e for e in values["stage_reports"] if e["stage"] == "summary")
    caveats = next(s["text"] for s in summary["sections"] if s["heading"] == "Caveats")
    assert "orders" not in caveats and "aggregat" not in caveats


def test_reject_joins_also_rejects_an_aggregate(fake_model, tmp_path, store):
    customers = _customers()
    tables = _bundle(tmp_path, store, {"customers": customers, "orders": _orders(customers)})
    values, _, _ = _drive(_app(store, outcome="age"), _state(tables), "c-helper", wrangling_approval=reject_joins)
    assert [a["chosen"] for a in _records(values, "stage1.clean.approval")] == ["reject"]


def test_skip_recommended_and_approved(fake_model, tmp_path, store):
    customers = _customers()
    tables = _bundle(tmp_path, store, {"customers": customers, "orders": _orders(customers)})
    values, _, _ = _drive(_app(store, choose({}, "skip")), _state(tables), "c-skip-approved")
    assert ("Why (the model's reason): fix aggregatable_table. The model recommended not joining this table. "
            + ALTERNATIVE) in _sections(values)["Changed"]
    assert values["analysis_table"] == "customers_orders"


# --- (d) the Stage 1 report --------------------------------------------------------------------------------------

def test_the_stage1_report(fake_model, tmp_path, store):
    customers = _customers()
    tables = _bundle(tmp_path, store, {"customers": customers, "orders": _orders(customers)})
    c = _counts(store, tables)
    values, _, _ = _drive(_app(store), _state(tables), "d", wrangling_approval=reject_join_missing)  # M2.4a: round 2
    sections = _sections(values)
    assert sections["Checked"].split("\n") == [
        "Checked 40 rows and 4 columns.",
        "orders: has several rows for each customer_id; it can be summarized and joined to the analysis table."]
    assert sections["Tables"].split("\n") == [
        "customers: 40 rows, 4 columns (main table; joined with orders into customers_orders)",
        "orders: 122 rows, 4 columns (summarized and joined into customers_orders)",
        "customers_orders: 40 rows, 9 columns (main table after the join)"]
    notes = "; ".join(_warnings(c)[1:])  # the unmatched-rows warning is not repeated in Notes
    assert sections["Changed"] == (
        "1. Summarize orders per customer_id (a row count, and the mean and sum of amount and quantity), then join the "
        f"summary to the analysis table, keeping every row. Adds 5 column(s): {', '.join(ADDED)}. Why (the model's "
        f"reason): fix aggregatable_table. {ALTERNATIVE} Your decision: approved. Effect: 40 rows stay 40; 5 column(s) "
        f"added; {4 * c['unmatched']} new missing value(s) from {c['unmatched']} row(s) with no match. Notes: {notes}."
        # M2.4a: the rejected round-2 fix of the unmatched rows, numbered across rounds and tagged
        f"\n2. (round 2) Remove the rows that have no rows in orders ({c['unmatched']} removed). Why (the model's "
        f"reason): fix join_missing. {ALTERNATIVE} Your decision: rejected: {JOIN_MISSING_REASON}.")
    assert sections["Rounds"].split("\n") == [  # M2.4a: the second round, and how Stage 1 ended
        "Round 2 found: orders: the columns the join added are missing for the rows that have no rows in orders.",
        "Stage 1 ended after 2 round(s): you declined every proposal of the last round."]
    assert sections["Left as is"] == ("orders: the columns the join added stay missing for the rows that have no rows "
                                      f"in orders. You rejected the fix: {JOIN_MISSING_REASON}.")  # M2.4a
    assert list(sections) == ["Checked", "Tables", "Data quality", "Links between tables", "Rounds", "Changed",
                              "Left as is", "Described"]  # M2.4a: Rounds before Changed
    assert notes == ("3 exact duplicate row(s) in orders are counted again in the summary; amount is "
                     f"{c['share']:.1%} missing in orders: its mean and sum use only the rows that have a value")


def test_no_notes_for_a_clean_carried_table(fake_model, tmp_path, store):
    customers = _customers()
    tables = _bundle(tmp_path, store, {"customers": customers, "orders": _orders(customers, clean=True)})
    values, _, _ = _drive(_app(store), _state(tables), "d-clean", wrangling_approval=reject_join_missing)  # M2.4a
    changed, rejected = _sections(values)["Changed"].split("\n")  # M2.4a: line 2 is the rejected round-2 fix
    assert "Notes:" not in changed and changed.endswith("new missing value(s) from 8 row(s) with no match.")
    assert rejected.startswith("2. (round 2) Remove the rows that have no rows in orders (8 removed).")


# --- (e) the execution order -------------------------------------------------------------------------------------

def test_the_execution_order_and_every_preview(fake_model, tmp_path, store):
    customers = _customers()
    customers["tier"] = np.where(customers.index % 4 == 0, "Basic ", customers["tier"])  # mergeable text
    customers.loc[customers.index % 8 == 0, "age"] = np.nan  # 12.5% missing
    messy = pd.concat([customers, customers.iloc[[3, 5]]], ignore_index=True)  # 2 exact duplicates
    tables = _bundle(tmp_path, store, {"customers": messy, "orders": _orders(customers)})
    pick = {"inconsistent_text": "trim_and_lowercase", "duplicate_rows": "dedupe_all", "high_missingness": "impute_median"}
    values, pauses, _ = _drive(_app(store, choose(pick)), _state(tables), "e",
                               wrangling_approval=reject_join_missing)  # M2.4a: round 2 is rejected
    ops = _wrangling(pauses)["proposed_operations"]
    executions = _records(values, "stage1.clean.execution")
    assert [o["op_type"] for o in ops] == [e["evidence"]["op_type"] for e in executions] == [
        "normalize_text", "dedupe", "join", "impute"]
    for op, execution in zip(ops, executions):
        preview, ev = op["impact_preview"], execution["evidence"]
        assert (preview["rows_before"], preview["rows_after"], preview["columns_before"], preview["columns_after"]) == (
            ev["rows_before"], ev["rows_after"], ev["columns_before"], ev["columns_after"]), op["op_type"]
        details = ev.get("details") or {}
        if op["op_type"] == "join":
            assert preview["cells_changed"] == details["new_missing_cells"] and preview["warnings"] == ev["warnings"]
            assert ev["params"]["aggregate"]["columns"] == ["amount", "quantity"]
        elif op["op_type"] == "impute":
            assert preview["cells_changed"] == details["cells_filled"]
        elif op["op_type"] == "normalize_text":
            assert preview["cells_changed"] == details["cells_changed"]
    assert values["analysis_table"] == "customers_orders" and _lineage_problems(values) == []
    _mirror(values)


# --- (f) stakeholder mode ----------------------------------------------------------------------------------------

def test_stakeholder_mode_always_interrupts(fake_model, tmp_path, store):
    customers = _customers()
    messy = pd.concat([customers, customers.iloc[[0]]], ignore_index=True)
    tables = _bundle(tmp_path, store, {"customers": messy, "orders": _orders(customers)})
    values, pauses, _ = _drive(_app(store), _state(tables, autonomy_mode="stakeholder"), "f",
                               wrangling_approval=reject_join_missing)  # M2.4a: round 2 is rejected
    assert [o["candidate_id"] for o in _wrangling(pauses)["proposed_operations"]] == ["aggregate_left_summary"]
    # M2.4a: the third approval is the rejected round-2 fix of the unmatched rows (it interrupted too)
    assert [a["chosen"] for a in _records(values, "stage1.clean.approval")] == ["auto_approve", "approve", "reject"]


# --- (g) privacy -------------------------------------------------------------------------------------------------

def test_privacy(fake_model, tmp_path, store):
    prompts, config = fake_model
    config.update(question_type=QuestionType.TWO_GROUP_COMPARISON, predictors=["region"])
    rng = np.random.default_rng(61)
    main = pd.DataFrame({"customer_id": range(1, 31), "region": rng.choice(["East", "West"], 30),
                         "total_spend": rng.normal(50, 5, 30)})
    events = pd.DataFrame({"customer_id": rng.integers(1, 26, 90), "zz_metric": rng.normal(10, 1, 90),
                           "zz_note": rng.choice(["NOTE-CANARY-A", "NOTE-CANARY-B"], 90)})
    events.loc[0, "zz_metric"] = 27182.818
    events.loc[events.index % 10 == 5, "zz_metric"] = np.nan
    events = pd.concat([events, events.iloc[[7]]], ignore_index=True)
    unlinked = pd.DataFrame({"zz_other": ["UNLINKED-CANARY", "x"], "zz_amount": [31415.926, 1.0]})
    tables = _bundle(tmp_path, store, {"main": main, "zz_events": events, "unlinked_extra": unlinked})
    values, pauses, _ = _drive(build_agent_graph(store), _state(tables), "g", wrangling_approval=reject_joins)
    (op,) = _wrangling(pauses)["proposed_operations"]
    assert op["candidate_id"] == "aggregate_left_summary"
    (proposer,) = [p["text"] for p in prompts if p["stage"] == "_WranglingProposalsOut"]
    others = [p for p in prompts if p["stage"] != "_WranglingProposalsOut"]
    forbidden = ["NOTE-CANARY", "UNLINKED-CANARY", "27182", "31415", "unlinked_extra", "zz_other", "zz_amount",
                 *op["impact_preview"]["warnings"], "counted again", "missing in zz_events",
                 f"{float(events['zz_metric'].mean()):.4g}", f"{float(events['zz_metric'].sum()):.4g}"]
    for p in prompts:
        for word in forbidden:
            assert word not in p["text"], (p["stage"], word)
    for p in others:
        assert "zz_events" not in p["text"] and "zz_metric" not in p["text"] and "zz_note" not in p["text"], p["stage"]
    lines = proposer.splitlines()
    (issue_line,) = [line for line in lines if re.match(r"^\[\d+\] aggregatable_table: ", line)]
    assert [line for line in lines if "zz_metric" in line or "zz_note" in line] == [issue_line]
    assert [line for line in lines if "zz_events" in line] == [
        issue_line,
        "      aggregate_left_summary: Summarize zz_events per customer_id (a row count, and the mean and sum of its "
        "numeric columns), then join the summary to the analysis table, keeping every row.",
        "      aggregate_left_count: Summarize zz_events per customer_id (a row count), then join the summary to the "
        "analysis table, keeping every row."]
    assert ("For a joinable_table or aggregatable_table issue, say in 'recommendation' whether the question needs the "
            "other table ('apply') or not ('skip'), and in 'rationale' why; you must still choose the best join option "
            "so the user can decide.") in proposer


# --- (h) an edit -------------------------------------------------------------------------------------------------

def _edit(change):
    def answer(payload):
        op = payload["proposed_operations"][0]
        return {"decisions": [{"index": op["index"], "action": "edit", "params": change(dict(op["params"]))}]}
    return answer


def test_an_edit_and_a_bad_edit(fake_model, tmp_path, store):
    customers = _customers()
    tables = _bundle(tmp_path, store, {"customers": customers, "orders": _orders(customers)})
    bad = _edit(lambda p: {**p, "aggregate": {"count": True, "columns": ["channel"], "stats": ["mean"]}})
    values, _, _ = _drive(_app(store, outcome="age"), _state(tables), "h-bad", wrangling_approval=bad)
    (execution,) = _records(values, "stage1.clean.execution")
    assert (execution["chosen"], execution["evidence"]["failure_reason"]) == ("failed", "Cannot summarize — channel is not numeric.")
    assert values["analysis_table"] == "customers" and values["plan"].status.value == "executed"
    c = _counts(store, tables)
    values, _, _ = _drive(_app(store), _state(tables), "h-inner", wrangling_approval=_edit(lambda p: {**p, "how": "inner"}))
    (execution,) = _records(values, "stage1.clean.execution")
    assert (execution["evidence"]["rows_before"], execution["evidence"]["rows_after"]) == (40, 40 - c["unmatched"])


# --- (i) both kinds of link in one carried table -----------------------------------------------------------------

def test_one_table_with_both_kinds_of_link(fake_model, tmp_path, store):
    customers = _customers().assign(account=lambda d: [f"acc{chr(97 + i)}" for i in range(len(d))])
    both = pd.DataFrame({"customer_id": range(1, 41), "account": ["acca", "accb"] * 20, "score": range(40)})
    tables = _bundle(tmp_path, store, {"customers": customers, "both": both})
    values, pauses, _ = _drive(_app(store, outcome="age"), _state(tables), "i")
    (op,) = _wrangling(pauses)["proposed_operations"]
    assert (op["candidate_id"], op["params"]["left_key"]) == ("join_left", "customer_id")  # the unique key wins the tie
    (fix,) = _records(values, "stage1.clean.fix")
    assert fix["evidence"]["issue_type"] == "joinable_table"


# --- (j) missing carried keys and a repeating left key -----------------------------------------------------------

def test_missing_keys_and_a_repeating_left_key(fake_model, tmp_path, store):
    rng = np.random.default_rng(71)
    # text keys: a numeric key with missing values is read as float, and the link scan skips float columns (M2.2)
    orders = pd.DataFrame({"customer_id": [f"c{k:02d}" for k in rng.integers(1, 21, 80)],
                           "region": rng.choice(["East", "West"], 80),
                           "amount": rng.normal(20, 2, 80)})
    # a key-like carried key that still repeats (19 distinct of 20 keyed rows), plus 4 rows with a missing key
    keys = [f"c{k:02d}" for k in range(1, 20)] + ["c01"] + [None] * 4
    visits = pd.DataFrame({"customer_id": keys, "minutes": rng.normal(5, 1, len(keys))})
    tables = _bundle(tmp_path, store, {"orders": orders, "visits": visits})
    values, _, _ = _drive(_app(store, outcome="amount"), _state(tables), "j",
                          wrangling_approval=reject_join_missing)  # M2.4a: round 2 (c20 has no visits) is rejected
    (execution,) = _records(values, "stage1.clean.execution")
    assert execution["evidence"]["details"]["aggregate"]["rows_summarized"] == len(visits)
    assert len(_store_frame(store, values["dataset"])) == len(orders) == execution["evidence"]["rows_after"]
    counted = _store_frame(store, values["dataset"]).drop_duplicates("customer_id")["rows_in_visits"].sum()
    assert counted == int(visits["customer_id"].notna().sum())  # rows with a missing key are not counted


# --- (k) the registry --------------------------------------------------------------------------------------------

def test_the_registry_is_unchanged():
    assert (len(registered_points()), len(DECISION_NODES)) == (32, 18)  # M2.4a: replan (planner.stage_replan)
