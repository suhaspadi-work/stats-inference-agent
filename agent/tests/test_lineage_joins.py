"""
M2.4b (G83): the applied aggregate joins are found from the analysis table's LINEAGE in the whole log (and the same
session), so on a follow-up question the structural missing values of an earlier aggregate join are still a join_missing
issue, not plain high_missingness issues. G86 and G88 are documented by tests named for them.
"""
import pytest

from core.dataset import Dataset, LocalDiskStore
from core.nodes import _lineage_aggregate_joins, _lineage_names
from tests.test_aggregate_run import _app, _customers, _is_join_missing_fix, _orders
from tests.test_decision_point_guard import choose
from tests.test_join_run import _bundle, _customers as _join_customers, _orders as _join_orders, _state, _store_frame
from tests.test_prompt_canaries import fake_model  # noqa: F401 (fixture)
from tests.test_stage1_loop import _run


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _ds(name, parents=(), session="s"):
    if parents:
        return Dataset.derived(session, name, f"{session}/{name}_v1.csv", parents=tuple(parents), operation_id="op")
    return Dataset(session_id=session, name=name, version=1, storage_key=f"{session}/{name}_v1.csv",
                   parent_version=None, created_from_operation=None)


# --- the helpers ------------------------------------------------------------------------------------------------

def test_lineage_names_follow_derived_parents():
    tables = {"orders": _ds("orders"), "customers": _ds("customers"), "regions": _ds("regions"),
              "oc": _ds("oc", ("orders@v1", "customers@v1")), "ocr": _ds("ocr", ("oc@v1", "regions@v1"))}
    assert _lineage_names({"tables": tables, "analysis_table": "ocr", "dataset": tables["ocr"]}) == {"ocr", "oc"}
    assert _lineage_names({"tables": tables, "analysis_table": "orders", "dataset": tables["orders"]}) == set()


def test_a_cycle_terminates_and_unknown_parents_are_ignored():
    tables = {"a": _ds("a", ("b@v1",)), "b": _ds("b", ("a@v1", "ghost@v1"))}
    assert _lineage_names({"tables": tables, "analysis_table": "a", "dataset": tables["a"]}) == {"a", "b"}


def _execution(result_name, session="s", aggregate=True, chosen="applied", **extra):
    params = {"right_table": "orders", "result_name": result_name, **({"aggregate": {"count": True}} if aggregate else {})}
    evidence = {"op_type": "join", "params": params, "details": {}, **extra}
    if session is not None:
        evidence["session_id"] = session
    return {"decision_point": "stage1.clean.execution", "chosen": chosen, "evidence": evidence}


def test_lineage_aggregate_joins_reads_the_whole_log_of_this_session():
    tables = {"customers": _ds("customers"), "orders": _ds("orders"),
              "customers_orders": _ds("customers_orders", ("customers@v1", "orders@v1"))}
    state = {"tables": tables, "analysis_table": "customers_orders", "dataset": tables["customers_orders"]}
    mine = _execution("customers_orders")
    log = [mine,
           {"decision_point": "planner.stage_plan", "evidence": {}},  # a later question: the join is still found
           _execution("customers_orders", session="other"),  # an earlier upload with the same names, another session
           _execution("customers_orders", session=None),  # a record from before M2.4b (no session_id)
           _execution("visits_summary"),  # a table that is not in the lineage
           _execution("customers_orders", aggregate=False),  # a plain join
           _execution("customers_orders", chosen="failed")]
    assert _lineage_aggregate_joins({**state, "decision_log": log}) == [mine["evidence"]]
    assert _lineage_aggregate_joins({**state, "decision_log": log[1:]}) == []  # only the other session's records


# --- end to end ---------------------------------------------------------------------------------------------------

def _decline_join_missing(payload):
    return {"decisions": [{"index": o["index"], "action": "reject", "reason": "keep them"} if _is_join_missing_fix(o)
                          else {"index": o["index"], "action": "approve"} for o in payload["proposed_operations"]]}


def _first_question(tmp_path, store, answer=_decline_join_missing, pick=None):
    customers = _customers()
    tables = _bundle(tmp_path, store, {"customers": customers, "orders": _orders(customers)})
    app = _app(store, pick or choose({}))
    values, _, _ = _run(app, _state(tables), "t", wrangling_approval=answer)
    return app, values


def _follow_up(app, values, answer=_decline_join_missing):
    return _run(app, {"request_text": "q", "session_id": "s-join", "dataset": values["dataset"]}, "t",
                wrangling_approval=answer)


def _question_fixes(values):
    log = values["decision_log"]
    start = max(i for i, e in enumerate(log) if e["decision_point"] == "planner.stage_plan")
    return [e["evidence"] for e in log[start:] if e["decision_point"] == "stage1.clean.fix"]


def test_the_follow_up_question_proposes_join_missing(fake_model, tmp_path, store):
    app, first = _first_question(tmp_path, store)
    second, pauses, _ = _follow_up(app, first)
    fixes = _question_fixes(second)
    assert [f["issue_type"] for f in fixes] == ["join_missing"]  # not four high_missingness issues
    issue = fixes[0]["issue_evidence"]
    joined = _store_frame(store, first["dataset"])
    missing = [c for c in ("amount_mean", "amount_sum", "quantity_mean", "quantity_sum") if joined[c].isna().any()]
    assert issue["columns"] == missing and issue["right_table"] == "orders"
    scans = [e for e in second["decision_log"] if e["decision_point"] == "stage1.clean.scan"][-1:]
    assert (f"The columns {', '.join(missing)} that the join of 'orders' added are missing for the rows that have no "
            "rows in 'orders'.") in scans[0]["rationale"]
    assert second["stage1_seen_keys"] == [["join_missing", [], "orders"]]  # each key once in that question
    assert not any(f["issue_type"] == "high_missingness" for f in fixes)  # the structural suppression holds


def test_after_a_fill_the_follow_up_offers_no_fill_for_the_means(fake_model, tmp_path, store):
    approve_all = lambda p: {"decisions": [{"index": o["index"], "action": "approve"} for o in p["proposed_operations"]]}
    def fill_when_eligible(issue, cands):  # the fill in question 1; in question 2 it is not eligible
        ids = [c["id"] for c in cands if c["eligible"] and c["id"] != "leave_as_is"]
        return "fill_empty_sums_with_zero" if "fill_empty_sums_with_zero" in ids else ids[0]
    app, first = _first_question(tmp_path, store, approve_all, choose(fill_when_eligible))
    second, pauses, _ = _follow_up(app, first)
    fixes = _question_fixes(second)
    assert [f["issue_type"] for f in fixes] == ["join_missing"]  # the means are still missing; nothing for the sums
    assert fixes[0]["issue_evidence"]["columns"] == ["amount_mean", "quantity_mean"]
    assert fixes[0]["issue_evidence"]["sum_columns"] == []
    op = next(p for p in pauses if p["type"] == "wrangling_approval")["proposed_operations"][0]
    fill = next(c for c in op["candidates"] if c["id"] == "fill_empty_sums_with_zero")
    assert (fill["eligible"], fill["reason"]) == (False, "none of the added columns is a sum")


def test_g86_a_plain_joins_missing_values_are_plain_high_missingness(fake_model, tmp_path, store):
    customers = _join_customers()
    orders = _join_orders(customers, extra_ids=list(range(900, 960)))  # 60 orders whose customer is unknown
    tables = _bundle(tmp_path, store, {"orders": orders, "customers": customers})
    from tests.test_join_run import _app as join_app
    app = join_app(store)
    first, _, _ = _run(app, _state(tables), "g86", wrangling_approval=lambda p: {"decisions": [
        {"index": o["index"], "action": "approve" if o["op_type"] == "join" else "reject"}
        for o in p["proposed_operations"]]})
    assert first["analysis_table"] == "orders_customers"
    second, _, _ = _follow_up(app, first, answer=lambda p: {"decisions": [
        {"index": o["index"], "action": "reject"} for o in p["proposed_operations"]]})
    # G86: a plain join leaves no row-count column, so its unmatched rows are plain high_missingness issues
    assert sorted(f["affected_columns"][0] for f in _question_fixes(second)) == ["region", "tier"]
    assert {f["issue_type"] for f in _question_fixes(second)} == {"high_missingness"}


def test_g88_the_outcome_caveat_is_per_question(fake_model, tmp_path, store):
    app, first = _first_question(tmp_path, store)
    assert "outcome_aggregated" in [u.issue_type for u in first["plan"].unaddressed_issues]
    second, _, _ = _follow_up(app, first)
    # G88: the follow-up question's outcome is the same summary column, but the caveat is not carried to it
    assert second["plan"].outcome_name == first["plan"].outcome_name == "amount_mean"
    assert "outcome_aggregated" not in [u.issue_type for u in second["plan"].unaddressed_issues]
