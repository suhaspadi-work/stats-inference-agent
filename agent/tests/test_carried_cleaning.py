"""
M2.4b: a LINKED carried table (one with a join or summary proposal) is cleaned in the Stage 1 loop before it enters
the analysis: the four integrity detectors run on it, its operations carry its "table" and make new versions of that
table, and its join or summary is withheld until its cleaning has been proposed. An unlinked carried table is never read
for cleaning and appears in no prompt or review payload. Expected values are computed independently with pandas.
"""
import dataclasses
import json
import re
import time

import numpy as np
import pandas as pd
import pytest

import core.nodes as nodes_module
from core import thresholds
from core.dataset import LocalDiskStore
from core.decision_points import DECISION_NODES, registered_points
from core.graph import build_agent_graph
from core.ingestion import profile_dataset
from core.plan import QuestionType
from tests.test_aggregate_run import (
    _app, _customers, _join_round, _orders, reject_carried, reject_carried_and_join_missing, reject_join_missing,
)
from tests.test_decision_point_guard import choose, confident, fake_draft
from tests.test_join_run import _bundle, _drive_checking_mirror, _lineage_problems, _state, _store_frame
from tests.test_operation_candidates import CountingStore
from tests.test_prompt_canaries import fake_model  # noqa: F401 (fixture)
from tests.test_stage1_loop import _run

TABLE_KEYS = {"rows_before", "rows_after", "columns_before", "columns_after"}


def _by_value(x):
    """A profile compared by value: a checkpoint round trip turns its tuples (top_values) into lists."""
    return json.loads(json.dumps(dataclasses.asdict(x), default=str))


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _wranglings(pauses):
    return [p for p in pauses if p["type"] == "wrangling_approval"]


def _records(values, point):
    return [e for e in values["decision_log"] if e["decision_point"] == point]


def _setup(tmp_path, store, customers=None, orders=None):
    customers = _customers() if customers is None else customers
    orders = _orders(_customers()) if orders is None else orders
    return customers, _bundle(tmp_path, store, {"customers": customers, "orders": orders})


def _cleaned(store, tables):
    """The orders after round 1 when every decision is approve: duplicates removed, then the rows missing amount."""
    return _store_frame(store, tables["orders"]).drop_duplicates().dropna(subset=["amount"])


def _reject_all(payload):
    return {"decisions": [{"index": o["index"], "action": "reject"} for o in payload["proposed_operations"]]}


# --- (a) the dependency rule ---------------------------------------------------------------------------------------

def test_the_join_waits_for_the_cleaning(fake_model, tmp_path, store):
    customers, tables = _setup(tmp_path, store)
    values, pauses, _ = _run(_app(store), _state(tables), "a", wrangling_approval=reject_join_missing)
    first, second = _wranglings(pauses)[:2]
    assert {o["table"] for o in first["proposed_operations"]} == {"orders"}
    assert [o["op_type"] for o in first["proposed_operations"]] == ["dedupe", "filter"]  # no join in round 1
    (join,) = second["proposed_operations"]
    assert (join["op_type"], join["table"], second["round"]) == ("join", None, 2)
    cleaned = _cleaned(store, tables)
    unmatched = len(set(customers["customer_id"]) - set(cleaned["customer_id"]))
    assert join["impact_preview"]["warnings"] == [f"{unmatched} row(s) have no match in orders"]  # nothing else
    (execution,) = [e for e in _records(values, "stage1.clean.execution") if e["evidence"]["op_type"] == "join"]
    assert execution["evidence"]["details"]["aggregate"]["rows_summarized"] == len(cleaned)
    joined = _store_frame(store, values["dataset"]).set_index("customer_id")
    summary = cleaned.groupby("customer_id").agg(rows=("amount", "size"), amount_mean=("amount", "mean"),
                                                  amount_sum=("amount", "sum"), quantity_sum=("quantity", "sum"))
    matched = joined.loc[summary.index]
    assert matched["rows_in_orders"].tolist() == summary["rows"].tolist()
    assert np.allclose(matched["amount_mean"], summary["amount_mean"])
    assert np.allclose(matched["amount_sum"], summary["amount_sum"])
    assert np.allclose(matched["quantity_sum"], summary["quantity_sum"])
    orders = values["tables"]["orders"]
    assert orders.version == 3 and _by_value(values["profiles"]["orders"]) == _by_value(profile_dataset(orders, store=store))
    assert execution["evidence"]["input_handles"] == ["customers@v1", "orders@v3"]
    assert values["tables"]["customers"] == tables["customers"] and _lineage_problems(values) == []
    json.dumps(values["decision_log"], allow_nan=False)


def test_the_mirror_holds_after_every_node(fake_model, tmp_path, store):
    _, tables = _setup(tmp_path, store)
    _, _, updates = _run(_app(store), _state(tables), "mirror", wrangling_approval=reject_join_missing)
    state, checked = {}, []
    for node, update in updates:  # the updates pattern: the state as each node leaves it
        if not isinstance(update, dict):
            continue
        state.update(update)
        if {"dataset", "profile", "tables", "profiles", "analysis_table"} & set(update) and "tables" in state:
            analysis = state["analysis_table"]
            assert state["tables"][analysis] == state["dataset"], node
            assert _by_value(state["profiles"][analysis]) == _by_value(state["profile"]), node
            assert _by_value(state["profiles"]["orders"]) == _by_value(
                profile_dataset(state["tables"]["orders"], store=store)), node
            checked.append(node)
    assert checked.count("wrangle_review") == 3  # round 1 cleaned only orders; the mirror held there too


# --- (b) the declined cleaning -----------------------------------------------------------------------------------

def test_a_declined_cleaning_lets_the_join_go_ahead(fake_model, tmp_path, store):
    _, tables = _setup(tmp_path, store)
    values, pauses, _ = _run(_app(store), _state(tables), "b", wrangling_approval=reject_carried_and_join_missing)
    first, second = _wranglings(pauses)[:2]
    assert {o["table"] for o in first["proposed_operations"]} == {"orders"}
    (join,) = second["proposed_operations"]
    assert join["impact_preview"]["warnings"] == [
        "8 row(s) have no match in orders",
        "3 exact duplicate row(s) in orders are counted again in the summary",
        "amount is 11.5% missing in orders: its mean and sum use only the rows that have a value"]
    descriptions = [u.description for u in values["plan"].unaddressed_issues if u.issue_type != "outcome_aggregated"]
    assert descriptions == ["In the table 'orders': 3 exact duplicate row(s) detected across all columns.",
                            "In the table 'orders': Column 'amount' is 11.5% missing."]
    assert values["stage1_end_reason"] == "declined_all"  # round 3 (the unmatched rows) applied nothing


def test_declining_both_rounds_ends_declined_all(fake_model, tmp_path, store):
    _, tables = _setup(tmp_path, store)
    values, pauses, _ = _run(_app(store, outcome="age"), _state(tables), "b2", wrangling_approval=_reject_all)
    assert [p["round"] for p in _wranglings(pauses)] == [1, 2]
    assert (values["stage1_end_reason"], values["analysis_table"]) == ("declined_all", "customers")


def test_finish_in_round_one_withholds_the_join(fake_model, tmp_path, store):
    _, tables = _setup(tmp_path, store)
    finish = lambda p: {"decisions": [{"index": o["index"], "action": "approve"} for o in p["proposed_operations"]],
                        "finish": True}
    values, pauses, _ = _run(_app(store, outcome="age"), _state(tables), "fin", wrangling_approval=finish)
    assert len(_wranglings(pauses)) == 1 and values["stage1_end_reason"] == "finished"
    assert not [e for e in _records(values, "stage1.clean.fix") if e["evidence"]["issue_type"] == "aggregatable_table"]
    assert values["analysis_table"] == "customers" and values["tables"]["orders"].version == 3


# --- (c) and (d) no withholding -------------------------------------------------------------------------------------

def test_a_clean_linked_table_is_joined_in_round_one(fake_model, tmp_path, store):
    customers = _customers()
    messy = pd.concat([customers, customers.iloc[[0]]], ignore_index=True)  # one exact duplicate in the analysis table
    _, tables = _setup(tmp_path, store, messy, _orders(customers, clean=True))
    values, pauses, _ = _run(_app(store), _state(tables), "c", wrangling_approval=reject_join_missing)
    first = _wranglings(pauses)[0]
    assert [(o["op_type"], o["table"]) for o in first["proposed_operations"]] == [("dedupe", None), ("join", None)]


def test_the_analysis_tables_issues_do_not_withhold_a_join(fake_model, tmp_path, store):
    customers = _customers()
    customers["tier"] = np.where(customers.index % 4 == 0, "Basic ", customers["tier"])
    messy = pd.concat([customers, customers.iloc[[3]]], ignore_index=True)
    _, tables = _setup(tmp_path, store, messy, _orders(_customers(), clean=True))
    pick = choose({"inconsistent_text": "trim_and_lowercase"})
    values, pauses, _ = _run(_app(store, pick), _state(tables), "d", wrangling_approval=reject_join_missing)
    ops = _wranglings(pauses)[0]["proposed_operations"]
    assert [o["op_type"] for o in ops] == ["normalize_text", "dedupe", "join"]
    executions = _records(values, "stage1.clean.execution")[:3]
    for op, execution in zip(ops, executions):
        assert {k: op["impact_preview"][k] for k in TABLE_KEYS} == {k: execution["evidence"][k] for k in TABLE_KEYS}


# --- (e) an unlinked carried table and the prompts -------------------------------------------------------------------

def test_an_unlinked_table_is_never_read_for_cleaning(fake_model, tmp_path, monkeypatch):
    prompts, config = fake_model
    config.update(question_type=QuestionType.TWO_GROUP_COMPARISON, predictors=["region"])
    store = CountingStore(tmp_path / "counting")
    customers = _customers().assign(total_spend=lambda d: d["age"] * 2)  # the fake chat drafts total_spend
    # the unlinked table shares the key's NAME but none of its values: the link scan reads it, nothing else does
    unlinked = pd.DataFrame({"customer_id": [901, 902, 903], "zz_secret": ["CANARY-ONE", "CANARY-TWO", "x"],
                             "zz_amount": [31415.926, 2.0, 3.0]})
    tables = _bundle(tmp_path, store, {"customers": customers, "orders": _orders(customers),
                                       "zz_unlinked": unlinked})
    scans = []
    real_scan = nodes_module.scan_links

    def counting_scan(*args, **kwargs):
        scans.append(1)
        return real_scan(*args, **kwargs)
    import core.table_links as table_links_module
    monkeypatch.setattr(nodes_module, "scan_links", counting_scan)
    monkeypatch.setattr(table_links_module, "scan_links", counting_scan)
    store.reads.clear()
    values, pauses, _ = _run(build_agent_graph(store), _state(tables), "e", wrangling_approval=_reject_all)
    key = tables["zz_unlinked"].storage_key
    assert store.reads.count(key) == 1 + len(scans)  # its profile, and one read per link scan; never cleaned
    payloads = json.dumps([p for p in pauses if p["type"] == "wrangling_approval"], default=str)
    for word in ("zz_unlinked", "zz_secret", "zz_amount", "CANARY", "31415"):
        assert word not in payloads, word
        for p in prompts:
            assert word not in p["text"], (p["stage"], word)
    # the linked table's names in the proposer prompt: only on its issue lines and options
    proposer = [p["text"] for p in prompts if p["stage"] == "_WranglingProposalsOut"]
    for text in proposer:
        for line in text.splitlines():
            if "orders" in line:
                assert re.match(r"^(\[\d+\] \w+: (In the table 'orders': |The table 'orders')|      \w+: )", line), line
    # the report prompts: only inside the rejected cleaning's caveat sentences
    for p in prompts:
        if p["stage"] == "report":
            text = p["text"]
            sentences = [f"{u.issue_type} ({u.description}) was identified but left unaddressed because: "
                         f"{u.rejection_reason}" for u in values["plan"].unaddressed_issues
                         if u.description.startswith("In the table 'orders'")]
            assert len(sentences) == 2 and all(sentence in text for sentence in sentences)
            for sentence in sentences:
                text = text.replace(sentence, "")
            assert "orders" not in text and "amount" not in text
        if p["stage"] in ("_ClassificationOut", "_DraftPlanOut"):
            assert "orders" not in p["text"]


# --- (f) per-table chains --------------------------------------------------------------------------------------------

def test_per_table_chains_and_previews(fake_model, tmp_path, store):
    customers = _customers()
    messy = pd.concat([customers, customers.iloc[[2]]], ignore_index=True)
    _, tables = _setup(tmp_path, store, messy, _orders(customers))
    pick = choose({"high_missingness": "impute_median"})
    values, pauses, _ = _run(_app(store, pick), _state(tables), "f", wrangling_approval=reject_join_missing)
    ops = _wranglings(pauses)[0]["proposed_operations"]
    assert [(o["op_type"], o["table"], o["impact_preview"]["basis"]) for o in ops] == [
        ("dedupe", None, "uploaded_data"), ("dedupe", "orders", "uploaded_data"),
        ("impute", "orders", "after_earlier_operations")]
    executions = _records(values, "stage1.clean.execution")[:3]
    assert [e["evidence"].get("table") for e in executions] == [None, "orders", "orders"]
    for op, execution in zip(ops, executions):
        ev = execution["evidence"]
        assert {k: op["impact_preview"][k] for k in TABLE_KEYS} == {k: ev[k] for k in TABLE_KEYS}, op["op_type"]
    impute = executions[2]["evidence"]
    deduped = _store_frame(store, tables["orders"]).drop_duplicates()
    assert ops[2]["impact_preview"]["cells_changed"] == impute["details"]["cells_filled"] == int(
        deduped["amount"].isna().sum())


# --- (g) stakeholder mode ------------------------------------------------------------------------------------------

def test_stakeholder_mode_never_auto_approves_a_carried_table(fake_model, tmp_path, store):
    customers = _customers()
    messy = pd.concat([customers, customers.iloc[[2]]], ignore_index=True)
    _, tables = _setup(tmp_path, store, messy, _orders(customers))
    values, pauses, _ = _run(_app(store), _state(tables, autonomy_mode="stakeholder"), "g",
                             wrangling_approval=reject_join_missing)
    first = _wranglings(pauses)[0]
    assert [(o["op_type"], o["table"]) for o in first["proposed_operations"]] == [("dedupe", "orders"),
                                                                                ("filter", "orders")]
    approvals = [(a["evidence"]["op_type"], a["chosen"], a["decided_by"]) for a in _records(values, "stage1.clean.approval")]
    assert approvals[:3] == [("dedupe", "auto_approve", "code_rule"), ("dedupe", "approve", "human"),
                             ("filter", "approve", "human")]


# --- (h) a round that only cleaned a carried table ----------------------------------------------------------------

def test_a_carried_only_round_continues(fake_model, tmp_path, store):
    _, tables = _setup(tmp_path, store)
    values, _, updates = _run(_app(store), _state(tables), "h", wrangling_approval=reject_join_missing)
    first_rediagnose = next(u for node, u in updates if node == "rediagnose")
    assert first_rediagnose == {"stage1_continue": True, "stage1_end_reason": None}  # no snapshot, no profile
    snapshots = _records(values, "stage1.quality.snapshot")
    assert [s["chosen"] for s in snapshots] == ["before", "after"]  # after the join only (round 3 was declined)


def test_the_round_limit_still_applies_and_the_report_is_silent_g84(fake_model, tmp_path, store, monkeypatch):
    monkeypatch.setattr(thresholds, "MAX_STAGE1_ROUNDS", 1)
    _, tables = _setup(tmp_path, store)
    values, pauses, _ = _run(_app(store, outcome="age"), _state(tables), "h1")
    assert len(_wranglings(pauses)) == 1 and values["stage1_end_reason"] == "round_limit"
    assert not [e for e in _records(values, "stage1.clean.fix") if e["evidence"]["issue_type"] == "aggregatable_table"]
    # G84: the analysis table is unchanged and the loop ran one round, so replan writes no record and the Stage 1
    # report has no Rounds section: it does not say that the limit stopped the loop before the join
    stage1 = next(e for e in values["stage_reports"] if e["stage"] == "stage1")
    assert _records(values, "planner.stage_replan") == []
    assert "Rounds" not in [s["heading"] for s in stage1["sections"]]


# --- (i) a carried issue that cleaning creates ----------------------------------------------------------------------

def _text_orders():
    """Orders whose channel differs only by spaces and case on three copied rows: no exact duplicates before trimming."""
    orders = _orders(_customers(), clean=True)
    copies = orders.iloc[[0, 1, 2]].assign(channel=lambda d: d["channel"].str.upper() + " ")
    return pd.concat([orders, copies], ignore_index=True)


def test_cleaning_that_creates_a_carried_issue(fake_model, tmp_path, store):
    orders = _text_orders()
    assert int(orders.duplicated().sum()) == 0
    normalized = orders.assign(channel=orders["channel"].str.strip().str.lower())
    assert int(normalized.duplicated().sum()) == 3
    _, tables = _setup(tmp_path, store, orders=orders)
    pick = choose({"inconsistent_text": "trim_and_lowercase"})
    values, pauses, _ = _run(_app(store, pick), _state(tables), "i", wrangling_approval=reject_join_missing)
    rounds = [[(o["op_type"], o["table"]) for o in p["proposed_operations"]] for p in _wranglings(pauses)]
    assert rounds[:3] == [[("normalize_text", "orders")], [("dedupe", "orders")], [("join", None)]]
    keys = values["stage1_seen_keys"]
    assert ["inconsistent_text", ["channel"], "orders"] in keys
    assert ["duplicate_rows", sorted(orders.columns), "orders"] in keys
    assert ["aggregatable_table", ["customer_id"], "orders"] in keys
    assert len(keys) == len({json.dumps(k) for k in keys})  # each proposed once


# --- (j) the next question ---------------------------------------------------------------------------------------------

def test_the_next_question_after_a_declined_and_an_approved_cleaning(fake_model, tmp_path, store):
    _, tables = _setup(tmp_path, store)
    app = _app(store, outcome="age")
    first, _, _ = _run(app, _state(tables), "j1", wrangling_approval=_reject_all)  # cleaning and join declined
    again = {"request_text": "q", "session_id": "s-join", "dataset": first["dataset"]}
    _, pauses, _ = _run(app, again, "j1", wrangling_approval=_reject_all)
    assert [(o["op_type"], o["table"]) for o in _wranglings(pauses)[0]["proposed_operations"]] == [
        ("dedupe", "orders"), ("filter", "orders")]  # still dirty: proposed again
    reject_joins_only = lambda p: {"decisions": [{"index": o["index"], "action": "reject" if o["op_type"] == "join"
                                                  else "approve"} for o in p["proposed_operations"]]}
    first, _, _ = _run(app, _state(tables), "j2", wrangling_approval=reject_joins_only)  # cleaning approved
    again = {"request_text": "q", "session_id": "s-join", "dataset": first["dataset"]}
    _, pauses, _ = _run(app, again, "j2", wrangling_approval=reject_joins_only)
    assert [(o["op_type"], o["table"]) for o in _wranglings(pauses)[0]["proposed_operations"]] == [("join", None)]


# --- (k) edits -----------------------------------------------------------------------------------------------------

def _edit_impute(params):
    def answer(payload):
        out = []
        for o in payload["proposed_operations"]:
            if o["op_type"] == "impute":
                out.append({"index": o["index"], "action": "edit", "params": {**o["params"], **params}})
            elif o.get("table") is not None:
                out.append({"index": o["index"], "action": "reject"})
            else:
                out.append({"index": o["index"], "action": "reject" if o["op_type"] in ("filter", "fill_unmatched")
                            else "approve"})
        return {"decisions": out}
    return answer


def test_an_edit_and_a_bad_edit_of_a_carried_operation(fake_model, tmp_path, store):
    _, tables = _setup(tmp_path, store)
    pick = choose({"high_missingness": "impute_median"})
    values, _, _ = _run(_app(store, pick), _state(tables), "k1", wrangling_approval=_edit_impute({"strategy": "mean"}))
    (impute,) = [e for e in _records(values, "stage1.clean.execution") if e["evidence"]["op_type"] == "impute"]
    assert (impute["chosen"], impute["evidence"]["table"], impute["evidence"]["details"]["strategy"]) == (
        "applied", "orders", "mean")
    assert values["tables"]["orders"].version == 2
    values, pauses, _ = _run(_app(store, pick), _state(tables), "k2", wrangling_approval=_edit_impute({"column": "nope"}))
    (impute,) = [e for e in _records(values, "stage1.clean.execution") if e["evidence"]["op_type"] == "impute"]
    assert impute["chosen"] == "failed" and values["tables"]["orders"].version == 1
    assert [p["round"] for p in _wranglings(pauses)][:2] == [1, 2]  # the declined rule: the join still goes ahead
    assert any(o["op_type"] == "join" for o in _wranglings(pauses)[1]["proposed_operations"])


# --- (l) cost -------------------------------------------------------------------------------------------------------

def test_three_linked_tables_of_three_hundred(fake_model, tmp_path, monkeypatch):
    store = LocalDiskStore(base_dir=tmp_path / "store")
    rng = np.random.default_rng(12)
    frames = {"customers": _customers()}
    for i in range(297):
        frames[f"t{i:03d}"] = pd.DataFrame({f"only_{i:03d}": [i, i + 1]})
    for name in ("visits", "returns", "reviews"):
        frames[name] = pd.DataFrame({"customer_id": range(1, 41), f"{name}_score": rng.normal(5, 1, 40)})
    tables = _bundle(tmp_path, store, frames)
    linked = []
    real = nodes_module.detect_carried_issues

    def recording(tables_, profiles_, store_, names):
        linked.extend(names)
        return real(tables_, profiles_, store_, names)
    monkeypatch.setattr(nodes_module, "detect_carried_issues", recording)
    reject_joins = lambda p: {"decisions": [{"index": o["index"], "action": "reject"} for o in p["proposed_operations"]]}
    started = time.perf_counter()
    _run(_app(store, outcome="age"), _state(tables), "l", wrangling_approval=reject_joins)
    assert time.perf_counter() - started < 10
    assert set(linked) == {"visits", "returns", "reviews"}


# --- (m) the registry -------------------------------------------------------------------------------------------------

def test_the_registry_is_unchanged(tmp_path):
    nodes = build_agent_graph(LocalDiskStore(base_dir=tmp_path)).get_graph().nodes
    assert (len(registered_points()), len(DECISION_NODES), len([n for n in nodes if not n.startswith("__")])) == (
        32, 18, 22)


# --- the sibling of test_join_run::test_a_normalized_join: the cleaning approved -----------------------------------

def test_a_cleaned_key_makes_the_plain_join_eligible(fake_model, tmp_path, store):
    rng = np.random.default_rng(41)
    main = pd.DataFrame({"region": rng.choice(["East", "West"], 120), "total_spend": rng.normal(50, 5, 120)})
    zones = pd.DataFrame({"region": ["East ", "West "], "manager": ["Ana", "Ben"]})  # stray spaces only
    tables = _bundle(tmp_path, store, {"main": main, "zones": zones})
    app = build_agent_graph(store, classify_fn=confident(QuestionType.TWO_GROUP_COMPARISON),
                            draft_fn=fake_draft(["region"]), propose_fn=choose({"inconsistent_text": "trim_text"}))
    values, pauses, _ = _run(app, _state(tables), "norm")
    first = _wranglings(pauses)[0]
    assert [(o["candidate_id"], o["table"]) for o in first["proposed_operations"]] == [("trim_text", "zones")]
    (op,) = _join_round([{"type": p["type"], "payload": p} for p in pauses])["proposed_operations"]
    by_id = {c["id"]: c for c in op["candidates"]}
    assert by_id["join_left"]["eligible"] and op["candidate_id"] == "join_left"
    assert _store_frame(store, values["dataset"])["manager"].notna().all()
