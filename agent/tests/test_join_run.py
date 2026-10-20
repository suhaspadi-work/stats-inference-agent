"""
M2.3b: joins through the real graph (guard fakes, fake chat model). A carried table that shares a unique key with
the analysis table is proposed as a join; when it is approved, a NEW derived table becomes the analysis table (the
originals stay), dataset, profile, tables, profiles and analysis_table stay in sync after every node, and the lineage
leads back to the uploaded tables; a declined join is not a data-quality caveat, and the Stage 1 report states both
reasons. The proposer sees the question and only the linked table's column names.
"""
import json
import re

import numpy as np
import pandas as pd
import pytest
from langgraph.types import Command

import core.classifier as classifier_module
from core.classifier import _ProposedOperationOut, _WranglingProposalsOut, propose_wrangling_operations
from core.dataset import LocalDiskStore
from core.decision_points import DECISION_NODES, registered_points
from core.graph import build_agent_graph
from core.ingestion import load_bundle
from core.lineage import check_lineage, edges_from_operations, lineage_graph
from core.operation_candidates import enumerate_candidates
from core.detection import detect_all_issues
from core.plan import QuestionType
from tests.test_decision_point_guard import choose, confident, fake_draft, reject_joins
from tests.test_prompt_canaries import fake_model  # noqa: F401 (fixture)
from tests.test_stage1_report import _answer, _drive

QUESTION = "Did West spend more?"
ALTERNATIVE = "Alternative the model suggested: Leave the data as it is (it would bias the result)."


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


# --- helpers -----------------------------------------------------------------------------------------------------

def _bundle(tmp_path, store, frames, session="s-join"):
    folder = tmp_path / f"files-{session}"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, frame in frames.items():
        paths.append(folder / f"{name}.csv")
        frame.to_csv(paths[-1], index=False)
    return load_bundle(paths, session_id=session, store=store)


def _state(tables, **extra):
    return {"request_text": QUESTION, "session_id": "s-join", "datasets": list(tables.values()), **extra}


def _app(store, propose_fn=None, predictors=("region",)):
    return build_agent_graph(store, classify_fn=confident(QuestionType.TWO_GROUP_COMPARISON),
                             draft_fn=fake_draft(list(predictors)), propose_fn=propose_fn or choose({}))


def _customers(n=150, seed=29):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"customer_id": range(1, n + 1), "region": rng.choice(["East", "West"], n),
                         "tier": rng.choice(["basic", "plus"], n)})


def _orders(customers, n=300, seed=30, extra_ids=()):
    rng = np.random.default_rng(seed)
    ids = rng.integers(1, len(customers) + 1, n)
    ids[:len(extra_ids)] = extra_ids  # ids that are not customers
    region = customers.set_index("customer_id")["region"].reindex(ids).to_numpy()
    return pd.DataFrame({"customer_id": ids, "total_spend": 50 + np.where(region == "West", 10.0, 0.0)
                         + rng.normal(0, 5, n)})


def _records(values, point):
    return [r for r in values["decision_log"] if r["decision_point"] == point]


def _wrangling(pauses):
    return next(p["payload"] for p in pauses if p["type"] == "wrangling_approval")


def _sections(values):
    stage1 = next(e for e in values["stage_reports"] if e["stage"] == "stage1")
    return {s["heading"]: s["text"] for s in stage1["sections"]}


def _mirror(values):
    analysis = values["analysis_table"]
    assert values["dataset"] == values["tables"][analysis] and values["profile"] == values["profiles"][analysis]


def _drive_checking_mirror(app, graph_input, thread, **overrides):
    """Run to the end in updates mode and check the mirror after every node that wrote a table key."""
    config = {"configurable": {"thread_id": thread}}
    state, checked = {}, []
    while True:
        for event in app.stream(graph_input, config, stream_mode="updates"):
            for node, update in event.items():
                if node == "__interrupt__" or not isinstance(update, dict):
                    continue
                state.update(update)
                if {"dataset", "profile", "tables", "profiles", "analysis_table"} & set(update) and "tables" in state:
                    analysis = state["analysis_table"]
                    assert state["tables"][analysis] == state["dataset"], node
                    if "profile" in state:
                        assert state["profiles"][analysis] == state["profile"], node
                    checked.append(node)
        snapshot = app.get_state(config)
        if not snapshot.interrupts:
            return snapshot.values, checked
        graph_input = Command(resume=_answer(snapshot.interrupts[0].value, overrides))


def _lineage_problems(values):
    applied = [r["evidence"] for r in _records(values, "stage1.clean.execution") if r["chosen"] == "applied"]
    derived = {e["params"].get("result_name") for e in applied if e["op_type"] == "join"}
    uploaded = {f"{name}@v1" for name in values["tables"] if name not in derived}  # as the live script does
    return check_lineage(values["dataset"].handle, lineage_graph(edges_from_operations(applied)), uploaded)


def _store_frame(store, ds):
    return store.read(ds.storage_key)


# --- (a) a join approved -----------------------------------------------------------------------------------------

def test_an_approved_join(fake_model, tmp_path, store):
    customers = _customers()
    orders = _orders(customers)
    tables = _bundle(tmp_path, store, {"orders": orders, "customers": customers})
    _, pauses, _ = _drive(_app(store), _state(tables), "a-pauses")
    (op,) = _wrangling(pauses)["proposed_operations"]
    assert (op["candidate_id"], op["op_type"], op["recommendation"]) == ("join_left", "join", "apply")
    assert op["what_was_run"] == ("Join customers to the analysis table on customer_id, keeping every row. Adds 2 "
                                  "column(s): region, tier.")
    assert op["impact_preview"]["rows_after"] == len(orders) and [c["id"] for c in op["candidates"]] == [
        "leave_as_is", "join_left", "join_inner", "join_left_normalized"]

    values, checked = _drive_checking_mirror(_app(store), _state(tables), "a")
    assert {"profile", "wrangle_review", "rediagnose"} <= set(checked)
    assert values["analysis_table"] == "orders_customers"
    assert set(values["tables"]) == {"orders", "customers", "orders_customers"}
    assert values["tables"]["customers"] == tables["customers"] and values["tables"]["orders"] == tables["orders"]
    _mirror(values)
    joined = _store_frame(store, values["dataset"])
    assert list(joined.columns) == ["customer_id", "total_spend", "region", "tier"] and len(joined) == len(orders)
    assert _store_frame(store, tables["customers"]).equals(_store_frame(store, values["tables"]["customers"]))
    plan = values["plan"]
    assert (plan.outcome_name, list(plan.candidate_predictors)) == ("total_spend", ["region"])
    assert list(plan.data_sources) == [values["dataset"].handle] == ["orders_customers@v1"]
    assert (values["route"]["capability"], values["route"]["status"]) == ("two_group", "ok")
    assert (values["test_result"]["group_a_label"], values["test_result"]["group_b_label"]) == ("East", "West")
    json.dumps(values["decision_log"], allow_nan=False)


# --- (b) lineage -------------------------------------------------------------------------------------------------

def test_the_lineage_of_a_join(fake_model, tmp_path, store):
    customers = _customers()
    tables = _bundle(tmp_path, store, {"orders": _orders(customers), "customers": customers})
    values, _, _ = _drive(_app(store), _state(tables), "b")
    (execution,) = _records(values, "stage1.clean.execution")
    assert execution["evidence"]["input_handles"] == ["orders@v1", "customers@v1"]
    assert execution["evidence"]["output_handle"] == "orders_customers@v1"
    assert values["dataset"].parents == ("orders@v1", "customers@v1")
    assert _lineage_problems(values) == []


# --- (c) and (d) a rejected join, and the model's recommendation -------------------------------------------------

def _reject(reason):
    def answer(payload):
        decision = {"index": payload["proposed_operations"][0]["index"], "action": "reject"}
        return {"decisions": [{**decision, **({"reason": reason} if reason is not None else {})}]}
    return answer


@pytest.mark.parametrize("recommendation, reason, line", [
    ("apply", "the question does not need this table",
     "customers was not joined. The model recommended joining: fix joinable_table. You rejected the join: the "
     "question does not need this table."),
    ("skip", "the question does not need this table",
     "customers was not joined. The model recommended against joining: fix joinable_table. You rejected the join: the "
     "question does not need this table."),
    ("apply", None,
     "customers was not joined. The model recommended joining: fix joinable_table. You rejected the join without "
     "giving a reason."),
], ids=["reason", "skip_and_reject", "no_reason"])
def test_a_rejected_join(recommendation, reason, line, fake_model, tmp_path, store):
    customers = _customers()
    orders = _orders(customers).assign(region=lambda d: np.where(d.index % 2, "East", "West"))  # analysable alone
    tables = _bundle(tmp_path, store, {"orders": orders, "customers": customers})
    values, _, _ = _drive(_app(store, choose({}, recommendation)), _state(tables), f"c-{recommendation}-{reason}",
                          wrangling_approval=_reject(reason))
    assert values["tables"] == tables and values["analysis_table"] == "orders"  # both tables untouched
    assert list(values["plan"].unaddressed_issues) == []
    (approval,) = _records(values, "stage1.clean.approval")
    assert approval["chosen"] == "reject" and approval["evidence"]["carried_forward_as_unaddressed"] is False
    assert approval["rationale"] == (reason or "no reason given")
    assert _records(values, "stage1.clean.execution") == []
    assert _sections(values)["Left as is"] == line
    summary = next(e for e in values["stage_reports"] if e["stage"] == "summary")
    caveats = next(s["text"] for s in summary["sections"] if s["heading"] == "Caveats")
    assert "customers" not in caveats and "joinable_table" not in caveats


def test_the_model_recommends_skip_and_the_user_approves(fake_model, tmp_path, store):
    customers = _customers()
    tables = _bundle(tmp_path, store, {"orders": _orders(customers), "customers": customers})
    values, _, _ = _drive(_app(store, choose({}, "skip")), _state(tables), "d")
    (execution,) = _records(values, "stage1.clean.execution")
    assert execution["chosen"] == "applied" and values["analysis_table"] == "orders_customers"
    changed = _sections(values)["Changed"]
    assert ("Why (the model's reason): fix joinable_table. The model recommended not joining this table. "
            + ALTERNATIVE) in changed
    assert "Left as is" not in _sections(values)


# --- (e) the execution order -------------------------------------------------------------------------------------

def _messy_orders(customers):
    df = _orders(customers, n=200)
    df["channel"] = np.where(np.arange(200) % 3 == 0, "Web ", np.where(np.arange(200) % 3 == 1, "web", "store"))
    df["visits"] = np.arange(200, dtype=float) % 9 + 1
    df.loc[df.index % 10 == 0, "visits"] = np.nan  # 10% missing
    return pd.concat([df, df.iloc[[5, 6, 7]]], ignore_index=True)


PICK = {"inconsistent_text": "trim_and_lowercase", "duplicate_rows": "dedupe_all", "high_missingness": "impute_median"}


def test_the_execution_order_and_every_preview(fake_model, tmp_path, store):
    customers = _customers()
    tables = _bundle(tmp_path, store, {"orders": _messy_orders(customers), "customers": customers})
    proposer = choose(PICK)
    values, pauses, _ = _drive(_app(store, lambda i, c, q=None: list(reversed(proposer(i, c, q)))), _state(tables), "e")
    ops = _wrangling(pauses)["proposed_operations"]
    executions = _records(values, "stage1.clean.execution")
    assert [o["op_type"] for o in ops] == [e["evidence"]["op_type"] for e in executions] == [
        "normalize_text", "dedupe", "join", "impute"]
    assert [o["preview_basis"] for o in ops] == ["uploaded_data"] + ["after_earlier_operations"] * 3
    for op, execution in zip(ops, executions):
        preview, ev = op["impact_preview"], execution["evidence"]
        assert (preview["rows_before"], preview["rows_after"], preview["columns_before"], preview["columns_after"]) == (
            ev["rows_before"], ev["rows_after"], ev["columns_before"], ev["columns_after"]), op["op_type"]
        details = ev.get("details") or {}
        expected = {"normalize_text": details.get("cells_changed"), "join": details.get("new_missing_cells"),
                    "impute": details.get("cells_filled"), "dedupe": preview["rows_changed"] * ev["columns_before"]}
        assert preview["cells_changed"] == expected[op["op_type"]], op["op_type"]
        assert preview["rows_changed"] == (details.get("cells_filled") if op["op_type"] == "impute" else
                                           details.get("cells_changed") if op["op_type"] == "normalize_text" else
                                           ev["rows_before"] - ev["rows_after"]), op["op_type"]
    join_ev = executions[2]["evidence"]
    assert join_ev["input_handles"] == [executions[1]["evidence"]["output_handle"], "customers@v1"]
    assert executions[3]["evidence"]["input_handles"] == ["orders_customers@v1"]  # the impute runs on the join
    assert values["dataset"].handle == "orders_customers@v2" and values["analysis_table"] == "orders_customers"
    _mirror(values)
    assert _lineage_problems(values) == []


# --- (f) an inner join -------------------------------------------------------------------------------------------

def test_an_inner_join(fake_model, tmp_path, store):
    customers = _customers()
    orders = _orders(customers, extra_ids=[901, 902, 903, 904, 905])
    tables = _bundle(tmp_path, store, {"orders": orders, "customers": customers})
    values, pauses, _ = _drive(_app(store, choose({"joinable_table": "join_inner"})), _state(tables), "f")
    (op,) = _wrangling(pauses)["proposed_operations"]
    removed = int((~orders["customer_id"].isin(customers["customer_id"])).sum())
    assert removed == 5 and op["impact_preview"]["rows_changed"] == removed
    assert op["impact_preview"]["warnings"] == [f"{removed} row(s) have no match in customers and are removed"]
    (execution,) = _records(values, "stage1.clean.execution")
    assert (execution["evidence"]["rows_before"], execution["evidence"]["rows_after"]) == (300, 300 - removed)
    assert f"Effect: 300 to {300 - removed} rows; 2 column(s) added." in _sections(values)["Changed"]


# --- (g) a normalized join ---------------------------------------------------------------------------------------

def test_a_normalized_join(fake_model, tmp_path, store):
    rng = np.random.default_rng(41)
    main = pd.DataFrame({"region": rng.choice(["East", "West"], 120), "total_spend": rng.normal(50, 5, 120)})
    zones = pd.DataFrame({"region": ["east ", "WEST"], "manager": ["Ana", "Ben"]})
    tables = _bundle(tmp_path, store, {"main": main, "zones": zones})
    from tests.test_aggregate_run import reject_carried  # (a lazy import: test_aggregate_run imports this module)
    # M2.4b: round 1 cleans the zones keys (stray spaces, case); declining it keeps them as uploaded, so in round 2 only
    # the normalized join can match, as before
    values, pauses, _ = _drive(_app(store), _state(tables), "g", wrangling_approval=reject_carried)
    (op,) = next(p["payload"] for p in pauses if p["type"] == "wrangling_approval"
                 and any(o["op_type"] == "join" for o in p["payload"]["proposed_operations"]))["proposed_operations"]
    by_id = {c["id"]: c for c in op["candidates"]}
    assert (by_id["join_left"]["eligible"], by_id["join_left"]["reason"]) == (False, "no key value matches")
    assert not by_id["join_inner"]["eligible"] and by_id["join_left_normalized"]["eligible"]
    assert op["candidate_id"] == "join_left_normalized"
    (execution,) = _records(values, "stage1.clean.execution")
    details = execution["evidence"]["details"]
    assert (details["matched_rows"], details["unmatched_left_rows"], details["normalize_keys"]) == (120, 0, True)
    assert _store_frame(store, values["dataset"])["manager"].notna().all()


# --- (h) an edit and a bad edit ----------------------------------------------------------------------------------

def _edit_join(params_change):
    def answer(payload):
        out = []
        for op in payload["proposed_operations"]:
            if op["op_type"] == "join":
                out.append({"index": op["index"], "action": "edit", "params": {**op["params"], **params_change}})
            else:
                out.append({"index": op["index"], "action": "approve"})
        return {"decisions": out}
    return answer


def test_an_edit_and_a_bad_edit(fake_model, tmp_path, store):
    customers = _customers()
    orders = _orders(customers, extra_ids=[901, 902]).assign(visits=lambda d: np.where(d.index % 10 == 0, np.nan, d.index % 7 + 1.0))
    tables = _bundle(tmp_path, store, {"orders": orders, "customers": customers})
    values, _, _ = _drive(_app(store, choose({"high_missingness": "impute_median"})), _state(tables), "h",
                          wrangling_approval=_edit_join({"how": "inner"}))
    join = next(e for e in _records(values, "stage1.clean.execution") if e["evidence"]["op_type"] == "join")
    assert join["evidence"]["details"]["how"] == "inner" and join["evidence"]["rows_after"] == 298

    regions = orders.assign(region=lambda d: np.where(d.index % 2, "East", "West"))
    tables = _bundle(tmp_path, store, {"orders": regions, "customers": customers.drop(columns="region")}, session="s-bad")
    values, _, _ = _drive(_app(store, choose({"high_missingness": "impute_median"})), _state(tables), "h-bad",
                          wrangling_approval=_edit_join({"right_table": "nowhere"}))
    executions = _records(values, "stage1.clean.execution")
    assert [(e["evidence"]["op_type"], e["chosen"]) for e in executions] == [("join", "failed"), ("impute", "applied")]
    assert executions[0]["evidence"]["failure_reason"] == "Cannot join — table not found: nowhere"
    assert values["analysis_table"] == "orders" and values["dataset"].handle == "orders@v2"
    assert set(values["tables"]) == {"orders", "customers"}
    _mirror(values)


# --- (i) two carried tables that both join -----------------------------------------------------------------------

def test_two_joins_in_a_row(fake_model, tmp_path, store):
    customers = _customers()
    products = pd.DataFrame({"product_id": range(1, 21), "category": ["toys", "books"] * 10})
    orders = _orders(customers).assign(product_id=lambda d: d.index % 20 + 1)
    tables = _bundle(tmp_path, store, {"orders": orders, "customers": customers, "products": products})
    values, pauses, _ = _drive(_app(store), _state(tables), "i")
    ops = _wrangling(pauses)["proposed_operations"]
    assert [o["params"]["right_table"] for o in ops] == ["customers", "products"]
    assert [o["params"]["result_name"] for o in ops] == ["orders_customers", "orders_products"]
    first, second = _records(values, "stage1.clean.execution")
    assert first["evidence"]["output_handle"] == "orders_customers@v1"
    assert second["evidence"]["input_handles"] == ["orders_customers@v1", "products@v1"]  # the left is the first result
    assert values["analysis_table"] == "orders_products"
    assert set(values["tables"]) == {"orders", "customers", "products", "orders_customers", "orders_products"}
    assert all(values["tables"][n] == tables[n] for n in ("orders", "customers", "products"))
    assert list(_store_frame(store, values["dataset"]).columns) == [
        "customer_id", "total_spend", "product_id", "region", "tier", "category"]
    _mirror(values)
    assert _lineage_problems(values) == []
    assert _sections(values)["Tables"].split("\n") == [
        "orders: 300 rows, 3 columns (main table; joined with customers into orders_customers)",
        "customers: 150 rows, 3 columns (joined into orders_customers)",
        "products: 20 rows, 2 columns (joined into orders_products)",
        "orders_customers: 300 rows, 5 columns (main table after the join; joined with products into orders_products)",
        "orders_products: 300 rows, 6 columns (main table after the join)"]


# --- (j) stakeholder mode ----------------------------------------------------------------------------------------

def test_stakeholder_mode_always_interrupts_for_a_join(fake_model, tmp_path, store):
    customers = _customers()
    orders = _orders(customers)
    orders = pd.concat([orders, orders.iloc[[0, 1]]], ignore_index=True)
    tables = _bundle(tmp_path, store, {"orders": orders, "customers": customers})
    values, pauses, _ = _drive(_app(store), _state(tables, autonomy_mode="stakeholder"), "j")
    assert [o["op_type"] for o in _wrangling(pauses)["proposed_operations"]] == ["join"]  # the dedupe was auto-approved
    assert [a["chosen"] for a in _records(values, "stage1.clean.approval")] == ["auto_approve", "approve"]


# --- (k) privacy -------------------------------------------------------------------------------------------------

LINKED, UNLINKED = "linked_lookup", "unlinked_extra"
LINKED_COLUMN, UNLINKED_COLUMN = "zz_linked_only", "zz_unlinked_only"
CANARIES = ("JOIN-CANARY-ONE", "JOIN-CANARY-TWO", "UNLINKED-CANARY", "27182.8182")


def test_privacy(fake_model, tmp_path, store):
    prompts, config = fake_model
    config.update(question_type=QuestionType.TWO_GROUP_COMPARISON, predictors=["region"])
    rng = np.random.default_rng(43)
    main = pd.DataFrame({"region": rng.choice(["East", "West"], 120), "total_spend": rng.normal(50, 5, 120)})
    linked = pd.DataFrame({"region": ["East", "West"], LINKED_COLUMN: ["JOIN-CANARY-ONE", "JOIN-CANARY-TWO"]})
    unlinked = pd.DataFrame({UNLINKED_COLUMN: ["UNLINKED-CANARY", "x"], "zz_amount": [27182.8182, 1.5]})
    tables = _bundle(tmp_path, store, {"main": main, LINKED: linked, UNLINKED: unlinked})
    values, pauses, _ = _drive(build_agent_graph(store), _state(tables), "k", wrangling_approval=reject_joins)
    assert "wrangling_approval" in [p["type"] for p in pauses] and values["report"]
    (proposer,) = [p["text"] for p in prompts if p["stage"] == "_WranglingProposalsOut"]
    others = [p for p in prompts if p["stage"] != "_WranglingProposalsOut"]
    assert {p["stage"] for p in others} == {"_ClassificationOut", "_DraftPlanOut", "report"}
    for p in prompts:
        for word in (*CANARIES, UNLINKED, UNLINKED_COLUMN, "zz_amount"):
            assert word not in p["text"], (p["stage"], word)
    for p in others:
        assert LINKED not in p["text"] and LINKED_COLUMN not in p["text"], p["stage"]
    lines = proposer.splitlines()
    (join_line,) = [line for line in lines if re.match(r"^\[\d+\] joinable_table: ", line)]
    assert [line for line in lines if LINKED_COLUMN in line] == [join_line]
    assert [line for line in lines if LINKED in line] == [
        join_line, f"      join_left: Join {LINKED} to the analysis table on region, keeping every row.",
        f"      join_inner: Join {LINKED} to the analysis table on region, keeping only the rows that have a match."]
    assert f"The user's question: {QUESTION}" in proposer
    assert ("For a joinable_table issue, say in 'recommendation' whether the question needs the other table ('apply') "
            "or not ('skip'), and in 'rationale' why; you must still choose the best join option so the user can "
            "decide.") in proposer
    for word in ("matched_distinct", "inner_join_rows", "orphan", "fan_out", "many_to_one", "Adds 1 column"):
        assert word not in proposer, word  # no link field or number, and no preview wording


# --- (l) the proposer --------------------------------------------------------------------------------------------

def test_the_schema():
    assert set(_ProposedOperationOut.model_fields) == {"issue_index", "candidate_id", "rationale",
                                                       "alternative_candidate_id", "alternative_rejected_because",
                                                       "recommendation"}
    assert _ProposedOperationOut.model_fields["recommendation"].default == "apply"
    proposal = _ProposedOperationOut(issue_index=0, candidate_id="x", rationale="r", alternative_candidate_id="y",
                                     alternative_rejected_because="z")
    assert proposal.recommendation == "apply"


class Model:
    def __init__(self, answers):
        self.answers, self.prompts = list(answers), []

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.prompts.append(messages)
        return _WranglingProposalsOut(proposals=self.answers.pop(0))


def _p(candidate):
    return _ProposedOperationOut(issue_index=0, candidate_id=candidate, rationale="because",
                                 alternative_candidate_id="leave_as_is", alternative_rejected_because="worse")


def test_the_prompt_without_a_question_or_a_join_is_unchanged(monkeypatch, tmp_path, store):
    from core.detection import CandidateIssue
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    tables = _bundle(tmp_path, store, {"t": df})
    issue = CandidateIssue(issue_type="duplicate_rows", description="1 exact duplicate rows.", affected_columns=("a", "b"),
                           evidence={"duplicate_row_count": 1})
    candidates = [enumerate_candidates(issue, tables["t"], None, store)]
    model = Model([[_p("dedupe_all")]])
    monkeypatch.setattr(classifier_module, "init_chat_model", lambda *a, **k: model)
    propose_wrangling_operations([issue], candidates)
    (messages,) = model.prompts
    previous = (  # the M2.3a step 2 prompt, written out
        "You are choosing data-cleaning operations for detected data-quality issues. For each issue, choose "
        "exactly one of its listed options by its id (candidate_id) and explain why. Then name one different "
        "option id of the same issue as the alternative you considered and did not choose (alternative_candidate_id); "
        "'leave_as_is' (keep the data as it is) is also a valid alternative. Choose exactly one option per issue, "
        "referencing its index. Copy each option id exactly as written, including every part of it.\n\n"
        "Detected issues:\n[0] duplicate_rows: 1 exact duplicate rows. (columns: a, b, evidence: "
        "{'duplicate_row_count': 1})\n    options:\n      dedupe_all: Remove exact duplicate rows (all columns).")
    assert messages == [("system", previous), ("human", "Choose a fix for each detected issue.")]


def test_the_retry_works_for_a_join(monkeypatch, tmp_path, store):
    customers = _customers()
    tables = _bundle(tmp_path, store, {"orders": _orders(customers), "customers": customers})
    from core.ingestion import profile_dataset
    profiles = {n: profile_dataset(ds, store=store) for n, ds in tables.items()}
    issues = detect_all_issues(tables["orders"], profiles["orders"], store, tables=tables, profiles=profiles,
                               analysis="orders")
    assert [i.issue_type for i in issues] == ["joinable_table"]
    candidates = [enumerate_candidates(issues[0], tables["orders"], None, store, tables=tables)]
    model = Model([[_p("join_outer")], [_p("join_left")]])
    monkeypatch.setattr(classifier_module, "init_chat_model", lambda *a, **k: model)
    (proposal,) = propose_wrangling_operations(issues, candidates, QUESTION)
    assert proposal.candidate_id == "join_left" and proposal._retried and len(model.prompts) == 2
    assert "'join_outer' for issue 0, which is not one of its eligible options" in model.prompts[1][-1][1]


# --- (m) the Stage 1 report --------------------------------------------------------------------------------------

def test_the_stage1_report_of_a_join(fake_model, tmp_path, store):
    customers = _customers()
    orders = _orders(customers)
    tables = _bundle(tmp_path, store, {"orders": orders, "customers": customers})
    values, _, _ = _drive(_app(store), _state(tables), "m")
    sections = _sections(values)
    assert sections["Tables"].split("\n") == [
        "orders: 300 rows, 2 columns (main table; joined with customers into orders_customers)",
        "customers: 150 rows, 3 columns (joined into orders_customers)",
        "orders_customers: 300 rows, 4 columns (main table after the join)"]
    assert sections["Checked"].split("\n") == ["Checked 300 rows and 2 columns.",
                                               "customers: can be joined to the analysis table on customer_id."]
    assert sections["Changed"] == (
        "1. Join customers to the analysis table on customer_id, keeping every row. Adds 2 column(s): region, tier. "
        f"Why (the model's reason): fix joinable_table. {ALTERNATIVE} Your decision: approved. Effect: 300 rows stay "
        "300; 2 column(s) added; no new missing values.")

    alone = orders.assign(region=lambda d: np.where(d.index % 2, "East", "West"))  # analysable without the join
    tables = _bundle(tmp_path, store, {"orders": alone, "customers": customers}, session="s-rejected")
    rejected, _, _ = _drive(_app(store), _state(tables), "m-rejected", wrangling_approval=reject_joins)
    assert _sections(rejected)["Tables"].split("\n") == [  # as before M2.3b
        "orders: 300 rows, 3 columns (main table)",
        "customers: 150 rows, 3 columns (carried along; not joined)"]  # M2.3b: the new wording


def test_the_effect_line_with_unmatched_rows(fake_model, tmp_path, store):
    customers = _customers()
    tables = _bundle(tmp_path, store, {"orders": _orders(customers, extra_ids=[901, 902, 903]), "customers": customers})
    values, _, _ = _drive(_app(store), _state(tables), "m-unmatched")
    assert ("Effect: 300 rows stay 300; 2 column(s) added; 6 new missing value(s) from 3 row(s) with no match."
            in _sections(values)["Changed"])


# --- (n) thirty tables -------------------------------------------------------------------------------------------

def test_thirty_tables(fake_model, tmp_path, store):
    prompts, config = fake_model
    config.update(question_type=QuestionType.TWO_GROUP_COMPARISON, predictors=["region"])
    customers = _customers()
    products = pd.DataFrame({"product_id": range(1, 21), "category": ["toys", "books"] * 10})
    frames = {"orders": _orders(customers).assign(product_id=lambda d: d.index % 20 + 1), "customers": customers,
              "products": products}
    frames.update({f"tiny_{chr(97 + i // 26)}{chr(97 + i % 26)}": pd.DataFrame({f"tinycol_{chr(97 + i)}": [i]})
                   for i in range(27)})
    tables = _bundle(tmp_path, store, frames)
    assert len(tables) == 30
    values, pauses, _ = _drive(build_agent_graph(store), _state(tables), "n")
    payload = _wrangling(pauses)
    assert [o["params"]["right_table"] for o in payload["proposed_operations"]] == ["customers", "products"]
    text = json.dumps(payload) + "\n".join(p["text"] for p in prompts)
    assert "tiny_" not in text and "tinycol_" not in text
    json.dumps(values["decision_log"], allow_nan=False)


# --- (o) the registry --------------------------------------------------------------------------------------------

def test_the_registry_is_unchanged():
    assert (len(registered_points()), len(DECISION_NODES)) == (32, 18)  # M2.4a: replan (planner.stage_replan)
