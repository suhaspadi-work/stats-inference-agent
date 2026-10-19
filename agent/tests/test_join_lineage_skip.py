"""
M2.3b: a carried table already joined into the analysis table (found through the analysis dataset's lineage) raises
no join issue on a later question; a rejected join is not remembered, so it is proposed again.
"""
import numpy as np
import pandas as pd
import pytest
from langgraph.types import Command

from core.dataset import Dataset, LocalDiskStore
from core.detection import _joined_names, detect_join_issues
from core.ingestion import load_dataset, profile_dataset
from tests.test_decision_point_guard import choose
from tests.test_join_run import QUESTION, _app, _bundle, _customers, _orders, _state
from tests.test_prompt_canaries import fake_model  # noqa: F401 (fixture)
from tests.test_stage1_report import _answer


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _run(app, graph_input, thread, **overrides):
    """As the product does it: invoke, answer each pause; returns the values and the pause types."""
    config = {"configurable": {"thread_id": thread}}
    result, pauses = app.invoke(graph_input, config), []
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        pauses.append(payload)
        result = app.invoke(Command(resume=_answer(payload, overrides)), config)
    return result, pauses


def _again(first):
    """The product's Ask another question: the returned dataset, no datasets."""
    return {"request_text": QUESTION, "session_id": "s-join", "dataset": first["dataset"]}


def _proposed(pauses):
    return [op["params"]["right_table"] for p in pauses if p["type"] == "wrangling_approval"
            for op in p["proposed_operations"] if op["op_type"] == "join"]


def _latest_scan(values):
    return next(r for r in reversed(values["decision_log"]) if r["decision_point"] == "stage1.clean.scan")


def _reject_table(name):
    def answer(payload):
        return {"decisions": [{"index": op["index"], "action": "reject", "reason": "not needed"}
                              if op["op_type"] == "join" and op["params"]["right_table"] == name
                              else {"index": op["index"], "action": "approve"} for op in payload["proposed_operations"]]}
    return answer


REGIONS = pd.DataFrame({"region": ["East", "West"], "manager": ["Ana", "Ben"]})


# --- (a) the joined table is not proposed again --------------------------------------------------------------------

def test_a_joined_table_is_not_proposed_again(fake_model, tmp_path, store):
    customers = _customers()
    tables = _bundle(tmp_path, store, {"orders": _orders(customers), "customers": customers})
    app = _app(store)
    first, pauses = _run(app, _state(tables), "a")
    assert _proposed(pauses) == ["customers"] and first["analysis_table"] == "orders_customers"
    second, pauses = _run(app, _again(first), "a")
    assert second["analysis_table"] == "orders_customers" and set(second["tables"]) == set(first["tables"])
    assert "wrangling_approval" not in [p["type"] for p in pauses]  # nothing else needs cleaning
    scan = _latest_scan(second)
    assert (scan["chosen"], scan["evidence"]["joinable_tables"]) == ("skipped", [])
    assert second["plan"].status.value == "executed"


# --- (b) a third table that links to the joined table --------------------------------------------------------------

def test_only_the_new_link_is_proposed(fake_model, tmp_path, store):
    customers = _customers()
    tables = _bundle(tmp_path, store, {"orders": _orders(customers), "customers": customers, "regions": REGIONS})
    app = _app(store)
    first, pauses = _run(app, _state(tables), "b")
    assert _proposed(pauses) == ["customers"]  # the orders have no region yet, so regions has no link
    second, pauses = _run(app, _again(first), "b", wrangling_approval=_reject_table("regions"))
    assert _proposed(pauses) == ["regions"]  # region came from customers; customers itself is not proposed again


# --- (c) a chain of joins --------------------------------------------------------------------------------------------

def test_a_chain_of_joins(fake_model, tmp_path, store):
    customers = _customers()
    products = pd.DataFrame({"product_id": range(1, 21), "category": ["toys", "books"] * 10})
    orders = _orders(customers).assign(product_id=lambda d: d.index % 20 + 1)
    tables = _bundle(tmp_path, store, {"orders": orders, "customers": customers, "products": products,
                                       "regions": REGIONS})
    app = _app(store)
    first, pauses = _run(app, _state(tables), "c")
    assert _proposed(pauses) == ["customers", "products"] and first["analysis_table"] == "orders_products"
    assert first["dataset"].parents == ("orders_customers@v1", "products@v1")
    second, pauses = _run(app, _again(first), "c", wrangling_approval=_reject_table("regions"))
    assert _proposed(pauses) == ["regions"]  # neither customers nor products, but the fourth table still


# --- (d) a rejected join is proposed again ----------------------------------------------------------------------------

def test_a_rejected_join_is_proposed_again(fake_model, tmp_path, store):
    customers = _customers()
    orders = _orders(customers).assign(region=lambda d: np.where(d.index % 2, "East", "West"))
    tables = _bundle(tmp_path, store, {"orders": orders, "customers": customers})
    app = _app(store)
    first, pauses = _run(app, _state(tables), "d", wrangling_approval=_reject_table("customers"))
    assert _proposed(pauses) == ["customers"] and first["analysis_table"] == "orders"
    second, pauses = _run(app, _again(first), "d", wrangling_approval=_reject_table("customers"))
    assert _proposed(pauses) == ["customers"]


# --- (e) unit tests of the lineage walk -------------------------------------------------------------------------------

def _load(tmp_path, store, name, frame):
    path = tmp_path / f"{name}.csv"
    frame.to_csv(path, index=False)
    return load_dataset(path, session_id="s", name=name, store=store)


def test_detect_skips_the_parents_of_a_derived_analysis_table(tmp_path, store):
    rng = np.random.default_rng(3)
    joined = pd.DataFrame({"customer_id": rng.integers(1, 21, 60), "region": rng.choice(["East", "West"], 60)})
    tables = {name: _load(tmp_path, store, name, frame) for name, frame in (
        ("joined", joined), ("customers", pd.DataFrame({"customer_id": range(1, 21), "tier": ["a", "b"] * 10})),
        ("accounts", pd.DataFrame({"customer_id": range(1, 21), "score": range(20)})))}
    plain = {n: profile_dataset(ds, store=store) for n, ds in tables.items()}
    assert [i.evidence["right_table"] for i in detect_join_issues("joined", tables, plain, store)] == [
        "customers", "accounts"]  # an uploaded analysis table: both are proposed
    tables["joined"] = Dataset.derived("s", "joined", tables["joined"].storage_key,
                                       parents=("orders@v1", "customers@v1", "ghost@v3"), operation_id="op")
    profiles = {**plain, "joined": profile_dataset(tables["joined"], store=store)}
    issues = detect_join_issues("joined", tables, profiles, store)  # orders and ghost are not in tables: ignored
    assert [i.evidence["right_table"] for i in issues] == ["accounts"]


def test_the_walk_follows_derived_parents_and_stops_on_a_cycle():
    def derived(name, *parents):
        return Dataset.derived("s", name, f"s/{name}_v1.csv", parents=parents, operation_id="op")
    tables = {"b": derived("b", "a@v1", "products@v1"), "a": derived("a", "orders@v1", "customers@v1"),
              "customers": Dataset(session_id="s", name="customers", version=1, storage_key="k", parent_version=None,
                                   created_from_operation=None)}
    assert _joined_names("b", tables) == {"a", "products", "orders", "customers"}
    cycle = {"x": derived("x", "y@v1"), "y": derived("y", "x@v1")}
    assert _joined_names("x", cycle) == {"x", "y"}  # terminates
    assert _joined_names("missing", cycle) == set() and _joined_names("customers", tables) == set()


# --- after later operations (M2.3b: next_version keeps the parents) --------------------------------------------------

def _with_visits(orders):
    return orders.assign(visits=lambda d: np.where(d.index % 10 == 0, np.nan, d.index % 7 + 1.0))  # 10% missing


def _reject_dedupe(payload):
    return {"decisions": [{"index": op["index"], "action": "reject", "reason": "keep the repeats"}
                          if op["op_type"] == "dedupe" else {"index": op["index"], "action": "approve"}
                          for op in payload["proposed_operations"]]}


def test_not_proposed_again_after_a_join_and_an_imputation(fake_model, tmp_path, store):
    customers = _customers()
    tables = _bundle(tmp_path, store, {"orders": _with_visits(_orders(customers)), "customers": customers})
    app = _app(store, choose({"high_missingness": "impute_median"}))
    first, pauses = _run(app, _state(tables), "later-impute")
    assert _proposed(pauses) == ["customers"] and first["dataset"].handle == "orders_customers@v2"
    second, pauses = _run(app, _again(first), "later-impute")
    assert _proposed(pauses) == [] and "wrangling_approval" not in [p["type"] for p in pauses]
    assert _latest_scan(second)["evidence"]["joinable_tables"] == []


def test_not_proposed_again_after_a_join_and_a_later_dedupe(fake_model, tmp_path, store):
    customers = _customers()
    orders = _orders(customers)
    orders = pd.concat([orders, orders.iloc[[0, 1, 2]]], ignore_index=True)
    tables = _bundle(tmp_path, store, {"orders": orders, "customers": customers})
    app = _app(store)
    # a dedupe always runs before a join in one review, so the dedupe that follows the join is a later question's
    first, pauses = _run(app, _state(tables), "later-dedupe", wrangling_approval=_reject_dedupe)
    assert _proposed(pauses) == ["customers"] and first["dataset"].handle == "orders_customers@v1"
    second, pauses = _run(app, _again(first), "later-dedupe")
    assert _proposed(pauses) == [] and second["dataset"].handle == "orders_customers@v2"  # the dedupe ran
    third, pauses = _run(app, _again(second), "later-dedupe")
    assert _proposed(pauses) == [] and "wrangling_approval" not in [p["type"] for p in pauses]


def test_a_rejected_join_is_proposed_again_after_an_imputation(fake_model, tmp_path, store):
    customers = _customers()
    orders = _with_visits(_orders(customers)).assign(region=lambda d: np.where(d.index % 2, "East", "West"))
    tables = _bundle(tmp_path, store, {"orders": orders, "customers": customers})
    app = _app(store, choose({"high_missingness": "impute_median"}))
    first, pauses = _run(app, _state(tables), "rejected-later", wrangling_approval=_reject_table("customers"))
    assert first["dataset"].handle == "orders@v2"
    second, pauses = _run(app, _again(first), "rejected-later", wrangling_approval=_reject_table("customers"))
    assert _proposed(pauses) == ["customers"]
