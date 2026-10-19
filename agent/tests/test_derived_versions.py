"""
M2.3b: the version chain of a derived table. next_version keeps the parents (the handles the chain's first version
was built from), so a join is not forgotten after another operation; parent_handles names the previous version
first. core.lineage is built from the operations and is unaffected.
"""
import numpy as np
import pandas as pd
import pytest

from core.checkpoint_types import make_serde
from core.dataset import Dataset, LocalDiskStore, parent_handles
from core.ingestion import load_dataset
from core.lineage import edges_from_operations, lineage_graph, roots
from core.operation import ApprovalStatus, Operation
from core.wrangling import execute_impute, execute_join


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _derived():
    return Dataset.derived("s", "orders_customers", "s/orders_customers_v1.csv", parents=("orders@v1", "customers@v1"),
                           operation_id="op1")


def test_next_version_keeps_the_parents():
    v2 = _derived().next_version("s/orders_customers_v2.csv", "op2")
    assert (v2.version, v2.parent_version, v2.parents) == (2, 1, ("orders@v1", "customers@v1"))
    upload = Dataset(session_id="s", name="orders", version=1, storage_key="s/orders_v1.csv", parent_version=None,
                     created_from_operation=None)
    assert upload.next_version("s/orders_v2.csv", "op").parents == ()


def test_parent_handles_name_the_previous_version_first():
    v1 = _derived()
    v2 = v1.next_version("s/orders_customers_v2.csv", "op2")
    assert parent_handles(v1) == ("orders@v1", "customers@v1") and parent_handles(v2) == ("orders_customers@v1",)
    upload = Dataset(session_id="s", name="orders", version=1, storage_key="k", parent_version=None,
                     created_from_operation=None)
    assert parent_handles(upload) == () and parent_handles(upload.next_version("k2", "op")) == ("orders@v1",)


def test_a_chain_of_three_versions():
    chain = [_derived()]
    for n in (2, 3):
        chain.append(chain[-1].next_version(f"s/orders_customers_v{n}.csv", f"op{n}"))
    assert [d.parents for d in chain] == [("orders@v1", "customers@v1")] * 3
    assert [parent_handles(d) for d in chain[1:]] == [("orders_customers@v1",), ("orders_customers@v2",)]


def test_a_checkpoint_round_trip_of_a_derived_v2():
    v2 = _derived().next_version("s/orders_customers_v2.csv", "op2")
    serde = make_serde()
    restored = serde.loads_typed(serde.dumps_typed(v2))
    assert restored == v2 and isinstance(restored.parents, tuple) and type(restored) is Dataset


def test_lineage_after_a_join_and_an_imputation(tmp_path, store):
    rng = np.random.default_rng(2)
    tables = {}
    for name, frame in (("orders", pd.DataFrame({"customer_id": rng.integers(1, 11, 40),
                                                 "visits": np.where(np.arange(40) % 5 == 0, np.nan, 3.0)})),
                        ("customers", pd.DataFrame({"customer_id": range(1, 11), "region": ["East", "West"] * 5}))):
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        tables[name] = load_dataset(path, session_id="s", name=name, store=store)
    join = Operation(op_type="join", session_id="s", input_handles=("orders@v1", "customers@v1"),
                     params={"right_table": "customers", "left_key": "customer_id", "right_key": "customer_id",
                             "how": "left", "result_name": "orders_customers"}, approval_status=ApprovalStatus.APPROVED)
    joined, finished_join = execute_join(tables["orders"], store, join, tables)
    impute = Operation(op_type="impute", session_id="s", input_handles=(joined.handle,),
                       params={"column": "visits", "strategy": "median"}, approval_status=ApprovalStatus.APPROVED)
    final, finished_impute = execute_impute(joined, store, impute)
    assert final.handle == "orders_customers@v2" and final.parents == joined.parents
    graph = lineage_graph(edges_from_operations([finished_join, finished_impute]))
    assert roots(final.handle, graph) == {"orders@v1", "customers@v1"}
