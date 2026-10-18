"""
M2.1: lineage across tables. Dataset.derived and parent_handles (core/dataset.py) and the pure lineage graph
(core/lineage.py): edges from applied operations, ancestors, roots and the check that a table leads back to
uploaded tables only. Every walk is iterative, so long chains are safe.
"""
import copy
import dataclasses
import time

import pytest

from core.dataset import Dataset, parent_handles
from core.lineage import ancestors, check_lineage, edges_from_operations, lineage_graph, roots
from core.operation import ApprovalStatus, Operation


def _upload(name="customers"):
    return Dataset(session_id="s", name=name, version=1, storage_key=f"s/{name}_v1.csv", parent_version=None,
                   created_from_operation=None, original_filename=f"{name}.csv")


def _op(inputs, output=None, op_type="dedupe"):
    return Operation(op_type=op_type, session_id="s", input_handles=tuple(inputs), params={},
                     approval_status=ApprovalStatus.APPROVED, output_handle=output)


# --- Dataset.derived, next_version, parent_handles ------------------------------------------------------------------

def test_derived():
    joined = Dataset.derived("s", "joined", "s/joined_v1.csv", ["customers@v2", "regions@v1"], "op-1")
    assert (joined.version, joined.parent_version, joined.created_from_operation) == (1, None, "op-1")
    assert joined.parents == ("customers@v2", "regions@v1") and isinstance(joined.parents, tuple)
    assert joined.handle == "joined@v1"
    assert joined == dataclasses.replace(joined) and joined == copy.deepcopy(joined)
    with pytest.raises(dataclasses.FrozenInstanceError):
        joined.parents = ()
    with pytest.raises(ValueError, match="at least one parent"):
        Dataset.derived("s", "joined", "k", [], "op-1")


def test_next_version_is_unchanged():
    upload = _upload()
    v2 = upload.next_version("s/customers_v2.csv", "op-1")
    assert (v2.version, v2.parent_version, v2.parents, v2.created_from_operation) == (2, 1, (), "op-1")
    assert upload.parents == ()


def test_parent_handles():
    upload = _upload()
    assert parent_handles(upload) == ()
    assert parent_handles(upload.next_version("k", "op")) == ("customers@v1",)
    joined = Dataset.derived("s", "joined", "k", ("customers@v2", "regions@v1"), "op")
    assert parent_handles(joined) == ("customers@v2", "regions@v1")


# --- edges and the graph ---------------------------------------------------------------------------------------------

def test_edges_from_operations():
    ops = [_op(["customers@v1"], "customers@v2"), _op(["customers@v2", "regions@v1"], "joined@v1", "join"),
           _op(["customers@v2"], None), _op(["joined@v1"], "joined@v2")]
    assert edges_from_operations(ops) == [("customers@v1", "customers@v2"), ("customers@v2", "joined@v1"),
                                          ("regions@v1", "joined@v1"), ("joined@v1", "joined@v2")]
    records = [{"input_handles": ["a@v1"], "output_handle": "a@v2"}, {"input_handles": ["a@v2"], "output_handle": None},
               {"input_handles": ["a@v2", "b@v1"], "output_handle": "j@v1"}]
    assert edges_from_operations(records) == [("a@v1", "a@v2"), ("a@v2", "j@v1"), ("b@v1", "j@v1")]
    assert edges_from_operations([]) == []


def test_lineage_graph():
    edges = [("a@v1", "a@v2"), ("b@v1", "j@v1"), ("a@v2", "j@v1"), ("b@v1", "j@v1"), ("c@v1", "j@v1")]
    assert lineage_graph(edges) == {"a@v2": ("a@v1",), "j@v1": ("b@v1", "a@v2", "c@v1")}
    assert lineage_graph([]) == {}


# --- ancestors and roots -------------------------------------------------------------------------------------------------

def test_ancestors():
    chain = lineage_graph([("a@v1", "a@v2"), ("a@v2", "a@v3")])
    assert ancestors("a@v3", chain) == {"a@v1", "a@v2"}
    diamond = lineage_graph([("g", "p1"), ("g", "p2"), ("p1", "c"), ("p2", "c")])
    assert ancestors("c", diamond) == {"p1", "p2", "g"}
    assert ancestors("lonely", chain) == set() and ancestors("a@v1", chain) == set()


def test_a_cycle():
    with pytest.raises(ValueError) as error:
        ancestors("x", {"x": ("x",)})
    assert str(error.value) == "Lineage cycle through 'x'."
    cycle = {"a": ("b",), "b": ("c",), "c": ("a",)}
    with pytest.raises(ValueError) as error:
        ancestors("a", cycle)
    message = str(error.value)
    assert message.startswith("Lineage cycle through '")
    assert message[len("Lineage cycle through '"):-2] in cycle


def test_roots():
    graph = lineage_graph([("a@v1", "a@v2"), ("b@v1", "j@v1"), ("a@v2", "j@v1")])
    assert roots("j@v1", graph) == {"a@v1", "b@v1"}
    assert roots("a@v2", graph) == {"a@v1"}
    assert roots("a@v1", graph) == {"a@v1"} and roots("unknown", graph) == {"unknown"}


# --- check_lineage --------------------------------------------------------------------------------------------------------

def test_check_lineage():
    graph = lineage_graph([("a@v1", "a@v2"), ("b@v1", "j@v1"), ("a@v2", "j@v1")])
    assert check_lineage("j@v1", graph, {"a@v1", "b@v1"}) == []
    assert check_lineage("j@v1", graph, {"a@v1"}) == ["'b@v1' is not an uploaded table"]
    assert check_lineage("j@v1", graph, set()) == ["'a@v1' is not an uploaded table", "'b@v1' is not an uploaded table"]
    assert check_lineage("x", {"x": ("x",)}, {"x"}) == ["Lineage cycle through 'x'."]
    assert check_lineage("a@v1", {}, {"a@v1"}) == []


def test_inputs_are_not_mutated():
    ops = [_op(["a@v1"], "a@v2"), {"input_handles": ["a@v2", "b@v1"], "output_handle": "j@v1"}]
    ops_before = copy.deepcopy(ops)
    edges = edges_from_operations(ops)
    edges_before = list(edges)
    graph = lineage_graph(edges)
    graph_before = copy.deepcopy(graph)
    uploaded = {"a@v1", "b@v1"}
    ancestors("j@v1", graph), roots("j@v1", graph), check_lineage("j@v1", graph, uploaded)
    assert ops == ops_before and edges == edges_before and graph == graph_before and uploaded == {"a@v1", "b@v1"}


def test_a_long_chain():
    edges = [(f"t@v{i}", f"t@v{i + 1}") for i in range(1, 5000)]
    start = time.perf_counter()
    graph = lineage_graph(edges)
    assert len(ancestors("t@v5000", graph)) == 4999
    assert roots("t@v5000", graph) == {"t@v1"}
    assert check_lineage("t@v5000", graph, {"t@v1"}) == []
    assert time.perf_counter() - start < 2.0
