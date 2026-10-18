"""
Lineage across tables (M2.1). Pure: no import of core.nodes, inputs are never mutated, and every walk is
iterative, so a graph of thousands of handles has no recursion-depth problem. An edge runs from an input handle
to the handle an operation produced; a table's roots must be uploaded tables.
"""
from __future__ import annotations


def _field(operation, key):
    return operation.get(key) if isinstance(operation, dict) else getattr(operation, key, None)


def edges_from_operations(operations) -> list:
    """(input_handle, output_handle) for every operation that produced an output; operations are Operation
    objects or dicts with input_handles and output_handle (the evidence of an applied execution record)."""
    edges = []
    for operation in operations:
        output = _field(operation, "output_handle")
        if not output:
            continue
        for handle in _field(operation, "input_handles") or ():
            edges.append((handle, output))
    return edges


def lineage_graph(edges) -> dict:
    """{output_handle: its input handles in first-seen order}."""
    graph = {}
    for source, target in edges:
        inputs = graph.setdefault(target, [])
        if source not in inputs:
            inputs.append(source)
    return {target: tuple(inputs) for target, inputs in graph.items()}


def ancestors(handle, graph) -> set:
    """Every handle reachable upwards from handle; ValueError on a cycle."""
    found, on_path = set(), set()
    stack = [(handle, iter(graph.get(handle, ())))]
    on_path.add(handle)
    while stack:
        node, children = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            on_path.discard(node)
            continue
        if child in on_path:
            raise ValueError(f"Lineage cycle through '{child}'.")
        if child in found:
            continue
        found.add(child)
        on_path.add(child)
        stack.append((child, iter(graph.get(child, ()))))
    return found


def roots(handle, graph) -> set:
    """The ancestors of handle that have no parent in the graph, or {handle} when it has none."""
    upwards = ancestors(handle, graph)
    if not upwards:
        return {handle}
    return {h for h in upwards if not graph.get(h)}


def check_lineage(handle, graph, uploaded_handles) -> list:
    """Problem strings: a root that is not an uploaded table, or a cycle; [] when the lineage is sound."""
    try:
        found = roots(handle, graph)
    except ValueError as e:
        return [str(e)]
    uploaded = set(uploaded_handles)
    return [f"'{h}' is not an uploaded table" for h in sorted(found) if h not in uploaded]
