"""
The allowlist of state types for LangGraph's checkpoint serializer (M0.4, G4).

LangGraph warns that it will block unregistered types in a future version.
Once it does, every resume would hand nodes plain dicts and strings instead of
the dataclasses and enums they expect. Measured with langgraph 1.2.12: with an
explicit allowed_msgpack_modules, an unlisted type is already blocked today
(it comes back as its raw payload, with a "Blocked deserialization" warning and
no exception), so the list must be complete. It is built from the modules that
define state types, and tests/test_checkpoint_types.py checks it covers every
type that appears in checkpointed state.

This module must not import core.nodes or core.graph.
"""
from __future__ import annotations
import inspect
from dataclasses import is_dataclass
from enum import Enum

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

import core.dataset
import core.detection
import core.model_safe_view
import core.operation
import core.plan

STATE_TYPE_MODULES = (core.plan, core.dataset, core.operation, core.detection, core.model_safe_view)


def state_types() -> list[tuple[str, str]]:
    """(module name, class name) for every dataclass or Enum defined in STATE_TYPE_MODULES, sorted."""
    found = set()
    for module in STATE_TYPE_MODULES:
        for name, obj in vars(module).items():
            if not inspect.isclass(obj) or obj.__module__ != module.__name__:
                continue
            if is_dataclass(obj) or issubclass(obj, Enum):
                found.add((module.__name__, obj.__name__))
    return sorted(found)


ALLOWED_MSGPACK_TYPES: tuple[tuple[str, str], ...] = tuple(state_types())


def make_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=list(ALLOWED_MSGPACK_TYPES))


def make_checkpointer() -> InMemorySaver:
    return InMemorySaver(serde=make_serde())
