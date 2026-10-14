"""
What the report generators may see (M0.5). Chart data holds raw values for the UI; it is removed
before a plan or result is handed to a model, and assert_model_safe then checks both.

Pure: imports only core.plan and core.model_safe_view, never core.nodes. Inputs are never mutated.
"""
from __future__ import annotations
import copy
from dataclasses import replace
from typing import Any

from core.model_safe_view import assert_model_safe
from core.plan import AnalysisPlan


def _without_chart_data(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _without_chart_data(v) for k, v in value.items() if k != "chart_data"}
    if isinstance(value, list):
        return [_without_chart_data(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_without_chart_data(v) for v in value)
    return copy.deepcopy(value)


def result_for_model(result: Any) -> Any:
    """A deep copy with every key named "chart_data" removed at any depth."""
    return _without_chart_data(result)


def plan_for_model(plan: AnalysisPlan) -> AnalysisPlan:
    """The same plan with chart_data emptied on every method decision."""
    return replace(plan, method_decisions=tuple(replace(d, chart_data=()) for d in plan.method_decisions))


def prepare_for_model(plan: AnalysisPlan, result: Any) -> tuple[AnalysisPlan, Any]:
    safe_plan, safe_result = plan_for_model(plan), result_for_model(result)
    assert_model_safe(safe_plan, "plan")
    assert_model_safe(safe_result, "result")
    return safe_plan, safe_result
