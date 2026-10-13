"""
Writes each node's new decision records to the event log as the node returns
(M0.2 step 13, D27, G25). One file per thread: <event_log_dir>/<thread id>/events.jsonl.

The wrapper takes (state, config): LangGraph injects the run config by that
parameter name, which is where the thread id comes from. functools.wraps is
deliberately not used: it sets __wrapped__, so inspect.signature would report
the inner (state) signature and LangGraph would stop passing config.
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable

from core.decision_points import ALLOWED_RECORD_CHART_TYPES
from core.event_log import EventLog, events_path


def _check_charts(records: list[dict]) -> None:
    for record in records:
        for chart in record.get("chart_data") or []:
            chart_type = chart.get("chart_type") if isinstance(chart, dict) else None
            if chart_type not in ALLOWED_RECORD_CHART_TYPES:
                raise ValueError(
                    f"Decision record {record.get('decision_point')!r} carries a {chart_type!r} chart; "
                    f"only {ALLOWED_RECORD_CHART_TYPES} may be written (decision-time charts hold raw values, M0.5)."
                )


def with_event_log(node_name: str, node_fn: Callable, event_log_dir) -> Callable:
    base_dir = Path(event_log_dir)

    def wrapped(state, config):
        update = node_fn(state)  # exceptions propagate untouched; a paused node raises before this returns
        records = update.get("decision_log") if isinstance(update, dict) else None
        if not records:
            return update

        thread_id = ((config or {}).get("configurable") or {}).get("thread_id")
        if thread_id is None:
            raise ValueError(
                f"Node '{node_name}' wrote decision records but the run config has no "
                f"configurable.thread_id, so they cannot be written to the event log."
            )
        _check_charts(records)  # before anything is written

        path = events_path(base_dir, thread_id)
        meta = {"node": node_name, "thread_id": thread_id, "session_id": state.get("session_id")}
        try:
            log = EventLog(log_path=path)  # created only now, so silent nodes create no directory
            for record in records:
                log.append("decision", record, meta=meta)
        except OSError as e:
            raise RuntimeError(f"Could not write decision records for node '{node_name}' to {path}: {e}") from e
        return update

    wrapped.__name__ = getattr(node_fn, "__name__", node_name)
    wrapped.__qualname__ = getattr(node_fn, "__qualname__", node_name)
    wrapped.__doc__ = getattr(node_fn, "__doc__", None)
    return wrapped
