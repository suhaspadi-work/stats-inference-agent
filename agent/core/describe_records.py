"""
The audit record for the describe step (M1.2), the last step of stage 1. Pure: no graph, no model calls,
and no import of core.nodes. The run-or-skip choice itself is the stage plan's planner.step.describe record;
this one records that the step ran and what it produced. It carries counts only, never a chart.

  stage1.describe.execution   the finished dataset was described
"""
from __future__ import annotations

from core.decision_record import Candidate, DecisionRecord
from core.ingestion import HIGH_CARDINALITY_THRESHOLD

DESCRIBE_EXECUTION_POINT = "stage1.describe.execution"


def describe_execution_record(result: dict, step: dict) -> DecisionRecord:
    """result is the dict from DESCRIPTIVE.compute; step is the stage plan's describe step."""
    withheld = sorted(c["name"] for c in result["column_stats"]
                      if not c["is_numeric"] and (c.get("distinct_count") or 0) > HIGH_CARDINALITY_THRESHOLD)
    return DecisionRecord(
        decision_point=DESCRIBE_EXECUTION_POINT,
        chosen="ran",
        decided_by="code_rule",
        candidates=(
            Candidate(option="ran", eligible=True,
                      reason="chosen: the stage plan's describe step says run (" + step["state"] + ")"),
            Candidate(option="skipped", eligible=step["state"] != "planned",
                      reason="not chosen: describe is part of stage 1 and runs by default"),
        ),
        rationale="the finished dataset is described as the last step of stage 1",
        evidence={
            "row_count": result["row_count"],
            "column_count": len(result["column_stats"]),
            "chart_count": len(result["chart_data"]),
            "withheld_text_columns": withheld,
            "step_state": step["state"],
            "exploratory": step["exploratory"],
        },
    )
