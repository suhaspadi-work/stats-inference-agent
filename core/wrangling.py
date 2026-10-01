from __future__ import annotations
import pandas as pd

from core.dataset import Dataset, DatasetStore
from core.operation import (
    Operation, OperationImpact, ApprovalStatus, WranglingRationale,
)


def _propose(op_type: str, dataset: Dataset, params: dict, rationale: WranglingRationale) -> Operation:
    """Shared helper: every wrangling function proposes an Operation the same way."""
    return Operation(
        op_type=op_type,
        session_id=dataset.session_id,
        input_handles=(dataset.handle,),
        params=params,
        approval_status=ApprovalStatus.PENDING,
        rationale=rationale,
    )


def _execute(dataset: Dataset, store: DatasetStore, operation: Operation, new_df: pd.DataFrame, old_df: pd.DataFrame) -> tuple[Dataset, Operation]:
    """
    Shared helper: write the new data, produce the next Dataset version, and
    record the impact on the Operation. Only called after approval, per the
    Operation lifecycle from Phase 1.
    """
    new_storage_key = f"{dataset.session_id}/{dataset.name}_v{dataset.version + 1}.csv"
    store.write(new_storage_key, new_df)
    new_dataset = dataset.next_version(new_storage_key=new_storage_key, operation_id=operation.id)

    impact = OperationImpact(
        rows_before=len(old_df), rows_after=len(new_df),
        columns_before=len(old_df.columns), columns_after=len(new_df.columns),
    )
    finished_operation = operation.with_result(impact=impact, output_handle=new_dataset.handle)
    return new_dataset, finished_operation


def propose_cast(dataset: Dataset, store: DatasetStore, column: str, to_type: str, rationale: WranglingRationale) -> Operation:
    """Propose casting one column to a new dtype. No execution yet — approval happens first."""
    return _propose("cast", dataset, {"column": column, "to_type": to_type}, rationale)


def execute_cast(dataset: Dataset, store: DatasetStore, operation: Operation) -> tuple[Dataset, Operation]:
    if operation.approval_status != ApprovalStatus.APPROVED:
        raise ValueError("Operation must be APPROVED before execution.")
    df = store.read(dataset.storage_key)
    column, to_type = operation.params["column"], operation.params["to_type"]
    new_df = df.copy()
    try:
        new_df[column] = new_df[column].astype(to_type)
    except (ValueError, TypeError) as e:
        return dataset, operation.mark_failed(f"Cast failed: {e}")
    return _execute(dataset, store, operation, new_df, df)


def propose_rename(dataset: Dataset, store: DatasetStore, mapping: dict[str, str], rationale: WranglingRationale) -> Operation:
    """Propose renaming one or more columns. Low-risk, but still logged and reviewable."""
    return _propose("rename", dataset, {"mapping": mapping}, rationale)


def execute_rename(dataset: Dataset, store: DatasetStore, operation: Operation) -> tuple[Dataset, Operation]:
    if operation.approval_status != ApprovalStatus.APPROVED:
        raise ValueError("Operation must be APPROVED before execution.")
    df = store.read(dataset.storage_key)
    mapping = operation.params["mapping"]
    missing = [c for c in mapping if c not in df.columns]
    if missing:
        return dataset, operation.mark_failed(f"Cannot rename — columns not found: {missing}")
    new_df = df.rename(columns=mapping)
    return _execute(dataset, store, operation, new_df, df)


def propose_filter(dataset: Dataset, store: DatasetStore, column: str, condition: str, value, rationale: WranglingRationale) -> Operation:
    """
    Propose filtering rows. `condition` is one of a small, safe whitelist —
    never an arbitrary expression string — per the declarative-operations
    design decision (no free-form code from the model).
    """
    allowed = {"equals", "not_equals", "greater_than", "less_than", "is_null", "not_null"}
    if condition not in allowed:
        raise ValueError(f"Unsupported filter condition: {condition}. Must be one of {allowed}.")
    return _propose("filter", dataset, {"column": column, "condition": condition, "value": value}, rationale)


def execute_filter(dataset: Dataset, store: DatasetStore, operation: Operation) -> tuple[Dataset, Operation]:
    if operation.approval_status != ApprovalStatus.APPROVED:
        raise ValueError("Operation must be APPROVED before execution.")
    df = store.read(dataset.storage_key)
    column = operation.params["column"]
    condition = operation.params["condition"]
    value = operation.params.get("value")

    if column not in df.columns:
        return dataset, operation.mark_failed(f"Column not found: {column}")

    if condition == "equals":
        new_df = df[df[column] == value]
    elif condition == "not_equals":
        new_df = df[df[column] != value]
    elif condition == "greater_than":
        new_df = df[df[column] > value]
    elif condition == "less_than":
        new_df = df[df[column] < value]
    elif condition == "is_null":
        new_df = df[df[column].isna()]
    elif condition == "not_null":
        new_df = df[df[column].notna()]

    return _execute(dataset, store, operation, new_df, df)


def propose_dedupe(dataset: Dataset, store: DatasetStore, subset: list[str] | None, rationale: WranglingRationale) -> Operation:
    """Propose dropping exact duplicate rows, optionally scoped to a subset of columns."""
    return _propose("dedupe", dataset, {"subset": subset}, rationale)


def execute_dedupe(dataset: Dataset, store: DatasetStore, operation: Operation) -> tuple[Dataset, Operation]:
    if operation.approval_status != ApprovalStatus.APPROVED:
        raise ValueError("Operation must be APPROVED before execution.")
    df = store.read(dataset.storage_key)
    subset = operation.params["subset"]

    if subset:
        missing = [c for c in subset if c not in df.columns]
        if missing:
            return dataset, operation.mark_failed(f"Cannot dedupe — columns not found: {missing}")

    new_df = df.drop_duplicates(subset=subset)
    return _execute(dataset, store, operation, new_df, df)