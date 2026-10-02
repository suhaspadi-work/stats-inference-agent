import pandas as pd
import pytest
from core.dataset import LocalDiskStore
from core.operation import ApprovalStatus, WranglingRationale
from core.wrangling import (
    propose_cast, execute_cast,
    propose_rename, execute_rename,
    propose_filter, execute_filter,
    propose_dedupe, execute_dedupe,
)
from core.ingestion import load_dataset
from core.operation import Operation


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


@pytest.fixture
def dataset(tmp_path, store):
    df = pd.DataFrame({
        "customer_id": [1, 2, 3, 3, 4],
        "age_str": ["25", "30", "35", "35", "40"],
        "region": ["East", "West", "East", "East", None],
    })
    path = tmp_path / "customers.csv"
    df.to_csv(path, index=False)
    return load_dataset(path, session_id="session-abc", name="customers", store=store)


def simple_rationale(text="test rationale"):
    return WranglingRationale(rationale=text, alternatives=(), evidence={})


def test_cast_column_type(dataset, store):
    op = propose_cast(dataset, store, column="age_str", to_type="int64", rationale=simple_rationale())
    approved = op.approve()
    new_dataset, finished = execute_cast(dataset, store, approved)

    assert finished.approval_status == ApprovalStatus.APPROVED
    assert new_dataset.version == 2
    df = store.read(new_dataset.storage_key)
    assert df["age_str"].dtype == "int64"


def test_cast_execution_requires_approval_first(dataset, store):
    op = propose_cast(dataset, store, column="age_str", to_type="int64", rationale=simple_rationale())
    with pytest.raises(ValueError):
        execute_cast(dataset, store, op)  # still PENDING -- should be blocked


def test_cast_records_failure_on_bad_conversion(dataset, store):
    op = propose_cast(dataset, store, column="region", to_type="int64", rationale=simple_rationale())
    approved = op.approve()
    same_dataset, finished = execute_cast(dataset, store, approved)

    assert finished.approval_status == ApprovalStatus.FAILED
    assert "Cast failed" in finished.failure_reason
    assert same_dataset.version == dataset.version  # no new version on failure


def test_rename_columns(dataset, store):
    op = propose_rename(dataset, store, mapping={"age_str": "age"}, rationale=simple_rationale())
    approved = op.approve()
    new_dataset, finished = execute_rename(dataset, store, approved)

    df = store.read(new_dataset.storage_key)
    assert "age" in df.columns
    assert "age_str" not in df.columns


def test_rename_fails_cleanly_on_missing_column(dataset, store):
    op = propose_rename(dataset, store, mapping={"nonexistent": "x"}, rationale=simple_rationale())
    approved = op.approve()
    _, finished = execute_rename(dataset, store, approved)
    assert finished.approval_status == ApprovalStatus.FAILED


def test_filter_rejects_unsupported_condition(dataset, store):
    with pytest.raises(ValueError):
        propose_filter(dataset, store, column="region", condition="contains", value="E", rationale=simple_rationale())


def test_filter_equals_with_real_rationale(dataset, store):
    from core.operation import AlternativeApproachConsidered

    rationale = WranglingRationale(
        rationale="Restricting to East region only, since the analysis question "
                  "specifically asks about East-region customer behavior.",
        alternatives=(
            AlternativeApproachConsidered(
                approach="keep_all_regions_and_add_region_as_covariate",
                rejected_because="the request scoped the question to East specifically, "
                                  "not a cross-region comparison",
            ),
        ),
        evidence={"total_rows_before_filter": 5, "east_region_rows": 3},
    )
    op = propose_filter(dataset, store, column="region", condition="equals", value="East", rationale=rationale)
    approved = op.approve()
    new_dataset, finished = execute_filter(dataset, store, approved)

    df = store.read(new_dataset.storage_key)
    assert len(df) == 3
    assert finished.impact.rows_before == 5
    assert finished.impact.rows_after == 3

    # The actual thing this test is for: the rationale and its alternative survive intact
    assert finished.rationale.rationale.startswith("Restricting to East region")
    assert len(finished.rationale.alternatives) == 1
    assert finished.rationale.alternatives[0].approach == "keep_all_regions_and_add_region_as_covariate"
    assert finished.rationale.evidence["east_region_rows"] == 3


def test_filter_not_null(dataset, store):
    op = propose_filter(dataset, store, column="region", condition="not_null", value=None, rationale=simple_rationale())
    approved = op.approve()
    new_dataset, _ = execute_filter(dataset, store, approved)
    df = store.read(new_dataset.storage_key)
    assert len(df) == 4  # drops the one None region


def test_dedupe_on_subset_with_real_rationale(dataset, store):
    from core.operation import AlternativeApproachConsidered

    rationale = WranglingRationale(
        rationale="customer_id=3 appears twice with identical values across all columns; "
                  "this looks like a duplicate submission, not two distinct customers or events.",
        alternatives=(
            AlternativeApproachConsidered(
                approach="keep_both_and_flag_as_potential_duplicate",
                rejected_because="the duplicate rows are byte-for-byte identical, so retaining "
                                  "both would double-count this customer with no offsetting information gained",
            ),
        ),
        evidence={"duplicate_customer_ids": [3], "exact_duplicate_row_count": 1},
    )
    op = propose_dedupe(dataset, store, subset=["customer_id"], rationale=rationale)
    approved = op.approve()
    new_dataset, finished = execute_dedupe(dataset, store, approved)

    df = store.read(new_dataset.storage_key)
    assert len(df) == 4
    assert finished.impact.rows_before == 5
    assert finished.impact.rows_after == 4

    # Confirm the reasoning is actually attached and inspectable, not just the mechanical result
    assert "duplicate submission" in finished.rationale.rationale
    assert finished.rationale.alternatives[0].approach == "keep_both_and_flag_as_potential_duplicate"
    assert finished.rationale.evidence["duplicate_customer_ids"] == [3]


def test_dedupe_fails_cleanly_on_missing_subset_column(dataset, store):
    op = propose_dedupe(dataset, store, subset=["nonexistent"], rationale=simple_rationale())
    approved = op.approve()
    _, finished = execute_dedupe(dataset, store, approved)
    assert finished.approval_status == ApprovalStatus.FAILED


def test_rejected_operation_is_never_executed(dataset, store):
    op = propose_filter(dataset, store, column="region", condition="equals", value="East", rationale=simple_rationale())
    rejected = op.reject(reason="Not sure this filter is appropriate")
    with pytest.raises(ValueError):
        execute_filter(dataset, store, rejected)  # REJECTED, not APPROVED -- must be blocked

def test_filter_not_null_works_without_a_value_param(dataset, store):
    """
    Regression test: not_null/is_null conditions don't need a 'value' key
    at all -- execute_filter must not crash on its absence. Found via the
    Phase 8 evaluation battery, where the model correctly proposed params
    without 'value' for a not_null filter.
    """
    op = Operation(
        op_type="filter", session_id=dataset.session_id,
        input_handles=(dataset.handle,), params={"column": "region", "condition": "not_null"},
        approval_status=ApprovalStatus.PENDING, rationale=simple_rationale(),
    )
    approved = op.approve()
    new_dataset, finished = execute_filter(dataset, store, approved)
    assert finished.approval_status == ApprovalStatus.APPROVED