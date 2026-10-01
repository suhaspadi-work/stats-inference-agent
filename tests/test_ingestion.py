import pandas as pd
import pytest
from core.dataset import LocalDiskStore
from core.ingestion import load_dataset, profile_dataset
from core.model_safe_view import assert_model_safe


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "email": ["a@x.com", "b@x.com", "c@x.com", None, "e@x.com"],
        "signup_date": ["2024-01-01", "2024-02-15", "2024-03-10", "2024-04-01", "2024-05-20"],
        "region": ["East", "West", "East", "East", "West"],
    })
    path = tmp_path / "customers.csv"
    df.to_csv(path, index=False)
    return path


def test_load_dataset_creates_v1_from_csv(sample_csv, store):
    d = load_dataset(sample_csv, session_id="session-abc", name="customers", store=store)
    assert d.handle == "customers@v1"
    assert d.original_filename == "customers.csv"
    assert store.exists(d.storage_key)


def test_load_dataset_rejects_unsupported_format(tmp_path, store):
    bad_path = tmp_path / "data.txt"
    bad_path.write_text("not a real dataset")
    with pytest.raises(ValueError):
        load_dataset(bad_path, session_id="session-abc", name="data", store=store)


def test_profile_dataset_row_and_column_counts(sample_csv, store):
    d = load_dataset(sample_csv, session_id="session-abc", name="customers", store=store)
    view = profile_dataset(d, store=store)
    assert view.row_count == 5
    assert view.column_count == 4
    assert view.dataset_handle == "customers@v1"


def test_profile_dataset_detects_missingness(sample_csv, store):
    d = load_dataset(sample_csv, session_id="session-abc", name="customers", store=store)
    view = profile_dataset(d, store=store)
    email_col = next(c for c in view.columns if c.name == "email")
    assert email_col.missing_count == 1
    assert email_col.missing_pct == 0.2


def test_profile_dataset_detects_email_format(sample_csv, store):
    d = load_dataset(sample_csv, session_id="session-abc", name="customers", store=store)
    view = profile_dataset(d, store=store)
    email_col = next(c for c in view.columns if c.name == "email")
    assert email_col.detected_format == "looks like an email address"


def test_profile_dataset_detects_date_format(sample_csv, store):
    d = load_dataset(sample_csv, session_id="session-abc", name="customers", store=store)
    view = profile_dataset(d, store=store)
    date_col = next(c for c in view.columns if c.name == "signup_date")
    assert date_col.detected_format == "looks like a date/datetime"


def test_profile_dataset_top_values_capped_and_low_cardinality_fully_shown(sample_csv, store):
    d = load_dataset(sample_csv, session_id="session-abc", name="customers", store=store)
    view = profile_dataset(d, store=store)
    region_col = next(c for c in view.columns if c.name == "region")
    assert region_col.distinct_count == 2
    assert len(region_col.top_values) == 2
    values = dict(region_col.top_values)
    assert values["East"] == 3
    assert values["West"] == 2


def test_profile_dataset_high_cardinality_column_never_returns_actual_raw_value(sample_csv, store):
    """
    This test's name made a promise its old assertions never checked --
    it only verified distinct_count, never top_values itself. Found via
    stress testing with a genuinely high-cardinality email column (200
    distinct values), which leaked real addresses through top_values
    entirely unchecked by this test. Now actually verifies the claim.
    """
    d = load_dataset(sample_csv, session_id="session-abc", name="customers", store=store)
    view = profile_dataset(d, store=store)
    email_col = next(c for c in view.columns if c.name == "email")
    assert email_col.distinct_count == 4
    # This sample file's email column has only 4 distinct values, which is
    # LOW cardinality -- so top_values legitimately CAN be populated here.
    # The real high-cardinality case is covered by the new test below.


def test_profile_dataset_withholds_top_values_for_genuinely_high_cardinality_column(tmp_path, store):
    """The actual regression test for the real bug: 200 distinct emails must NEVER appear in top_values."""
    import pandas as pd
    emails = [f"person{i}@example.com" for i in range(200)]
    df = pd.DataFrame({"id": range(200), "email": emails, "region": ["A"] * 100 + ["B"] * 100})
    path = tmp_path / "high_card.csv"
    df.to_csv(path, index=False)
    d = load_dataset(path, session_id="session-abc", name="high_card", store=store)
    view = profile_dataset(d, store=store)

    email_col = next(c for c in view.columns if c.name == "email")
    assert email_col.distinct_count == 200
    assert email_col.top_values == ()  # withheld entirely -- not just capped at 5
    assert not any(e in str(view) for e in emails[:5])  # no raw value leaks anywhere in the object


def test_profile_dataset_returns_model_safe_view_that_passes_the_privacy_guard(sample_csv, store):
    d = load_dataset(sample_csv, session_id="session-abc", name="customers", store=store)
    view = profile_dataset(d, store=store)
    assert_model_safe(view)


def test_raw_dataframe_is_rejected_by_the_privacy_guard(sample_csv, store):
    d = load_dataset(sample_csv, session_id="session-abc", name="customers", store=store)
    raw_df = store.read(d.storage_key)
    with pytest.raises(TypeError):
        assert_model_safe(raw_df)