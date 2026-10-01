import pandas as pd
import pytest
from core.dataset import LocalDiskStore
from core.ingestion import load_dataset, profile_dataset
from core.detection import detect_duplicate_rows, detect_high_missingness, detect_all_issues


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def make_dataset(df, tmp_path, store, name="data"):
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    return load_dataset(path, session_id="session-abc", name=name, store=store)


def test_detects_no_issues_on_clean_data(tmp_path, store):
    df = pd.DataFrame({"id": [1, 2, 3], "region": ["East", "West", "East"], "spend": [10.0, 20.0, 30.0]})
    ds = make_dataset(df, tmp_path, store)
    profile = profile_dataset(ds, store=store)

    assert detect_duplicate_rows(ds, store) is None
    assert detect_high_missingness(profile) == []
    assert detect_all_issues(ds, profile, store) == []


def test_detects_duplicate_rows(tmp_path, store):
    df = pd.DataFrame({
        "id": [1, 2, 2, 3], "region": ["East", "West", "West", "East"], "spend": [10.0, 20.0, 20.0, 30.0],
    })
    ds = make_dataset(df, tmp_path, store)

    issue = detect_duplicate_rows(ds, store)
    assert issue is not None
    assert issue.issue_type == "duplicate_rows"
    assert issue.evidence["duplicate_row_count"] == 1


def test_detects_high_missingness(tmp_path, store):
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "region": ["East", None, None, None, "West", "East", "West", "East", "West", "East"],  # 30% missing
        "spend": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],  # 0% missing
    })
    ds = make_dataset(df, tmp_path, store)
    profile = profile_dataset(ds, store=store)

    issues = detect_high_missingness(profile)
    assert len(issues) == 1
    assert issues[0].affected_columns == ("region",)
    assert issues[0].evidence["missing_pct"] == 0.3


def test_missingness_below_threshold_is_not_flagged(tmp_path, store):
    df = pd.DataFrame({
        "id": list(range(1, 101)),
        "region": ["East"] * 97 + [None, None, None],  # exactly 3% missing -- below 5% threshold
    })
    ds = make_dataset(df, tmp_path, store)
    profile = profile_dataset(ds, store=store)

    assert detect_high_missingness(profile) == []


def test_detect_all_issues_combines_both_checks(tmp_path, store):
    df = pd.DataFrame({
        "id": [1, 2, 2, 3, 4, 5, 6, 7, 8, 9],
        "region": ["East", "West", "West", None, None, None, "West", "East", "West", "East"],  # dup + missing
        "spend": [10.0] * 10,
    })
    ds = make_dataset(df, tmp_path, store)
    profile = profile_dataset(ds, store=store)

    issues = detect_all_issues(ds, profile, store)
    issue_types = {i.issue_type for i in issues}
    assert issue_types == {"duplicate_rows", "high_missingness"}