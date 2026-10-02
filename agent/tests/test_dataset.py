from pathlib import Path
import pandas as pd
import pytest
from core.dataset import Dataset, LocalDiskStore


@pytest.fixture
def store(tmp_path):
    """A fresh, isolated LocalDiskStore per test — tmp_path is pytest's built-in temp directory."""
    return LocalDiskStore(base_dir=tmp_path)


def test_handle_format():
    d = Dataset(
        session_id="session-abc", name="customers", version=1,
        storage_key="session-abc/customers_v1.csv",
        parent_version=None, created_from_operation=None,
    )
    assert d.handle == "customers@v1"


def test_next_version_links_lineage():
    d1 = Dataset(
        session_id="session-abc", name="customers", version=1,
        storage_key="session-abc/customers_v1.csv",
        parent_version=None, created_from_operation=None,
    )
    d2 = d1.next_version(new_storage_key="session-abc/customers_v2.csv", operation_id="op-001-dedupe")

    assert d2.handle == "customers@v2"
    assert d2.parent_version == 1
    assert d2.created_from_operation == "op-001-dedupe"
    assert d2.session_id == d1.session_id


def test_dataset_is_immutable():
    d = Dataset(
        session_id="session-abc", name="customers", version=1,
        storage_key="session-abc/customers_v1.csv",
        parent_version=None, created_from_operation=None,
    )
    with pytest.raises(Exception):
        d.version = 99  # frozen dataclass must reject this


def test_store_round_trip(store):
    df = pd.DataFrame({"customer_id": [1, 2, 3], "name": ["Alice", "Bob", "Carol"]})
    store.write("session-abc/customers_v1.csv", df)

    read_back = store.read("session-abc/customers_v1.csv")
    assert len(read_back) == 3
    assert list(read_back.columns) == ["customer_id", "name"]


def test_store_creates_nested_session_directories(store):
    df = pd.DataFrame({"a": [1]})
    # This would fail if write() didn't create the session subdirectory
    store.write("some-new-session/data.csv", df)
    assert store.exists("some-new-session/data.csv")