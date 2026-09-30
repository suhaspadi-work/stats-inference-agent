from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol
import uuid


class DatasetStore(Protocol):
    """
    Storage backend interface. v1 implements this with local disk; a real
    deployment could implement it with S3/GCS/etc. without touching any
    code that depends on Dataset or Operation.
    """
    def read(self, storage_key: str): ...
    def write(self, storage_key: str, data) -> None: ...
    def exists(self, storage_key: str) -> bool: ...


@dataclass(frozen=True)
class Dataset:
    """
    An immutable handle to one version of a dataset, scoped to a session/user
    so concurrent uploads never collide. Never holds the actual rows — only
    metadata and a storage_key a DatasetStore can resolve to real data.
    """
    session_id: str                # scopes this dataset to one user/session
    name: str                      # user-facing name, e.g. "customers"
    version: int                   # increments each time an operation is applied
    storage_key: str               # opaque key a DatasetStore resolves — NOT a bare local path
    parent_version: Optional[int]
    created_from_operation: Optional[str]  # None for the original upload
    original_filename: Optional[str] = None  # what the user actually uploaded, v1 only
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: int = 1

    @property
    def handle(self) -> str:
        """The string the agent/model refers to this dataset by, e.g. 'customers@v3'."""
        return f"{self.name}@v{self.version}"

    def next_version(self, new_storage_key: str, operation_id: str) -> "Dataset":
        """Produce the next immutable version, linked back to this one."""
        return Dataset(
            session_id=self.session_id,
            name=self.name,
            version=self.version + 1,
            storage_key=new_storage_key,
            parent_version=self.version,
            created_from_operation=operation_id,
        )


class LocalDiskStore:
    """
    v1 implementation of DatasetStore: storage_key is a relative path under
    a local data directory. Swappable later without touching Dataset itself.
    """
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, storage_key: str) -> Path:
        return self.base_dir / storage_key

    def read(self, storage_key: str):
        import pandas as pd
        return pd.read_csv(self._resolve(storage_key))

    def write(self, storage_key: str, data) -> None:
        resolved = self._resolve(storage_key)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(resolved, index=False)

    def exists(self, storage_key: str) -> bool:
        return self._resolve(storage_key).exists()