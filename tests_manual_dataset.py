from pathlib import Path
from core.dataset import Dataset, LocalDiskStore
import pandas as pd

store = LocalDiskStore(base_dir=Path("local_data"))

# Simulate an original upload
df = pd.DataFrame({"customer_id": [1, 2, 3], "name": ["Alice", "Bob", "Carol"]})
store.write("session-abc/customers_v1.csv", df)

d1 = Dataset(
    session_id="session-abc",
    name="customers",
    version=1,
    storage_key="session-abc/customers_v1.csv",
    parent_version=None,
    created_from_operation=None,
    original_filename="customers_upload.csv",
)
print("v1 handle:", d1.handle)
print("v1 storage_key:", d1.storage_key)

# Simulate applying an operation (e.g. a dedupe) to produce v2
deduped = df.drop_duplicates()
store.write("session-abc/customers_v2.csv", deduped)
d2 = d1.next_version(new_storage_key="session-abc/customers_v2.csv", operation_id="op-001-dedupe")

print("v2 handle:", d2.handle)
print("v2 parent_version:", d2.parent_version)
print("v2 session_id matches d1:", d2.session_id == d1.session_id)

# Confirm the store can actually read back what was written
read_back = store.read(d2.storage_key)
print("read back v2 rows:", len(read_back))

# Confirm immutability: uncommenting this should raise an error
# d1.version = 99