"""
M2.2: the data-quality snapshot of the analysis table (core/quality.py) on hand-made frames: counts, missingness,
constant columns, Tukey outlier fences and skew, computed independently here with numpy, pandas and scipy. The
snapshot holds numbers and column names only, never a cell value or a raw minimum or maximum.
"""
import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from core.dataset import LocalDiskStore
from core.ingestion import load_dataset
from core.model_safe_view import ColumnSummary, ModelSafeView
from core.quality import OUTLIER_IQR_MULTIPLIER, SKEW_THRESHOLD, profile_numbers, quality_snapshot

TEXT_CANARIES = ("QUALITY-CANARY-ONE", "QUALITY-CANARY-TWO")
MAX_CANARY = 98765.4321  # the maximum of the outlier column: it must never appear in a snapshot


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _dataset(df, tmp_path, store, name="quality"):
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    return load_dataset(path, session_id="s", name=name, store=store)


def _frame():
    rng = np.random.default_rng(5)
    n = 100
    out = rng.normal(50, 5, n)
    out[3], out[7], out[11] = 5.0, 150.0, MAX_CANARY  # one far below, two far above
    missing = rng.normal(10, 1, n)
    missing[:10] = np.nan  # 10% missing
    df = pd.DataFrame({
        "id": range(n),
        "label": [TEXT_CANARIES[i % 2] for i in range(n)],
        "constant": ["same"] * n,
        "out": out,
        "missing": missing,
        "skewed": rng.exponential(1.0, n) ** 3,
        "even": np.linspace(0, 1, n),
    })
    return pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)  # 2 exact duplicate rows


def _fences(values):
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return q1 - OUTLIER_IQR_MULTIPLIER * iqr, q3 + OUTLIER_IQR_MULTIPLIER * iqr


def test_every_field(tmp_path, store):
    df = _frame()
    snapshot = quality_snapshot(_dataset(df, tmp_path, store), store, "main")
    assert (snapshot["table"], snapshot["handle"]) == ("main", "quality@v1")
    assert (snapshot["rows"], snapshot["columns"], snapshot["duplicate_rows"]) == (102, 7, 2)
    assert snapshot["missing"] == [{"column": "missing", "missing": int(df["missing"].isna().sum()),
                                    "missing_pct": float(df["missing"].isna().mean())}]
    assert snapshot["columns_with_missing"] == 1
    assert snapshot["constant_columns"] == ["constant"]

    expected = []
    for column in ("id", "out", "missing", "skewed", "even"):
        values = df[column].dropna().to_numpy()
        lower, upper = _fences(values)
        below, above = int((values < lower).sum()), int((values > upper).sum())
        if below or above:
            expected.append({"column": column, "lower_fence": float(f"{lower:.6g}"), "upper_fence": float(f"{upper:.6g}"),
                             "below": below, "above": above})
    assert snapshot["outliers"] == expected and snapshot["columns_with_outliers"] == len(expected)
    out = next(o for o in snapshot["outliers"] if o["column"] == "out")
    assert out["below"] >= 1 and out["above"] >= 2

    skews = {c: float(scipy_stats.skew(df[c].dropna())) for c in ("id", "out", "missing", "skewed", "even")}
    assert snapshot["skewed_columns"] == [{"column": c, "skewness": round(v, 3)} for c, v in skews.items()
                                          if abs(v) > SKEW_THRESHOLD]
    assert "skewed" in [c["column"] for c in snapshot["skewed_columns"]]
    assert "even" not in [c["column"] for c in snapshot["skewed_columns"]]


def test_a_clean_frame(tmp_path, store):
    df = pd.DataFrame({"a": np.linspace(0, 1, 50), "b": ["x", "y"] * 25})
    snapshot = quality_snapshot(_dataset(df, tmp_path, store), store)
    assert snapshot["table"] == "quality"
    assert (snapshot["duplicate_rows"], snapshot["columns_with_missing"], snapshot["columns_with_outliers"]) == (0, 0, 0)
    assert snapshot["missing"] == snapshot["constant_columns"] == snapshot["outliers"] == snapshot["skewed_columns"] == []


def test_edge_frames_do_not_raise(tmp_path, store):
    empty_column = pd.DataFrame({"a": [1.0, 2.0, 3.0], "gone": [np.nan, np.nan, np.nan]})
    snapshot = quality_snapshot(_dataset(empty_column, tmp_path, store, "gone"), store)
    assert snapshot["missing"][0] == {"column": "gone", "missing": 3, "missing_pct": 1.0}
    assert "gone" in snapshot["constant_columns"]
    one_row = pd.DataFrame({"a": [1.0], "b": ["x"]})
    snapshot = quality_snapshot(_dataset(one_row, tmp_path, store, "one"), store)
    assert (snapshot["rows"], snapshot["duplicate_rows"], snapshot["outliers"]) == (1, 0, [])


def test_no_cell_value_and_no_raw_extreme(tmp_path, store):
    df = _frame()
    ds = _dataset(df, tmp_path, store)
    before = store.read(ds.storage_key)
    snapshot = quality_snapshot(ds, store)
    text = json.dumps(snapshot)
    for canary in TEXT_CANARIES:
        assert canary not in text
    for form in (str(MAX_CANARY), f"{MAX_CANARY:.6g}", f"{MAX_CANARY:.2f}", "98765"):
        assert form not in text
    for column in ("out", "missing", "skewed", "even"):
        for extreme in (df[column].min(), df[column].max()):
            form = f"{extreme:.6g}"
            if len(form) >= 5:  # a short form such as "5" is a substring of any number; the canary covers the max
                assert form not in text, column
    assert store.read(ds.storage_key).equals(before) and ds.version == 1


def _view(name, rows, columns):
    """columns: [(name, dtype, distinct, missing)]"""
    return ModelSafeView(dataset_handle=f"{name}@v1", row_count=rows, column_count=len(columns),
                         columns=tuple(ColumnSummary(name=c, dtype=d, missing_count=m, missing_pct=m / rows,
                                                     distinct_count=k) for c, d, k, m in columns))


def test_profile_numbers():
    profiles = {"main": _view("main", 10, [("a", "int64", 10, 0)]),
                "zeta": _view("zeta", 5, [("a", "int64", 5, 1), ("b", "str", 2, 0)]),
                "alpha": _view("alpha", 3, [("a", "int64", 3, 0)])}
    assert profile_numbers(profiles, "main") == [
        {"name": "zeta", "rows": 5, "columns": 2, "columns_with_missing": 1},
        {"name": "alpha", "rows": 3, "columns": 1, "columns_with_missing": 0}]
    assert profile_numbers({}, "main") == [] and profile_numbers(None, "main") == []
