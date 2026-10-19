"""
M2.3a step 2: the two text detectors of core/detection.py. detect_inconsistent_text flags a text column (at most
HIGH_CARDINALITY_THRESHOLD distinct values) whose levels merge after trimming spaces and ignoring case, or whose
values carry stray spaces; detect_numeric_as_text flags a text column whose values are numbers (at least
NUMERIC_TEXT_PARSE_RATE of them, never one with leading zeros). An issue holds counts only, never a level or a value.
Expected numbers are computed independently with pandas.
"""
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

import core.detection as detection
from core.dataset import LocalDiskStore
from core.detection import (
    detect_all_issues, detect_duplicate_rows, detect_high_missingness, detect_inconsistent_text, detect_numeric_as_text,
)
from core.ingestion import HIGH_CARDINALITY_THRESHOLD, load_dataset, profile_dataset
from core.thresholds import NUMERIC_TEXT_PARSE_RATE


class CountingStore(LocalDiskStore):
    def __init__(self, base_dir):
        super().__init__(base_dir)
        self.reads = []

    def read(self, storage_key):
        self.reads.append(storage_key)
        return super().read(storage_key)


@pytest.fixture
def store(tmp_path):
    return CountingStore(tmp_path / "store")


def _dataset(df, tmp_path, store, name="data"):
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    return load_dataset(path, session_id="s", name=name, store=store)


def _expected_text_evidence(series):
    """The five counts of an inconsistent_text issue, computed with pandas string methods."""
    values = series.dropna().astype(str)
    trimmed = values.str.replace(r"\s+", " ", regex=True).str.strip()
    lowered = trimmed.str.lower()
    return {"levels_before": values.nunique(), "levels_after_trim": trimmed.nunique(), "levels_after": lowered.nunique(),
            "cells_changed_trim": int((trimmed != values).sum()), "cells_changed_full": int((lowered != values).sum())}


@pytest.fixture
def trim_calls(monkeypatch):
    """Counts the values the inconsistent-text detector trims, to show a column was or was not examined."""
    calls = []
    real = detection._trim
    monkeypatch.setattr(detection, "_trim", lambda value: (calls.append(value), real(value))[1])
    return calls


# --- detect_inconsistent_text ----------------------------------------------------------------------------------------

def test_mergeable_levels():
    df = pd.DataFrame({"region": ["East", "east ", "EAST", "West"] * 5, "x": range(20)})
    (issue,) = detect_inconsistent_text(df)
    expected = _expected_text_evidence(df["region"])
    assert expected == {"levels_before": 4, "levels_after_trim": 4, "levels_after": 2, "cells_changed_trim": 5,
                        "cells_changed_full": 20}  # by hand: only "east " has a space; every value changes case or space
    assert (issue.issue_type, issue.affected_columns, issue.evidence) == ("inconsistent_text", ("region",), expected)
    assert issue.description == ("Column 'region' has 4 distinct values that reduce to 2 after trimming spaces and "
                                 "ignoring case.")


def test_stray_spaces_without_a_merge():
    df = pd.DataFrame({"c": [" a", "b ", "c"] * 4})
    (issue,) = detect_inconsistent_text(df)
    n = int((df["c"].str.strip() != df["c"]).sum())
    assert n == 8 and issue.evidence == _expected_text_evidence(df["c"])
    assert issue.evidence["levels_after"] == issue.evidence["levels_before"] == 3
    assert issue.evidence["cells_changed_trim"] == n
    assert issue.description == f"Column 'c' has {n} value(s) with leading, trailing or repeated spaces."


def test_inner_runs_of_spaces_merge():
    df = pd.DataFrame({"c": ["a  b", "a b"] * 5})
    (issue,) = detect_inconsistent_text(df)
    assert (issue.evidence["levels_before"], issue.evidence["levels_after_trim"], issue.evidence["levels_after"]) == (2, 1, 1)
    assert issue.evidence["cells_changed_trim"] == 5


def test_a_clean_text_column_gives_no_issue():
    assert detect_inconsistent_text(pd.DataFrame({"region": ["East", "West", "North"] * 4})) == []


def test_numeric_and_bool_columns_are_never_examined(trim_calls):
    df = pd.DataFrame({"x": np.linspace(0, 1, 12), "k": range(12), "flag": [True, False] * 6})
    assert detect_inconsistent_text(df) == [] and trim_calls == []


def test_missing_values_are_ignored():
    df = pd.DataFrame({"region": ["East", None, "east", np.nan, "West"] * 4})
    (issue,) = detect_inconsistent_text(df)
    assert issue.evidence == _expected_text_evidence(df["region"])
    assert (issue.evidence["levels_before"], issue.evidence["levels_after"]) == (3, 2)


def test_a_single_level_with_a_trailing_space():
    df = pd.DataFrame({"c": ["only "] * 10})
    (issue,) = detect_inconsistent_text(df)
    assert (issue.evidence["levels_before"], issue.evidence["levels_after"], issue.evidence["cells_changed_trim"]) == (1, 1, 10)
    assert issue.description == "Column 'c' has 10 value(s) with leading, trailing or repeated spaces."


@pytest.mark.parametrize("distinct, flagged", [(HIGH_CARDINALITY_THRESHOLD, True), (HIGH_CARDINALITY_THRESHOLD + 1, False)],
                         ids=["twenty", "twenty_one"])
def test_the_level_bound(distinct, flagged, trim_calls):
    assert HIGH_CARDINALITY_THRESHOLD == 20
    levels = ["level", "LEVEL"] + [f"v{i}" for i in range(distinct - 2)]  # one mergeable pair
    df = pd.DataFrame({"c": levels * 3})
    assert df["c"].nunique() == distinct
    issues = detect_inconsistent_text(df)
    if flagged:
        assert [i.affected_columns for i in issues] == [("c",)] and trim_calls
    else:
        assert issues == [] and trim_calls == []  # the withheld column is not read value by value


def test_no_level_appears_in_an_issue():
    df = pd.DataFrame({"c": ["CANARY-East ", "CANARY-east", "CANARY-West"] * 4})
    (issue,) = detect_inconsistent_text(df)
    text = json.dumps(dataclasses.asdict(issue))
    assert "CANARY" not in text and "CANARY" not in issue.description and "CANARY" not in json.dumps(issue.evidence)


# --- detect_numeric_as_text ------------------------------------------------------------------------------------------

def _numbers(n):
    return [f"{v:.1f}" for v in np.linspace(1, 99, n)]


@pytest.mark.parametrize("bad, flagged", [(5, True), (6, False)], ids=["rate_0.95", "rate_0.94"])
def test_the_parse_rate_boundary(bad, flagged):
    assert NUMERIC_TEXT_PARSE_RATE == 0.95
    df = pd.DataFrame({"t": _numbers(100 - bad) + ["unknown"] * bad})
    issues = detect_numeric_as_text(df)
    assert [i.affected_columns for i in issues] == ([("t",)] if flagged else [])


def test_all_numbers_are_flagged():
    (issue,) = detect_numeric_as_text(pd.DataFrame({"t": _numbers(20)}))
    assert issue.evidence == {"non_missing": 20, "parse_rate": 1.0, "unparseable_count": 0}


def test_fewer_than_ten_values_or_one_distinct_value():
    assert detect_numeric_as_text(pd.DataFrame({"t": _numbers(9) + [None] * 5})) == []
    assert detect_numeric_as_text(pd.DataFrame({"t": ["5"] * 20})) == []


@pytest.mark.parametrize("value, suppressed", [("02134", True), ("007", True), ("-05", True),
                                               ("0", False), ("0.5", False), ("10", False)])
def test_leading_zeros_suppress_the_column(value, suppressed):
    df = pd.DataFrame({"t": _numbers(19) + [value]})
    assert (detect_numeric_as_text(df) == []) is suppressed


def test_thousands_separators_and_whitespace():
    values = [f" {v} " for v in _numbers(96)] + ["1,234"] * 4  # spaces around numbers parse; "1,234" does not
    df = pd.DataFrame({"t": values})
    (issue,) = detect_numeric_as_text(df)
    parsed = pd.to_numeric(df["t"].str.strip(), errors="coerce")
    assert issue.evidence == {"non_missing": 100, "parse_rate": round(float(parsed.notna().mean()), 6),
                              "unparseable_count": int(parsed.isna().sum())}
    assert issue.evidence["unparseable_count"] == 4
    assert issue.description == "Column 't' is stored as text but 96.0% of its values are numbers."


def test_a_numeric_column_is_never_examined():
    assert detect_numeric_as_text(pd.DataFrame({"x": np.linspace(1, 99, 30), "k": range(30)})) == []


def test_no_value_appears_in_an_issue():
    df = pd.DataFrame({"t": ["31415.926"] + _numbers(30) + ["unknown"]})
    (issue,) = detect_numeric_as_text(df)
    text = json.dumps(dataclasses.asdict(issue))
    assert "31415.926" not in text and "31415" not in text and "unknown" not in text


# --- detect_all_issues -----------------------------------------------------------------------------------------------

def _all_four():
    n = 100
    rng = np.random.default_rng(3)
    tenure = [f"{v:.1f}" for v in rng.normal(24, 8, n)]
    tenure[3] = "unknown"  # stays text after the CSV round trip
    spend = rng.normal(50, 5, n)
    spend[:10] = np.nan  # 10% missing
    df = pd.DataFrame({"region": ["East", "east ", "West", "WEST"] * (n // 4), "tenure": tenure, "spend": spend})
    return pd.concat([df, df.iloc[[20, 21]]], ignore_index=True)


def test_the_order_and_one_read(tmp_path, store):
    ds = _dataset(_all_four(), tmp_path, store)
    profile = profile_dataset(ds, store=store)
    store.reads.clear()
    issues = detect_all_issues(ds, profile, store)
    assert [(i.issue_type, i.affected_columns) for i in issues] == [
        ("duplicate_rows", ("region", "tenure", "spend")), ("high_missingness", ("spend",)),
        ("inconsistent_text", ("region",)), ("numeric_as_text", ("tenure",))]
    assert store.reads == [ds.storage_key]


def test_without_the_new_problems_nothing_changes(tmp_path, store):
    df = pd.DataFrame({
        "id": [1, 2, 2, 3, 4, 5, 6, 7, 8, 9],
        "region": ["East", "West", "West", None, None, None, "West", "East", "West", "East"],
        "spend": [10.0] * 10,
    })
    ds = _dataset(df, tmp_path, store)
    profile = profile_dataset(ds, store=store)
    assert detect_all_issues(ds, profile, store) == [detect_duplicate_rows(ds, store), *detect_high_missingness(profile)]
