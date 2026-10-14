"""
M0.6 (G16): one statistics engine with two entry points. compute_descriptive_stats is the user-facing
description (statistics plus charts, unchanged); compute_diagnostics is the chart-free data check a later
Stage 1 step reads (missing counts and the duplicate count added). Results with NaN are compared by repr,
because NaN never equals itself.
"""
import json
from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

import core.capabilities as capabilities
import core.nodes as nodes_module
import core.visualization as viz
from core.agent_state import AgentState
from core.capabilities import DESCRIPTIVE, DIAGNOSE, Diagnostic, capability_named
from core.dataset import LocalDiskStore
from core.descriptive import (
    DescriptiveResult, compute_charts, compute_column_stats, compute_correlation, compute_descriptive_stats,
)
from core.detection import detect_duplicate_rows
from core.diagnostics import DiagnosticsResult, compute_diagnostics, diagnostics_to_dict
from core.ingestion import load_dataset, profile_dataset
from core.model_safe_view import assert_model_safe
from core.nodes import make_report_node
from core.operation import WranglingRationale
from core.report import generate_descriptive_report
from core.visualization import RAW_VALUE_CHART_TYPES, SUMMARY_CHART_TYPES
from core.wrangling import execute_dedupe, execute_filter, propose_dedupe, propose_filter
from tests.test_prompt_canaries import NAME_CANARIES, _base_df, _clean, _with_duplicates


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


@pytest.fixture
def load(tmp_path, store):
    counter = iter(range(1000))

    def load_df(df):
        name = f"data{next(counter)}"
        path = tmp_path / f"{name}.csv"
        df.to_csv(path, index=False)
        return load_dataset(path, session_id="session-abc", name=name, store=store)
    return load_df


def _has_key(value, key):
    if isinstance(value, dict):
        return key in value or any(_has_key(v, key) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_key(v, key) for v in value)
    return False


def _mixed():
    rng = np.random.default_rng(11)
    return pd.DataFrame({"region": rng.choice(["East", "West"], 40), "x": rng.normal(0, 1, 40),
                         "tier": rng.choice(["basic", "plus", "pro"], 40), "y": rng.normal(5, 2, 40)})


def _frames():
    rng = np.random.default_rng(12)
    return {
        "clean_canary": _clean(_base_df()),
        "numeric_only": pd.DataFrame({"a": rng.normal(0, 1, 30), "b": rng.normal(1, 2, 30), "c": rng.integers(0, 5, 30)}),
        "text_only": pd.DataFrame({"region": rng.choice(["East", "West"], 30), "tier": rng.choice(["basic", "pro"], 30)}),
        "one_row": pd.DataFrame({"x": [1.5], "y": [2.0], "label": ["a"]}),
    }


# --- 1. the engine split ------------------------------------------------------------------------

def test_compute_charts_keeps_the_order(store, load):
    df = store.read(load(_mixed()).storage_key)
    assert [c["chart_type"] for c in compute_charts(df)] == [
        "bar_chart", "histogram", "boxplot_single_column", "bar_chart", "histogram", "boxplot_single_column",
        "correlation_heatmap", "scatter_matrix", "missingness_chart", "pairwise_categorical_chart",
    ]


def test_the_parts_equal_the_descriptive_result(store, load):
    ds = load(_mixed())
    df = store.read(ds.storage_key)
    result = compute_descriptive_stats(ds, store)
    assert compute_column_stats(df) == result.column_stats
    assert compute_correlation(df) == result.correlation_matrix
    by_hand = DescriptiveResult(row_count=len(df), column_stats=compute_column_stats(df),
                                correlation_matrix=compute_correlation(df), chart_data=compute_charts(df))
    assert repr(result) == repr(by_hand)


# --- 2. diagnostics equal descriptive where they overlap ---------------------------------------------

@pytest.mark.parametrize("name", list(_frames()))
def test_diagnostics_overlap_equals_descriptive(name, store, load):
    df = _frames()[name]
    ds = load(df)
    diagnostics, descriptive = compute_diagnostics(ds, store), compute_descriptive_stats(ds, store)
    assert repr(diagnostics.column_stats) == repr(descriptive.column_stats)
    assert repr(diagnostics.correlation_matrix) == repr(descriptive.correlation_matrix)
    assert diagnostics.row_count == descriptive.row_count == len(df)
    assert diagnostics.column_count == len(df.columns)


# --- 3. missing values and duplicates ------------------------------------------------------------------

@pytest.mark.parametrize("df", [
    pd.DataFrame({"a": [1.0, None, 3.0, None, 5.0], "b": ["p", "q", None, "q", "p"], "c": [1, 2, 3, 4, 5]}),
    pd.DataFrame({"a": [1.0, 2.0, 3.0], "empty": [None, None, None], "label": ["x", "y", "x"]}),
], ids=["two_columns_missing", "all_missing_column"])
def test_missing_by_column_equals_the_profile(df, store, load):
    ds = load(df)
    missing = compute_diagnostics(ds, store).missing_by_column
    profile = profile_dataset(ds, store=store)
    assert [m.name for m in missing] == [c.name for c in profile.columns] == list(df.columns)
    for m, c in zip(missing, profile.columns):
        assert m.missing_count == c.missing_count, m.name
        assert m.missing_pct == pytest.approx(c.missing_pct, abs=1e-4), m.name
    assert any(m.missing_count for m in missing)


@pytest.mark.parametrize("duplicates", [0, 1, 4])
def test_duplicate_count_equals_detection(duplicates, store, load):
    df = pd.DataFrame({"id": range(10), "v": [float(i) for i in range(10)]})
    df = pd.concat([df, df.iloc[:duplicates]], ignore_index=True)
    ds = load(df)
    issue = detect_duplicate_rows(ds, store)
    detected = 0 if issue is None else issue.evidence["duplicate_row_count"]
    assert compute_diagnostics(ds, store).duplicate_row_count == detected == duplicates


# --- 4. no charts ----------------------------------------------------------------------------------------

def test_diagnostics_build_no_chart(monkeypatch, store, load):
    def refuse(*args, **kwargs):
        raise AssertionError("diagnostics must not build a chart")
    for name in (*RAW_VALUE_CHART_TYPES, *SUMMARY_CHART_TYPES):
        monkeypatch.setattr(viz, name, refuse)
    ds = load(_mixed())
    compute_diagnostics(ds, store)
    as_dict = DIAGNOSE.compute(ds, store)

    assert "chart_data" not in {f.name for f in fields(DiagnosticsResult)}
    assert not _has_key(as_dict, "chart_data")
    json.dumps(as_dict)
    assert_model_safe(as_dict, "diagnostics")
    with pytest.raises(AssertionError, match="must not build a chart"):
        compute_descriptive_stats(ds, store)  # the patch really covers the builders


def test_diagnostics_to_dict_shape(store, load):
    as_dict = diagnostics_to_dict(compute_diagnostics(load(_mixed()), store))
    assert sorted(as_dict) == ["column_count", "column_stats", "correlation_matrix", "duplicate_row_count",
                               "missing_by_column", "row_count", "schema_version"]
    assert as_dict["schema_version"] == 1
    assert sorted(as_dict["missing_by_column"][0]) == ["missing_count", "missing_pct", "name"]


# --- 5. privacy --------------------------------------------------------------------------------------------

def test_diagnostics_of_the_canary_data_hold_no_name(store, load):
    as_dict = DIAGNOSE.compute(load(_clean(_base_df())), store)
    text = json.dumps(as_dict, default=str)
    for name in (*NAME_CANARIES, "name-0001", "name-0150", "name-0297"):
        assert name not in text, name
    cols = {c["name"]: c for c in as_dict["column_stats"]}
    assert (cols["customer_name"]["distinct_count"], cols["customer_name"]["top_value_counts"],
            cols["customer_name"]["mode"]) == (300, [], None)
    assert cols["tier"]["top_value_counts"] and cols["tier"]["mode"] in ("basic", "plus", "pro")


@pytest.mark.parametrize("k, kept", [(20, 5), (21, 0)])
def test_diagnostics_threshold(k, kept, store, load):
    labels = [f"v{i:02d}" for i in range(k) for _ in range(k - i)]  # unequal counts
    (label,) = compute_diagnostics(load(pd.DataFrame({"label": labels})), store).column_stats
    assert label.distinct_count == k and len(label.top_value_counts) == kept
    assert (label.mode is None) == (kept == 0)


# --- 6. the Stage 1 use: a snapshot before and after each cleaning step ----------------------------------

def _rationale():
    return WranglingRationale(rationale="test", alternatives=(), evidence={})


def test_snapshots_track_cleaning(store, load):
    ds = load(_with_duplicates(_base_df()))
    before = diagnostics_to_dict(compute_diagnostics(ds, store))
    assert before == diagnostics_to_dict(compute_diagnostics(ds, store))  # same dataset, same snapshot
    assert before["duplicate_row_count"] == 4

    deduped, _ = execute_dedupe(ds, store, propose_dedupe(ds, store, subset=None, rationale=_rationale()).approve())
    after_dedupe = diagnostics_to_dict(compute_diagnostics(deduped, store))
    assert after_dedupe["duplicate_row_count"] == 0
    assert after_dedupe["row_count"] == before["row_count"] - 4

    notes_missing = next(m["missing_count"] for m in after_dedupe["missing_by_column"] if m["name"] == "notes")
    assert notes_missing > 0
    op = propose_filter(deduped, store, column="notes", condition="not_null", value=None, rationale=_rationale())
    filtered, _ = execute_filter(deduped, store, op.approve())
    after_filter = diagnostics_to_dict(compute_diagnostics(filtered, store))
    assert next(m["missing_count"] for m in after_filter["missing_by_column"] if m["name"] == "notes") == 0
    assert after_filter["row_count"] == after_dedupe["row_count"] - notes_missing


# --- 7. the registry ---------------------------------------------------------------------------------------

def test_diagnose_registry_entry():
    assert (DIAGNOSE.name, DIAGNOSE.stage, DIAGNOSE.kind, DIAGNOSE.result_key, DIAGNOSE.report) == (
        "diagnose", 1, "diagnostic", "diagnostics_result", None)
    assert capability_named("diagnose") is DIAGNOSE
    assert all(v is not DIAGNOSE for v in capabilities._BY_QUESTION_TYPE.values())
    assert DIAGNOSE not in capabilities.ANALYSES
    assert (DESCRIPTIVE.name, DESCRIPTIVE.result_key, DESCRIPTIVE.report) == (
        "descriptive", "descriptive_result", generate_descriptive_report)


def test_every_diagnostic_result_key_is_a_state_key():
    diagnostics = [v for v in vars(capabilities).values() if isinstance(v, Diagnostic)]
    assert {d.name for d in diagnostics} == {"descriptive", "diagnose"}
    for d in diagnostics:
        assert d.result_key in AgentState.__annotations__, d.name


def test_report_node_refuses_a_capability_without_a_report(monkeypatch):
    monkeypatch.setattr(nodes_module, "_capability_for_state", lambda state: DIAGNOSE)
    with pytest.raises(ValueError, match="has no report"):
        make_report_node()({"plan": None, "diagnostics_result": {}})
