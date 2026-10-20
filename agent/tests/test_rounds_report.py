"""
M2.4a: the Stage 1 report across cleaning rounds, on the G72 conversion shape and the aggregate shape: Checked is the
FIRST scan, a Rounds section states what later rounds found and how Stage 1 ended, Changed is numbered across rounds
with later-round lines tagged, Data quality runs from the first before snapshot to the LAST after snapshot. A one-round
report is byte-identical to the report before M2.4a. Expected numbers are computed independently with pandas.
"""
import numpy as np
import pandas as pd
import pytest

from core import thresholds
from core.dataset import LocalDiskStore
from core.graph import build_agent_graph
from core.plan import QuestionType
from tests.test_aggregate_run import (
    _app as _aggregate_app, _customers, _orders, reject_carried_and_join_missing, reject_join_missing,
)
from tests.test_decision_point_guard import _dataset, choose, confident, fake_draft
from tests.test_join_run import _bundle, _state
from tests.test_prompt_canaries import fake_model  # noqa: F401 (fixture)
from tests.test_replan import _join_tables
from tests.test_stage1_loop import _app, _approve_all, _conversion, _input, _run, _with_synthetic

# the Stage 1 report of this one-round run, produced by the code at b5b602d (before M2.4a)
PRE_M24A_TEXT = (
    'Stage 1: prepare and describe\n'
    '\n'
    'Checked\n'
    'Checked 82 rows and 4 columns.\n'
    'Found 2 exact duplicate rows.\n'
    'tenure: 7.3% missing (threshold 5%).\n'
    '\n'
    'Data quality\n'
    'Before: 82 rows, 4 columns, 2 exact duplicate rows.\n'
    'Missing values: tenure 7.3%.\n'
    'Outliers: none.\n'
    'After cleaning: 80 rows, 0 exact duplicate rows.\n'
    'Missing values after cleaning: none.\n'
    'Outliers after cleaning: none.\n'
    '\n'
    'Changed\n'
    "1. Remove 2 exact duplicate rows (all columns). Why (the model's reason): fix duplicate_rows. Alternative the model suggested: Leave the data as it is (it would bias the result). Your decision: approved. Effect: 82 to 80 rows.\n"
    "2. Fill the missing values of tenure with the median (22.85). Why (the model's reason): fix high_missingness. Alternative the model suggested: Leave the data as it is (it would bias the result). Your decision: approved. Effect: 6 value(s) filled; the mean of tenure moves from 22.88 to 22.88 and its spread from 5.84 to 5.614.\n"
    '\n'
    'Described\n'
    '80 rows and 4 columns described. Text values withheld for: none.')


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _stage1(values):
    return next(e for e in values["stage_reports"] if e["stage"] == "stage1")


def _sections(values):
    return {s["heading"]: s["text"] for s in _stage1(values)["sections"]}


def _records(values, point):
    return [e for e in values["decision_log"] if e["decision_point"] == point]


def test_a_one_round_report_is_byte_identical(fake_model, tmp_path, store):
    rng = np.random.default_rng(21)
    df = pd.DataFrame({"region": ["East", "West"] * 40, "channel": ["web", "store", "app", "web"] * 20,
                       "total_spend": rng.normal(50, 5, 80).round(2), "tenure": rng.normal(24, 6, 80).round(1)})
    df.loc[[2, 9, 17, 30, 44, 58], "tenure"] = np.nan
    df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)
    app = build_agent_graph(store, classify_fn=confident(QuestionType.TWO_GROUP_COMPARISON),
                            draft_fn=fake_draft(["region"]), propose_fn=choose({"high_missingness": "impute_median"}))
    values, _, _ = _run(app, _input(_dataset(df, tmp_path, store, "one")), "one")
    assert _stage1(values)["text"] == PRE_M24A_TEXT
    assert _records(values, "planner.stage_replan") == []
    (scan,) = _records(values, "stage1.clean.scan")  # one scan only: no Rounds section, no later scan in the sources
    assert scan["id"] in _stage1(values)["source_records"] and "Rounds" not in _sections(values)


def test_the_conversion_shape_across_two_rounds(fake_model, tmp_path, store):
    df, ds = _conversion(tmp_path, store)
    values, _, _ = _run(_app(store), _input(ds), "g72")
    sections = _sections(values)
    assert list(sections) == ["Checked", "Data quality", "Rounds", "Changed", "Described"]
    non_missing = df["score"].dropna()
    rate = pd.to_numeric(non_missing, errors="coerce").notna().mean()
    assert sections["Checked"] == (f"Checked {len(df)} rows and {len(df.columns)} columns.\n"
                                   f"score: stored as text, but {rate:.1%} of its values are numbers.")
    converted = pd.to_numeric(df["score"], errors="coerce")
    assert sections["Rounds"] == (f"Round 2 found: score: {converted.isna().mean():.1%} missing (threshold 5%).\n"
                                  "Stage 1 ended after 2 round(s): no new issues were found.")
    first, second = sections["Changed"].split("\n")
    assert first.startswith("1. Convert score to numbers") and "(round" not in first
    assert second.startswith(f"2. (round 2) Keep only rows where score has a value (removes {int(converted.isna().sum())} "
                             "rows).")
    kept = int(converted.notna().sum())
    quality = sections["Data quality"].split("\n")
    assert quality[:2] == [f"Before: {len(df)} rows, {len(df.columns)} columns, 0 exact duplicate rows.",
                           f"Missing values: score {df['score'].isna().mean():.1%}."]
    assert f"After cleaning: {kept} rows, 0 exact duplicate rows." in quality  # the LAST after snapshot
    scans, (replan,) = _records(values, "stage1.clean.scan"), _records(values, "planner.stage_replan")
    source = _stage1(values)["source_records"]
    assert scans[1]["id"] in source and replan["id"] in source and "The plan was updated" not in sections["Rounds"]


def test_finished(fake_model, tmp_path, store):
    _, ds = _conversion(tmp_path, store)
    values, _, _ = _run(_app(store), _input(ds), "fin", wrangling_approval=lambda p: _approve_all(p, finish=True))
    assert _sections(values)["Rounds"] == "Stage 1 ended after 1 round(s): you finished Stage 1."


def test_declined_round_one_keeps_cleaning_did_not_change(fake_model, tmp_path, store):
    _, ds = _conversion(tmp_path, store)
    reject = lambda p: {"decisions": [{"index": o["index"], "action": "reject"} for o in p["proposed_operations"]]}
    values, _, _ = _run(_app(store), _input(ds), "dec", wrangling_approval=reject)
    sections = _sections(values)
    assert "Rounds" not in sections and sections["Data quality"].endswith("Cleaning did not change the data.")


@pytest.mark.parametrize("limit", [1, "default"])  # M2.4b: the default limit is read from thresholds (7)
def test_the_round_limit(limit, fake_model, tmp_path, store, monkeypatch):
    limit = thresholds.MAX_STAGE1_ROUNDS if limit == "default" else limit
    if limit == 1:
        monkeypatch.setattr(thresholds, "MAX_STAGE1_ROUNDS", 1)
        _, ds = _conversion(tmp_path, store)
    else:  # the default limit, with an issue that is new in every round
        _with_synthetic(monkeypatch)
        df = pd.DataFrame({"region": ["East", "West"] * 30, "total_spend": np.arange(60.0), "channel": ["a", "b", "c"] * 20})
        ds = _dataset(df, tmp_path, store, "chain")
    values, _, _ = _run(_app(store), _input(ds), f"limit-{limit}")
    lines = _sections(values)["Rounds"].split("\n")
    assert lines[-1] == f"Stage 1 ended after {limit} round(s): the limit of {limit} rounds was reached."
    assert [line.split(" found")[0] for line in lines[:-1]] == [f"Round {n}" for n in range(2, limit + 1)]


def test_a_join_that_changes_modeling(fake_model, tmp_path, store):
    _, _, tables = _join_tables(tmp_path, store)
    values, _, _ = _run(_app(store), _state(tables), "join")
    assert _sections(values)["Rounds"] == ("Stage 1 ended after 1 round(s): no new issues were found.\n"
                                           "The plan was updated: modeling is now offered.")


def test_the_aggregate_shape_with_round_two_declined(fake_model, tmp_path, store):
    customers = _customers()
    tables = _bundle(tmp_path, store, {"customers": customers, "orders": _orders(customers)})
    # M2.4b: the orders cleaning (round 1) and the unmatched rows (round 3) are declined; the join runs in round 2
    values, _, _ = _run(_aggregate_app(store), _state(tables), "agg", wrangling_approval=reject_carried_and_join_missing)
    sections = _sections(values)
    orders = store.read(tables["orders"].storage_key)
    assert list(sections) == ["Checked", "Tables", "Data quality", "Links between tables", "Rounds", "Changed",
                              "Left as is", "Described"]
    assert sections["Checked"] == (f"Checked 40 rows and 4 columns.\nIn orders: Found {int(orders.duplicated().sum())} "
                                   f"exact duplicate rows.\nIn orders: amount: {orders['amount'].isna().mean():.1%} "
                                   "missing (threshold 5%).")  # M2.4b: the linked carried table's issues
    assert sections["Rounds"] == (
        "Round 2 found: orders: has several rows for each customer_id; it can be summarized and joined to the analysis "
        "table.\n"  # M2.4b: the join waited for the orders cleaning
        "Round 3 found: orders: the columns the join added are missing for the rows that have no rows in orders.\n"
        "Stage 1 ended after 3 round(s): you declined every proposal of the last round.")
    lines = sections["Changed"].split("\n")
    assert [line[:15] for line in lines] == ["1. In orders: R", "2. In orders: K", "3. (round 2) Su", "4. (round 3) Re"]
    assert sections["Left as is"].split("\n")[-1].startswith(
        "orders: the columns the join added stay missing for the rows that have no rows in orders. You rejected the fix: ")
    after = _records(values, "stage1.quality.snapshot")[-1]["evidence"]
    assert f"After cleaning: {after['rows']} rows, {after['duplicate_rows']} exact duplicate rows." in sections["Data quality"]
    assert after["rows"] == len(customers)
