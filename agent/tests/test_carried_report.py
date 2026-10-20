"""
M2.4b: the Stage 1 report with carried-table cleaning (the aggregate shape): Checked lists the analysis table's lines then
the linked carried table's ("In orders: ..."), the Rounds section states the later rounds, Changed names the table after
the number and round tag, Data quality ends with one line per cleaned carried table, and a report without carried
issues is unchanged. Expected numbers are computed independently with pandas.
"""
import numpy as np
import pandas as pd
import pytest

from core.dataset import LocalDiskStore
from core.graph import build_agent_graph
from core.plan import QuestionType
from tests.test_aggregate_run import (
    ALTERNATIVE, CARRIED_REASON, JOIN_MISSING_REASON, _app, _customers, _orders, reject_carried_and_join_missing,
    reject_join_missing,
)
from tests.test_carried_cleaning import _text_orders
from tests.test_decision_point_guard import _dataset, choose, confident, fake_draft
from tests.test_join_run import _bundle, _state, _store_frame
from tests.test_prompt_canaries import fake_model  # noqa: F401 (fixture)
from tests.test_rounds_report import PRE_M24A_TEXT
from tests.test_stage1_loop import _input, _run

# the Stage 1 report of the aggregate run with the unmatched rows declined, produced by the code at 568e49a (M2.4a)
M24A_AGGREGATE_TEXT = (
    'Stage 1: prepare and describe\n'
    '\n'
    'Checked\n'
    'Checked 40 rows and 4 columns.\n'
    'orders: has several rows for each customer_id; it can be summarized and joined to the analysis table.\n'
    '\n'
    'Tables\n'
    'customers: 40 rows, 4 columns (main table; joined with orders into customers_orders)\n'
    'orders: 122 rows, 4 columns (summarized and joined into customers_orders)\n'
    'customers_orders: 40 rows, 9 columns (main table after the join)\n'
    '\n'
    'Data quality\n'
    'Before: 40 rows, 4 columns, 0 exact duplicate rows.\n'
    'Missing values: none.\n'
    'Outliers: none.\n'
    'After cleaning: 40 rows, 0 exact duplicate rows.\n'
    'Missing values after cleaning: amount_mean 25.0%; amount_sum 25.0%; quantity_mean 20.0%; quantity_sum 20.0%.\n'
    'Outliers after cleaning: quantity_sum: 0 below, 1 above.\n'
    '\n'
    'Links between tables\n'
    'customers.customer_id with orders.customer_id: one to many; 32 of 40 keys in customers are found in orders; 8 rows of customers and 0 rows of orders have no match; an inner join would give 122 rows.\n'
    '\n'
    'Rounds\n'
    'Round 2 found: orders: the columns the join added are missing for the rows that have no rows in orders.\n'
    'Stage 1 ended after 2 round(s): you declined every proposal of the last round.\n'
    '\n'
    'Changed\n'
    "1. Summarize orders per customer_id (a row count, and the mean and sum of amount and quantity), then join the summary to the analysis table, keeping every row. Adds 5 column(s): rows_in_orders, amount_mean, amount_sum, quantity_mean, quantity_sum. Why (the model's reason): fix aggregatable_table. Alternative the model suggested: Leave the data as it is (it would bias the result). Your decision: approved. Effect: 40 rows stay 40; 5 column(s) added; 32 new missing value(s) from 8 row(s) with no match. Notes: 3 exact duplicate row(s) in orders are counted again in the summary; amount is 11.5% missing in orders: its mean and sum use only the rows that have a value.\n"
    "2. (round 2) Remove the rows that have no rows in orders (8 removed). Why (the model's reason): fix join_missing. Alternative the model suggested: Leave the data as it is (it would bias the result). Your decision: rejected: the unmatched customers stay as they are.\n"
    '\n'
    'Left as is\n'
    'orders: the columns the join added stay missing for the rows that have no rows in orders. You rejected the fix: the unmatched customers stay as they are.\n'
    '\n'
    'Described\n'
    '40 rows and 9 columns described. Text values withheld for: none.')


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _stage1(values):
    return next(e for e in values["stage_reports"] if e["stage"] == "stage1")


def _sections(values):
    return {s["heading"]: s["text"] for s in _stage1(values)["sections"]}


def _bundle_of(tmp_path, store, customers=None, **carried):
    customers = _customers() if customers is None else customers
    return _bundle(tmp_path, store, {"customers": customers, **(carried or {"orders": _orders(_customers())})})


def test_checked_rounds_and_changed(fake_model, tmp_path, store):
    customers = _customers()
    messy = pd.concat([customers, customers.iloc[[0]]], ignore_index=True)  # one duplicate in the analysis table
    tables = _bundle_of(tmp_path, store, messy)
    values, _, _ = _run(_app(store), _state(tables), "c", wrangling_approval=reject_join_missing)
    s = _sections(values)
    orders = _store_frame(store, tables["orders"])
    dupes, share = int(orders.duplicated().sum()), orders["amount"].isna().mean()
    assert s["Checked"].split("\n") == [
        "Checked 41 rows and 4 columns.", "Found 1 exact duplicate rows.",
        f"In orders: Found {dupes} exact duplicate rows.", f"In orders: amount: {share:.1%} missing (threshold 5%)."]
    assert s["Rounds"].split("\n") == [
        "Round 2 found: orders: has several rows for each customer_id; it can be summarized and joined to the analysis "
        "table.",
        "Round 3 found: orders: the columns the join added are missing for the rows that have no rows in orders.",
        "Stage 1 ended after 3 round(s): you declined every proposal of the last round."]
    lines = s["Changed"].split("\n")
    removed = int(orders.drop_duplicates()["amount"].isna().sum())
    assert lines[0].startswith("1. Remove 1 exact duplicate rows (all columns). Why")  # the analysis table: no prefix
    assert lines[1].startswith(f"2. In orders: Remove {dupes} exact duplicate rows (all columns). Why")
    assert lines[2].startswith(f"3. In orders: Keep only rows where amount has a value (removes {removed} rows). Why")
    assert lines[3].startswith("4. (round 2) Summarize orders per customer_id") and lines[4].startswith("5. (round 3) ")
    assert s["Tables"].split("\n") == [
        "customers: 41 rows, 4 columns (main table; joined with orders into customers_orders)",
        "orders: 122 rows, 4 columns (summarized and joined into customers_orders)",
        f"customers_orders: {len(messy.drop_duplicates())} rows, 9 columns (main table after the join)"]  # deduped
    cleaned = orders.drop_duplicates().dropna(subset=["amount"])
    assert s["Data quality"].split("\n")[-1] == f"orders: {len(orders)} rows before cleaning, {len(cleaned)} after."


def test_a_later_round_carried_line(fake_model, tmp_path, store):
    tables = _bundle_of(tmp_path, store, orders=_text_orders())
    pick = choose({"inconsistent_text": "trim_and_lowercase"})
    values, _, _ = _run(_app(store, pick), _state(tables), "later", wrangling_approval=reject_join_missing)
    lines = _sections(values)["Changed"].split("\n")
    assert lines[0].startswith("1. In orders: Trim spaces and ignore case in channel")
    assert lines[1].startswith("2. (round 2) In orders: Remove 3 exact duplicate rows (all columns). Why")
    assert lines[2].startswith("3. (round 3) Summarize orders per customer_id")
    assert "Round 2 found: In orders: Found 3 exact duplicate rows." in _sections(values)["Rounds"].split("\n")


def test_two_cleaned_carried_tables_and_a_declined_one(fake_model, tmp_path, store):
    rng = np.random.default_rng(83)
    visits = pd.DataFrame({"customer_id": rng.integers(1, 31, 80), "minutes": rng.normal(5, 1, 80).round(2)})
    visits = pd.concat([visits, visits.iloc[[0, 1]]], ignore_index=True)  # 2 duplicates
    tables = _bundle_of(tmp_path, store, orders=_orders(_customers()), visits=visits)
    values, _, _ = _run(_app(store), _state(tables), "two", wrangling_approval=reject_join_missing)
    quality = _sections(values)["Data quality"].split("\n")
    orders = _store_frame(store, tables["orders"])
    assert quality[-2:] == [
        f"orders: {len(orders)} rows before cleaning, {len(orders.drop_duplicates().dropna(subset=['amount']))} after.",
        f"visits: {len(visits)} rows before cleaning, {len(visits.drop_duplicates())} after."]
    # the visits cleaning declined: only orders gets a line
    decline_visits = lambda p: {"decisions": [{"index": o["index"], "action": "reject" if o.get("table") == "visits"
                                               or o["candidate_id"] in ("delete_unmatched_rows", "fill_empty_sums_with_zero")
                                               else "approve"} for o in p["proposed_operations"]]}
    values, _, _ = _run(_app(store), _state(tables), "two-declined", wrangling_approval=decline_visits)
    quality = _sections(values)["Data quality"].split("\n")
    assert [line for line in quality if "rows before cleaning" in line] == [quality[-1]]
    assert quality[-1].startswith("orders: ")


def test_no_line_when_no_carried_table_was_cleaned(fake_model, tmp_path, store):
    values, _, _ = _run(_app(store), _state(_bundle_of(tmp_path, store)), "none",
                        wrangling_approval=reject_carried_and_join_missing)
    assert "rows before cleaning" not in _sections(values)["Data quality"]


def test_a_report_without_carried_issues_is_byte_identical(fake_model, tmp_path, store):
    rng = np.random.default_rng(21)
    df = pd.DataFrame({"region": ["East", "West"] * 40, "channel": ["web", "store", "app", "web"] * 20,
                       "total_spend": rng.normal(50, 5, 80).round(2), "tenure": rng.normal(24, 6, 80).round(1)})
    df.loc[[2, 9, 17, 30, 44, 58], "tenure"] = np.nan
    df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)
    app = build_agent_graph(store, classify_fn=confident(QuestionType.TWO_GROUP_COMPARISON),
                            draft_fn=fake_draft(["region"]), propose_fn=choose({"high_missingness": "impute_median"}))
    values, _, _ = _run(app, _input(_dataset(df, tmp_path, store, "one")), "one")
    assert _stage1(values)["text"] == PRE_M24A_TEXT


def test_the_m24a_aggregate_report_with_the_cleaning_declined(fake_model, tmp_path, store):
    tables = _bundle_of(tmp_path, store)
    values, _, _ = _run(_app(store), _state(tables), "diff", wrangling_approval=reject_carried_and_join_missing)
    orders = _store_frame(store, tables["orders"])
    removed = int(orders.drop_duplicates()["amount"].isna().sum())
    old = M24A_AGGREGATE_TEXT.split("\n")
    summarize = next(i for i, line in enumerate(old) if line.startswith("1. Summarize"))
    unmatched = next(i for i, line in enumerate(old) if line.startswith("2. (round 2) Remove"))
    expected = list(old)
    # the exact differences from M2.4a: the carried lines in Checked, the later rounds, the renumbered Changed list
    # with the two declined orders operations first, and their generic Left as is lines
    checked = expected.index("orders: has several rows for each customer_id; it can be summarized and joined to the "
                             "analysis table.")
    expected[checked:checked + 1] = ["In orders: Found 3 exact duplicate rows.",
                                     "In orders: amount: 11.5% missing (threshold 5%)."]
    shift = 1
    rounds = expected.index("Round 2 found: orders: the columns the join added are missing for the rows that have no "
                            "rows in orders.")
    expected[rounds:rounds + 2] = [
        "Round 2 found: orders: has several rows for each customer_id; it can be summarized and joined to the analysis "
        "table.",
        "Round 3 found: orders: the columns the join added are missing for the rows that have no rows in orders.",
        "Stage 1 ended after 3 round(s): you declined every proposal of the last round."]
    shift += 1
    summarize += shift
    expected[summarize:summarize + 2] = [
        f"1. In orders: Remove 3 exact duplicate rows (all columns). Why (the model's reason): fix duplicate_rows. "
        f"{ALTERNATIVE} Your decision: rejected: {CARRIED_REASON}.",
        f"2. In orders: Keep only rows where amount has a value (removes {removed} rows). Why (the model's reason): fix "
        f"high_missingness. {ALTERNATIVE} Your decision: rejected: {CARRIED_REASON}.",
        "3. (round 2) " + old[summarize - shift][len("1. "):],
        "4. (round 3) " + old[unmatched][len("2. (round 2) "):]]
    left = expected.index("Left as is") + 1
    expected[left:left] = [  # M2.4b: the generic wording with the carried issue's description (it names the table)
        "duplicate_rows (In the table 'orders': 3 exact duplicate row(s) detected across all columns.) was left "
        f"unaddressed because: {CARRIED_REASON}",
        f"high_missingness (In the table 'orders': Column 'amount' is 11.5% missing.) was left unaddressed because: "
        f"{CARRIED_REASON}"]
    assert _stage1(values)["text"].split("\n") == expected


def test_the_declined_cleaning_caveat_reaches_plan_approval_and_the_summary(fake_model, tmp_path, store):
    tables = _bundle_of(tmp_path, store)
    values, pauses, _ = _run(_app(store), _state(tables), "caveat", wrangling_approval=reject_carried_and_join_missing)
    descriptions = ["In the table 'orders': 3 exact duplicate row(s) detected across all columns.",
                    "In the table 'orders': Column 'amount' is 11.5% missing."]
    approval = next(p for p in pauses if p["type"] == "plan_approval")
    assert [u["description"] for u in approval["unaddressed_issues"] if u["description"].startswith("In the table")] == (
        descriptions)
    summary = next(e for e in values["stage_reports"] if e["stage"] == "summary")
    caveats = next(s["text"] for s in summary["sections"] if s["heading"] == "Caveats")
    assert all(d in caveats for d in descriptions)


# --- the Left as is line of a declined carried fix, and the Data quality sentence -------------------------------------

def test_the_left_as_is_line_names_the_carried_table(fake_model, tmp_path, store):
    customers = _customers()
    customers.loc[customers.index % 6 == 0, "age"] = np.nan  # an analysis-table issue too (16.7% missing)
    tables = _bundle_of(tmp_path, store, customers)
    values, _, _ = _run(_app(store, outcome="age"), _state(tables), "left", wrangling_approval=lambda p: {"decisions": [
        {"index": o["index"], "action": "reject", "reason": "keep"} for o in p["proposed_operations"]]})
    left = _sections(values)["Left as is"].split("\n")
    assert "high_missingness (In the table 'orders': Column 'amount' is 11.5% missing.) was left unaddressed because: " \
           "keep" in left
    assert "high_missingness (age) was left unaddressed because: keep" in left  # the analysis table's line: unchanged
    fixes = [e["evidence"] for e in values["decision_log"] if e["decision_point"] == "stage1.clean.fix"]
    assert all(("description" in f) == ("table" in f) for f in fixes)  # only a carried issue's fix has a description


def test_a_carried_only_cleaning_does_not_say_cleaning_changed_nothing(fake_model, tmp_path, store):
    tables = _bundle_of(tmp_path, store)  # a clean analysis table
    decline_joins = lambda p: {"decisions": [{"index": o["index"], "action": "reject" if o["op_type"] == "join"
                                              else "approve"} for o in p["proposed_operations"]]}
    values, _, _ = _run(_app(store, outcome="age"), _state(tables), "carried-only", wrangling_approval=decline_joins)
    quality = _sections(values)["Data quality"]
    orders = _store_frame(store, tables["orders"])
    assert "Cleaning did not change the data." not in quality
    assert quality.split("\n")[-1] == (f"orders: {len(orders)} rows before cleaning, "
                                        f"{len(orders.drop_duplicates().dropna(subset=['amount']))} after.")


def test_nothing_cleaned_still_says_cleaning_changed_nothing(fake_model, tmp_path, store):
    tables = _bundle_of(tmp_path, store)
    values, _, _ = _run(_app(store, outcome="age"), _state(tables), "nothing", wrangling_approval=lambda p: {
        "decisions": [{"index": o["index"], "action": "reject"} for o in p["proposed_operations"]]})
    assert _sections(values)["Data quality"].endswith("Cleaning did not change the data.")

