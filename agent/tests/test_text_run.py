"""
M2.3a step 2: text normalization and the numeric conversion through the real graph (guard fakes, fake chat model).
Conversions run first, then text normalization, then exact duplicates, then missing values; every preview is
computed on the data as its operation will see it and equals the executed impact; the Stage 1 report states the
new issues and the executed effects; no level and no numeric-text value reaches a prompt, a payload, a record or a
report. Expected numbers are computed independently with pandas.
"""
import json
import re

import numpy as np
import pandas as pd
import pytest

from core.dataset import LocalDiskStore
from core.decision_points import DECISION_NODES, registered_points
from core.graph import build_agent_graph
from core.plan import QuestionType
from tests.test_cleaning_run import _app, _records, _reversed, _run, _sections, _wrangling
from tests.test_decision_point_guard import _dataset, choose
from tests.test_prompt_canaries import fake_model  # noqa: F401 (fixture)
from tests.test_stage1_report import _drive

PICK = {"inconsistent_text": "trim_and_lowercase", "numeric_as_text": "convert_to_number",
        "duplicate_rows": "dedupe_all", "high_missingness": "delete_rows"}
by_type = choose(lambda issue, candidates: PICK[issue.issue_type])
ALTERNATIVE = "Alternative the model suggested: Leave the data as it is (it would bias the result)."


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _stored(df, tmp_path):
    """The data as the graph sees it: after the CSV round trip."""
    path = tmp_path / "independent.csv"
    df.to_csv(path, index=False)
    return pd.read_csv(path)


def _four_issues(n=200, seed=9):
    """A mergeable region, numbers stored as text (3 'unknown'), 10% missing visits and 2 exact duplicate rows."""
    rng = np.random.default_rng(seed)
    region = np.array(["East", "east ", "West", "WEST"] * (n // 4))
    spend = 50 + np.where(np.isin(region, ["West", "WEST"]), 6.0, 0.0) + rng.normal(0, 5, n)
    tenure = [f"{v:.1f}" for v in rng.normal(24, 8, n)]
    for i in (3, 50, 120):
        tenure[i] = "unknown"
    df = pd.DataFrame({"region": region, "tenure": tenure, "total_spend": spend,
                       "visits": rng.integers(1, 9, n).astype(float)})
    df.loc[range(0, n, 10), "visits"] = np.nan
    return pd.concat([df, df.iloc[[7, 8]]], ignore_index=True)


# --- (a) the execution order ---------------------------------------------------------------------------------------

def test_the_execution_order(fake_model, tmp_path, store):
    values, pauses, _ = _run(store, tmp_path, _four_issues(), "a", _reversed(by_type))
    ops = _wrangling(pauses)["proposed_operations"]
    assert [(o["index"], o["op_type"], o["candidate_id"]) for o in ops] == [
        (0, "cast", "convert_to_number"), (1, "normalize_text", "trim_and_lowercase"), (2, "dedupe", "dedupe_all"),
        (3, "filter", "delete_rows")]
    assert [o["preview_basis"] for o in ops] == ["uploaded_data"] + ["after_earlier_operations"] * 3
    assert [e["evidence"]["op_type"] for e in _records(values, "stage1.clean.execution")] == [
        "cast", "normalize_text", "dedupe", "filter"]


# --- (b) normalizing creates duplicates ------------------------------------------------------------------------------

def test_normalizing_creates_duplicates(fake_model, tmp_path, store):
    rng = np.random.default_rng(12)
    n = 120
    region = rng.choice(["East", "West"], n)
    df = pd.DataFrame({"region": region, "total_spend": 50 + rng.normal(0, 5, n)})
    twins = df.iloc[[0, 1, 2]].copy()
    twins["region"] = twins["region"].str.lower() + " "  # exact duplicates of rows 0 to 2 only after normalizing
    df = pd.concat([df, twins, df.iloc[[10]]], ignore_index=True)  # plus one duplicate that exists before
    stored = _stored(df, tmp_path)
    before = int(stored.duplicated().sum())
    after = int(stored.assign(region=stored["region"].str.strip().str.lower()).duplicated().sum())
    assert (before, after) == (1, 4)
    values, pauses, _ = _run(store, tmp_path, df, "b", by_type)
    dedupe = next(o for o in _wrangling(pauses)["proposed_operations"] if o["op_type"] == "dedupe")
    assert dedupe["preview_basis"] == "after_earlier_operations"
    assert dedupe["impact_preview"]["rows_changed"] == after
    assert dedupe["what_was_run"] == f"Remove {after} exact duplicate rows (all columns)."
    (scan,) = _records(values, "stage1.clean.scan")
    assert scan["evidence"]["duplicate_row_count"] == before  # detected on the uploaded data
    execution = next(e for e in _records(values, "stage1.clean.execution") if e["evidence"]["op_type"] == "dedupe")
    assert execution["evidence"]["rows_before"] - execution["evidence"]["rows_after"] == after


# --- (c) and (d): every operation kind in one run --------------------------------------------------------------------

def _six_kinds():
    """296 rows plus 4 exact duplicates: region 4 levels that reduce to 2 (3 cells change); tenure 96% numbers; visits
    10% missing (median fill); notes 30% missing (dropped); score 6% missing (rows deleted)."""
    n = 296
    rng = np.random.default_rng(31)
    region = np.array(["east", "west"] * (n // 2), dtype=object)
    region[0], region[1], region[2] = "East", "East", "west "
    tenure = [f"{v:.1f}" for v in rng.normal(24, 8, n)]
    for i in range(20, 32):
        tenure[i] = "unknown"  # 12 of 300 values: 96.0% numbers
    visits = rng.integers(1, 9, n).astype(float)
    visits[100:130] = np.nan  # 30 of 300
    notes = np.array(["ok", "late"] * (n // 2), dtype=object)
    notes[130:220] = None  # 90 of 300
    score = rng.normal(70, 10, n)
    score[220:238] = np.nan  # 18 of 300
    spend = 50 + np.where(np.isin(region, ["west", "west "]), 6.0, 0.0) + rng.normal(0, 5, n)
    df = pd.DataFrame({"region": region, "tenure": tenure, "visits": visits, "notes": notes, "score": score,
                       "total_spend": spend})
    return pd.concat([df, df.iloc[[260, 261, 262, 263]]], ignore_index=True)


SIX = {"visits": "impute_median", "notes": "drop_column", "score": "delete_rows"}
six_kinds = choose(lambda issue, candidates: SIX[issue.affected_columns[0]] if issue.issue_type == "high_missingness"
                   else PICK[issue.issue_type])


def test_every_preview_equals_the_executed_impact(fake_model, tmp_path, store):
    values, pauses, _ = _run(store, tmp_path, _six_kinds(), "c", six_kinds)
    ops = _wrangling(pauses)["proposed_operations"]
    executions = _records(values, "stage1.clean.execution")
    assert [o["op_type"] for o in ops] == [e["evidence"]["op_type"] for e in executions] == [
        "cast", "normalize_text", "dedupe", "impute", "drop_column", "filter"]
    for op, execution in zip(ops, executions):
        preview, ev = op["impact_preview"], execution["evidence"]
        assert (preview["rows_before"], preview["rows_after"], preview["columns_before"], preview["columns_after"]) == (
            ev["rows_before"], ev["rows_after"], ev["columns_before"], ev["columns_after"]), op["op_type"]
        details = ev.get("details") or {}
        if op["op_type"] == "normalize_text":
            assert preview["cells_changed"] == details["cells_changed"]
        elif op["op_type"] == "cast":
            assert preview["cells_changed"] == details["non_missing_before"]
            assert preview["warnings"] == [f"{details['became_missing']} value(s) that are not numbers become missing"]
        elif op["op_type"] == "impute":
            assert preview["cells_changed"] == details["cells_filled"]
        elif op["op_type"] in ("dedupe", "filter"):
            assert preview["rows_changed"] == ev["rows_before"] - ev["rows_after"]
    json.dumps(values["decision_log"], allow_nan=False)  # (j) every record is strict JSON


def test_the_stage1_report(fake_model, tmp_path, store):
    df = _six_kinds()
    values, _, _ = _run(store, tmp_path, df, "d", six_kinds)
    d = _stored(df, tmp_path)
    rows, columns = len(d), len(d.columns)
    parsed = pd.to_numeric(d["tenure"].dropna().str.strip(), errors="coerce")
    rate, lost = float(parsed.notna().mean()), int(parsed.isna().sum())
    levels = d["region"].dropna()
    merged = levels.str.strip().str.lower()
    assert (rate, lost, levels.nunique(), merged.nunique(), int((merged != levels).sum())) == (0.96, 12, 4, 2, 3)
    pct = {c: float(d[c].isna().mean()) for c in ("visits", "notes", "score")}
    assert _sections(values)["Checked"] == "\n".join([
        f"Checked {rows} rows and {columns} columns.",
        f"Found {int(d.duplicated().sum())} exact duplicate rows.",
        *[f"{c}: {pct[c]:.1%} missing (threshold 5%)." for c in ("visits", "notes", "score")],
        "region: 4 distinct values reduce to 2 after trimming spaces and ignoring case.",
        f"tenure: stored as text, but {rate:.1%} of its values are numbers.",
    ])
    # replay the cleaning in pandas: convert, normalize, dedupe, then visits, notes and score in column order
    d["tenure"] = pd.to_numeric(d["tenure"].str.strip(), errors="coerce")
    d["region"] = d["region"].str.strip().str.lower()
    deduped = d.drop_duplicates()
    observed = deduped["visits"].dropna()
    median = observed.median()
    filled = deduped["visits"].fillna(median)
    kept = int(deduped["score"].notna().sum())
    approved = "Your decision: approved."
    assert _sections(values)["Changed"] == "\n".join([
        f"1. Convert tenure to numbers ({lost} value(s) that are not numbers become missing). Why (the model's "
        f"reason): fix numeric_as_text. {ALTERNATIVE} {approved} Effect: {lost} value(s) that were not numbers "
        "became missing.",
        f"2. Trim spaces and ignore case in region (4 distinct values become 2). Why (the model's reason): fix "
        f"inconsistent_text. {ALTERNATIVE} {approved} Effect: 4 distinct values became 2; 3 value(s) changed.",
        f"3. Remove {rows - len(deduped)} exact duplicate rows (all columns). Why (the model's reason): fix "
        f"duplicate_rows. {ALTERNATIVE} {approved} Effect: {rows} to {len(deduped)} rows.",
        f"4. Fill the missing values of visits with the median ({median:.4g}). Why (the model's reason): fix "
        f"high_missingness. {ALTERNATIVE} {approved} Effect: {int(deduped['visits'].isna().sum())} value(s) filled; "
        f"the mean of visits moves from {observed.mean():.4g} to {filled.mean():.4g} and its spread from "
        f"{observed.std():.4g} to {filled.std():.4g}.",
        f"5. Drop the column notes. Why (the model's reason): fix high_missingness. {ALTERNATIVE} {approved} "
        "Effect: the column notes was removed.",
        f"6. Keep only rows where score has a value (removes {len(deduped) - kept} rows). Why (the model's reason): "
        f"fix high_missingness. {ALTERNATIVE} {approved} Effect: {len(deduped)} to {kept} rows.",
    ])
    assert (rows, len(deduped)) == (300, 296)


@pytest.mark.parametrize("make, line", [
    (lambda: pd.DataFrame({"region": ["East", "east ", "West", "WEST"] * 30, "total_spend": np.linspace(40, 60, 120)}),
     "region: 4 distinct values reduce to 2 after trimming spaces and ignoring case."),
    (lambda: pd.DataFrame({"region": ["East", "West"] * 60, "total_spend": np.linspace(40, 60, 120),
                           "tenure": [f"{v}.5" for v in range(119)] + ["unknown"]}),
     f"tenure: stored as text, but {119 / 120:.1%} of its values are numbers."),
], ids=["inconsistent_text", "numeric_as_text"])
def test_no_issues_found_is_absent_with_only_a_new_issue(make, line, fake_model, tmp_path, store):
    values, _, _ = _run(store, tmp_path, make(), "only", by_type)
    checked = _sections(values)["Checked"].split("\n")
    assert checked[1:] == [line] and "No data-quality issues found." not in checked


# --- (e) privacy -----------------------------------------------------------------------------------------------------

LEVEL_CANARIES = ("CANARY-East ", "CANARY-east", "CANARY-West")
NUMBER_CANARIES = ("31415.926", "27182.818")


def _canary_frame(n=120):
    rng = np.random.default_rng(2)
    score = [f"{v:.1f}" for v in rng.normal(24, 8, n)]
    score[10], score[20], score[30] = NUMBER_CANARIES[0], NUMBER_CANARIES[1], "unknown"
    return pd.DataFrame({"region": rng.choice(["East", "West"], n), "segment": [LEVEL_CANARIES[i % 3] for i in range(n)],
                         "score": score, "total_spend": 50 + rng.normal(0, 5, n)})


def _hits(text):
    canaries = ["CANARY", *NUMBER_CANARIES, *(f"{float(c):.4g}" for c in NUMBER_CANARIES),
                *(f"{float(c):.2f}" for c in NUMBER_CANARIES)]
    return [c for c in canaries if c in text]


def test_no_level_or_value_reaches_anything(fake_model, tmp_path, store):
    prompts, config = fake_model
    config.update(question_type=QuestionType.TWO_GROUP_COMPARISON, predictors=["region"])
    state = {"request_text": "Did West spend more?", "session_id": "s",
             "dataset": _dataset(_canary_frame(), tmp_path, store, "e")}
    values, pauses, _ = _drive(build_agent_graph(store), state, "e")
    assert [p["type"] for p in pauses] == ["stage_plan_approval", "wrangling_approval", "plan_approval", "stage_gate"]
    assert prompts and all(_hits(p["text"]) == [] for p in prompts)
    assert all(_hits(json.dumps(p["payload"], default=str)) == [] for p in pauses)
    assert all(_hits(json.dumps(r)) == [] for r in values["decision_log"])  # the scan's issue evidence included
    assert all(_hits(json.dumps(e)) == [] for e in values["stage_reports"])
    (proposer,) = [p["text"] for p in prompts if p["stage"] == "_WranglingProposalsOut"]
    assert "Column 'segment' has 3 distinct values that reduce to 2 after trimming spaces and ignoring case." in proposer
    assert re.search(r"Column 'score' is stored as text but \d+\.\d% of its values are numbers\.", proposer)
    for line in ("      trim_text: Trim spaces in segment.\n", "      trim_and_lowercase: Trim spaces and ignore case in segment.\n",
                 "      convert_to_number: Convert score to numbers.\n"):
        assert line in proposer
    for op in _wrangling(pauses)["proposed_operations"]:
        for c in op["candidates"]:
            if c["eligible"] and "(" in c["what_was_run"]:
                assert c["what_was_run"] not in proposer, c["id"]  # the sentence with preview numbers
    assert "distinct values become" not in proposer and "become missing" not in proposer


# --- (f) stakeholder mode --------------------------------------------------------------------------------------------

def test_stakeholder_mode_interrupts_for_the_text_fixes(fake_model, tmp_path, store):
    df = _four_issues().drop(columns="visits")  # no missing values: the text issues and the duplicates
    state = {"request_text": "Did West spend more?", "session_id": "s", "dataset": _dataset(df, tmp_path, store, "f"),
             "autonomy_mode": "stakeholder"}
    values, pauses, _ = _drive(_app(store, by_type), state, "f")
    assert [o["candidate_id"] for o in _wrangling(pauses)["proposed_operations"]] == [
        "convert_to_number", "trim_and_lowercase"]  # the dedupe was auto-approved
    assert [a["chosen"] for a in _records(values, "stage1.clean.approval")].count("auto_approve") == 1
    assert [e["evidence"]["op_type"] for e in _records(values, "stage1.clean.execution")] == [
        "cast", "normalize_text", "dedupe"]


# --- (g) the analysis sees the cleaned data --------------------------------------------------------------------------

def test_the_analysis_sees_the_cleaned_data(fake_model, tmp_path, store):
    rng = np.random.default_rng(14)
    region = np.array(["East", "east ", "West", "WEST"] * 50)
    df = pd.DataFrame({"region": region,
                       "total_spend": 50 + np.where(np.isin(region, ["West", "WEST"]), 6.0, 0.0) + rng.normal(0, 5, 200)})
    values, _, _ = _run(store, tmp_path, df, "g", by_type)
    profile = values["profile"]
    assert next(c for c in profile.columns if c.name == "region").distinct_count == 2
    assert profile == values["profiles"][values["analysis_table"]]  # the mirror holds (compared by value, G67)
    assert (values["route"]["capability"], values["route"]["status"]) == ("two_group", "ok")
    result = values["test_result"]
    assert (result["group_a_label"], result["group_b_label"]) == ("east", "west")


# --- (h) a rejected conversion ---------------------------------------------------------------------------------------

def test_a_rejected_conversion_is_unaddressed(fake_model, tmp_path, store):
    df = pd.DataFrame({"region": ["East", "West"] * 60, "total_spend": np.linspace(40, 60, 120),
                       "tenure": [f"{v}.5" for v in range(119)] + ["unknown"]})

    def reject_convert(payload):
        return {"decisions": [{"index": op["index"], "reason": "the codes matter",
                               "action": "reject" if op["candidate_id"] == "convert_to_number" else "approve"}
                              for op in payload["proposed_operations"]]}
    values, pauses, _ = _run(store, tmp_path, df, "h", by_type, wrangling_approval=reject_convert)
    description = f"Column 'tenure' is stored as text but {119 / 120:.1%} of its values are numbers."
    approval = next(p["payload"] for p in pauses if p["type"] == "plan_approval")
    assert approval["unaddressed_issues"] == [{"issue_type": "numeric_as_text", "description": description,
                                               "affected_columns": ["tenure"], "reason": "the codes matter"}]
    summary = next(e for e in values["stage_reports"] if e["stage"] == "summary")
    caveats = next(s["text"] for s in summary["sections"] if s["heading"] == "Caveats").split("\n")
    assert f"numeric_as_text ({description}) was left unaddressed because: the codes matter" in caveats
    assert _records(values, "stage1.clean.execution") == []


# --- (i) a text column above the level bound -------------------------------------------------------------------------

def test_a_wide_text_column_is_not_examined(fake_model, tmp_path, store):
    prompts, _ = fake_model
    df = pd.DataFrame({"region": ["East", "West"] * 60, "total_spend": np.linspace(40, 60, 120),
                       "code": [f"WIDE-CANARY-{i % 25} " for i in range(120)]})  # 25 levels, every value with a space
    values, pauses, _ = _run(store, tmp_path, df, "i", by_type)
    assert "wrangling_approval" not in [p["type"] for p in pauses]
    (scan,) = _records(values, "stage1.clean.scan")
    assert (scan["chosen"], scan["evidence"]["text_issues"], scan["evidence"]["numeric_text_columns"]) == ("skipped", [], [])
    assert prompts and all("WIDE-CANARY" not in p["text"] for p in prompts)


# --- (j) the registry ------------------------------------------------------------------------------------------------

def test_the_registry_is_unchanged():
    assert (len(registered_points()), len(DECISION_NODES)) == (32, 18)  # M2.4a: replan (planner.stage_replan)
