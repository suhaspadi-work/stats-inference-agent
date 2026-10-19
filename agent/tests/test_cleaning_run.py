"""
M2.3a: cleaning runs through the real graph (guard fakes, fake chat model). The operation that runs is the chosen
code-enumerated candidate; duplicates run first; every preview is recomputed on the data as its operation will see
it, so with every operation approved and none edited each preview equals the executed impact; an imputed outcome
becomes a plain caveat on the plan, at plan approval, in the report prompt (count-free) and in the summary.
"""
import dataclasses
import json
import re

import numpy as np
import pandas as pd
import pytest

import core.capabilities as capabilities_module
import core.nodes as nodes_module
from core.dataset import LocalDiskStore
from core.decision_points import DECISION_NODES, registered_points
from core.graph import build_agent_graph
from core.plan import QuestionType
from core.wrangling import transform_filter
from tests.test_decision_point_guard import _dataset, choose, confident, fake_draft, fake_propose
from tests.test_exploratory_round import _frame
from tests.test_prompt_canaries import _base_df, _with_duplicates, canary_hits, fake_model  # noqa: F401 (fixture)
from tests.test_stage1_report import _drive

QUESTION = "Did West spend more?"
OUTCOME_CAVEAT = ("The outcome 'total_spend' had missing values filled (median) before the analysis, which changes its "
                  "distribution and the result.")


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _missing(df, column, rows):
    df = df.copy()
    df.loc[list(rows), column] = np.nan
    return df


def _tenure_missing(n_missing=15):
    return _missing(_frame(), "tenure", range(n_missing))


def _app(store, propose_fn, outcome="total_spend", predictors=("region",)):
    return build_agent_graph(store, classify_fn=confident(QuestionType.TWO_GROUP_COMPARISON),
                             draft_fn=fake_draft(list(predictors), outcome), propose_fn=propose_fn)


def _run(store, tmp_path, df, name, propose_fn, outcome="total_spend", **overrides):
    state = {"request_text": QUESTION, "session_id": "s", "dataset": _dataset(df, tmp_path, store, name)}
    return _drive(_app(store, propose_fn, outcome), state, name, **overrides)


def _records(values, point):
    return [r for r in values["decision_log"] if r["decision_point"] == point]


def _wrangling(pauses):
    return next(p["payload"] for p in pauses if p["type"] == "wrangling_approval")


def _sections(values):
    stage1 = next(e for e in values["stage_reports"] if e["stage"] == "stage1")
    return {s["heading"]: s["text"] for s in stage1["sections"]}


def _reversed(propose_fn):
    """The same choices, listed in reverse issue order (the missing-value issue first)."""
    return lambda issues, candidates: list(reversed(propose_fn(issues, candidates)))


# --- (a) the operation is the chosen candidate --------------------------------------------------------------------------

def test_the_operation_comes_from_the_chosen_candidate(fake_model, tmp_path, store):
    values, pauses, _ = _run(store, tmp_path, _tenure_missing(), "a", choose({"high_missingness": "impute_median"}))
    (op,) = _wrangling(pauses)["proposed_operations"]
    candidate = next(c for c in op["candidates"] if c["id"] == "impute_median")
    assert (op["op_type"], op["params"], op["candidate_id"]) == ("impute", {"column": "tenure", "strategy": "median"},
                                                                "impute_median")
    (execution,) = _records(values, "stage1.clean.execution")
    assert execution["evidence"]["op_type"] == "impute" and execution["evidence"]["params"] == op["params"]
    assert candidate["preview"] == op["impact_preview"]


# --- (b) the dependency order, and the preview equals the executed impact -----------------------------------------------

def _probe_frame():
    """10 missing tenure values, plus 20 exact copies of a row whose tenure is missing (the part 1 probe case)."""
    df = _missing(_frame(seed=5), "tenure", range(10))
    return pd.concat([df] + [df.iloc[[0]]] * 20, ignore_index=True)


def test_duplicates_run_first_and_previews_equal_the_impact(fake_model, tmp_path, store):
    df = _probe_frame()
    assert int(df["tenure"].isna().sum()) == 30 and int(df.drop_duplicates()["tenure"].isna().sum()) == 10
    proposer = _reversed(choose({"duplicate_rows": "dedupe_all", "high_missingness": "impute_median"}))
    values, pauses, _ = _run(store, tmp_path, df, "b", proposer)
    first, second = _wrangling(pauses)["proposed_operations"]
    assert (first["index"], first["op_type"], second["index"], second["op_type"]) == (0, "dedupe", 1, "impute")
    assert (first["preview_basis"], second["preview_basis"]) == ("uploaded_data", "after_earlier_operations")
    executions = _records(values, "stage1.clean.execution")
    assert [e["evidence"]["op_type"] for e in executions] == ["dedupe", "impute"]
    for op, execution in zip((first, second), executions):
        ev = execution["evidence"]
        assert (op["impact_preview"]["rows_before"], op["impact_preview"]["rows_after"]) == (ev["rows_before"], ev["rows_after"])
    assert first["impact_preview"]["rows_changed"] == executions[0]["evidence"]["rows_before"] - executions[0]["evidence"]["rows_after"]
    details = executions[1]["evidence"]["details"]
    assert second["impact_preview"]["cells_changed"] == details["cells_filled"] == 10  # the probe: 10, not 30
    assert second["impact_preview"]["shift"] == {k: details[k] if k != "column" else "tenure"
                                                 for k in ("column", "mean_before", "mean_after", "std_before", "std_after")}
    assert second["what_was_run"] == f"Fill the missing values of tenure with the median ({df.drop_duplicates()['tenure'].median():.4g})."


# --- (c) the first operation rejected ----------------------------------------------------------------------------------------

def _reject_first(payload):
    return {"decisions": [{"index": op["index"], "action": "reject" if op["index"] == 0 else "approve",
                           "reason": "keep the repeats"} for op in payload["proposed_operations"]]}


def test_a_rejected_first_operation(fake_model, tmp_path, store):
    df = _probe_frame()
    values, _, _ = _run(store, tmp_path, df, "c", choose({"duplicate_rows": "dedupe_all", "high_missingness": "impute_median"}),
                        wrangling_approval=_reject_first)
    (execution,) = _records(values, "stage1.clean.execution")
    ev = execution["evidence"]
    assert (ev["op_type"], ev["rows_before"], ev["rows_after"], ev["details"]["cells_filled"]) == ("impute", len(df), len(df), 30)
    assert f"Effect: {len(df)} to {len(df)} rows." in _sections(values)["Changed"]


# --- (d) an edit, and a bad edit -----------------------------------------------------------------------------------------------

def test_an_edit_and_a_bad_edit(fake_model, tmp_path, store):
    edited = {"column": "tenure", "strategy": "mean"}
    values, _, _ = _run(store, tmp_path, _tenure_missing(), "d", choose({"high_missingness": "impute_median"}),
                        wrangling_approval={"decisions": [{"index": 0, "action": "edit", "params": edited}]})
    (execution,) = _records(values, "stage1.clean.execution")
    assert execution["chosen"] == "applied" and execution["evidence"]["params"] == edited
    bad = {"column": "tenure", "strategy": "wild"}
    values, _, _ = _run(store, tmp_path, _tenure_missing(), "d2", choose({"high_missingness": "impute_median"}),
                        wrangling_approval={"decisions": [{"index": 0, "action": "edit", "params": bad}]})
    (execution,) = _records(values, "stage1.clean.execution")
    assert execution["chosen"] == "failed" and "Unsupported imputation strategy: wild" in execution["evidence"]["failure_reason"]
    assert values["plan"].status.value == "executed"


# --- (e) the review payload ------------------------------------------------------------------------------------------------------

def test_the_review_payload(fake_model, tmp_path, store):
    _, pauses, _ = _run(store, tmp_path, _missing(_tenure_missing(), "region", ()), "e",
                        choose({"high_missingness": "impute_median"}))
    payload = _wrangling(pauses)
    json.dumps(payload, allow_nan=False)
    (op,) = payload["proposed_operations"]
    assert set(op) == {"index", "op_type", "params", "rationale", "alternative", "candidate_id", "what_was_run",
                       "impact_preview", "preview_basis", "candidates"}
    assert op["alternative"] == "Leave the data as it is."
    for c in op["candidates"]:
        assert set(c) == {"id", "what_was_run", "eligible", "reason", "preview"}
        assert (c["preview"] is None) is (not c["eligible"])
    assert any(not c["eligible"] for c in op["candidates"])


# --- (f) the autonomy rule -------------------------------------------------------------------------------------------------------

def test_only_duplicates_are_auto_approved(fake_model, tmp_path, store):
    assert nodes_module._STAKEHOLDER_AUTO_APPROVE_ISSUE_TYPES == {"duplicate_rows"}
    df = _probe_frame()
    for choice in ("impute_median", "delete_rows", "drop_column"):
        state = {"request_text": QUESTION, "session_id": "s", "dataset": _dataset(df, tmp_path, store, f"f_{choice}"),
                 "autonomy_mode": "stakeholder"}
        app = _app(store, choose({"duplicate_rows": "dedupe_all", "high_missingness": choice}))
        _, pauses, _ = _drive(app, state, f"f_{choice}")
        ops = _wrangling(pauses)["proposed_operations"]
        assert [o["candidate_id"] for o in ops] == [choice], choice  # the dedupe was auto-approved


# --- (g) the fix record ----------------------------------------------------------------------------------------------------------

def test_the_fix_record(fake_model, tmp_path, store):
    values, pauses, _ = _run(store, tmp_path, _tenure_missing(), "g", choose({"high_missingness": "impute_median"}))
    (fix,) = _records(values, "stage1.clean.fix")
    (op,) = _wrangling(pauses)["proposed_operations"]
    assert fix["chosen"] == "impute_median"
    assert [(c["option"], c["eligible"]) for c in fix["candidates"]] == [(c["id"], c["eligible"]) for c in op["candidates"]]
    ev = fix["evidence"]
    assert (ev["candidate_id"], ev["what_was_run"], ev["preview"]) == ("impute_median", op["what_was_run"], op["impact_preview"])
    assert (ev["alternative_candidate_id"], ev["alternative_rejected_because"]) == ("leave_as_is", "it would bias the result")
    assert (len(registered_points()), len(DECISION_NODES)) == (31, 17)


# --- (h) the Stage 1 report of an imputation -----------------------------------------------------------------------------------

def test_the_stage1_report_of_an_imputation(fake_model, tmp_path, store):
    df = _tenure_missing()
    values, _, _ = _run(store, tmp_path, df, "h", choose({"high_missingness": "impute_median"}))
    median = df["tenure"].median()
    assert _sections(values)["Changed"] == (
        f"1. Fill the missing values of tenure with the median ({median:.4g}). Why (the model's reason): fix "
        "high_missingness. Alternative the model suggested: Leave the data as it is (it would bias the result). "
        f"Your decision: approved. Effect: {len(df)} to {len(df)} rows.")
    assert "Missing values after cleaning: none." in _sections(values)["Data quality"].split("\n")


# --- (i) the imputed outcome ---------------------------------------------------------------------------------------------------------

def test_an_imputed_outcome_is_a_caveat(fake_model, tmp_path, store):
    prompts, _ = fake_model
    df = _missing(_with_duplicates_of(_frame()), "total_spend", range(15))
    filled = int(df.drop_duplicates()["total_spend"].isna().sum())

    def reject_dedupe(payload):
        return {"decisions": [{"index": op["index"], "reason": "keep the repeats",
                               "action": "reject" if op["op_type"] == "dedupe" else "approve"}
                              for op in payload["proposed_operations"]]}
    values, pauses, _ = _run(store, tmp_path, df, "i", choose({"duplicate_rows": "dedupe_all",
                                                               "high_missingness": "impute_median"}),
                             wrangling_approval=reject_dedupe)
    caveats = [u for u in values["plan"].unaddressed_issues if u.issue_type == "outcome_imputed"]
    assert [u.description for u in caveats] == [OUTCOME_CAVEAT]
    approval = next(p["payload"] for p in pauses if p["type"] == "plan_approval")
    assert [u["issue_type"] for u in approval["unaddressed_issues"]] == ["duplicate_rows", "outcome_imputed"]
    assert approval["unaddressed_issues"][1] == {"issue_type": "outcome_imputed", "description": OUTCOME_CAVEAT,
                                                 "affected_columns": ["total_spend"],
                                                 "reason": "the values were filled in Stage 1, before the outcome was chosen"}
    report_prompts = [p["text"] for p in prompts if p["stage"] == "report"]
    assert report_prompts and all(OUTCOME_CAVEAT in t for t in report_prompts)
    assert all(not re.search(rf"\b{filled}\b missing", t) for t in report_prompts)
    summary = next(e for e in values["stage_reports"] if e["stage"] == "summary")
    lines = next(s["text"] for s in summary["sections"] if s["heading"] == "Caveats").split("\n")
    assert OUTCOME_CAVEAT in lines and not any(line.startswith("outcome_imputed") for line in lines)


def _with_duplicates_of(df):
    return pd.concat([df, df.iloc[[50, 51]]], ignore_index=True)


@pytest.mark.parametrize("column, choice", [("tenure", "impute_median"), ("total_spend", "delete_rows")],
                         ids=["another_column", "deleted_rows"])
def test_no_caveat(column, choice, fake_model, tmp_path, store):
    values, _, _ = _run(store, tmp_path, _missing(_frame(), column, range(15)), f"no_{column}",
                        choose({"high_missingness": choice}))
    assert [u for u in values["plan"].unaddressed_issues if u.issue_type == "outcome_imputed"] == []


# --- (j) privacy ------------------------------------------------------------------------------------------------------------------------

def test_no_preview_or_fill_value_reaches_a_prompt(fake_model, tmp_path, store):
    prompts, config = fake_model
    config.update(question_type=QuestionType.TWO_GROUP_COMPARISON, predictors=["region"])
    df = _missing(_with_duplicates(_base_df()), "tenure", range(0, 360, 9))
    state = {"request_text": QUESTION, "session_id": "s", "dataset": _dataset(df, tmp_path, store, "j")}
    values, _, _ = _drive(build_agent_graph(store), state, "j")
    assert canary_hits(prompts) == []
    numbers = []
    for fix in _records(values, "stage1.clean.fix"):
        for c in fix["candidates"]:
            preview = (c.get("evidence") or {}).get("preview") or {}
            shift = preview.get("shift") or {}
            numbers += [v for k, v in shift.items() if k != "column" and v is not None]
            numbers += re.findall(r"\(([-0-9.e+]+)\)", (c.get("evidence") or {}).get("what_was_run") or "")
    assert numbers
    for p in prompts:
        for number in numbers:
            for form in ({repr(number), f"{number:.4g}"} if isinstance(number, float) else {number}):
                assert form not in p["text"], (p["stage"], form)


# --- (k) a transform that fails during the preview recomputation ---------------------------------------------------------------------

def test_a_failing_recomputation_keeps_the_original_preview(fake_model, tmp_path, store, monkeypatch):
    calls = {"n": 0}

    def flaky(df, params):
        calls["n"] += 1
        if calls["n"] > 1:
            raise ValueError("flaky preview")
        return transform_filter(df, params)
    entry = capabilities_module.OPERATIONS["filter"]
    monkeypatch.setitem(capabilities_module.OPERATIONS, "filter", dataclasses.replace(entry, transform=flaky))
    df = _tenure_missing()
    values, pauses, _ = _run(store, tmp_path, df, "k", fake_propose)
    (op,) = _wrangling(pauses)["proposed_operations"]
    assert op["candidate_id"] == "delete_rows" and calls["n"] >= 2
    assert op["impact_preview"]["rows_after"] == int(df["tenure"].notna().sum()) and op["preview_basis"] == "uploaded_data"
    (execution,) = _records(values, "stage1.clean.execution")
    assert execution["chosen"] == "applied" and values["plan"].status.value == "executed"


# --- the mode of a withheld text column is never shown (M2.3a privacy fix) ----------------------------------------------------

def test_a_withheld_text_mode_is_shown_nowhere(fake_model, tmp_path, store):
    prompts, _ = fake_model
    df = _frame()
    df["customer_name"] = [f"MODE-CANARY-{i % 30}" for i in range(len(df))]
    df.loc[list(range(0, 120, 8)), "customer_name"] = None  # 15 of 120 missing, 30 distinct names: withheld
    values, pauses, _ = _run(store, tmp_path, df, "withheld", choose({"high_missingness": "impute_mode"}))
    canary = "MODE-CANARY"
    (op,) = _wrangling(pauses)["proposed_operations"]
    assert op["candidate_id"] == "impute_mode" and op["what_was_run"] == "Fill the missing values of customer_name with the mode."
    assert canary not in json.dumps(_wrangling(pauses))
    (fix,) = _records(values, "stage1.clean.fix")
    (execution,) = _records(values, "stage1.clean.execution")
    assert execution["chosen"] == "applied" and execution["evidence"]["details"]["fill_value"] is None
    assert canary not in json.dumps(fix) and canary not in json.dumps(execution)
    stage1 = next(e for e in values["stage_reports"] if e["stage"] == "stage1")
    assert canary not in stage1["text"] and canary not in json.dumps(stage1["sections"])
    assert prompts and all(canary not in p["text"] for p in prompts)


def test_the_exploratory_round_keeps_the_outcome_caveat(fake_model, tmp_path, store):
    prompts, _ = fake_model
    df = _missing(_frame(), "total_spend", range(15))
    values, pauses, _ = _run(store, tmp_path, df, "explore", choose({"high_missingness": "impute_median"}),
                             stage_gate={"action": "run"})
    assert "stage_gate" in [p["type"] for p in pauses] and len(values["rounds"]) == 2
    second = values["rounds"][1]
    assert second["exploratory"] and second["capability"] == "regression"
    assert [u.description for u in second["plan"].unaddressed_issues if u.issue_type == "outcome_imputed"] == [OUTCOME_CAVEAT]
    report_prompts = [p["text"] for p in prompts if p["stage"] == "report"]
    assert len(report_prompts) == 2 and OUTCOME_CAVEAT in report_prompts[1]
    assert "You write brief, plain-language summaries of a multiple linear regression" in report_prompts[1]
