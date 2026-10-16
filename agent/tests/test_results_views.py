"""
M1.5 step P3a: the pure view logic of the finished screen, the data overview and the history
(product/results_views.py), on REAL final states: the real graph with fake models.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "product"))

import results_views as rv  # noqa: E402
from pause_views import method_label  # noqa: E402

from core.dataset import LocalDiskStore  # noqa: E402
from core.method_glossary import GLOSSARY  # noqa: E402
from core.plan import QuestionType  # noqa: E402
from tests.test_analysis_reports import (  # noqa: E402
    BAD, GLOSSARY_DETAIL, _app, _run, _three_groups, _with_id, report_texts,  # noqa: F401 (fixture)
)
from tests.test_decision_point_guard import fake_propose  # noqa: E402
from tests.test_exploratory_round import _frame  # noqa: E402
from tests.test_prompt_canaries import fake_model  # noqa: E402,F401 (fixture)
from tests.test_stage1_report import _drive, _start  # noqa: E402

CANARY = "CANARY-CODE"


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _with_code(n=120, seed=51):
    df = _frame(seed=seed, n=n)
    df["customer_code"] = [f"{CANARY}-{i:04d}" for i in range(n)]  # 120 distinct strings: withheld
    return df


def _rows(rows):
    return {r["label"]: r["value"] for r in rows}


f = rv.format_number


# --- format_number and strip_charts -------------------------------------------------------------------------------------

def test_format_number():
    assert [f(None), f(True), f(False), f(3), f(np.int64(5)), f(0.123456), f(float("nan")), f(np.float64(2.5)),
            f(12345.678), f("x")] == ["n/a", "True", "False", "3", "5", "0.123", "n/a", "2.5", "1.23e+04", "x"]


def test_strip_charts():
    value = {"a": ({"chart_data": [1], "b": (1, 2)},), "chart_data": {"x": 1}, "c": [{"d": {"chart_data": 0, "e": 1}}]}
    before = copy.deepcopy(value)
    assert rv.strip_charts(value) == {"a": [{"b": [1, 2]}], "c": [{"d": {"e": 1}}]}
    assert value == before


# --- statistics_rows ---------------------------------------------------------------------------------------------------

def test_two_group_statistics(fake_model, tmp_path, store):
    values, _, _ = _run(store, tmp_path, _frame(), "two")
    r = values["test_result"]
    rows = _rows(rv.statistics_rows(r))
    assert rows["Method"] == GLOSSARY[r["method"]]["name"]
    a, b = r["group_a_label"], r["group_b_label"]
    assert r["method"] in ("student_t_test", "welch_t_test")
    assert rows["p-value"] == f(r["p_value"])
    assert rows[f"Effect size (Cohen's d, positive when {b} is higher)"] == f(r["effect_size"])
    lo, hi = r["confidence_interval"]
    assert rows[f"95% confidence interval for {b} minus {a}"] == f"{f(lo)} to {f(hi)}"
    assert rows[r["group_a_label"]] == f"mean {f(r['group_a_mean'])} (n = {r['group_a_n']})"
    assert rows[r["group_b_label"]] == f"mean {f(r['group_b_mean'])} (n = {r['group_b_n']})"


@pytest.mark.parametrize("means, seed, significant", [((50, 60, 70), 81, True), ((50, 50, 50), 82, False)],
                         ids=["significant", "not_significant"])
def test_multi_group_statistics(means, seed, significant, fake_model, tmp_path, store):
    values, _, _ = _run(store, tmp_path, _three_groups(means, seed), "multi", QuestionType.MULTI_GROUP_COMPARISON)
    r = values["test_result"]
    assert (r["p_value"] < 0.05) is significant
    rows = rv.statistics_rows(r)
    by = _rows(rows)
    assert by["p-value"] == f(r["p_value"])
    for group, mean in r["group_means"].items():
        assert by[group] == f"mean {f(mean)} (n = {r['group_ns'][group]})"
    pair_rows = [row for row in rows if " vs " in row["label"]]
    assert len(pair_rows) == len(r["pairwise_comparisons"]) == (3 if significant else 0)
    for row, c in zip(pair_rows, r["pairwise_comparisons"]):
        assert row["label"] == f"{c['group_a']} vs {c['group_b']}"
        assert row["value"].startswith(f"difference {f(c['mean_diff'])}, p = {f(c['p_value'])}")
        assert row["value"].endswith(", significant") is c["significant"]


def test_regression_statistics(fake_model, tmp_path, store):
    values, _, _ = _run(store, tmp_path, _frame(), "reg", QuestionType.DRIVER_ANALYSIS, ("tenure", "tier"))
    r = values["test_result"]
    rows = rv.statistics_rows(r)
    by = _rows(rows)
    assert (by["R-squared"], by["Adjusted R-squared"], by["Overall F-test p-value"], by["Observations"]) == (
        f(r["r_squared"]), f(r["adj_r_squared"]), f(r["f_p_value"]), str(r["n_observations"]))
    coefficient_rows = rows[5:]
    assert [row["label"] for row in coefficient_rows] == [c["name"] for c in r["coefficients"]]
    assert coefficient_rows[0]["value"] == f"estimate {f(r['coefficients'][0]['estimate'])}, p = {f(r['coefficients'][0]['p_value'])}"


def test_the_effect_size_labels():
    mw = {"result_type": "two_group", "method": "mann_whitney_u", "effect_size": 0.25, "confidence_interval": (1.0, 2.0),
          "group_a_label": "East", "group_b_label": "West"}
    labels = [r["label"] for r in rv.statistics_rows(mw)]
    assert "Effect size (rank-biserial correlation, positive when West tends higher)" in labels
    assert "95% confidence interval for West minus East" in labels
    bare = {"result_type": "two_group", "method": "welch_t_test", "effect_size": 0.5, "confidence_interval": [0.1, 0.9]}
    labels = [r["label"] for r in rv.statistics_rows(bare)]
    assert labels == ["Method", "Effect size (Cohen's d, positive when the second group is higher)",
                      "95% confidence interval for the second group minus the first group"]


def test_the_caps_and_the_fallback():
    pairs = [{"group_a": f"G{i}", "group_b": f"H{i}", "mean_diff": 1.0, "p_value": 0.01, "significant": True}
             for i in range(15)]
    rows = rv.statistics_rows({"result_type": "multi_group", "method": "one_way_anova", "pairwise_comparisons": pairs})
    assert len([r for r in rows if " vs " in r["label"]]) == 10 and rows[-1] == {"label": "and 5 more", "value": ""}
    coefficients = [{"name": f"x{i}", "estimate": 1.0, "p_value": 0.5} for i in range(20)]
    rows = rv.statistics_rows({"result_type": "regression", "coefficients": coefficients})
    assert len(rows) == 13 and rows[-1] == {"label": "and 8 more", "value": ""}
    assert rv.statistics_rows({"row_count": 3}) == [{"label": "Result", "value": "see the report"}]
    assert rv.statistics_rows(None) == [{"label": "Result", "value": "see the report"}]
    assert rv.statistics_rows({"result_type": "two_group"}) == []  # every key missing: every row skipped


# --- overview_rows ------------------------------------------------------------------------------------------------------

def test_overview_rows(fake_model, tmp_path, store):
    values, _, _ = _run(store, tmp_path, _with_code(), "overview")
    result = values["descriptive_result"]
    rows = rv.overview_rows(result)
    assert [r["column"] for r in rows] == [c["name"] for c in result["column_stats"]]
    by = {r["column"]: r for r in rows}
    spend = next(c for c in result["column_stats"] if c["name"] == "total_spend")
    assert by["total_spend"]["type"] == "numeric"
    assert by["total_spend"]["summary"] == (f"mean {f(spend['mean'])}, median {f(spend['median'])}, std {f(spend['std'])}, "
                                            f"range {f(spend['min'])} to {f(spend['max'])}")
    tier = next(c for c in result["column_stats"] if c["name"] == "tier")
    assert by["tier"]["type"] == "text" and by["tier"]["summary"] == "most common: " + ", ".join(
        f"{level} ({count})" for level, count in tier["top_value_counts"][:3])
    assert by["customer_code"]["summary"] == rv.WITHHELD_SUMMARY and by["customer_code"]["distinct"] == "120"
    assert CANARY not in json.dumps(rows).replace('"customer_code"', "")
    assert all(r["missing"] == "n/a" for r in rows)
    assert rv.overview_rows(None) is None and rv.overview_rows({"column_stats": []}) is None


# --- finished_view -------------------------------------------------------------------------------------------------------

def test_a_two_group_run(fake_model, tmp_path, store):
    values, _, _ = _run(store, tmp_path, _frame(), "finished")
    view = rv.finished_view(values)
    assert view["kind"] == "report" and view["question"] == values["plan"].request_text
    assert view["summary"]["title"] == "Summary" and view["stage1"]["stage"] == "stage1"
    (round_view,) = view["rounds"]
    entry = next(e for e in values["stage_reports"] if e["stage"] == "analysis")
    assert round_view["narrative"] == values["report"] and round_view["round"] == 1
    assert [s["heading"] for s in round_view["folded_sections"]] == GLOSSARY_DETAIL
    assert [s["heading"] for s in round_view["visible_sections"]] == [
        s["heading"] for s in entry["sections"] if not s.get("collapsed")]
    assert round_view["explain_options"] == entry["explain_options"] and round_view["explain_options"]
    assert round_view["statistics"] == rv.statistics_rows(values["test_result"])
    assert round_view["method_label"] == method_label(values["plan"].primary_method)
    assert round_view["fallback"] is False and round_view["fallback_note"] is None and view["fallback_any"] is False
    assert view["overview"] == rv.overview_rows(values["descriptive_result"]) and view["overview"]


def _has_chart_data(value):
    if isinstance(value, dict):
        return "chart_data" in value or any(_has_chart_data(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_chart_data(v) for v in value)
    return False


def test_details(fake_model, tmp_path, store):
    values, _, _ = _run(store, tmp_path, _three_groups((50, 60, 70), 81), "multi", QuestionType.MULTI_GROUP_COMPARISON)
    before = copy.deepcopy(values["rounds"])
    assert _has_chart_data(values["rounds"][0]["result"])
    (round_view,) = rv.finished_view(values)["rounds"]
    assert round_view["details"] and not _has_chart_data(round_view["details"])
    assert round_view["details"]["p_value"] == values["test_result"]["p_value"]
    assert values["rounds"] == before

    values, _, _ = _run(store, tmp_path, _frame(), "two")
    (round_view,) = rv.finished_view(values)["rounds"]
    assert round_view["details"] == rv.strip_charts(values["rounds"][0]["result"])
    assert round_view["details"]["confidence_interval"] == list(values["test_result"]["confidence_interval"])

    values, _, _ = _run(store, tmp_path, _frame(), "describe", QuestionType.DESCRIPTIVE, ())
    (round_view,) = rv.finished_view(values)["rounds"]
    assert round_view["details"] is None


def test_an_exploratory_run(fake_model, tmp_path, store):
    values, _, _ = _run(store, tmp_path, _with_id(), "explore", stage_gate={"action": "run"})
    first, second = rv.finished_view(values)["rounds"]
    assert (first["round"], first["exploratory"], second["round"], second["exploratory"]) == (1, False, 2, True)
    assert first["narrative"] == values["rounds"][0]["report"] != values["report"]
    assert second["narrative"] == values["rounds"][1]["report"] == values["report"]
    assert "R-squared" in _rows(second["statistics"])
    assert any(label.startswith("Effect size (") for label in _rows(first["statistics"]))


def test_a_fallback_narrative(report_texts, tmp_path, store):
    report_texts += [BAD]
    values, _, _ = _run(store, tmp_path, _frame(), "fallback")
    view = rv.finished_view(values)
    (round_view,) = view["rounds"]
    assert round_view["fallback"] is True and view["fallback_any"] is True
    assert round_view["fallback_note"] == ("This narrative was written from the computed facts, because the model's "
                                           "own text used causal language twice.")


def test_a_descriptive_question(fake_model, tmp_path, store):
    values, _, _ = _run(store, tmp_path, _frame(), "describe", QuestionType.DESCRIPTIVE, ())
    view = rv.finished_view(values)
    assert view["kind"] == "report" and view["summary"] is not None
    (round_view,) = view["rounds"]
    assert round_view["round"] == values["rounds"][-1]["round"] and round_view["analysis"] is None
    assert round_view["narrative"] == values["report"] and round_view["explain_options"] == []
    assert view["overview"] == rv.overview_rows(values["descriptive_result"]) and view["overview"]


def test_a_rejected_plan(fake_model, tmp_path, store):
    values, _, _ = _run(store, tmp_path, _frame(), "reject", plan_approval={"action": "reject", "reason": "no"})
    view = rv.finished_view(values)
    assert view == {"kind": "no_report", "message": values["clarification_needed"]}
    assert view["message"].startswith("Plan rejected")


def test_an_unsupported_question(fake_model, tmp_path, store):
    values, _, _ = _run(store, tmp_path, _frame(), "unsupported", QuestionType.ASSOCIATION, ("region", "tier"))
    view = rv.finished_view(values)
    assert view["kind"] == "no_report" and view["message"] == values["clarification_needed"] and view["message"]
    assert rv.finished_view({}) == {"kind": "no_report", "message": rv.NO_REPORT_MESSAGE}


def test_a_second_question_shows_only_its_own(fake_model, tmp_path, store):
    app = _app(store, propose_fn=fake_propose)
    _drive(app, _start(_frame(seed=61), tmp_path, store, "q1"), "thread")
    values, _, _ = _drive(app, _start(_frame(seed=62, n=100), tmp_path, store, "q2"), "thread")
    view = rv.finished_view(values)
    stage1_entries = [e for e in values["stage_reports"] if e["stage"] == "stage1"]
    summaries = [e for e in values["stage_reports"] if e["stage"] == "summary"]
    assert len(stage1_entries) == 2 and len(summaries) == 2
    assert view["stage1"] is stage1_entries[-1]
    assert view["summary"]["text"] == summaries[-1]["text"] != summaries[0]["text"]
    assert [r["round"] for r in view["rounds"]] == [2] and view["rounds"][0]["narrative"] == values["report"]


def test_a_stale_overview_is_not_shown(fake_model, tmp_path, store):
    app = _app(store, propose_fn=fake_propose)
    first, _, _ = _drive(app, _start(_frame(seed=61), tmp_path, store, "d1"), "describe-off")
    assert rv.finished_view(first)["overview"]
    values, _, _ = _drive(app, _start(_frame(seed=62, n=100), tmp_path, store, "d2"), "describe-off",
                          stage_plan_approval={"action": "approve", "run": {"describe": False}})
    assert values.get("descriptive_result") is not None  # the first question's result is still in state
    assert rv.finished_view(values)["overview"] is None


# --- history_view ---------------------------------------------------------------------------------------------------------

def _message(values, message_id=5):
    return {"id": message_id, "question_text": values["plan"].request_text, "report_text": values["report"],
            "test_result_json": json.dumps(values["test_result"], default=str),
            "stage_reports_json": json.dumps(values["stage_reports"], default=str), "created_at": "2026-10-16T10:00:00"}


def test_history_view(fake_model, tmp_path, store):
    values, _, _ = _run(store, tmp_path, _frame(), "history")
    live = rv.finished_view(values)
    explanations = [
        {"message_id": 5, "round": 1, "method_id": "student_t_test", "question": None, "answer": "a1", "source": "model"},
        {"message_id": 6, "round": 1, "method_id": "ols", "question": "q", "answer": "other", "source": "model"},
        {"message_id": 5, "round": 1, "method_id": "welch_t_test", "question": "why not?", "answer": "a2",
         "source": "model_retry"},
    ]
    view = rv.history_view(_message(values), explanations)
    assert (view["question"], view["report"], view["created_at"]) == (
        values["plan"].request_text, values["report"], "2026-10-16T10:00:00")
    assert view["summary"]["text"] == live["summary"]["text"]
    (analysis,) = view["analysis"]
    assert analysis["visible_sections"] == live["rounds"][0]["visible_sections"]
    assert analysis["folded_sections"] == live["rounds"][0]["folded_sections"]
    assert view["statistics"] == live["rounds"][0]["statistics"]
    b = values["test_result"]["group_b_label"]
    assert any(r["label"] == f"Effect size (Cohen's d, positive when {b} is higher)" for r in view["statistics"])
    assert any(r["label"].startswith(f"95% confidence interval for {b} minus ") for r in view["statistics"])
    assert [e["answer"] for e in view["explanations"]] == ["a1", "a2"]
    assert view["explanations"][1] == {"round": 1, "method_label": method_label("welch_t_test"), "question": "why not?",
                                       "answer": "a2", "source": "model_retry"}


def test_history_view_of_an_old_or_broken_message():
    old = {"id": 1, "question_text": "q", "report_text": "r", "test_result_json": None, "stage_reports_json": None,
           "created_at": "t"}
    view = rv.history_view(old, [])
    assert (view["summary"], view["analysis"], view["statistics"], view["explanations"]) == (None, [], [], [])
    broken = {**old, "test_result_json": "{not json", "stage_reports_json": "[{"}
    view = rv.history_view(broken, None)
    assert (view["summary"], view["analysis"], view["statistics"]) == (None, [], [])
    odd = {**old, "test_result_json": "[1, 2]", "stage_reports_json": '{"stage": "summary"}'}
    view = rv.history_view(odd, [])
    assert (view["summary"], view["analysis"], view["statistics"]) == (None, [], [])
    view = rv.history_view({"id": 2}, [])  # a row without the newer columns
    assert view["question"] is None and view["summary"] is None


# --- current_overview, overview_table, explanation captions and items (M1.5 step P3b1) -----------------------------------

def test_current_overview_matches_the_finished_view(fake_model, tmp_path, store):
    two, _, _ = _run(store, tmp_path, _frame(), "cur_two")
    described, _, _ = _run(store, tmp_path, _frame(), "cur_describe", QuestionType.DESCRIPTIVE, ())
    for values in (two, described):
        assert rv.current_overview(values) == rv.finished_view(values)["overview"] and rv.current_overview(values)
    app = _app(store, propose_fn=fake_propose)
    _drive(app, _start(_frame(seed=61), tmp_path, store, "c1"), "cur-thread")
    values, _, _ = _drive(app, _start(_frame(seed=62, n=100), tmp_path, store, "c2"), "cur-thread",
                          stage_plan_approval={"action": "approve", "run": {"describe": False}})
    assert values.get("descriptive_result") is not None and rv.current_overview(values) is None
    assert rv.current_overview({}) is None and rv.current_overview(None) is None


def test_current_overview_of_a_rejected_plan(fake_model, tmp_path, store):
    # describe runs in stage 1, before plan approval, so the rejected question did produce its own overview
    values, _, _ = _run(store, tmp_path, _frame(), "cur_reject", plan_approval={"action": "reject", "reason": "no"})
    assert any(r["decision_point"] == "stage1.describe.execution" for r in values["decision_log"])
    assert rv.finished_view(values)["kind"] == "no_report"
    assert rv.current_overview(values) == rv.overview_rows(values["descriptive_result"])


def test_overview_table(fake_model, tmp_path, store):
    assert rv.overview_table(None) is None and rv.overview_table([]) is None
    values, _, _ = _run(store, tmp_path, _with_code(), "table")
    rows = rv.current_overview(values)
    before = copy.deepcopy(rows)
    table = rv.overview_table(rows)
    assert table["columns"] == ["Column", "Type", "Distinct values", "Summary"]  # every "missing" is n/a (G62)
    assert [list(r) for r in table["rows"]] == [table["columns"]] * len(rows)
    assert [r["Column"] for r in table["rows"]] == [r["column"] for r in rows]
    assert table["rows"][0]["Summary"] == rows[0]["summary"]
    assert rows == before


def test_overview_table_keeps_a_real_missing_column():
    rows = [{"column": "a", "type": "numeric", "distinct": "3", "missing": "n/a", "summary": "mean 1"},
            {"column": "b", "type": "text", "distinct": "2", "missing": "1 (10.0%)", "summary": "n/a"}]
    table = rv.overview_table(rows)
    assert table["columns"] == ["Column", "Type", "Distinct values", "Missing", "Summary"]
    assert table["rows"][1] == {"Column": "b", "Type": "text", "Distinct values": "2", "Missing": "1 (10.0%)",
                                "Summary": "n/a"}


def test_explanation_caption():
    assert rv.explanation_caption("model") == "Written by the model from the computed facts and the recorded reasons."
    assert rv.explanation_caption("model_retry") == "Written by the model after one correction."
    assert rv.explanation_caption("glossary_fallback") == (
        "The model's answer did not pass the checks, so this is the glossary description.")
    assert rv.explanation_caption("other") == "" and rv.explanation_caption(None) == ""


def test_explanation_items():
    explanations = {
        "1|student_t_test|": {"method_id": "student_t_test", "round": 1, "question": None, "answer": "a1", "source": "model"},
        "2|ols|": {"method_id": "ols", "round": 2, "question": None, "answer": "b", "source": "model"},
        "1|welch_t_test|why?": {"method_id": "welch_t_test", "round": 1, "question": "why?", "answer": "a2",
                                "source": "glossary_fallback", "saved": False},
    }
    before = copy.deepcopy(explanations)
    items = rv.explanation_items(explanations, 1)
    assert items == [
        {"method_label": GLOSSARY["student_t_test"]["name"], "question": None, "answer": "a1",
         "caption": rv.explanation_caption("model"), "saved": True},
        {"method_label": GLOSSARY["welch_t_test"]["name"], "question": "why?", "answer": "a2",
         "caption": rv.explanation_caption("glossary_fallback"), "saved": False},
    ]
    assert [i["answer"] for i in rv.explanation_items(explanations, 2)] == ["b"]
    assert rv.explanation_items(explanations, 3) == [] and rv.explanation_items(None, 1) == []
    assert rv.explanation_items({}, 1) == [] and explanations == before
