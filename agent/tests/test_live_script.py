"""
G54: the live check's failure handling. An exhausted Groq quota is not a code failure: the script exits 3 when
it is the only failure, 1 when any run failed for another reason, and 0 when every run passed.
"""
import pytest

from core.report import CausalLanguageViolation
from eval.check_live_capabilities import _resume_for, classify_error, filter_runs, quota_reset_hint, summarize

GROQ_TPD = ("Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in "
            "organization `org_x` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199990, "
            "Requested 1200. Please try again in 1h2m3.5s. Need more tokens? Upgrade to Dev Tier', "
            "'type': 'tokens', 'code': 'rate_limit_exceeded'}}")


class RateLimitError(Exception):
    pass


# --- classify_error ------------------------------------------------------------------------------------------

def test_a_rate_limit_class_is_quota():
    assert classify_error(RateLimitError("anything")) == "quota"


def test_the_groq_tpd_message_is_quota():
    assert classify_error(Exception(GROQ_TPD)) == "quota"


@pytest.mark.parametrize("exc", [ValueError("429 items"), ConnectionError("Connection error."),
                                 CausalLanguageViolation("Report uses causal language ('causes')")],
                         ids=["value_error_429", "connection", "causal_language"])
def test_other_errors_are_code(exc):
    assert classify_error(exc) == "code"


def test_a_429_without_the_daily_limit_wording_is_code():
    assert classify_error(Exception("Error code: 429 - rate limit reached on requests per minute")) == "code"


# --- quota_reset_hint -------------------------------------------------------------------------------------------

def test_quota_reset_hint():
    assert quota_reset_hint("Please try again in 1h2m3.5s. Need more tokens?") == "1h2m3.5s"
    assert quota_reset_hint(GROQ_TPD) == "1h2m3.5s"
    assert quota_reset_hint("Please try again in 7m12.5s'}") == "7m12.5s"
    assert quota_reset_hint("Rate limit reached") is None


# --- summarize -----------------------------------------------------------------------------------------------------

def _ok(label):
    return {"label": label, "error_kind": None, "problems": []}


def _quota(label, hint=None):
    return {"label": label, "error_kind": "quota", "problems": [], "hint": hint}


def test_all_clean():
    s = summarize([_ok("a"), _ok("b")])
    assert s == {"exit_code": 0, "lines": ["SUMMARY: 2 runs, 2 passed, 0 code failures, 0 quota-blocked"]}


def test_a_code_failure():
    s = summarize([_ok("a"), {"label": "b", "error_kind": "code", "problems": ["ValueError: boom", "later"]}])
    assert s["exit_code"] == 1
    assert s["lines"] == ["SUMMARY: 2 runs, 1 passed, 1 code failures, 0 quota-blocked",
                          "  b: FAILED (code): ValueError: boom"]


def test_only_quota_failures():
    s = summarize([_ok("a"), _quota("b", "1h2m3.5s"), _quota("c")])
    assert s["exit_code"] == 3
    assert s["lines"] == ["SUMMARY: 3 runs, 1 passed, 0 code failures, 2 quota-blocked",
                          "  b: BLOCKED (quota): not a code failure; retry after the reset 1h2m3.5s",
                          "  c: BLOCKED (quota): not a code failure; retry after the reset"]


def test_quota_and_code_mixed():
    s = summarize([_quota("a"), {"label": "b", "error_kind": "code", "problems": ["X"]}])
    assert s["exit_code"] == 1
    assert s["lines"][0] == "SUMMARY: 2 runs, 0 passed, 1 code failures, 1 quota-blocked"


def test_a_problem_without_an_error():
    s = summarize([{"label": "messy", "error_kind": None, "problems": ["lineage: FAIL (x)"]}])
    assert s["exit_code"] == 1
    assert s["lines"] == ["SUMMARY: 1 runs, 0 passed, 1 code failures, 0 quota-blocked",
                          "  messy: FAILED (code): lineage: FAIL (x)"]


def test_a_quota_run_with_a_problem_counts_as_code():
    s = summarize([{"label": "a", "error_kind": "quota", "problems": ["event log: FAIL (x)"]}])
    assert s["exit_code"] == 1


# --- the answers of the runs -------------------------------------------------------------------------------------

def test_the_edit_run_answers():
    counts = {}
    spa = {"type": "stage_plan_approval"}
    plan = {"type": "plan_approval", "candidate_predictors": ["region", "tenure", "visits"]}
    answers = [_resume_for("edit", p, counts, "q") for p in (spa, spa, plan, plan, plan)]
    assert answers == [{"action": "approve", "run": {"clean": False}}, {"action": "approve"},
                       {"action": "edit", "candidate_predictors": ["customer_id"]},
                       {"action": "edit", "candidate_predictors": ["tenure", "visits"]}, {"action": "approve"}]
    one = {"type": "plan_approval", "candidate_predictors": ["region"]}
    assert _resume_for("edit", one, {"plan_approval": 1}, "q") == {"action": "edit", "candidate_predictors": ["region"]}


def test_other_runs_answer_as_before():
    counts = {}
    assert _resume_for("two_group", {"type": "stage_plan_approval"}, counts, "q") == {"action": "approve"}
    assert _resume_for("two_group", {"type": "plan_approval"}, counts, "q") == {"action": "approve"}
    assert _resume_for("two_group", {"type": "stage_gate"}, counts, "q") == {"action": "skip"}
    assert _resume_for("exploratory", {"type": "stage_gate", "refused": None}, counts, "q") == {
        "action": "run", "predictors": ["region", "customer_id"]}
    assert _resume_for("exploratory", {"type": "stage_gate", "refused": {"attempt": 1}}, counts, "q") == {"action": "run"}
    assert _resume_for("x", {"type": "clarification_needed"}, counts, "the question") == "the question"


def test_filter_runs_unchanged():
    runs = [("a", None, "q1"), ("b", None, "q2")]
    assert filter_runs(runs, []) == runs and filter_runs(runs, ["b"]) == [runs[1]]
