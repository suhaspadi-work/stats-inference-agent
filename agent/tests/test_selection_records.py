import inspect
import json

import numpy as np
import pandas as pd
import pytest
from langgraph.graph import StateGraph, START, END

import core.testing as testing_module
from core.agent_state import AgentState
from core.capabilities import TWO_GROUP, capability_for_question_type
from core.dataset import LocalDiskStore
from core.decision_log import log_update
from core.nodes import make_select_method_node
from core.plan import (
    AlternativeConsidered, AnalysisPlan, CausalStatus, MethodDecision, OutcomeType,
    QuestionType, UnaddressedIssue,
)
from core.selection_records import ALPHA, two_group_selection_record
from core.testing import select_two_group_method
from tests.test_testing import make_dataset


def _decision(method, a_p=0.21, b_p=0.35, levene=0.45, n_a=40, n_b=38,
              a_label="West", b_label="East", chart_data=(),
              a_passes=None, b_passes=None, variances_equal=None, **extra):
    """The selector's flags default to p > 0.05 on the given values; pass them to override (G30)."""
    evidence = {
        "shapiro_p_group_a": a_p, "shapiro_p_group_b": b_p, "levene_p": levene,
        "n_group_a": n_a, "n_group_b": n_b, "group_a_label": a_label, "group_b_label": b_label,
        "group_a_passes_normality": a_p > 0.05 if a_passes is None else a_passes,
        "group_b_passes_normality": b_p > 0.05 if b_passes is None else b_passes,
        "variances_equal": levene > 0.05 if variances_equal is None else variances_equal,
        **extra,
    }
    return MethodDecision(
        stage="test_selection", chosen_method=method, rationale="selector rationale",
        alternatives=(AlternativeConsidered(method="x", rejected_because="y"),),
        evidence=evidence, chart_data=chart_data,
    )


def _by_option(record):
    return {c.option: c for c in record.candidates}


# --- scenarios ----------------------------------------------------------------

def test_a_normal_equal_variance_chooses_student():
    r = two_group_selection_record(_decision("student_t_test"))
    assert r.decision_point == "stage2.test_selection"
    assert r.chosen == "student_t_test"
    assert r.decided_by == "code_rule"
    assert r.rationale == "selector rationale"
    c = _by_option(r)
    assert [c[m].eligible for m in ("student_t_test", "welch_t_test", "mann_whitney_u")] == [True, True, True]
    assert c["student_t_test"].reason == (
        "chosen: both groups pass the normality check (West n=40: Shapiro-Wilk p = 0.2100; "
        "East n=38: Shapiro-Wilk p = 0.3500) and Levene's p = 0.4500 is above 0.05")
    assert "Levene's p = 0.4500 is above 0.05" in c["welch_t_test"].reason
    assert "standard t-test is used" in c["welch_t_test"].reason
    assert c["mann_whitney_u"].reason.startswith("eligible, since it assumes no distribution, but both groups pass")
    assert c["mann_whitney_u"].reason.endswith("so the selection rule prefers a t test")
    assert r.evidence["family"] == "two_group"
    assert r.evidence["small_groups_unchecked"] == []


def test_b_normal_unequal_variance_chooses_welch():
    r = two_group_selection_record(_decision("welch_t_test", levene=0.003))
    c = _by_option(r)
    assert r.chosen == "welch_t_test"
    assert c["student_t_test"].eligible is False
    assert c["student_t_test"].reason == (
        "not eligible: assumes equal variances, but Levene's p = 0.0030 is at or below 0.05, "
        "so Welch's t-test is used")
    assert c["welch_t_test"].eligible is True
    assert c["welch_t_test"].reason.startswith("chosen: both groups pass the normality check")
    assert "Levene's p = 0.0030 is at or below 0.05" in c["welch_t_test"].reason
    assert c["mann_whitney_u"].eligible is True


def test_c_one_group_fails_normality_chooses_mann_whitney():
    r = two_group_selection_record(_decision("mann_whitney_u", b_p=0.012))
    c = _by_option(r)
    assert r.chosen == "mann_whitney_u"
    assert c["student_t_test"].eligible is False
    assert c["welch_t_test"].eligible is False
    for m in ("student_t_test", "welch_t_test", "mann_whitney_u"):
        assert "East fails the normality check (n=38, Shapiro-Wilk p = 0.0120)" in c[m].reason
    assert "West" not in c["welch_t_test"].reason
    assert c["mann_whitney_u"].eligible is True
    assert c["mann_whitney_u"].reason.startswith("chosen:")


def test_c2_both_fail_normality_and_unequal_variance_names_both():
    r = two_group_selection_record(_decision("mann_whitney_u", a_p=0.001, b_p=0.02, levene=0.01))
    student = _by_option(r)["student_t_test"]
    assert "West fails" in student.reason and "East fails" in student.reason
    assert "Levene's p = 0.0100 is at or below 0.05" in student.reason


def test_d_group_with_n_2_is_not_checked_and_treated_as_passing():
    r = two_group_selection_record(_decision("student_t_test", a_p=1.0, n_a=2))
    c = _by_option(r)
    assert r.evidence["small_groups_unchecked"] == ["West"]
    assert "West n = 2: normality not checked, treated as passing" in c["student_t_test"].reason
    assert "1.0000" not in c["student_t_test"].reason
    assert c["student_t_test"].eligible is True and c["welch_t_test"].eligible is True


def test_e_zero_p_prints_below_threshold():
    r = two_group_selection_record(_decision("mann_whitney_u", a_p=0.0, levene=0.0))
    c = _by_option(r)
    assert "Shapiro-Wilk p < 0.0001" in c["welch_t_test"].reason
    assert "Levene's p < 0.0001" in c["student_t_test"].reason


ALL_SCENARIOS = [
    _decision("student_t_test"),
    _decision("welch_t_test", levene=0.003),
    _decision("mann_whitney_u", b_p=0.012),
    _decision("mann_whitney_u", a_p=0.0, b_p=0.0, levene=0.0),
    _decision("student_t_test", a_p=1.0, n_a=2),
]


@pytest.mark.parametrize("decision", ALL_SCENARIOS)
def test_f_g_no_bootstrap_and_fixed_order(decision):
    r = two_group_selection_record(decision)
    assert [c.option for c in r.candidates] == ["student_t_test", "welch_t_test", "mann_whitney_u"]
    assert not any("bootstrap" in c.option or "bootstrap" in c.reason for c in r.candidates)
    assert all(c.generated_by == "code" for c in r.candidates)


@pytest.mark.parametrize("decision", ALL_SCENARIOS)
def test_candidate_evidence_holds_only_its_p_values(decision):
    c = _by_option(two_group_selection_record(decision))
    assert set(c["student_t_test"].evidence) == {"shapiro_p_group_a", "shapiro_p_group_b", "levene_p"}
    assert set(c["welch_t_test"].evidence) == {"shapiro_p_group_a", "shapiro_p_group_b"}
    assert c["mann_whitney_u"].evidence == {}


# --- validation ---------------------------------------------------------------

@pytest.mark.parametrize("key", ["shapiro_p_group_a", "levene_p", "n_group_b", "group_a_label"])
def test_h_missing_evidence_key_raises(key):
    d = _decision("student_t_test")
    evidence = dict(d.evidence)
    del evidence[key]
    bad = MethodDecision(stage=d.stage, chosen_method=d.chosen_method, rationale="r",
                         alternatives=(), evidence=evidence)
    with pytest.raises(ValueError, match=key):
        two_group_selection_record(bad)


@pytest.mark.parametrize("key", ["group_a_passes_normality", "group_b_passes_normality", "variances_equal"])
def test_h_missing_flag_raises(key):
    d = _decision("student_t_test")
    evidence = {k: v for k, v in d.evidence.items() if k != key}
    bad = MethodDecision(stage=d.stage, chosen_method=d.chosen_method, rationale="r",
                         alternatives=(), evidence=evidence)
    with pytest.raises(ValueError, match=key):
        two_group_selection_record(bad)


# --- G30: flags from unrounded values decide; rounded quotes carry a note -----

def test_g30_rounded_0_05_with_pass_flag_treated_as_passing():
    # Selector saw p = 0.05003 (> 0.05, passes); evidence stores 0.05 after rounding.
    r = two_group_selection_record(_decision("student_t_test", a_p=0.05, a_passes=True))
    c = _by_option(r)
    assert c["student_t_test"].eligible is True
    assert c["welch_t_test"].eligible is True
    assert "West n=40: Shapiro-Wilk p = 0.0500 [rounded; the selector used the unrounded value]" \
        in c["student_t_test"].reason


def test_g30_rounded_0_05_with_fail_flag_treated_as_failing():
    r = two_group_selection_record(_decision("mann_whitney_u", a_p=0.05, a_passes=False))
    c = _by_option(r)
    assert c["student_t_test"].eligible is False and c["welch_t_test"].eligible is False
    assert "West fails the normality check (n=40, Shapiro-Wilk p = 0.0500 [rounded;" in c["welch_t_test"].reason
    assert c["mann_whitney_u"].reason.startswith("chosen:")


def test_g30_levene_0_05_with_equal_flag():
    r = two_group_selection_record(_decision("student_t_test", levene=0.05, variances_equal=True))
    student = _by_option(r)["student_t_test"]
    assert student.eligible is True
    assert "Levene's p = 0.0500 [rounded; the selector used the unrounded value] is above 0.05" in student.reason


def test_h_unknown_method_raises():
    with pytest.raises(ValueError, match="bootstrap_test"):
        two_group_selection_record(_decision("bootstrap_test"))


# --- chart data, log_update, no mutation --------------------------------------

def test_i_chart_data_not_copied():
    d = _decision("student_t_test", chart_data=({"chart_type": "boxplot_by_group", "values": [1.0, 2.0]},))
    r = two_group_selection_record(d)
    assert r.chart_data == ()
    assert "chart_data" not in r.evidence


def test_j_passes_log_update_and_does_not_mutate():
    d = _decision("welch_t_test", levene=0.003,
                  unaddressed_issues=[{"issue_type": "duplicate_rows", "description": "d", "rejection_reason": "r"}])
    before = json.dumps(d.evidence, sort_keys=True)
    update = log_update(two_group_selection_record(d))
    json.dumps(update)
    assert update["decision_log"][0]["evidence"]["unaddressed_issues"][0]["issue_type"] == "duplicate_rows"
    assert json.dumps(d.evidence, sort_keys=True) == before
    assert "family" not in d.evidence


def test_k_two_group_capability_carries_the_record_builder():
    assert TWO_GROUP.record is two_group_selection_record
    assert capability_for_question_type("two_group_comparison").record is two_group_selection_record


def test_alpha_matches_the_selector_threshold():
    source = inspect.getsource(testing_module.select_two_group_method)
    assert ALPHA == 0.05
    assert source.count(f"> {ALPHA}") >= 3  # both Shapiro checks and Levene's


# --- consistency guard against the real selector -------------------------------

@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _two_groups(a, b):
    return pd.DataFrame({"region": ["East"] * len(a) + ["West"] * len(b),
                         "total_spend": np.concatenate([a, b])})


def _guard_frames():
    rng = np.random.default_rng(5)
    return {
        "normal_equal": _two_groups(rng.normal(50, 10, 150), rng.normal(55, 10, 150)),
        "normal_unequal": _two_groups(rng.normal(50, 3, 150), rng.normal(55, 30, 150)),
        "skewed": _two_groups(rng.exponential(5, 150), rng.exponential(5, 150) ** 2),
        "tiny_group": _two_groups(np.array([48.0, 53.0]), rng.normal(50, 10, 40)),
    }


def test_record_agrees_with_the_real_selector(tmp_path, store):
    records = {}
    for name, df in _guard_frames().items():
        dataset = make_dataset(df, tmp_path, store, name=name)
        choice = select_two_group_method(dataset, store, outcome_col="total_spend", group_col="region")
        r = two_group_selection_record(choice.decision)
        c = _by_option(r)
        assert r.chosen == choice.method, name
        assert c[r.chosen].eligible is True, name
        assert all(cand.option != r.chosen for cand in r.candidates if cand.eligible is False), name
        records[name] = r
    # The synthetic data exercises each branch of the selection rule.
    assert records["normal_equal"].chosen == "student_t_test"
    assert records["normal_unequal"].chosen == "welch_t_test"
    assert records["skewed"].chosen == "mann_whitney_u"
    assert records["tiny_group"].evidence["small_groups_unchecked"] == ["East"]


# --- node ---------------------------------------------------------------------

def _plan(dataset, issues=()):
    plan = AnalysisPlan(
        request_text="x", session_id="session-abc",
        question_type=QuestionType.TWO_GROUP_COMPARISON, causal_status=CausalStatus.OBSERVATIONAL,
        outcome_name="total_spend", outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="customer", data_sources=(dataset.handle,), candidate_predictors=("region",),
    )
    for issue in issues:
        plan = plan.record_unaddressed_issue(issue)
    return plan


def _run_select(store, plan, dataset):
    graph = StateGraph(AgentState)
    graph.add_node("n", make_select_method_node(store))
    graph.add_edge(START, "n")
    graph.add_edge("n", END)
    return graph.compile().invoke({"plan": plan, "dataset": dataset, "request_text": "x", "session_id": "session-abc"})


def test_select_node_logs_one_two_group_record(tmp_path, store):
    dataset = make_dataset(_guard_frames()["normal_equal"], tmp_path, store, name="node_check")
    result = _run_select(store, _plan(dataset), dataset)
    log = result["decision_log"]
    assert len(log) == 1
    assert log[0]["decision_point"] == "stage2.test_selection"
    assert log[0]["chosen"] == result["plan"].primary_method
    assert len(result["plan"].method_decisions) == 1


def test_select_node_record_carries_unaddressed_issues(tmp_path, store):
    dataset = make_dataset(_guard_frames()["normal_equal"], tmp_path, store, name="node_issues")
    issue = UnaddressedIssue(issue_type="duplicate_rows", description="3 duplicate rows",
                             proposed_operation="dedupe", rejection_reason="they are real repeat orders")
    result = _run_select(store, _plan(dataset, issues=(issue,)), dataset)
    evidence = result["decision_log"][0]["evidence"]
    assert evidence["unaddressed_issues"] == [
        {"issue_type": "duplicate_rows", "description": "3 duplicate rows",
         "rejection_reason": "they are real repeat orders"},
    ]
    assert len(result["plan"].method_decisions) == 1
