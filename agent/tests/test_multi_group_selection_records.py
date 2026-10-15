import copy
import json

import numpy as np
import pandas as pd
import pytest
from langgraph.graph import StateGraph, START, END

import core.multi_group as multi_group_module
import core.thresholds as thresholds
from core.agent_state import AgentState
from core.capabilities import MULTI_GROUP, REGRESSION, capability_for_question_type
from core.dataset import LocalDiskStore
from core.decision_log import log_update
from core.multi_group import select_multi_group_method
from core.nodes import make_select_method_node
from core.plan import AnalysisPlan, CausalStatus, MethodDecision, OutcomeType, QuestionType
from core.selection_records import multi_group_selection_record, regression_selection_records
from tests.test_testing import make_dataset

LABELS = ("East", "North", "West")


def _decision(method, ps=(0.21, 0.35, 0.48), ns=(40, 38, 41), levene=0.45, labels=LABELS,
              passes=None, variances_equal=None, chart_data=(), **extra):
    """Hand-made multi-group MethodDecision; flags default to p > 0.05 and can be overridden (G30)."""
    labels = list(labels)
    passes = passes if passes is not None else [p > 0.05 for p in ps]
    evidence = {
        **{f"shapiro_p_{label}": p for label, p in zip(labels, ps)},
        "levene_p": levene,
        "group_labels": labels,
        "n_by_group": dict(zip(labels, ns)),
        "normality_p_by_group": dict(zip(labels, ps)),
        "passes_normality_by_group": dict(zip(labels, passes)),
        "variances_equal": levene > 0.05 if variances_equal is None else variances_equal,
        **extra,
    }
    return MethodDecision(stage="test_selection", chosen_method=method, rationale="selector rationale",
                          alternatives=(), evidence=evidence, chart_data=chart_data)


def _by_option(record):
    return {c.option: c for c in record.candidates}


# --- scenarios ----------------------------------------------------------------

def test_a_all_pass_equal_variance_chooses_anova():
    r = multi_group_selection_record(_decision("one_way_anova"))
    c = _by_option(r)
    assert r.decision_point == "stage2.test_selection"
    assert r.decided_by == "code_rule"
    assert r.chosen == "one_way_anova"
    assert r.rationale == "selector rationale"
    assert [c["one_way_anova"].eligible, c["kruskal_wallis"].eligible] == [True, True]
    assert c["one_way_anova"].reason == (
        "chosen: all 3 groups pass the normality check (East n=40: Shapiro-Wilk p = 0.2100; "
        "North n=38: Shapiro-Wilk p = 0.3500; West n=41: Shapiro-Wilk p = 0.4800) and "
        "Levene's p = 0.4500 is above 0.05; if the omnibus test is significant, "
        "post-hoc comparisons use Tukey HSD")
    assert c["kruskal_wallis"].reason.startswith("eligible, since it assumes no distribution, but all 3 groups pass")
    assert c["kruskal_wallis"].reason.endswith("so the selection rule prefers ANOVA")
    assert r.evidence["family"] == "multi_group"


def test_b_all_pass_unequal_variance_chooses_kruskal():
    r = multi_group_selection_record(_decision("kruskal_wallis", levene=0.002))
    c = _by_option(r)
    assert r.chosen == "kruskal_wallis"
    assert c["one_way_anova"].eligible is False
    assert c["one_way_anova"].reason == (
        "not eligible: assumes equal variances, but Levene's p = 0.0020 is at or below 0.05")
    assert c["kruskal_wallis"].reason == (
        "chosen: Levene's p = 0.0020 is at or below 0.05; if the omnibus test is significant, "
        "post-hoc comparisons use Dunn's test with Holm adjustment")


def test_c_one_group_fails_names_only_that_group():
    r = multi_group_selection_record(_decision("kruskal_wallis", ps=(0.21, 0.012, 0.48)))
    c = _by_option(r)
    expected = "North fails the normality check (n=38, Shapiro-Wilk p = 0.0120)"
    assert c["one_way_anova"].reason == f"not eligible: assumes normality, but {expected}"
    assert c["kruskal_wallis"].reason.startswith(f"chosen: {expected};")
    for other in ("East", "West"):
        assert other not in c["one_way_anova"].reason


def test_d_two_groups_fail_and_variances_fail_names_both_and_levene():
    r = multi_group_selection_record(_decision("kruskal_wallis", ps=(0.003, 0.35, 0.01), levene=0.02))
    c = _by_option(r)
    anova = c["one_way_anova"].reason
    assert "East fails the normality check (n=40, Shapiro-Wilk p = 0.0030)" in anova
    assert "West fails the normality check (n=41, Shapiro-Wilk p = 0.0100)" in anova
    assert "North" not in anova
    assert anova.endswith("; assumes equal variances, but Levene's p = 0.0200 is at or below 0.05")
    kw = c["kruskal_wallis"].reason
    assert "East fails" in kw and "West fails" in kw and "Levene's p = 0.0200 is at or below 0.05" in kw


def test_e_group_with_n_2_not_checked():
    r = multi_group_selection_record(_decision("one_way_anova", ps=(0.21, 0.35, 1.0), ns=(40, 38, 2)))
    assert r.evidence["small_groups_unchecked"] == ["West"]
    assert "West n = 2: normality not checked, treated as passing" in _by_option(r)["one_way_anova"].reason


def test_f_zero_p_prints_below_threshold():
    r = multi_group_selection_record(_decision("kruskal_wallis", ps=(0.0, 0.35, 0.48), levene=0.0))
    reason = _by_option(r)["one_way_anova"].reason
    assert "Shapiro-Wilk p < 0.0001" in reason
    assert "Levene's p < 0.0001" in reason


SCENARIOS = [
    _decision("one_way_anova"),
    _decision("kruskal_wallis", levene=0.002),
    _decision("kruskal_wallis", ps=(0.21, 0.012, 0.48)),
    _decision("kruskal_wallis", ps=(0.003, 0.35, 0.01), levene=0.02),
    _decision("one_way_anova", ps=(0.21, 0.35, 1.0), ns=(40, 38, 2)),
]


@pytest.mark.parametrize("decision", SCENARIOS)
def test_g_exactly_two_candidates_in_order_nothing_else(decision):
    r = multi_group_selection_record(decision)
    assert [c.option for c in r.candidates] == ["one_way_anova", "kruskal_wallis"]
    assert all(c.generated_by == "code" for c in r.candidates)
    for c in r.candidates:
        text = f"{c.option} {c.reason}".lower()
        assert "bootstrap" not in text and "welch" not in text
    assert set(_by_option(r)["one_way_anova"].evidence) == {"normality_p_by_group", "levene_p"}
    assert _by_option(r)["kruskal_wallis"].evidence == {}


# --- validation ---------------------------------------------------------------

@pytest.mark.parametrize("key", ["group_labels", "n_by_group", "normality_p_by_group",
                                 "passes_normality_by_group", "levene_p", "variances_equal"])
def test_h_missing_key_raises(key):
    d = _decision("one_way_anova")
    evidence = {k: v for k, v in d.evidence.items() if k != key}
    bad = MethodDecision(stage=d.stage, chosen_method=d.chosen_method, rationale="r",
                         alternatives=(), evidence=evidence)
    with pytest.raises(ValueError, match=key):
        multi_group_selection_record(bad)


def test_h_label_missing_from_a_per_group_dict_raises():
    d = _decision("one_way_anova")
    evidence = copy.deepcopy(d.evidence)
    del evidence["n_by_group"]["North"]
    bad = MethodDecision(stage=d.stage, chosen_method=d.chosen_method, rationale="r",
                         alternatives=(), evidence=evidence)
    with pytest.raises(ValueError, match="n_by_group\\['North'\\]"):
        multi_group_selection_record(bad)


@pytest.mark.parametrize("method", ["welch_anova", "student_t_test"])
def test_h_unknown_method_raises(method):
    with pytest.raises(ValueError, match=method):
        multi_group_selection_record(_decision(method))


# --- chart data, log_update, labels, post-hoc ----------------------------------

def test_i_chart_data_not_copied():
    d = _decision("one_way_anova", chart_data=({"chart_type": "boxplot_by_group", "values": [1.0]},))
    r = multi_group_selection_record(d)
    assert r.chart_data == ()
    assert "chart_data" not in r.evidence


def test_j_log_update_no_mutation_and_awkward_labels_survive():
    labels = ("North_East", "South West", "by_group")
    d = _decision("kruskal_wallis", labels=labels, ps=(0.21, 0.01, 0.48))
    before = json.dumps(d.evidence, sort_keys=True)
    update = log_update(multi_group_selection_record(d))
    json.dumps(update)
    entry = update["decision_log"][0]
    assert entry["evidence"]["group_labels"] == list(labels)
    assert set(entry["evidence"]["normality_p_by_group"]) == set(labels)
    kw = next(c for c in entry["candidates"] if c["option"] == "kruskal_wallis")
    assert "South West fails the normality check (n=38" in kw["reason"]
    anova = next(c for c in entry["candidates"] if c["option"] == "one_way_anova")
    assert set(anova["evidence"]["normality_p_by_group"]) == set(labels)
    assert json.dumps(d.evidence, sort_keys=True) == before
    assert "family" not in d.evidence


def test_k_post_hoc_names():
    assert multi_group_selection_record(_decision("one_way_anova")).evidence[
        "post_hoc_if_omnibus_significant"] == "tukey_hsd"
    assert multi_group_selection_record(_decision("kruskal_wallis", levene=0.002)).evidence[
        "post_hoc_if_omnibus_significant"] == "dunn"


def test_l_g30_flag_overrides_rounded_value():
    passing = multi_group_selection_record(
        _decision("one_way_anova", ps=(0.05, 0.35, 0.48), passes=[True, True, True]))
    anova = _by_option(passing)["one_way_anova"]
    assert anova.eligible is True
    assert "East n=40: Shapiro-Wilk p = 0.0500 [rounded; the selector used the unrounded value]" in anova.reason

    failing = multi_group_selection_record(
        _decision("kruskal_wallis", ps=(0.05, 0.35, 0.48), passes=[False, True, True]))
    assert _by_option(failing)["one_way_anova"].eligible is False
    assert "East fails the normality check (n=40, Shapiro-Wilk p = 0.0500 [rounded;" in \
        _by_option(failing)["one_way_anova"].reason

    levene = multi_group_selection_record(_decision("one_way_anova", levene=0.05, variances_equal=True))
    assert _by_option(levene)["one_way_anova"].eligible is True


def test_m_capability_wiring():
    assert MULTI_GROUP.record is multi_group_selection_record
    assert capability_for_question_type("multi_group_comparison").record is multi_group_selection_record
    assert REGRESSION.record is regression_selection_records


def test_n_selector_threshold_guard():  # M1.4b 2a: replaces a source-text check; the selectors read named constants (test_thresholds.py proves the wiring)
    assert multi_group_module.NORMALITY_ALPHA == thresholds.NORMALITY_ALPHA
    assert multi_group_module.EQUAL_VARIANCE_ALPHA == thresholds.EQUAL_VARIANCE_ALPHA


# --- consistency guard against the real selector -------------------------------

@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _frame(groups):
    return pd.DataFrame({
        "region": [label for label, arr in groups.items() for _ in arr],
        "total_spend": np.concatenate(list(groups.values())),
    })


def _guard_frames():
    eq, uneq, skew = np.random.default_rng(3), np.random.default_rng(2), np.random.default_rng(11)
    tiny = np.random.default_rng(11)
    return {
        "normal_equal": _frame({"East": eq.normal(50, 10, 100), "North": eq.normal(52, 10, 100),
                                "West": eq.normal(55, 10, 100)}),
        "normal_unequal": _frame({"East": uneq.normal(50, 3, 100), "North": uneq.normal(52, 10, 100),
                                  "West": uneq.normal(55, 30, 100)}),
        "skewed": _frame({"East": skew.exponential(5, 100), "North": skew.exponential(5, 100) ** 2,
                          "West": skew.exponential(7, 100)}),
        "tiny_group": _frame({"East": tiny.normal(50, 5, 60), "North": tiny.normal(52, 5, 60),
                              "West": np.array([20.0, 80.0])}),
    }


def _select(df, tmp_path, store, name):
    dataset = make_dataset(df, tmp_path, store, name=name)
    return select_multi_group_method(dataset, store, outcome_col="total_spend", group_col="region")


def test_record_agrees_with_the_real_selector(tmp_path, store):
    records = {}
    for name, df in _guard_frames().items():
        choice = _select(df, tmp_path, store, name)
        r = multi_group_selection_record(choice.decision)
        c = _by_option(r)
        assert r.chosen == choice.method, name
        assert c[r.chosen].eligible is True, name
        assert all(cand.option != r.chosen for cand in r.candidates if cand.eligible is False), name
        records[name] = r
    assert records["normal_equal"].chosen == "one_way_anova"
    assert records["normal_unequal"].chosen == "kruskal_wallis"
    assert records["skewed"].chosen == "kruskal_wallis"
    assert records["tiny_group"].chosen == "kruskal_wallis"
    # The unequal-variance set reaches Kruskal-Wallis through Levene alone.
    assert all(records["normal_unequal"].evidence["passes_normality_by_group"].values())
    assert records["tiny_group"].evidence["small_groups_unchecked"] == ["West"]


def test_by_group_label_does_not_collide(tmp_path, store):
    rng = np.random.default_rng(3)
    df = _frame({"by_group": rng.normal(50, 10, 60), "North_East": rng.normal(52, 10, 60),
                 "South West": rng.normal(55, 10, 60)})
    ev = _select(df, tmp_path, store, "by_group_labels").decision.evidence
    assert type(ev["shapiro_p_by_group"]) is float
    assert ev["group_labels"] == ["North_East", "South West", "by_group"]
    assert set(ev["normality_p_by_group"]) == {"North_East", "South West", "by_group"}
    assert ev["normality_p_by_group"]["by_group"] == ev["shapiro_p_by_group"]


# --- node ---------------------------------------------------------------------

def test_select_node_logs_one_multi_group_record(tmp_path, store):
    dataset = make_dataset(_guard_frames()["normal_equal"], tmp_path, store, name="node_mg")
    plan = AnalysisPlan(
        request_text="x", session_id="session-abc",
        question_type=QuestionType.MULTI_GROUP_COMPARISON, causal_status=CausalStatus.OBSERVATIONAL,
        outcome_name="total_spend", outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="customer", data_sources=(dataset.handle,), candidate_predictors=("region",),
    )
    graph = StateGraph(AgentState)
    graph.add_node("n", make_select_method_node(store))
    graph.add_edge(START, "n")
    graph.add_edge("n", END)
    result = graph.compile().invoke({"plan": plan, "dataset": dataset, "request_text": "x",
                                     "session_id": "session-abc"})
    log = result["decision_log"]
    assert len(log) == 1
    assert log[0]["decision_point"] == "stage2.test_selection"
    assert log[0]["chosen"] == result["plan"].primary_method
    assert log[0]["evidence"]["family"] == "multi_group"
    assert len(result["plan"].method_decisions) == 1
