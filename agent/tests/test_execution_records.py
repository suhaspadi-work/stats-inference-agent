import copy
import json

import numpy as np
import pandas as pd
import pytest

import core.multi_group as multi_group_module
import core.thresholds as thresholds
from core.capabilities import MULTI_GROUP, REGRESSION, TWO_GROUP
from core.dataset import LocalDiskStore
from core.decision_log import log_update
from core.execution_records import ALPHA, multi_group_execution_records, post_hoc_record
from core.ingestion import load_dataset
from core.plan import AnalysisPlan, CausalStatus, OutcomeType, QuestionType

SELECTION = {"id": "sel-1", "decision_point": "stage2.test_selection",
             "evidence": {"post_hoc_if_omnibus_significant": "tukey_hsd"}}


def _result(method="one_way_anova", p=0.0012, post_hoc="tukey_hsd", statistic=12.3456, pairs=3, chart_data=None):
    result = {
        "result_type": "multi_group", "method": method, "statistic": statistic, "p_value": p,
        "post_hoc_method": post_hoc,
        "pairwise_comparisons": [{"group_a": "A", "group_b": "B"} for _ in range(pairs)],
        "group_ns": {"A": 30, "B": 30, "C": 30}, "group_means": {"A": 1.0, "B": 2.0, "C": 3.0},
    }
    if chart_data is not None:
        result["chart_data"] = chart_data
    return result


def _by_option(record):
    return {c.option: c for c in record.candidates}


def _eligibility(record):
    return [c.eligible for c in record.candidates]


# --- choice and eligibility --------------------------------------------------------

def test_anova_significant_chooses_tukey():
    r = post_hoc_record(_result(), SELECTION)
    assert r.decision_point == "stage2.post_hoc"
    assert (r.chosen, r.decided_by) == ("tukey_hsd", "code_rule")
    assert _eligibility(r) == [True, False, False]
    assert r.rationale == ("chosen: the omnibus test (one_way_anova, statistic = 12.3456, p = 0.0012) is "
                           "significant, below 0.05, so Tukey HSD compares each pair of groups")
    c = _by_option(r)
    assert c["dunn"].reason == "not eligible: Dunn's test follows Kruskal-Wallis, but the omnibus test was ANOVA"
    assert c["none"].reason == ("not eligible: the omnibus test is significant (p = 0.0012, below 0.05), "
                                "so post-hoc comparisons run")


def test_kruskal_significant_chooses_dunn():
    r = post_hoc_record(_result("kruskal_wallis", p=0.0003, post_hoc="dunn", statistic=20.5), None)
    assert r.chosen == "dunn"
    assert _eligibility(r) == [False, True, False]
    assert r.rationale == ("chosen: the omnibus test (kruskal_wallis, statistic = 20.5000, p = 0.0003) is "
                           "significant, below 0.05, so Dunn's test with Holm adjustment compares each pair of groups")
    assert _by_option(r)["tukey_hsd"].reason == ("not eligible: Tukey HSD follows ANOVA, but the omnibus test "
                                                 "was Kruskal-Wallis")


@pytest.mark.parametrize("method, follow_up", [("one_way_anova", "tukey_hsd"), ("kruskal_wallis", "dunn")])
def test_not_significant_chooses_none(method, follow_up):
    r = post_hoc_record(_result(method, p=0.42, post_hoc="none", pairs=0), None)
    assert r.chosen == "none"
    assert _eligibility(r) == [False, False, True]
    assert r.rationale == ("chosen: the omnibus test is not significant (p = 0.4200, at or above 0.05), "
                           "so no post-hoc comparisons are run")
    assert _by_option(r)[follow_up].reason == ("not eligible: the omnibus test is not significant "
                                               "(p = 0.4200, at or above 0.05)")


def test_p_exactly_0_05_is_not_significant():
    r = post_hoc_record(_result(p=0.05, post_hoc="none", pairs=0), None)
    assert r.chosen == "none"
    assert _by_option(r)["none"].eligible is True


def test_p_just_below_0_05_is_significant_with_rounded_note():
    r = post_hoc_record(_result(p=0.049996), None)
    assert r.chosen == "tukey_hsd"
    assert _by_option(r)["tukey_hsd"].eligible is True
    assert "p = 0.0500 [rounded; the selector used the unrounded value]" in r.rationale
    assert r.evidence["omnibus_p_value"] == 0.049996


# --- consistency with the selection record -----------------------------------------

def test_consistent_when_prediction_matches():
    r = post_hoc_record(_result(), SELECTION)
    assert r.evidence["selection_record_id"] == "sel-1"
    assert r.evidence["post_hoc_if_significant_per_selection"] == "tukey_hsd"
    assert r.evidence["consistent_with_selection"] is True


def test_consistent_when_not_significant_and_none_ran():
    r = post_hoc_record(_result(p=0.3, post_hoc="none", pairs=0), SELECTION)
    assert r.evidence["consistent_with_selection"] is True


def test_inconsistent_when_prediction_differs():
    selection = {"id": "sel-2", "evidence": {"post_hoc_if_omnibus_significant": "dunn"}}
    r = post_hoc_record(_result(), selection)
    assert r.evidence["consistent_with_selection"] is False


def test_no_selection_entry():
    r = post_hoc_record(_result(), None)
    assert r.evidence["selection_record_id"] is None
    assert r.evidence["post_hoc_if_significant_per_selection"] is None
    assert r.evidence["consistent_with_selection"] is None


def test_evidence_fields():
    r = post_hoc_record(_result(pairs=3), SELECTION)
    assert r.evidence == {
        "omnibus_method": "one_way_anova", "omnibus_statistic": 12.3456, "omnibus_p_value": 0.0012,
        "alpha": 0.05, "n_pairwise_comparisons": 3, "selection_record_id": "sel-1",
        "post_hoc_if_significant_per_selection": "tukey_hsd", "consistent_with_selection": True,
    }


# --- chart data -------------------------------------------------------------------

def test_chart_data_carried_with_nan_and_passes_log_update():
    chart = {"chart_type": "post_hoc_comparison_chart",
             "data": {"comparisons": ["A-B"], "ci_lower": [float("nan")], "significant": [True]}}
    r = post_hoc_record(_result(chart_data=[chart]), None)
    assert len(r.chart_data) == 1 and r.chart_data[0]["chart_type"] == "post_hoc_comparison_chart"
    entry = log_update(r)["decision_log"][0]
    assert entry["chart_data"][0]["data"]["comparisons"] == ["A-B"]


def test_chart_data_empty_when_absent():
    assert post_hoc_record(_result(), None).chart_data == ()


# --- shape and validation -------------------------------------------------------------

def test_candidate_order_and_generated_by():
    r = post_hoc_record(_result(), None)
    assert [(c.option, c.generated_by) for c in r.candidates] == [
        ("tukey_hsd", "code"), ("dunn", "code"), ("none", "code")]


@pytest.mark.parametrize("key", ["method", "statistic", "p_value", "post_hoc_method", "pairwise_comparisons"])
def test_missing_result_key_raises(key):
    result = _result()
    del result[key]
    with pytest.raises(ValueError, match=key):
        post_hoc_record(result, None)


def test_unknown_values_raise():
    with pytest.raises(ValueError, match="welch_anova"):
        post_hoc_record(_result(method="welch_anova"), None)
    with pytest.raises(ValueError, match="games_howell"):
        post_hoc_record(_result(post_hoc="games_howell"), None)


def test_passes_log_update_and_does_not_mutate():
    result = _result(chart_data=[{"chart_type": "post_hoc_comparison_chart", "data": {"x": (1.0, 2.0)}}])
    before = copy.deepcopy(result)
    update = log_update(*multi_group_execution_records(result, SELECTION))
    json.dumps(update)
    assert len(update["decision_log"]) == 1
    assert result == before


def test_alpha_matches_the_executor_threshold():  # M1.4b 2a: replaces a source-text check; the selectors read named constants (test_thresholds.py proves the wiring)
    assert ALPHA == thresholds.OMNIBUS_ALPHA == multi_group_module.OMNIBUS_ALPHA


def test_execution_records_wiring():
    assert MULTI_GROUP.execution_records is multi_group_execution_records
    assert TWO_GROUP.execution_records is None
    assert REGRESSION.execution_records is None


# --- consistency guard against the real executor ---------------------------------------

@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _frame(arrays):
    return pd.DataFrame({"region": [k for k, a in arrays.items() for _ in a],
                         "total_spend": np.concatenate(list(arrays.values()))})


def _guard_frames():
    a, k, n = (np.random.default_rng(1) for _ in range(3))
    return {
        "anova_significant": _frame({"East": a.normal(50, 5, 60), "North": a.normal(55, 5, 60),
                                     "West": a.normal(65, 5, 60)}),
        "kruskal_significant": _frame({"East": k.exponential(1, 60), "North": k.exponential(3, 60),
                                       "West": k.exponential(6, 60)}),
        "not_significant": _frame({"East": n.normal(50, 10, 60), "North": n.normal(50, 10, 60),
                                   "West": n.normal(50, 10, 60)}),
    }


def test_record_agrees_with_the_real_executor(tmp_path, store):
    chosen = {}
    for name, df in _guard_frames().items():
        path = tmp_path / f"{name}.csv"
        df.to_csv(path, index=False)
        ds = load_dataset(path, session_id="session-abc", name=name, store=store)
        plan = AnalysisPlan(
            request_text="x", session_id="session-abc", question_type=QuestionType.MULTI_GROUP_COMPARISON,
            causal_status=CausalStatus.OBSERVATIONAL, outcome_name="total_spend",
            outcome_type=OutcomeType.CONTINUOUS, unit_of_analysis="row", data_sources=(ds.handle,),
            candidate_predictors=("region",),
        )
        plan = plan.record_method_decision(MULTI_GROUP.select(ds, store, plan))
        result = MULTI_GROUP.execute(ds, store, plan)
        r = post_hoc_record(result, None)
        assert r.chosen == result["post_hoc_method"], name
        assert _by_option(r)[r.chosen].eligible is True, name
        chosen[name] = r.chosen
    assert chosen == {"anova_significant": "tukey_hsd", "kruskal_significant": "dunn", "not_significant": "none"}
