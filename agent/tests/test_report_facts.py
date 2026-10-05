import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

import core.report as report_module
from core.capabilities import TWO_GROUP
from core.dataset import LocalDiskStore
from core.ingestion import load_dataset
from core.plan import AnalysisPlan, QuestionType, CausalStatus, OutcomeType
from core.report import generate_report
from core.report_facts import two_group_facts
from core.testing import execute_two_group_test


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def result(**over):
    base = {
        "method": "student_t_test", "statistic": -2.0, "p_value": 0.001, "effect_size": 0.41,
        "confidence_interval": (2.10, 8.60), "group_a_n": 174, "group_b_n": 120,
        "group_a_mean": 85.42, "group_b_mean": 90.77,
        "group_a_label": "East", "group_b_label": "West",
    }
    base.update(over)
    return base


def make_plan(request_text="Did West spend more than East?"):
    return AnalysisPlan(
        request_text=request_text, session_id="session-abc",
        question_type=QuestionType.TWO_GROUP_COMPARISON, causal_status=CausalStatus.OBSERVATIONAL,
        outcome_name="total_spend", outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="customer", data_sources=("x@v1",), candidate_predictors=("region",),
    )


def east_west_dataset(tmp_path, store, west_shift=10.0, seed=4):
    rng = np.random.default_rng(seed)
    n = 150
    df = pd.DataFrame({
        "region": ["East"] * n + ["West"] * n,
        "total_spend": np.concatenate([rng.normal(50, 8, n), rng.normal(50 + west_shift, 8, n)]),
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return load_dataset(path, session_id="session-abc", name="data", store=store)


# ------------------------------------------------------------- facts content

def test_facts_name_both_groups_and_the_direction_when_the_second_group_is_higher():
    facts = two_group_facts(result(), "total_spend", "region")
    assert "East: n=174, mean=85.42" in facts
    assert "West: n=120, mean=90.77" in facts
    assert "Higher mean: West, by 5.35" in facts
    assert "Difference (West minus East): +5.35, 95% CI (2.10, 8.60)" in facts
    assert "The interval excludes zero: yes" in facts
    assert "Cohen's d, positive when West is higher" in facts
    assert "p-value: 0.0010" in facts


def test_facts_reverse_correctly_when_the_first_group_is_higher():
    facts = two_group_facts(
        result(group_a_mean=90.77, group_b_mean=85.42, confidence_interval=(-8.60, -2.10)),
        "total_spend", "region",
    )
    assert "Higher mean: East, by 5.35" in facts
    assert "Difference (West minus East): -5.35, 95% CI (-8.60, -2.10)" in facts
    assert "The interval excludes zero: yes" in facts


def test_facts_handle_equal_means_and_an_interval_that_touches_zero():
    facts = two_group_facts(
        result(group_a_mean=50.0, group_b_mean=50.0, confidence_interval=(0.0, 3.0)),
        "total_spend", "region",
    )
    assert "Means are equal" in facts
    assert "Difference (West minus East): 0.00" in facts
    assert "The interval excludes zero: no" in facts


def test_interval_exclusion_for_each_sign():
    assert "excludes zero: no" in two_group_facts(result(confidence_interval=(-1.0, 3.0)), "o", "g")
    assert "excludes zero: yes" in two_group_facts(result(confidence_interval=(-5.0, -1.0)), "o", "g")
    assert "excludes zero: yes" in two_group_facts(result(confidence_interval=(1.0, 5.0)), "o", "g")


def test_mann_whitney_names_the_rank_biserial_effect_size():
    facts = two_group_facts(result(method="mann_whitney_u", effect_size=0.2), "total_spend", "region")
    assert "rank-biserial correlation, positive when West tends higher" in facts
    assert "Cohen" not in facts


def test_facts_fall_back_to_generic_names_for_results_without_labels():
    old = result()
    del old["group_a_label"], old["group_b_label"]
    facts = two_group_facts(old, "total_spend", "region")
    assert "Group A: n=174" in facts
    assert "Difference (Group B minus Group A)" in facts


def test_number_formatting_for_tiny_values_and_p_values():
    tiny = two_group_facts(result(group_a_mean=10.0, group_b_mean=10.004), "o", "g")
    assert "by 0.0040" in tiny and "+0.0040" in tiny
    assert "p-value: < 0.0001" in two_group_facts(result(p_value=0.00001), "o", "g")
    assert "p-value: 0.0123" in two_group_facts(result(p_value=0.0123), "o", "g")


# ------------------------------------------------- what the model is given

def test_the_prompt_gives_the_model_labeled_facts_and_forbids_inferring_direction(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.messages = None

        def invoke(self, messages):
            self.messages = messages
            return SimpleNamespace(content="In this observational data, West is associated with higher spending than East.")

    fake = FakeModel()
    monkeypatch.setattr(report_module, "init_chat_model", lambda *a, **k: fake)

    text = generate_report(make_plan(), result())
    system, human = fake.messages[0][1], fake.messages[1][1]
    assert "West: n=120, mean=90.77" in human
    assert "East: n=174, mean=85.42" in human
    assert "Higher mean: West, by 5.35" in human
    assert "Group A" not in human and "Group B" not in human
    assert "never recompute or reverse" in system
    assert text.startswith("In this observational")


# ----------------------------------- sign conventions against real computation

def test_t_test_result_has_real_labels_and_second_minus_first_signs(tmp_path, store):
    ds = east_west_dataset(tmp_path, store, west_shift=10.0)
    r = execute_two_group_test(ds, store, "total_spend", "region", "student_t_test")
    assert (r.group_a_label, r.group_b_label) == ("East", "West")
    assert r.group_b_mean > r.group_a_mean
    lo, hi = r.confidence_interval
    assert lo > 0 and hi > 0          # interval is group B minus group A
    assert r.effect_size > 0          # Cohen's d is positive when B is higher
    assert r.statistic < 0            # the t statistic is computed on (A, B): negative when B is higher


def test_mann_whitney_effect_size_is_positive_when_the_second_group_is_higher(tmp_path, store):
    ds = east_west_dataset(tmp_path, store, west_shift=10.0)
    r = execute_two_group_test(ds, store, "total_spend", "region", "mann_whitney_u")
    assert r.effect_size > 0


def test_the_registry_result_carries_labels_into_correct_facts(tmp_path, store):
    ds = east_west_dataset(tmp_path, store, west_shift=10.0)
    plan = make_plan()
    decision = TWO_GROUP.select(ds, store, plan)
    res = TWO_GROUP.execute(ds, store, plan.record_method_decision(decision))
    assert (res["group_a_label"], res["group_b_label"]) == ("East", "West")
    facts = two_group_facts(res, "total_spend", "region")
    assert "Higher mean: West" in facts
    assert "Difference (West minus East): +" in facts