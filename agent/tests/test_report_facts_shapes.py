import json
import re
import numpy as np
import pandas as pd
import pytest

from core.capabilities import MULTI_GROUP, REGRESSION
from core.dataset import LocalDiskStore
from core.ingestion import load_dataset
from core.plan import AnalysisPlan, QuestionType, CausalStatus, OutcomeType
from core.report_facts import (
    MAX_LISTED, two_group_facts, multi_group_facts, multi_group_pairs,
    regression_facts, regression_terms,
)
from eval.check_report_fidelity import judge


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def make_dataset(df, tmp_path, store):
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return load_dataset(path, session_id="session-abc", name="data", store=store)


def make_plan(question_type, outcome, predictors, dataset):
    return AnalysisPlan(
        request_text="x", session_id="session-abc",
        question_type=question_type, causal_status=CausalStatus.OBSERVATIONAL,
        outcome_name=outcome, outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="row", data_sources=(dataset.handle,),
        candidate_predictors=tuple(predictors),
    )


LABEL_STYLES = {
    "plain": ["East", "West", "North", "South", "Central", "Coast", "Inland", "Island", "Border"],
    "numeric": ["2", "10", "100", "7", "30", "55", "9", "400", "12"],
    "symbols": ["Plan A (basic)", "Plan B, premium", "Plan C [new]", "A vs B", "100% free", "Tier #1", "Q&A group", "x/y", "o'brien"],
    "unicode": ["Zürich", "São Paulo", "東京", "Köln", "Málaga", "Łódź", "Ōsaka", "Åre", "Kraków"],
}


# ----------------------------------------------- multi-group across dataset shapes

@pytest.mark.parametrize("style", list(LABEL_STYLES))
@pytest.mark.parametrize("k", [3, 6, 9])
@pytest.mark.parametrize("balanced", [True, False])
@pytest.mark.parametrize("scale", [1.0, 0.01])
def test_multi_group_facts_hold_across_dataset_shapes(tmp_path, store, style, k, balanced, scale):
    labels = LABEL_STYLES[style][:k]
    rng = np.random.default_rng(1000 + k)
    sizes = [60] * k if balanced else [int(s) for s in np.linspace(30, 90, k)]
    means = np.linspace(0, 3, k) * scale + 10 * scale
    rng.shuffle(means)  # label order is unrelated to mean order
    df = pd.DataFrame({
        "grp": np.repeat(labels, sizes),
        "outcome": np.concatenate([rng.normal(m, scale, n) for m, n in zip(means, sizes)]),
    })
    ds = make_dataset(df, tmp_path, store)
    plan = make_plan(QuestionType.MULTI_GROUP_COMPARISON, "outcome", ("grp",), ds)
    result = MULTI_GROUP.execute(ds, store, plan.record_method_decision(MULTI_GROUP.select(ds, store, plan)))

    raw = {str(g): float(m) for g, m in df.groupby("grp")["outcome"].mean().items()}
    raw_n = {str(g): int(n) for g, n in df.groupby("grp")["outcome"].size().items()}

    pairs = multi_group_pairs(result)
    assert len(pairs) == k * (k - 1) // 2
    for p in pairs:
        a, b = p["group_a"], p["group_b"]
        assert p["higher"] == (b if raw[b] > raw[a] else a)
        assert p["size"] == pytest.approx(abs(raw[b] - raw[a]), rel=1e-9)

    facts = multi_group_facts(result, "outcome", "grp")
    for i, g in enumerate(sorted(raw, key=lambda g: (-raw[g], g)), 1):
        assert f"  {i}. {g}: n={raw_n[g]}, mean=" in facts
    assert not re.search(r"\bnan\b", facts.lower())
    n_sig = sum(p["significant"] for p in pairs)
    assert f"of {len(pairs)} pairs differ significantly" in facts
    assert ("not listed" in facts) == (n_sig > MAX_LISTED)
    assert len(facts.splitlines()) <= k + 2 * MAX_LISTED + 8
    json.dumps(result)


def _multi(pairs, means, method="kruskal_wallis", post="dunn", p=0.001):
    return {
        "result_type": "multi_group", "method": method, "statistic": 1.0, "p_value": p,
        "post_hoc_method": post, "pairwise_comparisons": pairs,
        "group_ns": {k: 50 for k in means}, "group_means": means,
    }


def _pair(a, b, mean_diff, lo=float("nan"), hi=float("nan"), p=0.01, sig=True):
    return {"group_a": a, "group_b": b, "mean_diff": mean_diff, "ci_lower": lo, "ci_upper": hi, "p_value": p, "significant": sig}


def test_dunn_pairs_without_an_interval_print_no_interval_and_no_nan():
    facts = multi_group_facts(_multi([_pair("A", "B", 2.0)], {"A": 1.0, "B": 3.0}), "o", "g")
    assert "B is higher than A by 2.00" in facts
    assert "95% CI" not in facts and "nan" not in facts.lower()


def test_interval_is_reoriented_to_higher_minus_lower():
    # B minus A is -2 (A is higher), interval (-3, -1) for B minus A, so A minus B is (1, 3)
    facts = multi_group_facts(_multi([_pair("A", "B", -2.0, -3.0, -1.0)], {"A": 3.0, "B": 1.0}, method="one_way_anova", post="tukey_hsd"), "o", "g")
    assert "A is higher than B by 2.00 (95% CI 1.00 to 3.00)" in facts


def test_an_inconsistent_interval_is_omitted_not_shown():
    facts = multi_group_facts(_multi([_pair("A", "B", 5.0, 4.0, 6.0)], {"A": 1.0, "B": 3.0}), "o", "g")
    assert "B is higher than A by 2.00" in facts and "95% CI" not in facts


def test_equal_means_and_no_posthoc():
    facts = multi_group_facts(_multi([_pair("A", "B", 0.0)], {"A": 2.0, "B": 2.0}), "o", "g")
    assert "A and B have equal means" in facts
    none = multi_group_facts(_multi([], {"A": 1.0, "B": 2.0, "C": 3.0}, method="one_way_anova", post="none", p=0.4), "o", "g")
    assert "No post-hoc comparisons were run" in none


def test_many_pairs_are_capped_but_the_total_is_stated():
    means = {f"G{i}": float(i) for i in range(9)}
    pairs = [_pair(a, b, means[b] - means[a]) for i, a in enumerate(means) for b in list(means)[i + 1:]]
    facts = multi_group_facts(_multi(pairs, means), "o", "g")
    assert "36 of 36 pairs differ significantly" in facts
    assert f"and {36 - MAX_LISTED} more significant pairs not listed" in facts
    assert len(facts.splitlines()) <= 9 + 2 * MAX_LISTED + 8


# ------------------------------------------------- regression across shapes

CAT_STYLES = {
    "plain": ["Base plan", "Plus", "Pro"],
    "symbols": ["Base (default)", "Plus [new], 2024", "Pro vs Max"],
}


@pytest.mark.parametrize("cat_style", list(CAT_STYLES))
@pytest.mark.parametrize("n_noise", [0, 3, 10])
@pytest.mark.parametrize("scale", [1.0, 0.01])
def test_regression_facts_hold_across_dataset_shapes(tmp_path, store, cat_style, n_noise, scale):
    base, second, third = CAT_STYLES[cat_style]
    rng = np.random.default_rng(2000 + n_noise)
    n = 400
    cat = rng.choice([base, second, third], n, p=[0.5, 0.3, 0.2])
    x1, x2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    flag = rng.integers(0, 2, n)
    effect = {base: 0.0, second: 4.0, third: -2.0}
    y = scale * (5 + 2 * x1 - 3 * x2 + 1.5 * flag + np.array([effect[c] for c in cat]) + rng.normal(0, 1, n))
    data = {"cat": cat, "x1": x1, "x2": x2, "flag": flag, "y": y}
    for i in range(n_noise):
        data[f"noise{i}"] = rng.normal(0, 1, n)
    df = pd.DataFrame(data)
    predictors = ["cat", "x1", "x2", "flag"] + [f"noise{i}" for i in range(n_noise)]

    ds = make_dataset(df, tmp_path, store)
    plan = make_plan(QuestionType.DRIVER_ANALYSIS, "y", predictors, ds)
    result = REGRESSION.execute(ds, store, plan.record_method_decision(REGRESSION.select(ds, store, plan)))

    terms = {t["name"]: t for t in regression_terms(result)}
    assert "const" not in terms
    assert terms["x1"]["kind"] == "numeric" and terms["x1"]["direction"] == "higher"
    assert terms["x1"]["estimate"] == pytest.approx(2 * scale, abs=0.3 * scale)
    assert terms["x2"]["direction"] == "lower"
    assert terms["x2"]["estimate"] == pytest.approx(-3 * scale, abs=0.4 * scale)
    assert terms["flag"]["kind"] == "binary" and terms["flag"]["direction"] == "higher"
    up = terms[f"cat[{second} vs {base}]"]
    assert (up["kind"], up["level"], up["reference"], up["direction"]) == ("level", second, base, "higher")
    down = terms[f"cat[{third} vs {base}]"]
    assert (down["level"], down["reference"], down["direction"]) == (third, base, "lower")

    facts = regression_facts(result, "y")
    assert f"{second} has" in facts and f"than {base}" in facts
    assert "Intercept: not interpreted" in facts
    assert len(facts.splitlines()) <= 12 + 2 * MAX_LISTED
    assert not re.search(r"\bnan\b", facts.lower())
    json.dumps(result)


# -------------------------------------------------------------- small outcomes

def test_small_scale_outcomes_keep_distinguishable_means():
    r = {
        "method": "student_t_test", "p_value": 0.01, "effect_size": 0.3,
        "confidence_interval": (0.0005, 0.0040), "group_a_n": 500, "group_b_n": 500,
        "group_a_mean": 0.0123, "group_b_mean": 0.0145, "group_a_label": "Control", "group_b_label": "Variant",
    }
    facts = two_group_facts(r, "conversion_rate", "arm")
    assert "mean=0.0123" in facts and "mean=0.0145" in facts
    assert "Higher mean: Variant, by 0.0022" in facts


# ---------------------------------------- the fidelity check's own judgement

def test_judge_accepts_the_two_sentences_the_old_check_wrongly_flagged():
    s1 = "In the data, the 120 customers in the East region had an average total spend of 59.85, while the 120 customers in the West region averaged 49.95."
    s2 = "In an observational comparison of total spending, the East region (n = 120) had a mean spend of 60.67, while the West region (n = 120) had a mean spend of 51.00."
    assert judge(s1, {"East": 59.85, "West": 49.95})[0] == "ok"
    assert judge(s2, {"East": 60.67, "West": 51.0})[0] == "ok"


def test_judge_flags_a_real_swap_and_handles_many_groups_and_no_numbers():
    swapped = "East averaged 49.95 while West averaged 59.85."
    assert judge(swapped, {"East": 59.85, "West": 49.95})[0] == "SWAP?"
    assert judge("Gold (80.1) beat Silver (70.2) and Bronze (60.3).", {"Gold": 80.1, "Silver": 70.2, "Bronze": 60.3})[0] == "ok"
    assert judge("Silver (80.1) beat Gold (70.2) and Bronze (60.3).", {"Gold": 80.1, "Silver": 70.2, "Bronze": 60.3})[0] == "SWAP?"
    assert judge("There was a clear difference.", {"East": 1.0, "West": 2.0})[0] == "unclear"