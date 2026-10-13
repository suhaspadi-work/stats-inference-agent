import json

import numpy as np
import pandas as pd
import pytest
from langgraph.graph import StateGraph, START, END

from core.agent_state import AgentState
from core.dataset import LocalDiskStore
from core.decision_log import log_update
from core.nodes import make_select_method_node
from core.plan import AnalysisPlan, CausalStatus, MethodDecision, OutcomeType, QuestionType
from core.regression import select_regression_method
from core.selection_records import regression_selection_records
from tests.test_regression import make_dataset


def _enc(scheme="treatment", reference="East", order=None, source="default", levels=None):
    return {"scheme": scheme, "reference": reference, "order": order, "source": source,
            "levels": levels if levels is not None else {"East": 120, "North": 100, "West": 80}}


def _decision(method="ols", bp=0.42, hetero=None, n=300, encodings=None, chart_data=()):
    evidence = {
        "n_observations": n, "n_dropped_missing": 0, "n_parameters": 4,
        "breusch_pagan_p": bp,
        "heteroscedasticity_detected": bp <= 0.05 if hetero is None else hetero,
        "shapiro_p_residuals": 0.3, "max_vif": 1.2, "high_vif_columns": [], "rare_levels": {},
        "encodings": encodings if encodings is not None else {},
    }
    return MethodDecision(stage="test_selection", chosen_method=method, rationale="selector rationale",
                          alternatives=(), evidence=evidence, chart_data=chart_data)


def _by_option(record):
    return {c.option: c for c in record.candidates}


# --- method record ------------------------------------------------------------

def test_homoscedastic_chooses_ols():
    (r,) = regression_selection_records(_decision("ols", bp=0.42))
    c = _by_option(r)
    assert r.decision_point == "stage3.regression_method"
    assert r.decided_by == "code_rule"
    assert r.chosen == "ols"
    assert r.rationale == "selector rationale"
    assert [c["ols"].eligible, c["ols_hc3_robust"].eligible] == [True, True]
    assert c["ols"].reason == ("chosen: Breusch-Pagan p = 0.4200 (n=300) is above 0.05, so classical "
                               "standard errors are valid and more efficient")
    assert c["ols_hc3_robust"].reason == (
        "eligible, and also valid when variance is constant, but Breusch-Pagan p = 0.4200 is above 0.05, "
        "so classical standard errors are used, which are more efficient")
    assert r.evidence["family"] == "regression"
    assert all(cand.evidence == {"breusch_pagan_p": 0.42} for cand in r.candidates)


def test_heteroscedastic_chooses_hc3_and_ols_ineligible():
    (r,) = regression_selection_records(_decision("ols_hc3_robust", bp=0.003, n=250))
    c = _by_option(r)
    assert r.chosen == "ols_hc3_robust"
    assert c["ols"].eligible is False
    assert c["ols"].reason == (
        "not eligible: Breusch-Pagan p = 0.0030 is at or below 0.05, so the residual variance is not "
        "constant and classical standard errors would be unreliable")
    assert c["ols_hc3_robust"].reason == (
        "chosen: Breusch-Pagan p = 0.0030 (n=250) is at or below 0.05, so the residual variance is not "
        "constant and heteroscedasticity-robust (HC3) standard errors are used")


def test_g30_rounded_0_05_follows_the_flag():
    (ols,) = regression_selection_records(_decision("ols", bp=0.05, hetero=False))
    assert ols.chosen == "ols"
    assert _by_option(ols)["ols"].eligible is True
    assert "p = 0.0500 [rounded; the selector used the unrounded value] (n=300) is above 0.05" in \
        _by_option(ols)["ols"].reason

    (hc3,) = regression_selection_records(_decision("ols_hc3_robust", bp=0.05, hetero=True))
    assert hc3.chosen == "ols_hc3_robust"
    assert _by_option(hc3)["ols"].eligible is False
    assert "[rounded; the selector used the unrounded value]" in _by_option(hc3)["ols_hc3_robust"].reason


@pytest.mark.parametrize("decision", [_decision("ols"), _decision("ols_hc3_robust", bp=0.001)])
def test_only_two_method_candidates(decision):
    r = regression_selection_records(decision)[0]
    assert [c.option for c in r.candidates] == ["ols", "ols_hc3_robust"]
    assert all(c.generated_by == "code" for c in r.candidates)


@pytest.mark.parametrize("key", ["n_observations", "n_parameters", "breusch_pagan_p",
                                 "heteroscedasticity_detected", "encodings"])
def test_missing_key_raises(key):
    d = _decision()
    evidence = {k: v for k, v in d.evidence.items() if k != key}
    bad = MethodDecision(stage=d.stage, chosen_method=d.chosen_method, rationale="r",
                         alternatives=(), evidence=evidence)
    with pytest.raises(ValueError, match=key):
        regression_selection_records(bad)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="wls"):
        regression_selection_records(_decision("wls"))


def test_chart_data_not_copied():
    records = regression_selection_records(
        _decision(encodings={"region": _enc()}, chart_data=({"chart_type": "residual_plot", "x": [1.0]},)))
    assert all(r.chart_data == () for r in records)
    assert "chart_data" not in records[0].evidence


# --- encoding records -------------------------------------------------------

def test_no_categorical_columns_gives_one_record():
    records = regression_selection_records(_decision())
    assert len(records) == 1


def test_two_categorical_columns_give_three_records_in_order():
    records = regression_selection_records(_decision(encodings={
        "region": _enc(),
        "plan": _enc(reference="Basic", levels={"Basic": 200, "Pro": 100}),
    }))
    assert [r.decision_point for r in records] == ["stage3.regression_method", "stage3.encoding", "stage3.encoding"]
    assert [r.evidence.get("column") for r in records[1:]] == ["region", "plan"]


def test_default_treatment_encoding():
    r = regression_selection_records(_decision(encodings={"region": _enc()}))[1]
    c = _by_option(r)
    assert r.chosen == "treatment"
    assert r.decided_by == "code_rule"
    assert r.overridden_by_human is False and r.original_choice is None
    assert r.rationale == ("treatment coding against 'East', the largest of 3 levels (n=120); this is a "
                           "fixed default rule and no ordering evidence was considered (G33)")
    assert c["treatment"].reason == "chosen: treatment coding against 'East', the largest of 3 levels (n=120)"
    assert c["effect"].eligible is True
    assert c["effect"].reason.endswith("not chosen because no effect coding was requested for this column")
    assert c["ordinal"].eligible is False
    assert c["ordinal"].reason == "no category order was supplied, and an order cannot be inferred from the data"
    assert r.evidence == {
        "column": "region", "scheme": "treatment", "reference": "East", "source": "default",
        "n_levels": 3, "reference_n": 120, "levels": {"East": 120, "North": 100, "West": 80},
    }
    assert [o.option for o in r.candidates] == ["treatment", "effect", "ordinal"]


def test_user_effect_override():
    r = regression_selection_records(_decision(encodings={"region": _enc(scheme="effect", source="user")}))[1]
    assert r.chosen == "effect"
    assert r.decided_by == "human"
    assert r.overridden_by_human is True
    assert r.original_choice == "default: treatment coding against the largest level"
    assert r.rationale == "effect coding was explicitly requested for this column"
    c = _by_option(r)
    assert c["effect"].reason.startswith("chosen: effect coding was explicitly requested")
    assert c["treatment"].eligible is True
    assert c["treatment"].reason.endswith("not chosen because effect coding was requested for this column")


def test_user_treatment_override_with_chosen_reference():
    r = regression_selection_records(_decision(encodings={"region": _enc(reference="West", source="user")}))[1]
    assert r.decided_by == "human"
    assert r.rationale == "treatment coding against 'West', the level you chose"
    assert "(n=80), the level you chose, of 3 levels" in _by_option(r)["treatment"].reason
    assert r.evidence["reference_n"] == 80


def test_supplied_order_makes_ordinal_chosen():
    enc = _enc(scheme="ordinal", reference=None, order=["low", "medium", "high"], source="user",
               levels={"low": 50, "medium": 80, "high": 40})
    r = regression_selection_records(_decision(encodings={"size": enc}))[1]
    c = _by_option(r)
    assert r.chosen == "ordinal"
    assert c["ordinal"].eligible is True
    assert r.rationale == "ordinal coding using the supplied order low < medium < high"
    assert c["treatment"].reason == ("eligible; not chosen because an order was supplied, so one slope "
                                     "per step is used and equal spacing is assumed")
    assert c["effect"].reason == c["treatment"].reason
    assert r.evidence["reference"] is None and r.evidence["reference_n"] is None


def test_awkward_names_survive_log_update_and_no_mutation():
    levels = {"North_East": 90, "South West": 60, "by_group": 30}
    d = _decision(encodings={"home region_2": _enc(reference="North_East", levels=levels)})
    before = json.dumps(d.evidence, sort_keys=True)
    update = log_update(*regression_selection_records(d))
    json.dumps(update)
    enc_entry = update["decision_log"][1]
    assert enc_entry["evidence"]["column"] == "home region_2"
    assert enc_entry["evidence"]["levels"] == levels
    assert "'North_East', the largest of 3 levels (n=90)" in enc_entry["rationale"]
    assert json.dumps(d.evidence, sort_keys=True) == before
    assert "family" not in d.evidence


# --- consistency guard against the real selector -------------------------------

@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _frame(rng, sizes, hetero=False):
    region = np.repeat(list(sizes), list(sizes.values()))
    x = rng.normal(0, 1, len(region))
    shift = {k: i * 2.0 for i, k in enumerate(sizes)}
    noise = rng.normal(0, 1, len(region))
    if hetero:
        noise = noise * np.exp(1.2 * x)  # noise grows with the predictor
    y = 5 + 2 * x + np.array([shift[r] for r in region]) + noise
    return pd.DataFrame({"x": x, "region": region, "y": y})


def _guard_frames():
    return {
        "homoscedastic": _frame(np.random.default_rng(4), {"East": 120, "North": 100, "West": 80}),
        "heteroscedastic": _frame(np.random.default_rng(2), {"East": 120, "North": 100, "West": 80}, hetero=True),
        "rare_level": _frame(np.random.default_rng(3), {"East": 150, "North": 145, "West": 3}),
    }


def test_records_agree_with_the_real_selector(tmp_path, store):
    chosen = {}
    for name, df in _guard_frames().items():
        dataset = make_dataset(df, tmp_path, store, name=name)
        choice = select_regression_method(dataset, store, "y", ["x", "region"])
        records = regression_selection_records(choice.decision)
        method = records[0]
        assert method.chosen == choice.method, name
        assert _by_option(method)[method.chosen].eligible is True, name
        assert len(records) == 1 + len(choice.encodings), name
        for enc_record in records[1:]:
            column = enc_record.evidence["column"]
            assert enc_record.evidence["reference"] == choice.encodings[column]["reference"], name
            assert _by_option(enc_record)[enc_record.chosen].eligible is True, name
        chosen[name] = (method.chosen, choice.decision.evidence["rare_levels"])
    assert chosen["homoscedastic"][0] == "ols"
    assert chosen["heteroscedastic"][0] == "ols_hc3_robust"
    assert chosen["rare_level"][1] == {"region": ["West"]}


# --- node ---------------------------------------------------------------------

def test_select_node_logs_method_and_encoding_records(tmp_path, store):
    dataset = make_dataset(_guard_frames()["homoscedastic"], tmp_path, store, name="node_reg")
    plan = AnalysisPlan(
        request_text="x", session_id="session-abc",
        question_type=QuestionType.DRIVER_ANALYSIS, causal_status=CausalStatus.OBSERVATIONAL,
        outcome_name="y", outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="row", data_sources=(dataset.handle,), candidate_predictors=("x", "region"),
    )
    graph = StateGraph(AgentState)
    graph.add_node("n", make_select_method_node(store))
    graph.add_edge(START, "n")
    graph.add_edge("n", END)
    result = graph.compile().invoke({"plan": plan, "dataset": dataset, "request_text": "x",
                                     "session_id": "session-abc"})
    log = result["decision_log"]
    assert [e["decision_point"] for e in log] == ["stage3.regression_method", "stage3.encoding"]
    assert log[0]["chosen"] == result["plan"].primary_method
    assert log[1]["evidence"]["column"] == "region"
    assert len(result["plan"].method_decisions) == 1
