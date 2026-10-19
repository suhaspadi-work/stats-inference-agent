"""
M2.3a: the report prompts of the three analysis generators keep the rejected-fix wording exactly ("identified but NOT
fixed, by explicit user decision") and state a plain caveat (an imputed outcome) in its own sentence, which never
claims a user decision. Without a plain caveat the prompt is byte-identical to the one before M2.3a.
"""
import pytest

import core.report as report_module
from core.plan import AnalysisPlan, CausalStatus, OutcomeType, QuestionType, UnaddressedIssue
from core.report import generate_multi_group_report, generate_regression_report, generate_report
from tests.test_report_facts_shapes import _multi, _pair

REJECTED = UnaddressedIssue(issue_type="duplicate_rows", description="4 exact duplicate rows",
                            proposed_operation="dedupe({'subset': None})", rejection_reason="repeat visits are real",
                            affected_columns=("id",))
CAVEAT = UnaddressedIssue(issue_type="outcome_imputed",
                          description="The outcome 'total_spend' had missing values filled (median) before the analysis, "
                                      "which changes its distribution and the result.",
                          proposed_operation="impute({...})",
                          rejection_reason="the values were filled in Stage 1, before the outcome was chosen",
                          affected_columns=("total_spend",))
OLD_INTRO = "\n\nIMPORTANT: The following data-quality issue(s) were identified but NOT fixed, by explicit user decision: "
NEW_SENTENCE = ("\n\nAlso state this caveat plainly, in your own words, and do not describe it as something the user "
                "chose not to fix: The outcome 'total_spend' had missing values filled (median) before the analysis, "
                "which changes its distribution and the result.")
LONG_TAIL = (" You MUST mention this plainly in your summary and note that the results should be interpreted with this "
             "caveat in mind. Do not omit this or bury it -- a reader should not come away with more confidence in the "
             "result than is warranted given this known, unaddressed issue.")
REJECTED_TEXT = "duplicate_rows (4 exact duplicate rows) was identified but left unaddressed because: repeat visits are real."


class Model:
    def __init__(self):
        self.system = None

    def invoke(self, messages):
        self.system = dict(messages)["system"]
        return type("Response", (), {"content": "ok"})()


def _plan(issues):
    plan = AnalysisPlan(request_text="Did West spend more?", session_id="s", question_type=QuestionType.TWO_GROUP_COMPARISON,
                        causal_status=CausalStatus.OBSERVATIONAL, outcome_name="total_spend",
                        outcome_type=OutcomeType.CONTINUOUS, unit_of_analysis="customer", data_sources=("c@v1",),
                        candidate_predictors=("region",))
    for issue in issues:
        plan = plan.record_unaddressed_issue(issue)
    return plan


TWO_GROUP = {"method": "welch_t_test", "statistic": 2.5, "p_value": 0.012, "effect_size": 0.35,
             "confidence_interval": (1.2, 8.4), "group_a_n": 200, "group_b_n": 200, "group_a_mean": 50.1,
             "group_b_mean": 55.3}
REGRESSION = {"method": "ols", "n_observations": 100, "n_dropped_missing": 0, "r_squared": 0.3, "adj_r_squared": 0.28,
              "aic": 1.0, "f_statistic": 5.0, "f_p_value": 0.001, "encodings": {}, "alternative_fits": [],
              "coefficients": [{"name": "Intercept", "estimate": 1.0, "std_error": 0.1, "t_statistic": 10.0,
                                "p_value": 0.001, "ci_lower": 0.8, "ci_upper": 1.2, "vif": None, "kind": "intercept",
                                "column": None, "level": None, "reference": None}]}
GENERATORS = [(generate_report, TWO_GROUP, LONG_TAIL),
              (generate_multi_group_report, _multi([_pair("A", "B", 2.0)], {"A": 1.0, "B": 3.0}), " You MUST mention this plainly."),
              (generate_regression_report, REGRESSION, " You MUST mention this plainly.")]
IDS = ["two_group", "multi_group", "regression"]


def _system(monkeypatch, generate, result, issues):
    model = Model()
    monkeypatch.setattr(report_module, "init_chat_model", lambda *a, **k: model)
    generate(_plan(issues), result)
    return model.system


@pytest.mark.parametrize("generate, result, tail", GENERATORS, ids=IDS)
def test_only_a_rejected_fix(generate, result, tail, monkeypatch):
    system = _system(monkeypatch, generate, result, [REJECTED])
    assert OLD_INTRO + REJECTED_TEXT + tail in system and "Also state this caveat" not in system


@pytest.mark.parametrize("generate, result, tail", GENERATORS, ids=IDS)
def test_only_an_imputed_outcome(generate, result, tail, monkeypatch):
    system = _system(monkeypatch, generate, result, [CAVEAT])
    assert NEW_SENTENCE in system and ".." not in system
    assert "by explicit user decision" not in system and "left unaddressed" not in system


@pytest.mark.parametrize("generate, result, tail", GENERATORS, ids=IDS)
def test_both(generate, result, tail, monkeypatch):
    system = _system(monkeypatch, generate, result, [REJECTED, CAVEAT])
    assert OLD_INTRO + REJECTED_TEXT + tail + NEW_SENTENCE in system
    assert "outcome_imputed" not in system


@pytest.mark.parametrize("generate, result, tail", GENERATORS, ids=IDS)
def test_no_issue_adds_nothing(generate, result, tail, monkeypatch):
    system = _system(monkeypatch, generate, result, [])
    assert "IMPORTANT: The following data-quality" not in system and "Also state this caveat" not in system
