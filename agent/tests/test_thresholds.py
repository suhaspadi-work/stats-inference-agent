"""
M1.4b step 2a: the named selection cut-offs (core.thresholds). Each selector reads its constant at call time, so
patching the constant in the module that uses it changes the choice; the decision records and the report facts
write their words from the same constants; every text still prints 0.05 exactly.
"""
import numpy as np
import pytest

import core.execution_records as execution_records
import core.multi_group as multi_group
import core.regression as regression
import core.report_facts as report_facts
import core.selection_records as selection_records
import core.testing as testing
import core.thresholds as thresholds
from core.dataset import LocalDiskStore
from core.multi_group import execute_multi_group_test, select_multi_group_method
from core.regression import select_regression_method
from core.report_facts import multi_group_facts
from core.selection_records import regression_selection_records
from core.testing import select_two_group_method
from tests.test_multi_group_selection_records import _frame as _groups_frame, _guard_frames as _multi_frames
from tests.test_regression_selection_records import _guard_frames as _regression_frames
from tests.test_report_facts_shapes import _multi, _pair
from tests.test_selection_records import _guard_frames as _two_group_frames
from tests.test_testing import make_dataset

NAMES = ("NORMALITY_ALPHA", "EQUAL_VARIANCE_ALPHA", "HETEROSKEDASTICITY_ALPHA", "OMNIBUS_ALPHA", "POST_HOC_ALPHA")
# M2.3a: the cleaning-candidate cut-offs, with their values
CLEANING = {"IMPUTE_MAX_MISSING_PCT": 0.40, "IMPUTE_MIN_OBSERVED": 10, "DISCRETE_MAX_DISTINCT": 20, "GROUP_MAX_LEVELS": 12,
            "GROUP_MIN_OBSERVED": 5, "MAX_GROUP_COLUMNS": 3,
            "NUMERIC_TEXT_PARSE_RATE": 0.95}  # M2.3a step 2: the numeric-text cut-off


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _ds(df, tmp_path, store, name):
    return make_dataset(df, tmp_path, store, name=name)


def _two_group(df, tmp_path, store, name="two"):
    return select_two_group_method(_ds(df, tmp_path, store, name), store, outcome_col="total_spend", group_col="region")


def _multi_group(df, tmp_path, store, name="multi"):
    return select_multi_group_method(_ds(df, tmp_path, store, name), store, outcome_col="total_spend", group_col="region")


def _execute(df, tmp_path, store, name):
    ds = _ds(df, tmp_path, store, name)
    method = select_multi_group_method(ds, store, outcome_col="total_spend", group_col="region").method
    return execute_multi_group_test(ds, store, "total_spend", "region", method)


def _regression(df, tmp_path, store, name="reg"):
    return select_regression_method(_ds(df, tmp_path, store, name), store, "y", ["x", "region"])


def _clear_anova():
    rng = np.random.default_rng(71)
    return _groups_frame({"East": rng.normal(50, 10, 60), "North": rng.normal(60, 10, 60),
                          "West": rng.normal(70, 10, 60)})


def _clear_ranks():
    rng = np.random.default_rng(72)
    return _groups_frame({"East": rng.exponential(5, 60), "North": rng.exponential(5, 60) + 10,
                          "West": rng.exponential(5, 60) + 20})


# --- the constants -------------------------------------------------------------------------------------------------

def test_the_constants():
    public = sorted(n for n in vars(thresholds) if n.isupper())
    assert public == sorted([*NAMES, *CLEANING])  # M2.3a: and the seven cleaning cut-offs
    assert {n: getattr(thresholds, n) for n in CLEANING} == CLEANING
    for name in NAMES:
        value = getattr(thresholds, name)
        assert type(value) is float and value == 0.05, name


# --- the selectors read them ---------------------------------------------------------------------------------------

def test_two_group_normality(tmp_path, store, monkeypatch):
    df = _two_group_frames()["normal_equal"]
    assert _two_group(df, tmp_path, store, "a").method == "student_t_test"
    monkeypatch.setattr(testing, "NORMALITY_ALPHA", 1.0)
    assert _two_group(df, tmp_path, store, "b").method == "mann_whitney_u"


def test_two_group_equal_variance(tmp_path, store, monkeypatch):
    df = _two_group_frames()["normal_equal"]
    monkeypatch.setattr(testing, "EQUAL_VARIANCE_ALPHA", 1.0)
    assert _two_group(df, tmp_path, store).method == "welch_t_test"


@pytest.mark.parametrize("name", ["NORMALITY_ALPHA", "EQUAL_VARIANCE_ALPHA"])
def test_multi_group_selection(name, tmp_path, store, monkeypatch):
    df = _multi_frames()["normal_equal"]
    assert _multi_group(df, tmp_path, store, "a").method == "one_way_anova"
    monkeypatch.setattr(multi_group, name, 1.0)
    assert _multi_group(df, tmp_path, store, "b").method == "kruskal_wallis"


@pytest.mark.parametrize("frame, method, post_hoc", [(_clear_anova, "one_way_anova", "tukey_hsd"),
                                                     (_clear_ranks, "kruskal_wallis", "dunn")], ids=["anova", "ranks"])
def test_omnibus(frame, method, post_hoc, tmp_path, store, monkeypatch):
    result = _execute(frame(), tmp_path, store, "a")
    assert (result.method, result.post_hoc_method) == (method, post_hoc) and result.p_value < 1e-6
    monkeypatch.setattr(multi_group, "OMNIBUS_ALPHA", 0.0)
    result = _execute(frame(), tmp_path, store, "b")
    assert result.post_hoc_method == "none" and result.pairwise_comparisons == ()


def test_post_hoc_dunn(tmp_path, store, monkeypatch):
    assert any(c.significant for c in _execute(_clear_ranks(), tmp_path, store, "a").pairwise_comparisons)
    monkeypatch.setattr(multi_group, "POST_HOC_ALPHA", 0.0)
    result = _execute(_clear_ranks(), tmp_path, store, "b")
    assert result.post_hoc_method == "dunn" and result.pairwise_comparisons
    assert not any(c.significant for c in result.pairwise_comparisons)


def test_post_hoc_tukey_receives_the_constant(tmp_path, store, monkeypatch):
    calls = []
    real = multi_group.pairwise_tukeyhsd

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)
    monkeypatch.setattr(multi_group, "pairwise_tukeyhsd", spy)
    assert _execute(_clear_anova(), tmp_path, store, "a").post_hoc_method == "tukey_hsd"
    monkeypatch.setattr(multi_group, "POST_HOC_ALPHA", 0.01)
    _execute(_clear_anova(), tmp_path, store, "b")
    assert [c["alpha"] for c in calls] == [thresholds.POST_HOC_ALPHA, 0.01]


def test_heteroskedasticity(tmp_path, store, monkeypatch):
    frames = _regression_frames()
    assert _regression(frames["homoscedastic"], tmp_path, store, "a").method == "ols"
    assert _regression(frames["heteroscedastic"], tmp_path, store, "b").method == "ols_hc3_robust"
    monkeypatch.setattr(regression, "HETEROSKEDASTICITY_ALPHA", 1.0)
    assert _regression(frames["homoscedastic"], tmp_path, store, "c").method == "ols_hc3_robust"
    monkeypatch.setattr(regression, "HETEROSKEDASTICITY_ALPHA", 0.0)
    assert _regression(frames["heteroscedastic"], tmp_path, store, "d").method == "ols"


def test_residual_normality(tmp_path, store, monkeypatch):
    df = _regression_frames()["homoscedastic"]
    caveat = "Residuals depart from normality"
    monkeypatch.setattr(regression, "NORMALITY_ALPHA", 1.0)
    assert caveat in _regression(df, tmp_path, store, "a").decision.rationale
    monkeypatch.setattr(regression, "NORMALITY_ALPHA", 0.0)
    assert caveat not in _regression(df, tmp_path, store, "b").decision.rationale


# --- the record modules and the facts read them ----------------------------------------------------------------------

def test_record_aliases():
    assert selection_records.ALPHA == thresholds.NORMALITY_ALPHA
    assert execution_records.ALPHA == thresholds.OMNIBUS_ALPHA


def _bp_reasons(tmp_path, store, name):
    choice = _regression(_regression_frames()["homoscedastic"], tmp_path, store, name)
    method_record = regression_selection_records(choice.decision)[0]
    return [c.reason for c in method_record.candidates]


def test_breusch_pagan_reasons(tmp_path, store, monkeypatch):
    assert "is above 0.05, so classical standard errors" in " ".join(_bp_reasons(tmp_path, store, "a"))
    monkeypatch.setattr(selection_records, "HETEROSKEDASTICITY_ALPHA", 0.1)
    reasons = " ".join(_bp_reasons(tmp_path, store, "b"))
    assert "is above 0.1, so classical standard errors" in reasons and "0.05" not in reasons


def _facts():
    return multi_group_facts(_multi([_pair("A", "B", 2.0)], {"A": 1.0, "B": 3.0}), "o", "g")


def test_post_hoc_facts_line(monkeypatch):
    assert "(p < 0.05, adjusted for multiple comparisons)" in _facts()
    monkeypatch.setattr(report_facts, "POST_HOC_ALPHA", 0.1)
    assert "(p < 0.1, adjusted for multiple comparisons)" in _facts()


# --- the texts are unchanged ---------------------------------------------------------------------------------------------

def test_two_group_rationale_text(tmp_path, store):
    rationale = _two_group(_two_group_frames()["normal_equal"], tmp_path, store).decision.rationale
    assert "Shapiro-Wilk p > 0.05" in rationale and "Levene's p > 0.05" in rationale


def test_multi_group_rationale_text(tmp_path, store):
    rationale = _multi_group(_multi_frames()["normal_equal"], tmp_path, store).decision.rationale
    assert "(Shapiro-Wilk p > 0.05) and have equal variances (Levene's p > 0.05)" in rationale
