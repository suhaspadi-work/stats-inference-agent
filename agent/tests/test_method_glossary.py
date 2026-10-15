"""
M1.4b step 2a: the method glossary (core.method_glossary). One plain-language entry per method the agent can
choose; its "when the agent uses it" text is built from the selectors' own named cut-offs; association language
only. These tests also hold the coverage check: every choosable method has an entry and no entry is an orphan.
"""
import copy
import importlib
import json

import pytest

import core.method_glossary as method_glossary
import core.thresholds as thresholds
from core.dataset import LocalDiskStore
from core.multi_group import select_multi_group_method
from core.regression import SCHEMES, select_regression_method
from core.report import find_causal_violation
from core.selection_records import _POST_HOC, MULTI_GROUP_METHODS, REGRESSION_METHODS, TWO_GROUP_METHODS
from core.testing import select_two_group_method
from tests.test_multi_group_selection_records import _guard_frames as _multi_frames
from tests.test_regression_selection_records import _guard_frames as _regression_frames
from tests.test_selection_records import _guard_frames as _two_group_frames
from tests.test_testing import make_dataset

KEYS = {"name", "kind", "what", "answers", "when_used", "assumes", "alternatives", "reading"}
KINDS = {"two_group_test", "multi_group_test", "post_hoc", "regression", "encoding"}
HEADINGS = ["What it is", "What it answers", "When the agent uses it", "What it assumes", "How to read the result",
            "Other options"]
CAUSAL_WORDS = ("cause", "drive", "lead to", "result in")
GROUPS = {
    "two_group": set(TWO_GROUP_METHODS), "multi_group": set(MULTI_GROUP_METHODS), "post_hoc": set(_POST_HOC.values()),
    "regression": set(REGRESSION_METHODS), "encoding": set(SCHEMES),
}


def _g(value):
    return f"{value:g}"


@pytest.fixture
def patched_thresholds():
    """A MonkeyPatch for core.thresholds; afterwards the patches are undone and the glossary is reloaded, so other
    tests see the default texts even when an assertion fails."""
    mp = pytest.MonkeyPatch()
    try:
        yield mp
    finally:
        mp.undo()
        importlib.reload(method_glossary)


def _texts(entry):
    return [entry["name"], entry["what"], entry["answers"], entry["reading"], *entry["when_used"], *entry["assumes"]]


# --- coverage (replaces the import-time check) ------------------------------------------------------------------------

def test_every_choosable_method_has_an_entry_and_none_is_an_orphan():
    choosable = set().union(*GROUPS.values())
    assert choosable - set(method_glossary.GLOSSARY) == set()
    assert set(method_glossary.GLOSSARY) - choosable == set()


def test_method_ids():
    ids = method_glossary.method_ids()
    assert ids == sorted(ids) == sorted(method_glossary.GLOSSARY) and len(ids) == 12


# --- shape -------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("method_id", sorted(method_glossary.GLOSSARY))
def test_entry_shape(method_id):
    entry = method_glossary.GLOSSARY[method_id]
    assert set(entry) == KEYS
    assert entry["kind"] in KINDS
    for key in ("name", "what", "answers", "reading"):
        assert isinstance(entry[key], str) and entry[key].strip(), key
    for key in ("when_used", "assumes"):
        assert isinstance(entry[key], list) and entry[key], key
        assert all(isinstance(line, str) and line.strip() for line in entry[key]), key
    assert entry["alternatives"] and all(a in method_glossary.GLOSSARY and a != method_id for a in entry["alternatives"])


def test_kinds_match_the_method_groups():
    kind_of = {"two_group": "two_group_test", "multi_group": "multi_group_test", "post_hoc": "post_hoc",
               "regression": "regression", "encoding": "encoding"}
    for group, ids in GROUPS.items():
        assert {method_glossary.GLOSSARY[i]["kind"] for i in ids} == {kind_of[group]}, group


@pytest.mark.parametrize("ids", [TWO_GROUP_METHODS, MULTI_GROUP_METHODS], ids=["two_group", "multi_group"])
def test_alternatives_are_symmetric(ids):
    for a in ids:
        for b in ids:
            if a != b:
                assert b in method_glossary.GLOSSARY[a]["alternatives"], (a, b)


# --- the constants -------------------------------------------------------------------------------------------------------

def _when(method_id):
    return "\n".join(method_glossary.GLOSSARY[method_id]["when_used"])


def _check_numbers(normality, variance, hetero, omnibus, post_hoc):
    for method_id in ("student_t_test", "welch_t_test", "mann_whitney_u", "one_way_anova", "kruskal_wallis"):
        assert f"Shapiro-Wilk p {'>' if method_id in ('student_t_test', 'welch_t_test', 'one_way_anova') else '<='} " \
               f"{normality}" in _when(method_id), method_id
    for method_id in ("student_t_test", "one_way_anova"):
        assert f"Levene's p > {variance}" in _when(method_id), method_id
    for method_id in ("welch_t_test", "kruskal_wallis"):
        assert f"Levene's p <= {variance}" in _when(method_id), method_id
    assert f"(p > {hetero})" in _when("ols") and f"(p <= {hetero})" in _when("ols_hc3_robust")
    for method_id in ("tukey_hsd", "dunn"):
        assert f"(p < {omnibus})" in _when(method_id), method_id
        assert f"below {post_hoc}." in method_glossary.GLOSSARY[method_id]["reading"], method_id
    for method_id in ("one_way_anova", "kruskal_wallis"):
        assert f"below {omnibus}." in method_glossary.GLOSSARY[method_id]["reading"], method_id


def test_texts_use_the_constants():
    _check_numbers(_g(thresholds.NORMALITY_ALPHA), _g(thresholds.EQUAL_VARIANCE_ALPHA),
                   _g(thresholds.HETEROSKEDASTICITY_ALPHA), _g(thresholds.OMNIBUS_ALPHA), _g(thresholds.POST_HOC_ALPHA))


def test_texts_follow_patched_constants(patched_thresholds):
    values = {"NORMALITY_ALPHA": 0.11, "EQUAL_VARIANCE_ALPHA": 0.12, "HETEROSKEDASTICITY_ALPHA": 0.13,
              "OMNIBUS_ALPHA": 0.14, "POST_HOC_ALPHA": 0.15}
    for name, value in values.items():
        patched_thresholds.setattr(thresholds, name, value)
    importlib.reload(method_glossary)
    _check_numbers("0.11", "0.12", "0.13", "0.14", "0.15")
    assert "0.05" not in json.dumps(method_glossary.GLOSSARY)


def test_the_reload_restored_the_defaults():
    # runs after the patched test in file order; the fixture reloaded the glossary with the real constants
    assert "Shapiro-Wilk p > 0.05" in _when("student_t_test")


# --- language --------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("method_id", sorted(method_glossary.GLOSSARY))
def test_association_language_only(method_id):
    render = method_glossary.render_glossary(method_id)
    texts = _texts(method_glossary.GLOSSARY[method_id]) + [render["title"]] + [s["text"] for s in render["sections"]]
    for text in texts:
        assert find_causal_violation(text) is None, text
        assert not any(word in text.lower() for word in CAUSAL_WORDS), text


# --- the functions -------------------------------------------------------------------------------------------------------

def test_glossary_entry_is_a_deep_copy():
    before = copy.deepcopy(method_glossary.GLOSSARY)
    entry = method_glossary.glossary_entry("ols")
    entry["when_used"].append("changed")
    entry["name"] = "changed"
    assert method_glossary.GLOSSARY == before


def test_unknown_id():
    with pytest.raises(KeyError, match="made_up_test"):
        method_glossary.glossary_entry("made_up_test")
    with pytest.raises(KeyError, match="made_up_test"):
        method_glossary.render_glossary("made_up_test")


def test_render_glossary():
    entry = method_glossary.GLOSSARY["student_t_test"]
    render = method_glossary.render_glossary("student_t_test")
    assert render["title"] == "Student's t-test"
    assert [s["heading"] for s in render["sections"]] == HEADINGS
    text = {s["heading"]: s["text"] for s in render["sections"]}
    assert text["What it is"] == entry["what"] and text["What it answers"] == entry["answers"]
    assert text["When the agent uses it"] == "\n".join(entry["when_used"])
    assert text["What it assumes"] == "\n".join(entry["assumes"])
    assert text["How to read the result"] == entry["reading"]
    assert text["Other options"] == "Welch's t-test, Mann-Whitney U test"


def test_render_without_alternatives(monkeypatch):
    lonely = {**copy.deepcopy(method_glossary.GLOSSARY["ols"]), "alternatives": []}
    monkeypatch.setitem(method_glossary.GLOSSARY, "lonely", lonely)
    assert [s["heading"] for s in method_glossary.render_glossary("lonely")["sections"]] == HEADINGS[:-1]


@pytest.mark.parametrize("method_id", sorted(method_glossary.GLOSSARY))
def test_every_render_serializes(method_id):
    json.dumps(method_glossary.render_glossary(method_id))


# --- consistency with the real selectors ----------------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def _chosen(tmp_path, store):
    chosen = {}
    two = _two_group_frames()
    for name in ("normal_equal", "normal_unequal", "skewed"):
        ds = make_dataset(two[name], tmp_path, store, name=f"two_{name}")
        chosen[f"two_{name}"] = select_two_group_method(ds, store, outcome_col="total_spend", group_col="region").method
    multi = _multi_frames()
    for name in ("normal_equal", "skewed"):
        ds = make_dataset(multi[name], tmp_path, store, name=f"multi_{name}")
        chosen[f"multi_{name}"] = select_multi_group_method(ds, store, outcome_col="total_spend",
                                                            group_col="region").method
    regression = _regression_frames()
    for name in ("homoscedastic", "heteroscedastic"):
        ds = make_dataset(regression[name], tmp_path, store, name=f"reg_{name}")
        choice = select_regression_method(ds, store, "y", ["x", "region"])
        chosen[f"reg_{name}"] = choice.method
        chosen[f"enc_{name}"] = choice.encodings["region"]["scheme"]
    return chosen


def test_the_selectors_emit_glossary_ids(tmp_path, store):
    chosen = _chosen(tmp_path, store)
    assert chosen == {
        "two_normal_equal": "student_t_test", "two_normal_unequal": "welch_t_test", "two_skewed": "mann_whitney_u",
        "multi_normal_equal": "one_way_anova", "multi_skewed": "kruskal_wallis",
        "reg_homoscedastic": "ols", "reg_heteroscedastic": "ols_hc3_robust",
        "enc_homoscedastic": "treatment", "enc_heteroscedastic": "treatment",
    }
    for method_id in chosen.values():
        assert method_glossary.render_glossary(method_id)["title"] == method_glossary.GLOSSARY[method_id]["name"]
