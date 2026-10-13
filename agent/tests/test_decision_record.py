import dataclasses
import json
from datetime import datetime

import pytest

from core.decision_record import Candidate, DecisionRecord
from core.event_log import EventLog


def _full_record() -> DecisionRecord:
    return DecisionRecord(
        decision_point="stage2.test_selection",
        chosen="welch_t_test",
        decided_by="code_rule",
        candidates=(
            Candidate(option="welch_t_test", eligible=True,
                      reason="both groups have n >= 30", evidence={"n_min": 42}),
            Candidate(option="student_t_test", eligible=False,
                      reason="variance ratio 3.10 exceeds 2.0", evidence={"variance_ratio": 3.1}),
            Candidate(option="bootstrap_test", eligible=None, reason="not evaluated"),
        ),
        rationale="Welch handles unequal variances.",
        evidence={"shapiro_p": [0.21, 0.04], "levene_p": 0.003},
        chart_data=({"type": "boxplot", "values": [1.0, 2.0, 3.0]},),
    )


def _contains_tuple_or_dataclass(obj) -> bool:
    if isinstance(obj, tuple) or dataclasses.is_dataclass(obj):
        return True
    if isinstance(obj, list):
        return any(_contains_tuple_or_dataclass(v) for v in obj)
    if isinstance(obj, dict):
        return any(_contains_tuple_or_dataclass(v) for v in obj.values())
    return False


# --- defaults -------------------------------------------------------------

def test_record_defaults():
    r = DecisionRecord(decision_point="stage2.test_selection", chosen="welch_t_test", decided_by="code_rule")
    assert r.candidates == ()
    assert r.rationale == ""
    assert r.evidence == {}
    assert r.overridden_by_human is False
    assert r.original_choice is None
    assert r.chart_data == ()
    assert r.supersedes is None
    assert r.schema_version == 1
    assert len(r.id) == 36
    assert datetime.fromisoformat(r.created_at).utcoffset().total_seconds() == 0


def test_record_ids_are_unique():
    a = DecisionRecord(decision_point="p", chosen="x", decided_by="model")
    b = DecisionRecord(decision_point="p", chosen="x", decided_by="model")
    assert a.id != b.id


def test_candidate_defaults():
    c = Candidate(option="ols", eligible=None, reason="not evaluated")
    assert c.evidence == {}
    assert c.generated_by == "code"
    assert c.eligible is None


# --- immutability ---------------------------------------------------------

def test_record_is_frozen():
    r = _full_record()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.chosen = "student_t_test"


def test_candidate_is_frozen():
    c = Candidate(option="ols", eligible=True, reason="r")
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.eligible = False


# --- validation -----------------------------------------------------------

@pytest.mark.parametrize("value", ["", "   "])
def test_empty_decision_point_rejected(value):
    with pytest.raises(ValueError, match="decision_point"):
        DecisionRecord(decision_point=value, chosen="x", decided_by="code_rule")


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_chosen_rejected(value):
    with pytest.raises(ValueError, match="chosen"):
        DecisionRecord(decision_point="p", chosen=value, decided_by="code_rule")


def test_invalid_decided_by_rejected():
    with pytest.raises(ValueError, match="decided_by"):
        DecisionRecord(decision_point="p", chosen="x", decided_by="robot")


def test_invalid_generated_by_rejected():
    with pytest.raises(ValueError, match="generated_by"):
        Candidate(option="ols", eligible=True, reason="r", generated_by="human")


def test_override_requires_human_decided_by():
    with pytest.raises(ValueError, match="decided_by must be 'human'"):
        DecisionRecord(decision_point="p", chosen="x", decided_by="code_rule",
                       overridden_by_human=True, original_choice="y")


def test_override_requires_original_choice():
    with pytest.raises(ValueError, match="original_choice"):
        DecisionRecord(decision_point="p", chosen="x", decided_by="human", overridden_by_human=True)


def test_valid_human_override():
    r = DecisionRecord(decision_point="stage2.regression_method", chosen="logit",
                       decided_by="human", overridden_by_human=True, original_choice="ols")
    assert r.original_choice == "ols"


# --- serialization --------------------------------------------------------

def test_to_dict_is_json_safe_and_plain():
    d = _full_record().to_dict()
    json.dumps(d)
    assert not _contains_tuple_or_dataclass(d)
    assert isinstance(d["candidates"], list)
    assert isinstance(d["candidates"][0], dict)
    assert isinstance(d["chart_data"], list)


def test_to_dict_does_not_alias_evidence():
    r = _full_record()
    d = r.to_dict()
    d["evidence"]["levene_p"] = 0.9
    assert r.evidence["levene_p"] == 0.003


def test_round_trip_equal():
    r = _full_record()
    assert DecisionRecord.from_dict(r.to_dict()) == r


def test_round_trip_through_json():
    r = _full_record()
    assert DecisionRecord.from_dict(json.loads(json.dumps(r.to_dict()))) == r


def test_from_dict_ignores_unknown_key():
    d = _full_record().to_dict()
    d["not_a_field"] = 123
    d["candidates"][0]["also_unknown"] = "x"
    rebuilt = DecisionRecord.from_dict(d)
    assert not hasattr(rebuilt, "not_a_field")
    del d["not_a_field"]
    del d["candidates"][0]["also_unknown"]
    assert rebuilt == DecisionRecord.from_dict(d)


def test_from_dict_accepts_missing_optional_keys():
    r = DecisionRecord.from_dict({"decision_point": "p", "chosen": "x", "decided_by": "model"})
    assert r.candidates == ()
    assert r.chart_data == ()
    assert r.overridden_by_human is False
    assert r.schema_version == 1


@pytest.mark.parametrize("key", ["decision_point", "chosen", "decided_by"])
def test_from_dict_names_missing_required_key(key):
    d = {"decision_point": "p", "chosen": "x", "decided_by": "model"}
    del d[key]
    with pytest.raises(ValueError, match=key):
        DecisionRecord.from_dict(d)


def test_candidate_from_dict_names_missing_required_key():
    d = _full_record().to_dict()
    del d["candidates"][1]["reason"]
    with pytest.raises(ValueError, match="Candidate.*reason"):
        DecisionRecord.from_dict(d)


# --- skipped decision -----------------------------------------------------

def test_skipped_decision():
    r = DecisionRecord(
        decision_point="stage1.clean.skew_transform",
        chosen="skipped",
        decided_by="code_rule",
        rationale="max absolute skew 0.42 is below threshold 1.0",
        evidence={"max_abs_skew": 0.42, "threshold": 1.0},
    )
    assert r.chosen == "skipped"
    assert DecisionRecord.from_dict(r.to_dict()) == r


# --- event log ------------------------------------------------------------

def test_event_log_round_trip(tmp_path):
    log = EventLog(log_path=tmp_path / "session.jsonl")
    r = _full_record()
    log.append("decision", r.to_dict())
    stored = log.read_by_type("decision")
    assert stored == [r.to_dict()]
    assert DecisionRecord.from_dict(stored[0]) == r
