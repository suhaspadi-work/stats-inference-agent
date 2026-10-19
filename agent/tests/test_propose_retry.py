"""
M2.3a: the proposer corrects an invalid choice once. When the model's answer fails validation, one more call carries
the error (option ids and indexes only) and the same options; a second failure raises. A corrected answer is marked
(a private flag, not in the schema the model sees), recorded on the fix record, and the Stage 1 report says so.
"""
import json
import re

import numpy as np
import pandas as pd
import pytest

import core.classifier as classifier_module
from core.classifier import _ProposedOperationOut, _WranglingProposalsOut, propose_wrangling_operations
from core.dataset import LocalDiskStore
from core.graph import build_agent_graph
from core.plan import QuestionType
from tests.test_decision_point_guard import _dataset, confident, fake_draft
from tests.test_exploratory_round import _frame
from tests.test_prompt_canaries import fake_model  # noqa: F401 (fixture)
from tests.test_propose_choices import CANARY, _p, _setup
from tests.test_stage1_report import _drive

SUFFIX = " (the model was asked once to correct its choice and did not)."


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


class Script:
    """A fake chat model answering with the scripted proposal lists in order (or raising a scripted error)."""

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def with_structured_output(self, schema):
        assert schema is _WranglingProposalsOut
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return _WranglingProposalsOut(proposals=answer)


def _run(monkeypatch, issues, candidates, *answers):
    model = Script(*answers)
    monkeypatch.setattr(classifier_module, "init_chat_model", lambda *a, **k: model)
    return model, propose_wrangling_operations(issues, candidates)


def _valid():
    """Fresh proposal objects each time (the retry marks the objects it returns)."""
    return [_p(0, "impute_median"), _p(1, "delete_rows")]


def test_a_shortened_id_is_corrected_once(monkeypatch, tmp_path, store):
    issues, candidates = _setup(tmp_path, store)
    first = [_p(0, "impute_group_median"), _p(1, "delete_rows")]
    model, result = _run(monkeypatch, issues, candidates, first, _valid())
    assert len(model.calls) == 2 and [p.candidate_id for p in result] == ["impute_median", "delete_rows"]
    assert all(p._retried for p in result)
    error = ("The model chose 'impute_group_median' for issue 0, which is not one of its eligible options: "
             "['delete_rows', 'drop_column', 'impute_group_median_g', 'impute_mean', 'impute_median', "
             "'impute_median_indicator'].")
    system, human, correction = model.calls[1]
    assert model.calls[1][:2] == model.calls[0]
    assert correction == ("human", f"Your previous answer was rejected: {error}. Choose again for every issue, "
                                   "copying each option id exactly as listed.")
    assert "      impute_group_median_g: " in system[1]
    assert "Copy each option id exactly as written, including every part of it." in system[1]


def test_a_valid_first_answer_makes_one_call(monkeypatch, tmp_path, store):
    issues, candidates = _setup(tmp_path, store)
    model, result = _run(monkeypatch, issues, candidates, _valid())
    assert len(model.calls) == 1 and not any(p._retried for p in result)


def test_two_invalid_answers_raise(monkeypatch, tmp_path, store):
    issues, candidates = _setup(tmp_path, store)
    bad = [_p(0, "impute_group_median"), _p(1, "delete_rows")]
    worse = [_p(0, "impute_median"), _p(1, "made_up")]
    with pytest.raises(ValueError) as error:
        _run(monkeypatch, issues, candidates, bad, worse)
    message = str(error.value)
    assert message.endswith(SUFFIX) and "The model chose 'made_up' for issue 1" in message


def test_a_model_error_is_not_retried(monkeypatch, tmp_path, store):
    issues, candidates = _setup(tmp_path, store)
    model = Script(RuntimeError("service unavailable"), _valid())
    monkeypatch.setattr(classifier_module, "init_chat_model", lambda *a, **k: model)
    with pytest.raises(RuntimeError, match="service unavailable"):
        propose_wrangling_operations(issues, candidates)
    assert len(model.calls) == 1


@pytest.mark.parametrize("first", [
    [_p(0, "impute_median"), _p(5, "delete_rows")],
    [_p(0, "impute_median")],
    [_p(0, "impute_mode"), _p(1, "delete_rows")],
    [_p(0, "made_up"), _p(1, "delete_rows")],
    [_p(0, "leave_as_is"), _p(1, "delete_rows")],
    [_p(0, "impute_median", "impute_median"), _p(1, "delete_rows")],
    [_p(0, "impute_median", "nope"), _p(1, "delete_rows")],
    [_p(0, "impute_median"), _p(0, "delete_rows"), _p(1, "delete_rows")],
], ids=["out_of_range", "missing_issue", "ineligible", "unknown", "leave_as_is", "same_alternative",
        "unknown_alternative", "duplicate"])
def test_every_invalid_choice_is_retried(first, monkeypatch, tmp_path, store):
    issues, candidates = _setup(tmp_path, store)
    model, result = _run(monkeypatch, issues, candidates, first, _valid())
    assert len(model.calls) == 2 and all(p._retried for p in result)


def test_the_retry_holds_no_number_or_value(monkeypatch, tmp_path, store):
    issues, candidates = _setup(tmp_path, store)
    model, _ = _run(monkeypatch, issues, candidates, [_p(0, "impute_group_median"), _p(1, "delete_rows")], _valid())
    correction = model.calls[1][-1][1]
    assert CANARY not in correction
    assert re.findall(r"\d+(?:\.\d+)?", correction) == ["0"]  # only the issue index
    for c in candidates[0] + candidates[1]:
        shift = (c["preview"] or {}).get("shift") or {}
        for value in shift.values():
            if isinstance(value, float):
                assert f"{value:.4g}" not in correction


def test_the_schema_has_no_retried_field():
    assert "retried" not in json.dumps(_WranglingProposalsOut.model_json_schema())
    assert "_retried" not in _ProposedOperationOut.model_fields


# --- through the real graph --------------------------------------------------------------------------------------------

class GraphModel:
    """Answers the proposer: the first answer shortens the group id, the second (or the only one) is valid."""

    def __init__(self, fail_first):
        self.fail_first, self.calls = fail_first, 0

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.calls += 1
        candidate = "impute_group_median" if self.fail_first and self.calls == 1 else "impute_group_median_region"
        return _WranglingProposalsOut(proposals=[_p(0, candidate)])


@pytest.mark.parametrize("fail_first", [True, False], ids=["corrected", "first_time"])
def test_the_record_and_the_report(fail_first, fake_model, monkeypatch, tmp_path, store):
    model = GraphModel(fail_first)
    monkeypatch.setattr(classifier_module, "init_chat_model", lambda *a, **k: model)
    df = _frame()
    df.loc[list(range(15)), "tenure"] = np.nan
    app = build_agent_graph(store, classify_fn=confident(QuestionType.TWO_GROUP_COMPARISON),
                            draft_fn=fake_draft(["region"]), propose_fn=propose_wrangling_operations)
    state = {"request_text": "Did West spend more?", "session_id": "s", "dataset": _dataset(df, tmp_path, store, "retry")}
    values, _, _ = _drive(app, state, "retry")
    assert model.calls == (2 if fail_first else 1)
    (fix,) = [r for r in values["decision_log"] if r["decision_point"] == "stage1.clean.fix"]
    assert fix["chosen"] == "impute_group_median_region" and fix["evidence"]["proposal_retried"] is fail_first
    stage1 = next(e for e in values["stage_reports"] if e["stage"] == "stage1")
    changed = next(s["text"] for s in stage1["sections"] if s["heading"] == "Changed")
    label = "Why (the model's reason, after one correction): " if fail_first else "Why (the model's reason): "
    assert changed.startswith(f"1. Fill the missing values of tenure with the median of each region. {label}because.")
