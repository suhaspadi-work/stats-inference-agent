"""
M2.3a: the wrangling proposer (core/classifier.py propose_wrangling_operations) with a fake model. The prompt lists,
per issue, only the eligible candidates other than leave_as_is as "id: description" lines (descriptions without
details, so no preview number, fill value, group level or cell value reaches it); the model's choice is validated.
"""
import numpy as np
import pandas as pd
import pytest

import core.classifier as classifier_module
from core.classifier import _ProposedOperationOut, _WranglingProposalsOut, _validate_choices, propose_wrangling_operations
from core.dataset import LocalDiskStore
from core.detection import CandidateIssue
from core.ingestion import load_dataset
from core.operation_candidates import enumerate_candidates

CANARY = "PROPOSE-CANARY"


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


class Model:
    """A fake chat model: records the prompt and answers with the given proposals."""

    def __init__(self, proposals):
        self.proposals, self.prompts = proposals, []

    def with_structured_output(self, schema):
        assert schema is _WranglingProposalsOut
        return self

    def invoke(self, messages):
        self.prompts.append("\n".join(content for _, content in messages))
        return _WranglingProposalsOut(proposals=self.proposals)


def _p(index, candidate, alternative="leave_as_is"):
    return _ProposedOperationOut(issue_index=index, candidate_id=candidate, rationale="because",
                                 alternative_candidate_id=alternative, alternative_rejected_because="it is worse")


def _setup(tmp_path, store):
    n = 100
    x = np.linspace(31415.9, 32718.2, n)
    x[:12] = np.nan
    df = pd.DataFrame({"x": x, "g": [f"{CANARY}-level{i % 2}" for i in range(n)],
                       "t": [f"{CANARY}-{i % 4}" if i % 9 else None for i in range(n)]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    ds = load_dataset(path, session_id="s", name="data", store=store)
    issues = [CandidateIssue(issue_type="high_missingness", description="x is 12% missing", affected_columns=("x",),
                             evidence={"missing_pct": 0.12}),
              CandidateIssue(issue_type="high_missingness", description="t is 11% missing", affected_columns=("t",),
                             evidence={"missing_pct": 0.11})]
    candidates = [enumerate_candidates(issue, ds, None, store) for issue in issues]
    return issues, candidates


def _run(monkeypatch, issues, candidates, proposals):
    model = Model(proposals)
    monkeypatch.setattr(classifier_module, "init_chat_model", lambda *a, **k: model)
    return model, propose_wrangling_operations(issues, candidates)


# --- the prompt ----------------------------------------------------------------------------------------------------------

def test_the_prompt_lists_only_eligible_options(monkeypatch, tmp_path, store):
    issues, candidates = _setup(tmp_path, store)
    model, result = _run(monkeypatch, issues, candidates, [_p(0, "impute_median"), _p(1, "delete_rows")])
    assert [p.candidate_id for p in result] == ["impute_median", "delete_rows"]
    (prompt,) = model.prompts
    blocks = prompt.split("\n[1] ")  # issue 0's options, then issue 1's
    assert len(blocks) == 2
    for block, cands in zip(blocks, candidates):
        for c in cands:
            line = f"      {c['id']}: "
            if c["eligible"] and c["id"] != "leave_as_is":
                assert line in block, c["id"]
            else:
                assert line not in block, c["id"]
    assert "      impute_median: Fill the missing values of x with the median.\n" in prompt
    assert "'leave_as_is' (keep the data as it is) is also a valid alternative" in prompt
    assert "different option id of the same issue" in prompt


def test_no_preview_number_or_cell_value_reaches_the_prompt(monkeypatch, tmp_path, store):
    issues, candidates = _setup(tmp_path, store)
    model, _ = _run(monkeypatch, issues, candidates, [_p(0, "impute_median"), _p(1, "delete_rows")])
    (prompt,) = model.prompts
    assert CANARY not in prompt and "level0" not in prompt
    for cands in candidates:
        for c in cands:
            preview = c["preview"] or {}
            shift = preview.get("shift") or {}
            for number in [shift.get(k) for k in ("mean_before", "mean_after", "std_before", "std_after")]:
                if number is not None:
                    for form in (repr(number), f"{number:.4g}", f"{number:.2f}"):
                        assert form not in prompt, (c["id"], form)
            if preview.get("rows_changed"):
                assert f"removes {preview['rows_changed']} rows" not in prompt
            assert c["what_was_run"] not in prompt or "(" not in c["what_was_run"], c["id"]


def test_assert_model_safe_is_applied(monkeypatch, tmp_path, store):
    issues, candidates = _setup(tmp_path, store)
    seen = []
    real = classifier_module.assert_model_safe
    monkeypatch.setattr(classifier_module, "assert_model_safe", lambda value, name: (seen.append(name), real(value, name)))
    _run(monkeypatch, issues, candidates, [_p(0, "impute_median"), _p(1, "delete_rows")])
    assert seen == ["issues", "options"]


def test_the_schema_has_no_params():
    assert "params" not in _ProposedOperationOut.model_fields and "op_type" not in _ProposedOperationOut.model_fields


# --- validation ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("proposals, message", [
    ([_p(0, "impute_median"), _p(5, "delete_rows")], "for issue index 5, which does not exist"),
    ([_p(0, "impute_median")], r"no choice for issue\(s\) \[1\]"),
    ([_p(0, "impute_mode"), _p(1, "delete_rows")], "'impute_mode' for issue 0, which is not one of its eligible options"),
    ([_p(0, "made_up"), _p(1, "delete_rows")], "'made_up' for issue 0, which is not one of its eligible options"),
    ([_p(0, "leave_as_is"), _p(1, "delete_rows")], "'leave_as_is' for issue 0, which is not one of its eligible options"),
    ([_p(0, "impute_median", "impute_median"), _p(1, "delete_rows")],
     "alternative 'impute_median' for issue 0 must be a different option"),
    ([_p(0, "impute_median", "fill_missing_category_x"), _p(1, "delete_rows")],
     "alternative 'fill_missing_category_x' for issue 0 must be a different option"),
    ([_p(0, "impute_median"), _p(0, "delete_rows"), _p(1, "delete_rows")], "more than once for issue 0"),
], ids=["out_of_range", "missing_issue", "ineligible", "unknown", "leave_as_is", "same_alternative",
        "unknown_alternative", "duplicate"])
def test_invalid_choices(proposals, message, monkeypatch, tmp_path, store):
    issues, candidates = _setup(tmp_path, store)
    assert not next(c for c in candidates[0] if c["id"] == "impute_mode")["eligible"]  # continuous: mode ineligible
    with pytest.raises(ValueError, match=message):
        _run(monkeypatch, issues, candidates, proposals)
    with pytest.raises(ValueError, match=message):
        _validate_choices(proposals, issues, candidates)


def test_a_valid_choice_passes(tmp_path, store):
    issues, candidates = _setup(tmp_path, store)
    _validate_choices([_p(1, "fill_missing_category", "impute_mode"), _p(0, "impute_group_median_g", "drop_column")],
                      issues, candidates)


def test_no_issues_never_calls_the_model(monkeypatch):
    monkeypatch.setattr(classifier_module, "init_chat_model", lambda *a, **k: pytest.fail("the model was called"))
    assert propose_wrangling_operations([], []) == []
    with pytest.raises(ValueError, match="one candidate list per issue"):
        propose_wrangling_operations([CandidateIssue("duplicate_rows", "d", ("a",), {})], [])
