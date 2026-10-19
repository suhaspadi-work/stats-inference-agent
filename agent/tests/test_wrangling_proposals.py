import numpy as np
import pandas as pd
import pytest
from core.classifier import propose_wrangling_operations
from core.dataset import LocalDiskStore
from core.detection import CandidateIssue
from core.ingestion import load_dataset
from core.operation_candidates import enumerate_candidates


def test_empty_issues_returns_empty_proposals():
    assert propose_wrangling_operations([], []) == []  # M2.3a: the candidates come with the issues


# M2.3a: the live tests build real candidates on a small dataset; the model chooses a candidate id

def _candidates(tmp_path, issues, df):
    store = LocalDiskStore(base_dir=tmp_path / "store")
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    ds = load_dataset(path, session_id="s", name="data", store=store)
    return [enumerate_candidates(issue, ds, None, store) for issue in issues]


def _eligible(candidates):
    return {c["id"] for c in candidates if c["eligible"] and c["id"] != "leave_as_is"}


def _frame(n=100):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"customer_id": range(n), "region": rng.choice(["East", "West"], n).astype(object),
                       "spend": rng.normal(50, 5, n)})
    return df


@pytest.mark.live
def test_live_proposes_dedupe_for_duplicate_rows(tmp_path):
    issue = CandidateIssue(
        issue_type="duplicate_rows",
        description="3 exact duplicate rows detected across all columns.",
        affected_columns=("customer_id", "region", "spend"),
        evidence={"duplicate_row_count": 3, "total_rows": 100},
    )
    df = _frame(97)
    df = pd.concat([df, df.iloc[[0, 1, 2]]], ignore_index=True)
    candidates = _candidates(tmp_path, [issue], df)
    proposals = propose_wrangling_operations([issue], candidates)

    assert len(proposals) == 1
    assert proposals[0].candidate_id == "dedupe_all"
    assert proposals[0].issue_index == 0
    assert len(proposals[0].rationale) > 0
    assert proposals[0].alternative_candidate_id == "leave_as_is"  # the only other candidate of a duplicate issue
    assert len(proposals[0].alternative_rejected_because) > 0


@pytest.mark.live
def test_live_proposes_a_fix_for_high_missingness(tmp_path):
    issue = CandidateIssue(
        issue_type="high_missingness",
        description="Column 'region' is 30.0% missing.",
        affected_columns=("region",),
        evidence={"missing_pct": 0.3, "missing_count": 30},
    )
    df = _frame()
    df.loc[:29, "region"] = None
    candidates = _candidates(tmp_path, [issue], df)
    proposals = propose_wrangling_operations([issue], candidates)

    assert len(proposals) == 1
    assert proposals[0].candidate_id in _eligible(candidates[0])  # every candidate of this issue acts on region
    assert proposals[0].alternative_candidate_id in {c["id"] for c in candidates[0]} - {proposals[0].candidate_id}


@pytest.mark.live
def test_live_handles_multiple_issues_in_one_call(tmp_path):
    issues = [
        CandidateIssue(
            issue_type="duplicate_rows", description="2 duplicate rows detected.",
            affected_columns=("customer_id",), evidence={"duplicate_row_count": 2, "total_rows": 50},
        ),
        CandidateIssue(
            issue_type="high_missingness", description="Column 'spend' is 15.0% missing.",
            affected_columns=("spend",), evidence={"missing_pct": 0.15, "missing_count": 7},
        ),
    ]
    df = _frame(48)
    df.loc[:6, "spend"] = np.nan
    df = pd.concat([df, df.iloc[[10, 11]]], ignore_index=True)
    candidates = _candidates(tmp_path, issues, df)
    proposals = propose_wrangling_operations(issues, candidates)

    assert len(proposals) == 2
    issue_indices = {p.issue_index for p in proposals}
    assert issue_indices == {0, 1}
    for p in proposals:
        assert p.candidate_id in _eligible(candidates[p.issue_index])
