import pytest
from core.classifier import propose_wrangling_operations
from core.detection import CandidateIssue


def test_empty_issues_returns_empty_proposals():
    assert propose_wrangling_operations([]) == []


@pytest.mark.live
def test_live_proposes_dedupe_for_duplicate_rows():
    issue = CandidateIssue(
        issue_type="duplicate_rows",
        description="3 exact duplicate rows detected across all columns.",
        affected_columns=("customer_id", "region", "spend"),
        evidence={"duplicate_row_count": 3, "total_rows": 100},
    )
    proposals = propose_wrangling_operations([issue])

    assert len(proposals) == 1
    assert proposals[0].op_type == "dedupe"
    assert proposals[0].issue_index == 0
    assert len(proposals[0].rationale) > 0
    assert len(proposals[0].alternative_approach) > 0


@pytest.mark.live
def test_live_proposes_filter_for_high_missingness():
    issue = CandidateIssue(
        issue_type="high_missingness",
        description="Column 'region' is 30.0% missing.",
        affected_columns=("region",),
        evidence={"missing_pct": 0.3, "missing_count": 30},
    )
    proposals = propose_wrangling_operations([issue])

    assert len(proposals) == 1
    assert proposals[0].op_type == "filter"
    assert proposals[0].params.get("column") == "region"


@pytest.mark.live
def test_live_handles_multiple_issues_in_one_call():
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
    proposals = propose_wrangling_operations(issues)

    assert len(proposals) == 2
    issue_indices = {p.issue_index for p in proposals}
    assert issue_indices == {0, 1}