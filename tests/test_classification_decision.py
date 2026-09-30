from core.plan import (
    AnalysisPlan, ClassificationDecision, AlternativeConsidered,
    QuestionType, CausalStatus, OutcomeType,
)


def test_plan_built_from_classification_carries_the_decision():
    classification = ClassificationDecision(
        question_type=QuestionType.TWO_GROUP_COMPARISON,
        causal_status=CausalStatus.OBSERVATIONAL,
        rationale="Request compares exactly two named groups (East vs West) on a "
                  "continuous outcome, with no mention of random assignment.",
        alternatives=(
            AlternativeConsidered(
                method="association",
                rejected_because="the request names two specific groups to compare, "
                                  "not a general correlation question",
            ),
        ),
        confidence=0.93,
        evidence={"groups_mentioned": 2, "randomization_language_present": False},
    )

    plan = AnalysisPlan.from_classification(
        request_text="Did customers in the West region spend more than the East region?",
        session_id="session-abc",
        classification=classification,
        outcome_name="total_spend",
        outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="customer",
        data_sources=("customers@v2",),
    )

    assert plan.question_type == QuestionType.TWO_GROUP_COMPARISON
    assert plan.causal_status == CausalStatus.OBSERVATIONAL
    assert plan.classification_decision is classification
    assert plan.classification_decision.confidence == 0.93
    assert len(plan.classification_decision.alternatives) == 1


def test_low_confidence_classification_is_a_real_distinguishable_state():
    """
    This test doesn't enforce the interrupt behavior yet (that's the agent
    loop's job, built next) -- it just confirms the schema can represent a
    genuinely low-confidence classification distinctly from a confident one,
    which is the prerequisite for the agent loop to act on it.
    """
    ambiguous = ClassificationDecision(
        question_type=QuestionType.ASSOCIATION,
        causal_status=CausalStatus.OBSERVATIONAL,
        rationale="Could plausibly be read as either a general association question "
                  "or a driver analysis; the request doesn't clearly specify which.",
        alternatives=(
            AlternativeConsidered(method="driver_analysis", rejected_because="tied, not clearly rejected"),
        ),
        confidence=0.42,
        evidence={},
    )
    assert ambiguous.confidence < 0.5