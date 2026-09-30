from core.plan import (
    AnalysisPlan, QuestionType, CausalStatus, OutcomeType, PlanStatus,
    MethodDecision, AlternativeConsidered,
)

plan = AnalysisPlan(
    request_text="Did customers in the West region spend more than the East region?",
    session_id="session-abc",
    question_type=QuestionType.TWO_GROUP_COMPARISON,
    causal_status=CausalStatus.OBSERVATIONAL,
    outcome_name="total_spend",
    outcome_type=OutcomeType.CONTINUOUS,
    unit_of_analysis="customer",
    data_sources=("customers@v2",),
)
print("status:", plan.status)

# Try to freeze before any method decision is recorded -- should fail
try:
    plan.freeze()
except ValueError as e:
    print("correctly blocked:", e)

# First method decision, based on a preview-sample assumption check
first_decision = MethodDecision(
    stage="test_selection",
    chosen_method="welch_t_test",
    rationale="Two independent groups, continuous outcome, roughly equal variances observed",
    alternatives=(
        AlternativeConsidered(
            method="mann_whitney_u",
            rejected_because="parametric test is more powerful when normality holds",
        ),
    ),
    evidence={"shapiro_p_group_a": 0.34, "shapiro_p_group_b": 0.29, "levene_p": 0.61},
)
refined = plan.refine(candidate_predictors=("region",)).record_method_decision(first_decision)
print("status after refine:", refined.status)
print("primary_method:", refined.primary_method)

frozen = refined.freeze()
print("status after freeze:", frozen.status)

# Try to refine after freezing -- should fail
try:
    frozen.refine(alpha=0.10)
except ValueError as e:
    print("correctly blocked:", e)

# Simulate the assumption check failing once run on the FULL dataset, not the preview sample
second_decision = MethodDecision(
    stage="test_selection",
    chosen_method="mann_whitney_u",
    rationale="Re-running the normality check on the full dataset showed non-normal residuals "
              "in group B; Mann-Whitney does not require this assumption",
    alternatives=(
        AlternativeConsidered(method="welch_t_test", rejected_because="normality assumption violated per re-check"),
        AlternativeConsidered(method="bootstrap_test", rejected_because="sample size sufficient for a standard rank test"),
    ),
    evidence={"shapiro_p_group_b_full_data": 0.02},
    supersedes=first_decision.id,
)
deviated = frozen.record_deviation(
    description="Switched from Welch's t-test to Mann-Whitney U",
    reason="Assumption check on the full dataset invalidated the original choice",
).record_method_decision(second_decision)

print("deviations logged:", len(deviated.deviations))
print("current primary_method:", deviated.primary_method)
print("current decision supersedes:", deviated.latest_decision.supersedes == first_decision.id)
print("total method decisions recorded:", len(deviated.method_decisions))

executed = deviated.mark_executed()
print("final status:", executed.status)

# Confirm an observational plan still can't be frozen with causal claims allowed
bad_plan = plan.refine(allowed_claims="causal").record_method_decision(first_decision)
try:
    bad_plan.freeze()
except ValueError as e:
    print("correctly blocked causal claim on observational data:", e)