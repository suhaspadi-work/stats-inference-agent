"""
The report fallback (M1.4a, G52): when the model's text uses causal language twice, the report is written by
code from the computed facts, so a report can no longer fail the run. Pure: no model, no data access.
"""
from __future__ import annotations

from core.plan import AnalysisPlan, unaddressed_text
from core.report_facts import multi_group_facts, regression_facts, two_group_facts

MINIMAL_FALLBACK_TEXT = ("The narrative could not be written safely for this result. The computed results are recorded "
                         "with the analysis.")


def facts_fallback_report(plan: AnalysisPlan, result: dict, capability_name: str) -> str:
    """A report from the code-computed facts, with the unaddressed-issue caveats and the association note."""
    predictor = plan.candidate_predictors[0] if plan.candidate_predictors else "N/A"
    if capability_name == "two_group":
        facts = two_group_facts(result, plan.outcome_name, predictor)
    elif capability_name == "multi_group":
        facts = multi_group_facts(result, plan.outcome_name, predictor)
    elif capability_name == "regression":
        facts = regression_facts(result, plan.outcome_name)
    else:
        raise ValueError(f"No fact-based fallback for capability '{capability_name}'.")
    lines = ["Summary of the computed results (written from the computed facts, without narrative text):\n" + facts]
    # M2.3a: a plain caveat (an imputed outcome) renders as its description alone
    lines += [f"Caveat: {unaddressed_text(i, 'was identified but left unaddressed').rstrip('.')}. "
              "Interpret the results with this in mind." for i in plan.unaddressed_issues]
    if plan.allowed_claims == "associations only":
        lines.append("These results describe associations, not effects.")
    return "\n".join(lines)
