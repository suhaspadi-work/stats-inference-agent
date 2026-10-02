from __future__ import annotations
import re
from langchain.chat_models import init_chat_model

from core.plan import AnalysisPlan

# A deliberately narrow, high-precision list -- catching obvious causal
# claims, not every possible phrasing. False negatives are more acceptable
# here than false positives that would block a legitimate report.
_CAUSAL_PATTERNS = [
    r"\bcauses?\b", r"\bcaused\b", r"\bcausing\b",
    r"\bdrives?\b", r"\bdriving\b",
    r"\bleads? to\b", r"\bled to\b",
    r"\bresults? in\b", r"\bresulted in\b",
]
_CAUSAL_RE = re.compile("|".join(_CAUSAL_PATTERNS), re.IGNORECASE)


class CausalLanguageViolation(ValueError):
    """Raised when a report uses causal language on data that only supports associations."""


_NEGATION_MARKERS = (
    "not", "n't", "cannot", "can't", "no evidence", "does not", "doesn't", "isn't", "wasn't",
    "rather than", "as opposed to", "instead of",
)


def check_for_causal_language(report_text: str, plan: AnalysisPlan) -> None:
    """
    Independent guard on the FINAL TEXT, separate from the plan-level
    allowed_claims check. The plan's freeze() enforces the rule structurally
    on the plan object; this enforces it on what the model actually wrote,
    since free text generation is a second place the rule could be violated
    even when the plan itself is correctly configured.

    A causal-sounding word preceded closely by a negation ("not that it
    causes", "does not cause") is EXCLUDED from the violation -- that
    phrasing correctly disclaims causation rather than asserting it. This
    keyword check is deliberately narrow: it catches the common unsafe
    pattern, not every possible phrasing, and false negatives here are more
    acceptable than blocking a report that is actually safe and correct.
    """
    if plan.allowed_claims != "associations only":
        return

    for match in _CAUSAL_RE.finditer(report_text):
        window_start = max(0, match.start() - 40)
        preceding_text = report_text[window_start:match.start()].lower()
        if any(marker in preceding_text for marker in _NEGATION_MARKERS):
            continue  # correctly disclaiming causation, not asserting it
        raise CausalLanguageViolation(
            f"Report uses causal language ('{match.group()}') on data where "
            f"allowed_claims is 'associations only'. Rewrite using association "
            f"language (e.g. 'is associated with') instead."
        )


def generate_report(plan: AnalysisPlan, test_result: dict, model_name: str = "openai/gpt-oss-120b") -> str:
    """
    Generate the plain-language report. The model writes the prose, but
    every number in the prompt comes from real computation (test_result),
    and the output is checked by check_for_causal_language before being
    trusted -- the model's job is to explain real numbers, not to decide
    what language is allowed.
    """
    model = init_chat_model(model_name, model_provider="groq")

    language_constraint = (
        "IMPORTANT: This data is observational, not from a randomized experiment. "
        "You MUST describe the result as an association, not a cause. Never use words "
        "like 'causes', 'drives', 'leads to', or 'results in'. Use phrasing like "
        "'is associated with' or 'is correlated with' instead."
        if plan.allowed_claims == "associations only"
        else "This data comes from a randomized experiment, so causal language is appropriate here."
    )

    unaddressed_constraint = ""
    if plan.unaddressed_issues:
        issues_text = "; ".join(
            f"{i.issue_type} ({i.description}) was identified but left unaddressed because: {i.rejection_reason}"
            for i in plan.unaddressed_issues
        )
        unaddressed_constraint = (
            "\n\nIMPORTANT: The following data-quality issue(s) were identified but NOT fixed, "
            f"by explicit user decision: {issues_text}. You MUST mention this plainly in your "
            "summary and note that the results should be interpreted with this caveat in mind. "
            "Do not omit this or bury it -- a reader should not come away with more confidence "
            "in the result than is warranted given this known, unaddressed issue."
        )

    system_prompt = (
        "You write brief, plain-language summaries of statistical analyses for a "
        "non-technical audience. Report the actual numbers given -- never invent or "
        "round away meaningful precision. Keep it to 4-6 sentences if there is an "
        "unaddressed data-quality caveat to mention, otherwise 3-5 sentences.\n\n"
        f"{language_constraint}"
        f"{unaddressed_constraint}"
    )

    human_prompt = (
        f"Original question: {plan.request_text}\n"
        f"Outcome variable: {plan.outcome_name}\n"
        f"Comparing groups on: {plan.candidate_predictors[0] if plan.candidate_predictors else 'N/A'}\n"
        f"Method used: {test_result['method']}\n"
        f"Group A (n={test_result['group_a_n']}): mean = {test_result['group_a_mean']:.2f}\n"
        f"Group B (n={test_result['group_b_n']}): mean = {test_result['group_b_mean']:.2f}\n"
        f"p-value: {test_result['p_value']:.4f}\n"
        f"Effect size: {test_result['effect_size']:.3f}\n"
        f"95% CI on the difference: ({test_result['confidence_interval'][0]:.2f}, "
        f"{test_result['confidence_interval'][1]:.2f})"
    )

    response = model.invoke([("system", system_prompt), ("human", human_prompt)])
    report_text = response.content

    check_for_causal_language(report_text, plan)
    return report_text