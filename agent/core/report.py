from __future__ import annotations
import re
from langchain.chat_models import init_chat_model
from core.report_facts import two_group_facts
from core.report_facts import multi_group_facts, regression_facts

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
    and the labels, direction, and differences come from code too
    (report_facts), so the model restates facts instead of inferring them.
    The output is checked by check_for_causal_language before being trusted.
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
        "non-technical audience. You are given facts computed by code. Report the actual "
        "numbers given -- never invent or round away meaningful precision. State which "
        "group is higher and the direction of every difference exactly as the facts give "
        "it, refer to groups by the names in the facts, and never recompute or reverse a "
        "comparison. Keep it to 4-6 sentences if there is an unaddressed data-quality "
        "caveat to mention, otherwise 3-5 sentences.\n\n"
        f"{language_constraint}"
        f"{unaddressed_constraint}"
    )

    predictor = plan.candidate_predictors[0] if plan.candidate_predictors else "N/A"
    human_prompt = (
        f"Original question: {plan.request_text}\n\n"
        f"Facts computed from the data:\n{two_group_facts(test_result, plan.outcome_name, predictor)}"
    )

    response = model.invoke([("system", system_prompt), ("human", human_prompt)])
    report_text = response.content

    check_for_causal_language(report_text, plan)
    return report_text

def generate_descriptive_report(plan: AnalysisPlan, descriptive_result: dict, model_name: str = "openai/gpt-oss-120b") -> str:
    """
    Generates a plain-language EDA summary -- structurally different from
    generate_report: no p-value, no causal-language guard needed (a
    description makes no inferential claim to guard against), and the
    content is a dataset overview rather than a single test's conclusion.
    """
    model = init_chat_model(model_name, model_provider="groq")

    column_summaries = []
    for col in descriptive_result["column_stats"]:
        if col["is_numeric"]:
            column_summaries.append(
                f"- {col['name']} (numeric): mean={col['mean']:.2f}, median={col['median']:.2f}, "
                f"std={col['std']:.2f}, range=[{col['min']:.2f}, {col['max']:.2f}], skewness={col['skewness']:.2f}"
            )
        else:
            top = ", ".join(f"{v} ({c})" for v, c in col["top_value_counts"][:3])
            column_summaries.append(f"- {col['name']} (categorical): most common values: {top}")

    correlation_text = ""
    if descriptive_result["correlation_matrix"]:
        cm = descriptive_result["correlation_matrix"]
        pairs = []
        for i, col_a in enumerate(cm["columns"]):
            for j, col_b in enumerate(cm["columns"]):
                if i < j:
                    pairs.append(f"{col_a}-{col_b}: {cm['matrix'][i][j]:.2f}")
        correlation_text = f"\n\nCorrelations between numeric columns: {', '.join(pairs)}"

    system_prompt = (
        "You write brief, plain-language summaries describing a dataset for a "
        "non-technical audience. This is pure description -- there is no hypothesis "
        "being tested and no causal or associational claim being made, so do not use "
        "language implying a statistical test or comparison was run. Simply describe "
        "what is in the data: typical values, spread, notable patterns, and any strong "
        "correlations worth a reader's attention. Report the actual numbers given -- "
        "never invent or round away meaningful precision. Keep it to 4-6 sentences."
    )

    human_prompt = (
        f"Dataset has {descriptive_result['row_count']} rows.\n\n"
        f"Columns:\n" + "\n".join(column_summaries) + correlation_text
    )

    response = model.invoke([("system", system_prompt), ("human", human_prompt)])
    return response.content

def generate_multi_group_report(plan: AnalysisPlan, test_result: dict, model_name: str = "openai/gpt-oss-120b") -> str:
    """
    Plain-language report for a multi-group comparison. Groups, their order, and
    the direction and size of every listed difference come from code
    (report_facts.multi_group_facts); the model only restates them.
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
            f"by explicit user decision: {issues_text}. You MUST mention this plainly."
        )

    system_prompt = (
        "You write brief, plain-language summaries of a multi-group statistical comparison for a "
        "non-technical audience. You are given facts computed by code. Report the actual numbers "
        "given and never invent any. State which groups are higher and which pairs differ exactly as "
        "the facts give them, refer to groups by the names in the facts, and never recompute or "
        "reverse a comparison. If the facts say some pairs are not listed, say that other pairs are "
        "not described here instead of implying the list is complete. Keep it to 5-7 sentences.\n\n"
        f"{language_constraint}{unaddressed_constraint}"
    )

    predictor = plan.candidate_predictors[0] if plan.candidate_predictors else "N/A"
    human_prompt = (
        f"Original question: {plan.request_text}\n\n"
        f"Facts computed from the data:\n{multi_group_facts(test_result, plan.outcome_name, predictor)}"
    )
    response = model.invoke([("system", system_prompt), ("human", human_prompt)])
    return response.content


def generate_regression_report(plan: AnalysisPlan, test_result: dict, model_name: str = "openai/gpt-oss-120b") -> str:
    """
    Plain-language regression report. The direction and size of every association,
    and what each categorical coefficient is compared against, come from code
    (report_facts.regression_facts); the model only restates them.
    """
    model = init_chat_model(model_name, model_provider="groq")

    language_constraint = (
        "IMPORTANT: This data is observational, not from a randomized experiment. "
        "You MUST describe every result as an association, not a cause. Never use words "
        "like 'causes', 'drives', 'leads to', 'results in', 'increases' or 'effect of'. Use "
        "phrasing like 'is associated with' instead."
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
            f"by explicit user decision: {issues_text}. You MUST mention this plainly."
        )

    system_prompt = (
        "You write brief, plain-language summaries of a multiple linear regression for a non-technical "
        "audience. You are given facts computed by code. Report the actual numbers given and never "
        "invent any. State each association's direction and size exactly as the facts give it, say what "
        "each categorical coefficient is compared against, and never recompute or reverse a comparison. "
        "Mention every caution listed. If the facts say some terms are not listed, say that other terms "
        "are not described here. Keep it to 6-9 sentences.\n\n"
        f"{language_constraint}{unaddressed_constraint}"
    )

    human_prompt = (
        f"Original question: {plan.request_text}\n\n"
        f"Facts computed from the data:\n{regression_facts(test_result, plan.outcome_name)}"
    )
    response = model.invoke([("system", system_prompt), ("human", human_prompt)])
    return response.content