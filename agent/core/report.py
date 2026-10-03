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
    Generates a plain-language report for a multi-group comparison --
    reuses the same causal-language guard as generate_report (a real
    inferential claim is still being made here), but the content differs:
    an omnibus p-value plus, when significant, which SPECIFIC group pairs
    the post-hoc test found to differ.
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

    group_summary = "\n".join(
        f"- {label}: n={test_result['group_ns'][label]}, mean={test_result['group_means'][label]:.2f}"
        for label in test_result["group_ns"]
    )

    post_hoc_summary = ""
    if test_result["pairwise_comparisons"]:
        sig_pairs = [c for c in test_result["pairwise_comparisons"] if c["significant"]]
        non_sig_pairs = [c for c in test_result["pairwise_comparisons"] if not c["significant"]]
        post_hoc_summary = (
            f"\n\nPost-hoc test ({test_result['post_hoc_method']}) results:\n"
            f"Significantly different pairs: "
            + (", ".join(f"{c['group_a']} vs {c['group_b']} (p={c['p_value']:.4f})" for c in sig_pairs) if sig_pairs else "none")
            + "\nNot significantly different pairs: "
            + (", ".join(f"{c['group_a']} vs {c['group_b']}" for c in non_sig_pairs) if non_sig_pairs else "none")
        )

    system_prompt = (
        "You write brief, plain-language summaries of a multi-group statistical "
        "comparison for a non-technical audience. Report the actual numbers given -- "
        "never invent or round away meaningful precision. If the omnibus test was "
        "significant and post-hoc results are given, explicitly state WHICH group "
        "pairs differ and which don't -- this is the key finding, not just the "
        "overall p-value. Keep it to 5-7 sentences.\n\n"
        f"{language_constraint}"
        f"{unaddressed_constraint}"
    )

    human_prompt = (
        f"Comparing {plan.outcome_name} across groups:\n{group_summary}\n\n"
        f"Omnibus test: {test_result['method']}, statistic={test_result['statistic']:.4f}, "
        f"p-value={test_result['p_value']:.6f}"
        f"{post_hoc_summary}"
    )

    response = model.invoke([("system", system_prompt), ("human", human_prompt)])
    return response.content