from __future__ import annotations
import re
from langchain.chat_models import init_chat_model
from core.report_facts import two_group_facts
from core.report_facts import multi_group_facts, regression_facts

from core.ingestion import HIGH_CARDINALITY_THRESHOLD
from core.plan import PLAIN_CAVEAT_TYPES, AnalysisPlan, unaddressed_text

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

    def __init__(self, matched: str, sentence: str):
        self.matched, self.sentence = matched, sentence
        super().__init__(
            f"Report uses causal language ('{matched}') on data where allowed_claims is 'associations only'. "
            f"Rewrite using association language (e.g. 'is associated with') instead. Sentence: \"{sentence}\""
        )


# M1.4a (G52): whole-word negation markers (so 'another' no longer counts as 'not'), plus never, without,
# neither and nor; a word ending in n't counts too.
_NEGATION_RE = re.compile(
    r"(?<!\w)(?:not|no|never|cannot|without|neither|nor|no evidence|rather than|as opposed to|instead of)(?!\w)"
    r"|\b\w+n't\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_NEGATION_WINDOW = 40


def find_causal_violation(report_text: str):
    """
    The first causal word that is not disclaimed, as {"matched", "sentence"}, or None. A causal word is
    excused when a negation marker appears in the up to 40 characters before it in the SAME sentence.
    """
    for sentence in _SENTENCE_SPLIT_RE.split(report_text or ""):
        # Curly apostrophes count as straight ones for the negation markers ("doesn’t"); one character for
        # one, so the positions match. The returned sentence and matched text stay the original.
        straight = sentence.replace("\u2019", "'").replace("\u2018", "'")
        for match in _CAUSAL_RE.finditer(sentence):
            before = straight[max(0, match.start() - _NEGATION_WINDOW):match.start()]
            if _NEGATION_RE.search(before):
                continue  # correctly disclaiming causation, not asserting it
            return {"matched": match.group(), "sentence": sentence.strip()}
    return None


def violation_feedback(violation) -> str:
    """The correction appended to the prompt for the one retry (violation is the exception or its dict)."""
    matched = getattr(violation, "matched", None) or violation["matched"]
    sentence = getattr(violation, "sentence", None) or violation["sentence"]
    return (f"Correction needed: your previous text used causal language ('{matched}') in this sentence: "
            f"\"{sentence}\". Rewrite the whole summary using association language only, such as 'is associated "
            "with'. Do not use the words cause, drive, lead to or result in.")


_LONG_TAIL = (" You MUST mention this plainly in your "
              "summary and note that the results should be interpreted with this caveat in mind. "
              "Do not omit this or bury it -- a reader should not come away with more confidence "
              "in the result than is warranted given this known, unaddressed issue.")
_SHORT_TAIL = " You MUST mention this plainly."


def _unaddressed_constraint(plan, tail: str) -> str:
    """
    The prompt text for the plan's unaddressed issues. Rejected fixes keep the wording they always had (a user
    decision); a plain caveat (M2.3a: an imputed outcome) gets its own sentence that does not claim a user decision.
    Without plain caveats the text is exactly what it was before M2.3a.
    """
    rejected = [i for i in plan.unaddressed_issues if i.issue_type not in PLAIN_CAVEAT_TYPES]
    plain = [i for i in plan.unaddressed_issues if i.issue_type in PLAIN_CAVEAT_TYPES]
    text = ""
    if rejected:
        issues_text = "; ".join(unaddressed_text(i, "was identified but left unaddressed") for i in rejected)
        text = ("\n\nIMPORTANT: The following data-quality issue(s) were identified but NOT fixed, "
                f"by explicit user decision: {issues_text}." + tail)
    if plain:
        caveats = "; ".join(i.description.rstrip(".") for i in plain)
        text += ("\n\nAlso state this caveat plainly, in your own words, and do not describe it as something the user "
                 f"chose not to fix: {caveats}.")
    return text


def check_for_causal_language(report_text: str, plan: AnalysisPlan) -> None:
    """
    Independent guard on the FINAL TEXT, separate from the plan-level
    allowed_claims check. The plan's freeze() enforces the rule structurally
    on the plan object; this enforces it on what the model actually wrote,
    since free text generation is a second place the rule could be violated
    even when the plan itself is correctly configured.

    A causal-sounding word preceded closely by a negation in the same sentence ("not that it causes",
    "does not cause", "never causes") is EXCLUDED from the violation -- that phrasing correctly disclaims
    causation rather than asserting it (see find_causal_violation). This keyword check is deliberately
    narrow: it catches the common unsafe pattern, not every possible phrasing.
    """
    if plan.allowed_claims != "associations only":
        return
    violation = find_causal_violation(report_text)
    if violation is not None:
        raise CausalLanguageViolation(violation["matched"], violation["sentence"])


def generate_report(plan: AnalysisPlan, test_result: dict, model_name: str = "openai/gpt-oss-120b",
                    feedback: str = "") -> str:
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

    unaddressed_constraint = _unaddressed_constraint(plan, _LONG_TAIL)

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
    if feedback:
        human_prompt += "\n\n" + feedback  # M1.4a: the one retry after a causal-language violation

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
        elif col.get("distinct_count", 0) > HIGH_CARDINALITY_THRESHOLD:
            column_summaries.append(
                f"- {col['name']} (categorical): {col['distinct_count']} distinct values; "
                "individual values are withheld"
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

def generate_multi_group_report(plan: AnalysisPlan, test_result: dict, model_name: str = "openai/gpt-oss-120b",
                                feedback: str = "") -> str:
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

    unaddressed_constraint = _unaddressed_constraint(plan, _SHORT_TAIL)

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
    if feedback:
        human_prompt += "\n\n" + feedback  # M1.4a: the one retry after a causal-language violation
    response = model.invoke([("system", system_prompt), ("human", human_prompt)])
    return response.content


def generate_regression_report(plan: AnalysisPlan, test_result: dict, model_name: str = "openai/gpt-oss-120b",
                               feedback: str = "") -> str:
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

    unaddressed_constraint = _unaddressed_constraint(plan, _SHORT_TAIL)

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
    if feedback:
        human_prompt += "\n\n" + feedback  # M1.4a: the one retry after a causal-language violation
    response = model.invoke([("system", system_prompt), ("human", human_prompt)])
    return response.content