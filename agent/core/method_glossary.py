"""
The method glossary (M1.4b step 2a): one plain-language entry per method the agent can choose, for a reader who
is not a statistician. Pure: no model, no data. Every cut-off in "when_used" is built from the same named constant
the selector uses (core.thresholds, core.regression), so the glossary cannot drift from the rule. Association
language only.

Each entry: name, kind (two_group_test, multi_group_test, post_hoc, regression, encoding), what, answers,
when_used (lines), assumes (lines), alternatives (ids in GLOSSARY), reading.
"""
from __future__ import annotations

import copy

from core.regression import MAX_CATEGORICAL_LEVELS, MIN_LEVEL_COUNT
from core.thresholds import (
    EQUAL_VARIANCE_ALPHA, HETEROSKEDASTICITY_ALPHA, NORMALITY_ALPHA, OMNIBUS_ALPHA, POST_HOC_ALPHA,
)

_N = f"{NORMALITY_ALPHA:g}"
_V = f"{EQUAL_VARIANCE_ALPHA:g}"
_H = f"{HETEROSKEDASTICITY_ALPHA:g}"
_OMNIBUS = f"{OMNIBUS_ALPHA:g}"
_POST_HOC_P = f"{POST_HOC_ALPHA:g}"

_BELL_SHAPED = "Each group's values are roughly bell-shaped, or the groups are large."
_INDEPENDENT = "Rows are independent of each other."
_RANKABLE = "Values can be ranked from lowest to highest."
_THREE_GROUPS = "There are at least three groups."
_STRAIGHT_LINE = "The relationship between each numeric predictor and the outcome is roughly a straight line."
_LEVEL_ROWS = f"Each category has enough rows ({MIN_LEVEL_COUNT} or more) to be estimated reliably."
_LEVEL_LIMIT = f"A text predictor has at most {MAX_CATEGORICAL_LEVELS} categories."

_T_READING = ("A small p-value (below the significance level in your plan) means the two averages differ by more than "
              "chance would typically produce. The confidence interval shows the plausible size of the difference.")
_TWO_AVERAGES = "Do the two group averages differ by more than chance would typically produce?"
_REGRESSION_ANSWERS = "Which predictors are associated with the outcome, and by how much?"
_REGRESSION_READING = ("Each coefficient is the expected difference in the outcome for a one-unit step in that predictor "
                       "(or against the reference category), with the others held fixed. It describes an association "
                       "in these data, not an effect.")

GLOSSARY: dict[str, dict] = {
    "student_t_test": {
        "name": "Student's t-test", "kind": "two_group_test",
        "what": "Compares the average of a numeric outcome between two groups, assuming both groups vary about equally.",
        "answers": _TWO_AVERAGES,
        "when_used": [f"Both groups pass a normality check (Shapiro-Wilk p > {_N}).",
                      f"The groups have similar variance (Levene's p > {_V})."],
        "assumes": [_BELL_SHAPED, "The two groups vary about equally.", _INDEPENDENT],
        "alternatives": ["welch_t_test", "mann_whitney_u"],
        "reading": _T_READING,
    },
    "welch_t_test": {
        "name": "Welch's t-test", "kind": "two_group_test",
        "what": "Compares two group averages without assuming the groups vary equally.",
        "answers": _TWO_AVERAGES,
        "when_used": [f"Both groups pass a normality check (Shapiro-Wilk p > {_N}).",
                      f"The groups do not have similar variance (Levene's p <= {_V})."],
        "assumes": [_BELL_SHAPED, _INDEPENDENT],
        "alternatives": ["student_t_test", "mann_whitney_u"],
        "reading": _T_READING,
    },
    "mann_whitney_u": {
        "name": "Mann-Whitney U test", "kind": "two_group_test",
        "what": "A rank-based comparison of two groups that does not need bell-shaped data.",
        "answers": "Do values in one group tend to be higher than in the other?",
        "when_used": [f"At least one group fails a normality check (Shapiro-Wilk p <= {_N})."],
        "assumes": [_INDEPENDENT, _RANKABLE],
        "alternatives": ["student_t_test", "welch_t_test"],
        "reading": "A small p-value means values in one group tend to rank higher than values in the other.",
    },
    "one_way_anova": {
        "name": "One-way ANOVA", "kind": "multi_group_test",
        "what": "Compares the averages of three or more groups at once.",
        "answers": "Does at least one group's average differ from the others?",
        "when_used": [f"Every group passes a normality check (Shapiro-Wilk p > {_N}).",
                      f"The groups have similar variance (Levene's p > {_V})."],
        "assumes": [_BELL_SHAPED, "The groups vary about equally.", _INDEPENDENT, _THREE_GROUPS],
        "alternatives": ["kruskal_wallis", "tukey_hsd"],
        "reading": ("A small p-value says at least one average differs, but not which one. The pairwise follow-up "
                    f"(Tukey) says which, and it runs when the overall p-value is below {_OMNIBUS}."),
    },
    "kruskal_wallis": {
        "name": "Kruskal-Wallis test", "kind": "multi_group_test",
        "what": "A rank-based comparison of three or more groups that does not need bell-shaped data.",
        "answers": "Do values in at least one group tend to be higher or lower than in the others?",
        "when_used": [f"Some group fails a normality check (Shapiro-Wilk p <= {_N}), or the groups do not have "
                      f"similar variance (Levene's p <= {_V})."],
        "assumes": [_INDEPENDENT, _RANKABLE, _THREE_GROUPS],
        "alternatives": ["one_way_anova", "dunn"],
        "reading": ("A small p-value says values in at least one group tend to rank higher or lower than in the others, "
                    "but not which one. The pairwise follow-up (Dunn's test) says which, and it runs when the overall "
                    f"p-value is below {_OMNIBUS}."),
    },
    "tukey_hsd": {
        "name": "Tukey's HSD", "kind": "post_hoc",
        "what": ("Compares every pair of group averages after a significant ANOVA, adjusted so that making many "
                 "comparisons does not inflate false alarms."),
        "answers": "Which pairs of groups have different averages?",
        "when_used": [f"Runs after a significant one-way ANOVA (p < {_OMNIBUS})."],
        "assumes": ["The same as the one-way ANOVA it follows.", _INDEPENDENT],
        "alternatives": ["dunn"],
        "reading": f"A pair is flagged when its adjusted p-value is below {_POST_HOC_P}.",
    },
    "dunn": {
        "name": "Dunn's test", "kind": "post_hoc",
        "what": ("Compares every pair of groups by rank after a significant Kruskal-Wallis test, with Holm's "
                 "adjustment so that making many comparisons does not inflate false alarms."),
        "answers": "Which pairs of groups tend to have higher or lower values than each other?",
        "when_used": [f"Runs after a significant Kruskal-Wallis test (p < {_OMNIBUS})."],
        "assumes": ["The same as the Kruskal-Wallis test it follows.", _INDEPENDENT],
        "alternatives": ["tukey_hsd"],
        "reading": f"A pair is flagged when its Holm-adjusted p-value is below {_POST_HOC_P}.",
    },
    "ols": {
        "name": "Linear regression (OLS)", "kind": "regression",
        "what": ("Multiple linear regression with classical standard errors: estimates how the outcome varies with "
                 "each predictor while the other predictors are held fixed."),
        "answers": _REGRESSION_ANSWERS,
        "when_used": [f"The Breusch-Pagan check finds no sign that the spread of the residuals changes (p > {_H})."],
        "assumes": [_STRAIGHT_LINE, _INDEPENDENT, "The spread of the residuals is roughly constant."],
        "alternatives": ["ols_hc3_robust"],
        "reading": _REGRESSION_READING,
    },
    "ols_hc3_robust": {
        "name": "Linear regression with robust (HC3) standard errors", "kind": "regression",
        "what": "The same regression with standard errors that stay valid when the spread of the residuals is not constant.",
        "answers": _REGRESSION_ANSWERS,
        "when_used": [f"The Breusch-Pagan check finds the spread of the residuals changes (p <= {_H})."],
        "assumes": [_STRAIGHT_LINE, _INDEPENDENT],
        "alternatives": ["ols"],
        "reading": _REGRESSION_READING,
    },
    "treatment": {
        "name": "Treatment coding", "kind": "encoding",
        "what": ("Each category of a text predictor is compared against a reference category (by default the largest "
                 "group)."),
        "answers": "How does each category differ from the reference?",
        "when_used": ["The default for text predictors."],
        "assumes": [_LEVEL_ROWS, _LEVEL_LIMIT],
        "alternatives": ["effect", "ordinal"],
        "reading": "A coefficient is the difference from the reference category.",
    },
    "effect": {
        "name": "Effect coding", "kind": "encoding",
        "what": ("Each category's coefficient is its difference from the average of all categories (the coefficients "
                 "add up to zero)."),
        "answers": "How does each category differ from the average of all categories?",
        "when_used": ["Used only when requested as an override; the agent's default is treatment coding."],
        "assumes": [_LEVEL_ROWS, _LEVEL_LIMIT],
        "alternatives": ["treatment", "ordinal"],
        "reading": "A coefficient is the difference from the average of all categories.",
    },
    "ordinal": {
        "name": "Ordinal coding", "kind": "encoding",
        "what": "Categories with a natural order get one slope per step instead of one coefficient each.",
        "answers": "How does the outcome change with each step up the order?",
        "when_used": ["Used only when you supply the order of the categories."],
        "assumes": ["The steps are evenly spaced in how they relate to the outcome."],
        "alternatives": ["treatment", "effect"],
        "reading": "The coefficient is the expected change in the outcome per step up the order.",
    },
}

def glossary_entry(method_id: str) -> dict:
    """A deep copy of the entry; KeyError naming the id for an unknown one."""
    if method_id not in GLOSSARY:
        raise KeyError(f"No glossary entry for method '{method_id}'.")
    return copy.deepcopy(GLOSSARY[method_id])


def method_ids() -> list[str]:
    return sorted(GLOSSARY)


def render_glossary(method_id: str) -> dict:
    """{"title", "sections": [{"heading", "text"}]}; "Other options" is omitted when there are none."""
    entry = glossary_entry(method_id)
    sections = [
        {"heading": "What it is", "text": entry["what"]},
        {"heading": "What it answers", "text": entry["answers"]},
        {"heading": "When the agent uses it", "text": "\n".join(entry["when_used"])},
        {"heading": "What it assumes", "text": "\n".join(entry["assumes"])},
        {"heading": "How to read the result", "text": entry["reading"]},
    ]
    if entry["alternatives"]:
        sections.append({"heading": "Other options",
                         "text": ", ".join(GLOSSARY[a]["name"] for a in entry["alternatives"])})
    return {"title": entry["name"], "sections": sections}
