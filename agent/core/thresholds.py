"""
The named cut-offs of the method selectors and the post-hoc step (M1.4b step 2a). Pure constants, no imports.
The selectors, the decision records and the method glossary all read them, so the words always match the rule.
These are not the plan's significance level (AnalysisPlan.alpha), which governs the reported test itself.
"""

NORMALITY_ALPHA = 0.05  # Shapiro-Wilk: a group (or the regression residuals) passes the normality check above it
EQUAL_VARIANCE_ALPHA = 0.05  # Levene: the groups have similar variance above it
HETEROSKEDASTICITY_ALPHA = 0.05  # Breusch-Pagan: at or below it, the regression uses HC3-robust standard errors
OMNIBUS_ALPHA = 0.05  # multi-group: the post-hoc test runs only when the omnibus p-value is below it
POST_HOC_ALPHA = 0.05  # post-hoc: a pair is flagged significant when its adjusted p-value is below it

# --- Stage 1 cleaning candidates (M2.3a): when an imputation or a group-wise fill is offered -------------------------
IMPUTE_MAX_MISSING_PCT = 0.40  # no imputation candidate when more than this share of a column is missing
IMPUTE_MIN_OBSERVED = 10       # an imputation needs at least this many observed values
DISCRETE_MAX_DISTINCT = 20     # mode imputation of a numeric column only at or below this many distinct values
GROUP_MAX_LEVELS = 12          # a group column for a group-wise fill has 2 to this many levels
GROUP_MIN_OBSERVED = 5         # observed values of the target column needed in every group
MAX_GROUP_COLUMNS = 3          # group-wise candidates are offered for at most this many group columns
NUMERIC_TEXT_PARSE_RATE = 0.95  # the share of non-missing values that must parse as numbers before a text column is flagged
MAX_AGG_COLUMNS = 5            # the numeric columns of a carried table that an aggregate-then-join summarizes

# --- the Stage 1 loop (M2.4a) -----------------------------------------------------------------------------------------
MAX_STAGE1_ROUNDS = 3  # the most cleaning rounds one question runs; a round only starts when the previous one created new issues
