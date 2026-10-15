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
