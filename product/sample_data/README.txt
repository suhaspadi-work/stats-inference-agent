Sample datasets for testing the Stats Inference Agent product.
Every file is synthetic (fixed random seeds). The "planted truth" says what the data were built to contain,
so you can check that the app finds it. Numbers are approximate; the p-values are tiny where the difference is real.

01_spend_clean.csv            240 rows. East/West, tier, tenure, visits, total_spend, customer_id, customer_name.
                              Truth: West spends about 8 more than East (means ~66.5 vs ~75.2). Normal, similar spread.
                              Expect: Student's t-test. Boundary gate proposes: region + tier, tenure, visits.
                              Left out (identifier-like): customer_id, customer_name (shown as "values are not shown").
                              Driver question: ols; tenure ~0.6, visits ~1.2, West ~+8.
02_spend_messy.csv            206 rows = 200 + 6 exact duplicate rows; tenure missing in 25 rows (12.1%).
                              Expect: cleaning screen with 2 proposals (remove duplicates, keep rows with a tenure value).
03a_spend_three_regions_anova.csv   240 rows, East/North/West (means ~66.5 / 71.8 / 80.1), normal, equal spread.
                              Expect: one-way ANOVA, then Tukey's HSD; all 3 pairs significant.
03b_spend_three_regions_rank_test.csv  240 rows, East/North/West with large differences (means ~67.6 / 77.0 / 84.9),
                              a group fails the normality check. Expect: Kruskal-Wallis, then Dunn's test; 3 pairs significant.
04_spend_three_regions_null.csv     240 rows, three regions with NO real difference (means ~68.5 / 67.3 / 66.5).
                              Expect: one-way ANOVA, p ~0.27, no pairwise follow-up ("No pairwise follow-up ran").
05_spend_skewed.csv           160 rows, East/West, outcome skewed (exponential). West ~+9 higher on average.
                              Expect: Mann-Whitney U (rank-based).
06_spend_unequal_variance.csv 200 rows, East/West, normal but West is far more spread out than East.
                              Expect: Welch's t-test.
07_text_outcome.csv           120 rows; the outcome "satisfaction" is text (low/medium/high), not a number.
                              Expect: a clear error message, not a crash.
08_spend_heteroskedastic.csv  300 rows; the spread of total_spend grows with tenure.
                              Driver question. Expect: linear regression with robust (HC3) standard errors.
09_tiny_sample.csv            10 rows (5 per region). Expect: a result with very small groups (n = 5 each).
10_two_columns_only.csv       80 rows with only region and total_spend. Expect: NO boundary gate (no extra predictor).
