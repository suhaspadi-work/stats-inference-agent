from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticTwoGroupSpec:
    """
    Describes exactly what ground truth was planted, so an evaluation can
    check the agent's output against a KNOWN answer, not just "did it run."
    """
    n_per_group: int
    true_mean_a: float
    true_mean_b: float
    true_std: float
    true_effect: float  # true_mean_b - true_mean_a, the number a correct analysis should recover
    seed: int


def generate_two_group_continuous(
    n_per_group: int = 200,
    mean_a: float = 50.0,
    effect: float = 0.0,       # 0.0 -> an A/A test; any nonzero -> a planted, known effect
    std: float = 10.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, SyntheticTwoGroupSpec]:
    """
    Generate a two-group dataset with a continuous outcome and a KNOWN true
    effect (which may be exactly zero, for A/A testing). Returns both the
    data and the spec describing the ground truth, so a test can assert
    against `spec.true_effect` rather than trusting anything the agent says.
    """
    rng = np.random.default_rng(seed)
    mean_b = mean_a + effect

    group_a = rng.normal(loc=mean_a, scale=std, size=n_per_group)
    group_b = rng.normal(loc=mean_b, scale=std, size=n_per_group)

    df = pd.DataFrame({
        "customer_id": range(1, 2 * n_per_group + 1),
        "group": ["A"] * n_per_group + ["B"] * n_per_group,
        "outcome": np.concatenate([group_a, group_b]),
    })

    spec = SyntheticTwoGroupSpec(
        n_per_group=n_per_group,
        true_mean_a=mean_a,
        true_mean_b=mean_b,
        true_std=std,
        true_effect=effect,
        seed=seed,
    )
    return df, spec

def generate_two_group_continuous_with_quality_issues(
    n_per_group: int = 200,
    mean_a: float = 50.0,
    effect: float = 8.0,
    std: float = 10.0,
    seed: int = 5,
    n_duplicate_rows: int = 3,
    missingness_pct: float = 0.0,
    missingness_column: str = "group",
) -> tuple[pd.DataFrame, SyntheticTwoGroupSpec]:
    """
    Same ground truth as generate_two_group_continuous, but with
    deliberately injected duplicate rows and/or missing values layered on
    top -- for testing the wrangling/detection pipeline against a KNOWN
    effect that must still be statistically recoverable even with these
    quality issues present. The returned spec's true_effect is unchanged;
    the issues are injected AFTER generation, not baked into the underlying
    distributions, so ground-truth correctness and data-quality testing
    stay cleanly separable.
    """
    df, spec = generate_two_group_continuous(
        n_per_group=n_per_group, mean_a=mean_a, effect=effect, std=std, seed=seed,
    )

    rng = np.random.default_rng(seed + 1)  # separate stream from the data generation itself

    if n_duplicate_rows > 0:
        dup_rows = df.sample(n=n_duplicate_rows, random_state=seed + 1)
        df = pd.concat([df, dup_rows], ignore_index=True)

    if missingness_pct > 0:
        n_missing = int(len(df) * missingness_pct)
        missing_indices = rng.choice(df.index, size=n_missing, replace=False)
        # Cast to plain object dtype first -- newer pandas string dtypes
        # don't always register a None assignment as a missing value under
        # isna(), so this guarantees standard NaN semantics regardless of
        # pandas version.
        df[missingness_column] = df[missingness_column].astype(object)
        df.loc[missing_indices, missingness_column] = None

    return df, spec