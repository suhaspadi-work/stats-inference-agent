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