"""
Deterministic synthetic cohort for Layer 1 tests.

Design
------
Three environments (site_A, site_B, site_C) with:
- 2 invariant features: genuinely predictive of label in all sites
- 2 spurious features: correlated with label only via site-specific confounders

This means a model that fits the spurious features will score well in-sample
but the cross-environment representation will show high v-REx variance on
those features, and SCI should be measurably > 0.

All randomness is seeded — the cohort is byte-for-byte reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SEED = 42
N_PER_ENV = 200
N_INVARIANT = 2
N_SPURIOUS = 2


@dataclass
class SyntheticCohort:
    representations: dict[str, np.ndarray]   # env → (n, d) float32
    labels: dict[str, np.ndarray]             # env → (n,)   int8 {0,1}
    feature_names: list[str]
    env_names: list[str]
    invariant_indices: list[int]
    spurious_indices: list[int]
    seed: int


def make_synthetic_cohort(
    n_per_env: int = N_PER_ENV,
    seed: int = SEED,
) -> SyntheticCohort:
    """
    Build a reproducible multi-environment cohort with known invariant/spurious
    structure.

    Environment-specific confounders
    ---------------------------------
    site_A: spurious features positively correlated with label (r ≈ 0.6)
    site_B: spurious features uncorrelated with label
    site_C: spurious features negatively correlated with label (r ≈ -0.4)

    Invariant features have identical signal across all three sites.
    """
    rng = np.random.default_rng(seed)

    env_configs = {
        "site_A": {"spurious_strength": +0.6,  "n": n_per_env},
        "site_B": {"spurious_strength":  0.0,  "n": n_per_env},
        "site_C": {"spurious_strength": -0.4,  "n": n_per_env},
    }

    representations: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}

    for env, cfg in env_configs.items():
        n = cfg["n"]
        s = cfg["spurious_strength"]

        # Invariant features: same generative process in every environment
        inv_signal = rng.standard_normal((n, N_INVARIANT)).astype(np.float32)

        # Label: logistic function of invariant features only
        log_odds = inv_signal @ np.array([1.5, -1.0], dtype=np.float32)
        prob = 1.0 / (1.0 + np.exp(-log_odds))
        y = rng.binomial(1, prob).astype(np.int8)

        # Spurious features: correlated with label via site-specific strength
        noise = rng.standard_normal((n, N_SPURIOUS)).astype(np.float32)
        spur = s * y[:, None] + noise

        X = np.concatenate([inv_signal, spur], axis=1)
        representations[env] = X
        labels[env] = y

    feature_names = (
        [f"inv_{i}" for i in range(N_INVARIANT)]
        + [f"spur_{i}" for i in range(N_SPURIOUS)]
    )

    return SyntheticCohort(
        representations=representations,
        labels=labels,
        feature_names=feature_names,
        env_names=list(env_configs.keys()),
        invariant_indices=list(range(N_INVARIANT)),
        spurious_indices=list(range(N_INVARIANT, N_INVARIANT + N_SPURIOUS)),
        seed=seed,
    )
