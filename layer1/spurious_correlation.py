"""
Spurious Correlation Index (SCI).

SCI = fraction of predictive variance attributable to features that violate
      invariance across environments.

Range: [0, 1].  SCI = 0 means all predictive signal is invariant.
                SCI = 1 means all predictive signal is spurious.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import Tensor


@dataclass(frozen=True)
class StratumMetricResult:
    value: float | None
    warnings: list[str] = field(default_factory=list)
    assumption_violations: list[str] = field(default_factory=list)


def _fit_and_variance(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 0,
) -> float:
    """Return variance of predicted probabilities from a logistic classifier."""
    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X, y)
    probs = clf.predict_proba(X)[:, 1]
    return float(np.var(probs))


def compute_sci(
    representations: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    *,
    vrex_threshold: float = 0.05,
    random_state: int = 0,
) -> StratumMetricResult:
    """
    Compute the Spurious Correlation Index.

    Strategy
    --------
    1. Pool all environments → fit a full logistic model → get total
       predictive variance V_total.
    2. Identify which feature dimensions are *invariant* via v-REx: a feature
       is invariant if its per-environment mean does not vary significantly
       (surrogate for the IRM identification condition without requiring
       gradient access to the model).
    3. Fit a logistic model on invariant features only → V_invariant.
    4. SCI = 1 - V_invariant / V_total.
       Clipped to [0, 1] to guard against numerical noise.

    Args:
        representations: mapping env_name → np.ndarray shape (n_e, d).
        labels:          mapping env_name → np.ndarray shape (n_e,).
        vrex_threshold:  feature is flagged as spurious if its cross-env mean
                         variance exceeds this fraction of its overall variance.
        random_state:    passed to sklearn for reproducibility.

    Returns:
        StratumMetricResult with .value in [0, 1] or None on failure.
    """
    envs = list(representations.keys())

    # ── Assumption: need ≥ 2 environments ──────────────────────────────────
    if len(envs) < 2:
        return StratumMetricResult(
            value=None,
            warnings=[],
            assumption_violations=[
                "single_environment: invariance untestable"
            ],
        )

    warnings: list[str] = []

    # ── Stack data ──────────────────────────────────────────────────────────
    X_all = np.concatenate([representations[e] for e in envs], axis=0)
    y_all = np.concatenate([labels[e] for e in envs], axis=0)

    if X_all.shape[0] < 20:
        warnings.append("low_sample_count: SCI estimate may be unreliable (n < 20)")

    n_features = X_all.shape[1]

    # ── Identify invariant features via per-environment mean variance ───────
    env_means = np.stack(
        [representations[e].mean(axis=0) for e in envs], axis=0
    )  # shape (E, d)

    overall_var = X_all.var(axis=0)                    # shape (d,)
    cross_env_var = env_means.var(axis=0)              # shape (d,)

    # Avoid division by zero for constant features
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_shift = np.where(
            overall_var > 0,
            cross_env_var / overall_var,
            0.0,
        )

    invariant_mask = relative_shift <= vrex_threshold  # shape (d,)
    n_invariant = int(invariant_mask.sum())

    if n_invariant == 0:
        warnings.append(
            "no_invariant_features: all features flagged as spurious; "
            "SCI=1.0 but treat with caution"
        )
        return StratumMetricResult(
            value=1.0,
            warnings=warnings,
            assumption_violations=[],
        )

    if n_invariant == n_features:
        warnings.append(
            "all_features_invariant: no spurious signal detected; SCI=0.0"
        )

    # ── Predictive variance of full vs. invariant representation ───────────
    v_total = _fit_and_variance(X_all, y_all, random_state=random_state)

    X_invariant = X_all[:, invariant_mask]
    v_invariant = _fit_and_variance(X_invariant, y_all, random_state=random_state)

    if v_total < 1e-9:
        # Model is essentially constant — SCI is undefined
        return StratumMetricResult(
            value=None,
            warnings=warnings,
            assumption_violations=[
                "zero_predictive_variance: model produces constant predictions"
            ],
        )

    sci = float(np.clip(1.0 - v_invariant / v_total, 0.0, 1.0))

    return StratumMetricResult(
        value=sci,
        warnings=warnings,
        assumption_violations=[],
    )
