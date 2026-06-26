"""
stratum/layer3/performativity.py

Layer 3: Sociotechnical and Temporal Validity — Performative Drift

Performative drift is the case the protocol singles out as distinct from
ordinary covariate/concept drift: the model's own deployment changes the
joint distribution it was evaluated on, typically because clinicians adapt
their behavior in response to (or in spite of) the model's outputs.

The defining symptom: a performance metric (AUROC, calibration) can stay
flat or even improve while the model's *causal effect* on the outcome it
was meant to improve reverses sign. A model that is "ignored" by clinicians
who have learned to distrust its false positives can look statistically
identical on AUROC across two periods while its real-world effect on patient
outcomes goes from positive to negative (or to zero, the most common case).

This module does not try to detect performative drift via P(X) or P(Y|X)
shift tests — those are exactly the tests that can stay silent in this
regime. Instead it re-estimates the model's causal effect on the outcome
in each monitoring window and compares the estimates over time.

METRIC_NAME: drift.performative_effect_reversal
METRIC_LAYER: 3
PAPER_REF: sec:temporal-validity
THRESHOLD_CITE: Perdomo et al. (2020) "Performative Prediction"; Pearl (2009) ch.3
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np


class EffectEstimator(str, Enum):
    IPW = "inverse_propensity_weighting"
    AIPW = "augmented_ipw"
    REGRESSION_ADJUSTMENT = "regression_adjustment"


@dataclass
class CausalEffectEstimate:
    """One window's causal effect estimate, with the inputs needed to
    re-derive it (for audit) and the variance needed to compare windows."""
    window_label: str
    estimator: EffectEstimator
    ate: float                  # average treatment effect estimate
    ate_variance: float
    n: int
    propensity_overlap_ok: bool  # positivity diagnostic; see _check_overlap


@dataclass
class PerformativityResult:
    estimate_ref: CausalEffectEstimate
    estimate_new: CausalEffectEstimate
    effect_delta: float
    z_statistic: float
    p_value: float
    sign_reversed: bool
    fired: bool
    caveats: list[str]


# --------------------------------------------------------------------------- #
# Treatment definition for this setting
# --------------------------------------------------------------------------- #
#
# "Treatment" here is "clinician acted in accordance with the model's
# recommendation" (a binary indicator derived from comparing model_output
# to actual_clinical_decision in the shadow-deployment log), NOT "model
# was deployed" as a single global switch. This lets the effect estimate
# be computed within a single deployment period, which is what makes
# window-over-window comparison meaningful: we are asking "when clinicians
# follow the model, what happens to outcomes — and has that changed?"
# rather than re-running a pre/post natural experiment every monitoring
# cycle.

def derive_concordance_treatment(
    model_output: np.ndarray,
    clinical_decision: np.ndarray,
    *,
    decision_threshold: float = 0.5,
) -> np.ndarray:
    """
    Binary indicator: did the clinician's actual decision match what the
    model recommended (model_output thresholded)? This is the "treatment"
    whose causal effect on outcomes we track over time.

    Both inputs must already be on the same binary decision scale (e.g.
    "flag for review" / "do not flag"). If your model emits a continuous
    risk score, threshold it consistently with how it's used clinically —
    not at a value chosen to make this test pass.
    """
    model_decision = (np.asarray(model_output) >= decision_threshold).astype(int)
    clinical_decision = np.asarray(clinical_decision).astype(int)
    return (model_decision == clinical_decision).astype(int)


# --------------------------------------------------------------------------- #
# Causal effect estimation
# --------------------------------------------------------------------------- #

def estimate_ate_ipw(
    treatment: np.ndarray,
    outcome: np.ndarray,
    covariates: np.ndarray,
    *,
    clip_propensity: tuple[float, float] = (0.05, 0.95),
) -> CausalEffectEstimate:
    """
    Inverse propensity weighting estimate of the ATE of `treatment`
    (clinician concordance with model) on `outcome`, adjusting for
    `covariates` via a logistic propensity model.

    This is intentionally the simpler of the two estimators offered here.
    It is more sensitive to propensity model misspecification than AIPW,
    but has a more transparent failure mode (positivity violations are
    directly visible in the propensity scores) — useful as a first check
    before trusting AIPW's doubly-robust but more opaque combination.
    """
    propensity = _fit_propensity_scores(covariates, treatment)
    propensity = np.clip(propensity, clip_propensity[0], clip_propensity[1])

    overlap_ok = _check_overlap(propensity)

    weights_treated = treatment / propensity
    weights_control = (1 - treatment) / (1 - propensity)

    mu1 = np.sum(weights_treated * outcome) / np.sum(weights_treated)
    mu0 = np.sum(weights_control * outcome) / np.sum(weights_control)
    ate = float(mu1 - mu0)

    # Sandwich-type variance for IPW ATE (Lunceford & Davidian 2004, simplified)
    n = len(treatment)
    influence = (
        weights_treated * (outcome - mu1) - weights_control * (outcome - mu0)
    )
    variance = float(np.var(influence) / n)

    return CausalEffectEstimate(
        window_label="unlabeled",
        estimator=EffectEstimator.IPW,
        ate=ate,
        ate_variance=variance,
        n=n,
        propensity_overlap_ok=overlap_ok,
    )


def estimate_ate_aipw(
    treatment: np.ndarray,
    outcome: np.ndarray,
    covariates: np.ndarray,
    *,
    clip_propensity: tuple[float, float] = (0.05, 0.95),
) -> CausalEffectEstimate:
    """
    Augmented (doubly-robust) IPW estimate: combines a propensity model
    with an outcome regression so the estimate stays consistent if EITHER
    model is correctly specified. Preferred default for the validity-horizon
    pipeline; falls back to plain IPW estimates' overlap diagnostic so
    positivity violations are still surfaced even though AIPW is more
    robust to other misspecification.
    """
    propensity = _fit_propensity_scores(covariates, treatment)
    propensity = np.clip(propensity, clip_propensity[0], clip_propensity[1])
    overlap_ok = _check_overlap(propensity)

    mu1_hat, mu0_hat = _fit_outcome_regression(covariates, treatment, outcome)

    n = len(treatment)
    aipw_treated = (
        mu1_hat + (treatment / propensity) * (outcome - mu1_hat)
    )
    aipw_control = (
        mu0_hat + ((1 - treatment) / (1 - propensity)) * (outcome - mu0_hat)
    )
    per_unit_effect = aipw_treated - aipw_control
    ate = float(np.mean(per_unit_effect))
    variance = float(np.var(per_unit_effect) / n)

    return CausalEffectEstimate(
        window_label="unlabeled",
        estimator=EffectEstimator.AIPW,
        ate=ate,
        ate_variance=variance,
        n=n,
        propensity_overlap_ok=overlap_ok,
    )


def _fit_propensity_scores(covariates: np.ndarray, treatment: np.ndarray) -> np.ndarray:
    """Logistic regression propensity model, Newton-Raphson, no external deps."""
    x = np.asarray(covariates, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n, d = x.shape
    x_design = np.hstack([np.ones((n, 1)), x])
    t = np.asarray(treatment, dtype=float).ravel()

    beta = np.zeros(d + 1)
    for _ in range(100):
        eta = x_design @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        w = p * (1 - p) + 1e-8
        grad = x_design.T @ (t - p) - 1e-6 * beta
        hessian = -(x_design.T * w) @ x_design - 1e-6 * np.eye(d + 1)
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ grad
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new

    eta = x_design @ beta
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))


def _fit_outcome_regression(
    covariates: np.ndarray, treatment: np.ndarray, outcome: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fits separate linear outcome models for treated and control arms,
    returns predicted mu1(x), mu0(x) for every unit (used for AIPW's
    augmentation term, evaluated at each unit's own covariates regardless
    of which arm they were actually in).
    """
    x = np.asarray(covariates, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n, d = x.shape
    x_design = np.hstack([np.ones((n, 1)), x])
    t = np.asarray(treatment, dtype=float).ravel()
    y = np.asarray(outcome, dtype=float).ravel()

    def _ridge_fit(mask: np.ndarray) -> np.ndarray:
        xm, ym = x_design[mask], y[mask]
        reg = 1e-6 * np.eye(d + 1)
        return np.linalg.solve(xm.T @ xm + reg, xm.T @ ym)

    treated_mask = t == 1
    control_mask = t == 0

    beta1 = _ridge_fit(treated_mask) if treated_mask.sum() > d else np.zeros(d + 1)
    beta0 = _ridge_fit(control_mask) if control_mask.sum() > d else np.zeros(d + 1)

    mu1_hat = x_design @ beta1
    mu0_hat = x_design @ beta0
    return mu1_hat, mu0_hat


def _check_overlap(propensity: np.ndarray, *, min_frac_in_range: float = 0.95) -> bool:
    """
    Positivity diagnostic: flags whether enough of the sample has propensity
    scores comfortably away from 0/1. This does NOT silently fix overlap
    violations (e.g. by trimming) — it reports them, because trimming
    changes the estimand and that decision belongs to the analyst, not to
    this function.
    """
    in_range = np.mean((propensity > 0.1) & (propensity < 0.9))
    return bool(in_range >= min_frac_in_range)


# --------------------------------------------------------------------------- #
# Window-over-window comparison: the actual performativity check
# --------------------------------------------------------------------------- #

def compare_causal_effects(
    estimate_ref: CausalEffectEstimate,
    estimate_new: CausalEffectEstimate,
    *,
    significance_level: float = 0.05,
) -> PerformativityResult:
    """
    Compares two causal effect estimates from different monitoring windows.
    Fires when the change is statistically significant AND, separately,
    flags sign reversal regardless of significance — a sign flip with a
    wide confidence interval is still worth a human looking at it, even if
    it wouldn't survive a strict significance cut on its own.
    """
    delta = estimate_new.ate - estimate_ref.ate
    se = math.sqrt(estimate_ref.ate_variance + estimate_new.ate_variance)
    z = delta / se if se > 0 else 0.0
    p_value = _two_sided_normal_pvalue(z)

    sign_reversed = (
        (estimate_ref.ate > 0 and estimate_new.ate < 0)
        or (estimate_ref.ate < 0 and estimate_new.ate > 0)
    )

    caveats = []
    if not estimate_ref.propensity_overlap_ok:
        caveats.append(
            "Reference window has poor propensity overlap; effect estimate may be "
            "unreliable for the affected covariate region."
        )
    if not estimate_new.propensity_overlap_ok:
        caveats.append(
            "New window has poor propensity overlap; effect estimate may be "
            "unreliable for the affected covariate region."
        )
    if estimate_ref.n < 100 or estimate_new.n < 100:
        caveats.append(
            "One or both windows have fewer than 100 units; variance estimates "
            "and the resulting z-test should be treated as approximate."
        )

    return PerformativityResult(
        estimate_ref=estimate_ref,
        estimate_new=estimate_new,
        effect_delta=float(delta),
        z_statistic=float(z),
        p_value=p_value,
        sign_reversed=sign_reversed,
        fired=(p_value < significance_level) or sign_reversed,
        caveats=caveats,
    )


def _two_sided_normal_pvalue(z: float) -> float:
    """Two-sided p-value from a standard normal z-statistic, via erf."""
    return float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2)))))


# --------------------------------------------------------------------------- #
# Convenience: build both estimates from raw shadow-deployment-style arrays
# --------------------------------------------------------------------------- #

def run_performativity_check(
    *,
    ref_window: dict,
    new_window: dict,
    estimator: EffectEstimator = EffectEstimator.AIPW,
    decision_threshold: float = 0.5,
    significance_level: float = 0.05,
) -> PerformativityResult:
    """
    End-to-end helper. Each window dict must provide:
        'model_output', 'clinical_decision', 'outcome', 'covariates'
    as array-likes of matching length (this is exactly the shape logged by
    shadow_deployment.ShadowLog.to_arrays(), so the shadow-mode log can be
    fed in directly without manual reshaping).
    """
    fit_fn = estimate_ate_ipw if estimator == EffectEstimator.IPW else estimate_ate_aipw

    def _build(window: dict, label: str) -> CausalEffectEstimate:
        treatment = derive_concordance_treatment(
            np.asarray(window["model_output"]),
            np.asarray(window["clinical_decision"]),
            decision_threshold=decision_threshold,
        )
        est = fit_fn(treatment, np.asarray(window["outcome"]), np.asarray(window["covariates"]))
        est.window_label = label
        return est

    estimate_ref = _build(ref_window, "reference")
    estimate_new = _build(new_window, "new")

    return compare_causal_effects(
        estimate_ref, estimate_new, significance_level=significance_level
    )
