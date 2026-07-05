"""
stratum/layer3/drift.py

Layer 3: Sociotechnical and Temporal Validity — Drift Detection

Implements detectors for the three drift types defined in the Temporal
Validity Protocol:

    1. Covariate shift   — P(X) changes           -> MMD / KL on inputs
    2. Concept drift      — P(Y|X) changes          -> refit label model, compare coefficients
    3. Performative drift — model changes P(X)      -> see performativity.py

This module implements the *statistical machinery* shared across drift
types: a CUSUM sequential test and a Bayesian online changepoint detector.
Subgroup-level monitoring with FDR correction is implemented here; the
specific covariate/concept comparisons live in their own functions so they
can be unit tested independently of the sequential testing logic.

METRIC_NAME: drift.cusum
METRIC_LAYER: 3
PAPER_REF: sec:temporal-validity
THRESHOLD_CITE: Page & Hinkley (1954); Basseville & Nikiforov (1993) ch.2
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Shared types
# --------------------------------------------------------------------------- #

class DriftType(str, Enum):
    COVARIATE = "covariate_shift"
    CONCEPT = "concept_drift"
    PERFORMATIVE = "performative_drift"


@dataclass
class DriftAlert:
    """A single drift detection event for one subgroup/stream."""
    subgroup: str
    drift_type: DriftType
    statistic: float
    threshold: float
    index: int                # sample index (or time index) at which alert fired
    raw_p_value: float | None = None
    corrected_p_value: float | None = None
    fired: bool = False


@dataclass
class CUSUMResult:
    """Full trace of a CUSUM run, plus the alert if one fired."""
    statistic_trace: np.ndarray
    threshold: float
    alert_index: int | None
    fired: bool

    def to_alert(self, subgroup: str, drift_type: DriftType) -> DriftAlert:
        stat_at_alert = (
            float(self.statistic_trace[self.alert_index])
            if self.alert_index is not None
            else float(self.statistic_trace[-1])
        )
        return DriftAlert(
            subgroup=subgroup,
            drift_type=drift_type,
            statistic=stat_at_alert,
            threshold=self.threshold,
            index=self.alert_index if self.alert_index is not None else -1,
            fired=self.fired,
        )


# --------------------------------------------------------------------------- #
# CUSUM: sequential likelihood ratio test
# --------------------------------------------------------------------------- #
#
# Design note: the threshold is an argument supplied by the caller, not
# something this function tunes. Per the protocol, alert thresholds must be
# pre-specified — typically derived analytically from a target false-alarm
# rate (e.g. via Wald's approximation) or fixed in a pre-registration
# document, BEFORE the detector ever sees production data. This module
# refuses to expose any "auto-threshold-from-this-data" path, by design.

def cusum_threshold_from_target_arl(
    target_arl: float,
    *,
    delta: float = 1.0,
) -> float:
    """
    Compute a CUSUM decision threshold analytically from a target Average
    Run Length (ARL) under the null (in-control) regime, using Wald's
    approximation for the two-sided Gaussian CUSUM:

        h ≈ ln(ARL) / delta

    This is meant to be called ONCE, before deployment, to pre-register a
    threshold. It must not be called against streaming production data to
    "tune" the alarm rate after the fact — that would invalidate the
    sequential test's false-alarm guarantees.

    Parameters
    ----------
    target_arl : desired average run length to false alarm under H0.
    delta : standardized mean shift the CUSUM is tuned to detect quickly
        (in units of the reference distribution's standard deviation).
    """
    if target_arl <= 1:
        raise ValueError("target_arl must be > 1")
    if delta <= 0:
        raise ValueError("delta must be > 0")
    return math.log(target_arl) / delta


def cusum_gaussian(
    x: Sequence[float] | np.ndarray,
    *,
    mu0: float,
    sigma0: float,
    delta: float,
    threshold: float,
    two_sided: bool = True,
) -> CUSUMResult:
    """
    Sequential likelihood-ratio CUSUM for a shift in the mean of a Gaussian
    stream, standardized by a pre-specified in-control (mu0, sigma0).

    This is the textbook page-hinkley / CUSUM recursion:

        S+_t = max(0, S+_{t-1} + (x_t - mu0)/sigma0 - delta/2)
        S-_t = min(0, S-_{t-1} + (x_t - mu0)/sigma0 + delta/2)

    Alarm fires the first time |S| exceeds `threshold`. The recursion stops
    accumulating evidence (resets to 0) after an alarm, consistent with
    standard sequential changepoint practice: once flagged, the segment is
    handed off for changepoint localization (see `bayesian_online_changepoint`)
    rather than continuing to accumulate under a now-invalid null.

    Parameters
    ----------
    x : standardized or raw observations (one per time step).
    mu0, sigma0 : in-control mean/std, estimated from a held-out reference
        window BEFORE monitoring starts. Never re-estimated from the
        monitored stream itself.
    delta : minimum standardized shift magnitude considered "drift" worth
        detecting — this is the sensitivity knob, set by domain judgement,
        not fit to the data.
    threshold : pre-specified decision boundary (see
        `cusum_threshold_from_target_arl`).
    """
    x = np.asarray(x, dtype=float)
    if sigma0 <= 0:
        raise ValueError("sigma0 must be > 0")
    if threshold <= 0:
        raise ValueError("threshold must be > 0")

    z = (x - mu0) / sigma0
    n = len(z)
    s_pos = np.zeros(n)
    s_neg = np.zeros(n)
    stat = np.zeros(n)

    alert_index = None
    running_pos, running_neg = 0.0, 0.0

    for t in range(n):
        running_pos = max(0.0, running_pos + z[t] - delta / 2.0)
        if two_sided:
            running_neg = min(0.0, running_neg + z[t] + delta / 2.0)
        s_pos[t] = running_pos
        s_neg[t] = running_neg
        stat[t] = max(running_pos, abs(running_neg))

        if alert_index is None and stat[t] >= threshold:
            alert_index = t
            running_pos, running_neg = 0.0, 0.0  # reset post-alarm

    return CUSUMResult(
        statistic_trace=stat,
        threshold=threshold,
        alert_index=alert_index,
        fired=alert_index is not None,
    )


def cusum_bernoulli(
    x: Sequence[int] | np.ndarray,
    *,
    p0: float,
    p1: float,
    threshold: float,
) -> CUSUMResult:
    """
    Sequential likelihood-ratio CUSUM for a shift in a Bernoulli rate
    (e.g. error rate, positive-prediction rate) from p0 (in-control) to
    p1 (the smallest rate considered worth alarming on).

    Log-likelihood-ratio increment per observation x_t in {0,1}:

        llr_t = x_t * ln(p1/p0) + (1 - x_t) * ln((1-p1)/(1-p0))

    S_t = max(0, S_{t-1} + llr_t); alarm when S_t >= threshold.
    """
    x = np.asarray(x, dtype=float)
    if not (0 < p0 < 1) or not (0 < p1 < 1):
        raise ValueError("p0, p1 must be in (0, 1)")
    if threshold <= 0:
        raise ValueError("threshold must be > 0")

    llr1 = math.log(p1 / p0)
    llr0 = math.log((1 - p1) / (1 - p0))
    increments = x * llr1 + (1 - x) * llr0

    n = len(x)
    stat = np.zeros(n)
    alert_index = None
    running = 0.0
    for t in range(n):
        running = max(0.0, running + increments[t])
        stat[t] = running
        if alert_index is None and running >= threshold:
            alert_index = t
            running = 0.0

    return CUSUMResult(
        statistic_trace=stat,
        threshold=threshold,
        alert_index=alert_index,
        fired=alert_index is not None,
    )


# --------------------------------------------------------------------------- #
# Bayesian online changepoint detection (BOCPD)
# --------------------------------------------------------------------------- #
#
# Used to *localize* a changepoint once CUSUM has signalled that one exists,
# or to run continuously as a complementary detector with different
# sensitivity characteristics (BOCPD is better at multiple/gradual
# changepoints; CUSUM is better at clean single-shift detection with
# certified false-alarm control).

@dataclass
class BOCPDResult:
    run_length_posterior: np.ndarray   # shape (n, n) lower-triangular-ish
    changepoint_probs: np.ndarray      # P(changepoint at t) per step
    map_changepoints: list[int]


def _studentt_logpdf(x: float, mu: float, kappa: float, alpha: float, beta: float) -> float:
    """Posterior predictive log-density under Normal-Inverse-Gamma prior."""
    from math import lgamma, log, pi

    nu = 2 * alpha
    var = beta * (kappa + 1) / (alpha * kappa)
    t_scaled = (x - mu) / math.sqrt(var)
    return (
        lgamma((nu + 1) / 2)
        - lgamma(nu / 2)
        - 0.5 * log(nu * pi * var)
        - ((nu + 1) / 2) * log(1 + (t_scaled ** 2) / nu)
    )


def bayesian_online_changepoint(
    x: Sequence[float] | np.ndarray,
    *,
    hazard: float = 1.0 / 250.0,
    mu0: float = 0.0,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    map_threshold: float = 0.5,
) -> BOCPDResult:
    """
    Adams & MacKay (2007) online changepoint detection with a Normal-Inverse-
    Gamma conjugate prior on each regime's (mean, variance).

    `hazard` is the prior changepoint rate (1 / expected run length) and,
    like the CUSUM threshold, should be pre-specified from domain knowledge
    about how often the *clinical or data pipeline* is expected to change
    (model updates, EHR migrations, guideline revisions) — not fit to make
    the detector agree with a particular labeling of historical data.

    Returns the full run-length posterior plus a MAP changepoint list:
    indices where P(run length resets to 0) exceeds `map_threshold`.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)

    # R[t, l] = P(run length = l | x_1:t)
    R = np.zeros((n + 1, n + 1))
    R[0, 0] = 1.0

    mu_params = np.full(n + 1, mu0)
    kappa_params = np.full(n + 1, kappa0)
    alpha_params = np.full(n + 1, alpha0)
    beta_params = np.full(n + 1, beta0)

    changepoint_probs = np.zeros(n)
    map_changepoints = []

    for t in range(1, n + 1):
        x_t = x[t - 1]
        pred_logprobs = np.array([
            _studentt_logpdf(x_t, mu_params[l], kappa_params[l], alpha_params[l], beta_params[l])
            for l in range(t)
        ])
        pred_probs = np.exp(pred_logprobs - pred_logprobs.max())  # numerically stable
        pred_probs /= pred_probs.sum() if pred_probs.sum() > 0 else 1.0
        pred_probs = np.exp(pred_logprobs)  # restore true scale for the recursion below

        growth_probs = R[t - 1, :t] * pred_probs * (1 - hazard)
        cp_prob = np.sum(R[t - 1, :t] * pred_probs * hazard)

        new_R = np.zeros(t + 1)
        new_R[1:] = growth_probs
        new_R[0] = cp_prob

        total = new_R.sum()
        if total > 0:
            new_R /= total
        R[t, :t + 1] = new_R

        changepoint_probs[t - 1] = new_R[0]
        if new_R[0] >= map_threshold:
            map_changepoints.append(t - 1)

        # Update sufficient statistics for each run length (including new l=0 regime)
        new_mu = np.zeros(t + 1)
        new_kappa = np.zeros(t + 1)
        new_alpha = np.zeros(t + 1)
        new_beta = np.zeros(t + 1)

        new_mu[0], new_kappa[0], new_alpha[0], new_beta[0] = mu0, kappa0, alpha0, beta0
        for l in range(t):
            k = kappa_params[l]
            new_kappa[l + 1] = k + 1
            new_mu[l + 1] = (k * mu_params[l] + x_t) / (k + 1)
            new_alpha[l + 1] = alpha_params[l] + 0.5
            new_beta[l + 1] = beta_params[l] + (k * (x_t - mu_params[l]) ** 2) / (2 * (k + 1))

        mu_params[: t + 1] = new_mu
        kappa_params[: t + 1] = new_kappa
        alpha_params[: t + 1] = new_alpha
        beta_params[: t + 1] = new_beta

    return BOCPDResult(
        run_length_posterior=R,
        changepoint_probs=changepoint_probs,
        map_changepoints=map_changepoints,
    )


# --------------------------------------------------------------------------- #
# Covariate shift: MMD / KL on input distributions
# --------------------------------------------------------------------------- #

def mmd_rbf(
    x_ref: np.ndarray,
    x_new: np.ndarray,
    *,
    gamma: float | None = None,
) -> float:
    """
    Squared Maximum Mean Discrepancy between reference and new covariate
    samples using an RBF kernel. Unbiased estimator (Gretton et al. 2012,
    eq. 3). `gamma` defaults to the median-heuristic bandwidth if not given.

    A larger value indicates greater divergence between P_ref(X) and
    P_new(X). This statistic alone is NOT a drift alarm — pair it with
    a permutation test (`mmd_permutation_pvalue`) or feed it through
    `cusum_gaussian` against a held-out null distribution of MMD values.
    """
    x_ref = np.atleast_2d(x_ref)
    x_new = np.atleast_2d(x_new)
    if x_ref.ndim == 1:
        x_ref = x_ref.reshape(-1, 1)
    if x_new.ndim == 1:
        x_new = x_new.reshape(-1, 1)

    if gamma is None:
        combined = np.vstack([x_ref, x_new])
        dists = _pairwise_sq_dists(combined, combined)
        median_dist = np.median(dists[dists > 0]) if np.any(dists > 0) else 1.0
        gamma = 1.0 / (2 * median_dist) if median_dist > 0 else 1.0

    k_xx = _rbf_kernel(x_ref, x_ref, gamma)
    k_yy = _rbf_kernel(x_new, x_new, gamma)
    k_xy = _rbf_kernel(x_ref, x_new, gamma)

    m, n = len(x_ref), len(x_new)
    sum_xx = (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1)) if m > 1 else 0.0
    sum_yy = (k_yy.sum() - np.trace(k_yy)) / (n * (n - 1)) if n > 1 else 0.0
    sum_xy = k_xy.sum() / (m * n)

    return float(sum_xx + sum_yy - 2 * sum_xy)


def mmd_permutation_pvalue(
    x_ref: np.ndarray,
    x_new: np.ndarray,
    *,
    n_permutations: int = 500,
    gamma: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Permutation test p-value for MMD-based covariate shift.

    Returns (observed_mmd, p_value). Null distribution is built by
    repeatedly reshuffling the pooled sample into two groups of the
    original sizes and recomputing MMD.
    """
    rng = rng or np.random.default_rng()
    observed = mmd_rbf(x_ref, x_new, gamma=gamma)

    pooled = np.vstack([np.atleast_2d(x_ref).reshape(len(x_ref), -1),
                         np.atleast_2d(x_new).reshape(len(x_new), -1)])
    m = len(x_ref)
    n_total = len(pooled)

    null_stats = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(n_total)
        a, b = pooled[perm[:m]], pooled[perm[m:]]
        null_stats[i] = mmd_rbf(a, b, gamma=gamma)

    p_value = float((np.sum(null_stats >= observed) + 1) / (n_permutations + 1))
    return observed, p_value


def kl_divergence_gaussian(
    x_ref: np.ndarray,
    x_new: np.ndarray,
    *,
    eps: float = 1e-8,
) -> float:
    """
    KL(P_new || P_ref) for univariate covariates under a Gaussian
    approximation to each distribution. Cheap, interpretable, and a
    reasonable default when MMD's permutation cost is prohibitive for
    high-frequency per-subgroup monitoring. Asymmetric — direction matters
    here: this measures how surprising the new distribution is to a model
    that "expects" the reference distribution.
    """
    x_ref = np.asarray(x_ref, dtype=float).ravel()
    x_new = np.asarray(x_new, dtype=float).ravel()

    mu_ref, sigma_ref = x_ref.mean(), x_ref.std() + eps
    mu_new, sigma_new = x_new.mean(), x_new.std() + eps

    return float(
        math.log(sigma_ref / sigma_new)
        + (sigma_new ** 2 + (mu_new - mu_ref) ** 2) / (2 * sigma_ref ** 2)
        - 0.5
    )


def _pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_sq = np.sum(a ** 2, axis=1, keepdims=True)
    b_sq = np.sum(b ** 2, axis=1, keepdims=True)
    return a_sq + b_sq.T - 2 * a @ b.T


def _rbf_kernel(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-gamma * _pairwise_sq_dists(a, b))


# --------------------------------------------------------------------------- #
# Concept drift: refit label model, compare coefficients
# --------------------------------------------------------------------------- #

@dataclass
class ConceptDriftResult:
    coef_ref: np.ndarray
    coef_new: np.ndarray
    coef_delta: np.ndarray
    wald_statistic: float
    p_value: float
    fired: bool


def concept_drift_logistic(
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    x_new: np.ndarray,
    y_new: np.ndarray,
    *,
    significance_level: float = 0.05,
    l2_penalty: float = 1e-6,
    max_iter: int = 200,
) -> ConceptDriftResult:
    """
    Detect concept drift — a change in P(Y|X) — by refitting a logistic
    label model on the new window and comparing coefficients against the
    reference-window fit via a Wald test.

    This deliberately does NOT use AUROC or accuracy: per the protocol,
    performance metrics can stay flat under concept drift if errors shift
    location without changing aggregate calibration (e.g. the model gets
    worse for one subgroup and incidentally better for another). Comparing
    P(Y|X) coefficients directly catches structural relationship changes
    that performance metrics can mask.
    """
    coef_ref, cov_ref = _fit_logistic_with_cov(x_ref, y_ref, l2_penalty, max_iter)
    coef_new, cov_new = _fit_logistic_with_cov(x_new, y_new, l2_penalty, max_iter)

    delta = coef_new - coef_ref
    cov_sum = cov_ref + cov_new
    # Wald statistic: delta' (cov_ref + cov_new)^-1 delta ~ chi2(df=len(delta))
    try:
        cov_inv = np.linalg.inv(cov_sum)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_sum)

    wald_stat = float(delta @ cov_inv @ delta)
    df = len(delta)
    p_value = float(_chi2_sf(wald_stat, df))

    return ConceptDriftResult(
        coef_ref=coef_ref,
        coef_new=coef_new,
        coef_delta=delta,
        wald_statistic=wald_stat,
        p_value=p_value,
        fired=p_value < significance_level,
    )


def _fit_logistic_with_cov(
    x: np.ndarray, y: np.ndarray, l2_penalty: float, max_iter: int
) -> tuple[np.ndarray, np.ndarray]:
    """Newton-Raphson logistic regression returning coefficients + their
    asymptotic covariance (inverse Fisher information), with an L2 ridge
    term for numerical stability in small/unbalanced subgroup windows."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n, d = x.shape
    x_design = np.hstack([np.ones((n, 1)), x])  # intercept

    beta = np.zeros(d + 1)
    for _ in range(max_iter):
        eta = x_design @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        w = p * (1 - p) + 1e-8
        grad = x_design.T @ (y - p) - l2_penalty * beta
        hessian = -(x_design.T * w) @ x_design - l2_penalty * np.eye(d + 1)
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
    p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
    w = p * (1 - p) + 1e-8
    fisher_info = (x_design.T * w) @ x_design + l2_penalty * np.eye(d + 1)
    try:
        cov = np.linalg.inv(fisher_info)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(fisher_info)

    return beta, cov


def _chi2_sf(x: float, df: int) -> float:
    """Survival function of chi-squared without scipy dependency, via the
    regularized upper incomplete gamma function (series/continued-fraction
    hybrid, sufficient precision for drift-test p-values)."""
    if x <= 0:
        return 1.0
    a = df / 2.0
    z = x / 2.0
    return float(_gammaincc(a, z))


def _gammaincc(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x), via continued fraction
    (Numerical Recipes 6.2), avoiding a scipy dependency for this module."""
    if x < a + 1.0:
        return 1.0 - _gammainc_lower_series(a, x)

    # Continued fraction for Q(a, x)
    eps = 1e-12
    max_iter = 500
    b = x + 1.0 - a
    c = 1.0 / 1e-30
    d = 1.0 / b
    h = d
    for i in range(1, max_iter):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def _gammainc_lower_series(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x) via series expansion."""
    if x == 0:
        return 0.0
    ap = a
    summ = 1.0 / a
    delta = summ
    for _ in range(500):
        ap += 1.0
        delta *= x / ap
        summ += delta
        if abs(delta) < abs(summ) * 1e-12:
            break
    return summ * math.exp(-x + a * math.log(x) - math.lgamma(a))


# --------------------------------------------------------------------------- #
# Per-subgroup monitoring with FDR correction
# --------------------------------------------------------------------------- #

def benjamini_hochberg(p_values: Sequence[float], *, alpha: float = 0.05) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction. Returns a boolean array of which
    hypotheses are rejected (i.e. which subgroup alerts survive multiple-
    comparisons correction) at the given false discovery rate.

    Required whenever drift is tested across more than one subgroup: testing
    k subgroups at raw alpha=0.05 each gives a substantially higher chance
    that at least one fires spuriously, and "all subgroups individually
    significant" results are exactly the artefact this protocol exists to
    prevent.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool)

    order = np.argsort(p)
    ranked = p[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha

    below = ranked <= thresholds
    if not np.any(below):
        reject_sorted = np.zeros(n, dtype=bool)
    else:
        max_rank = np.max(np.where(below)[0])
        reject_sorted = np.zeros(n, dtype=bool)
        reject_sorted[: max_rank + 1] = True

    reject = np.zeros(n, dtype=bool)
    reject[order] = reject_sorted
    return reject


def monitor_subgroups(
    subgroup_data: dict[str, tuple[np.ndarray, np.ndarray]],
    detector_fn: Callable[[np.ndarray, np.ndarray], tuple[float, float]],
    *,
    drift_type: DriftType,
    alpha: float = 0.05,
) -> list[DriftAlert]:
    """
    Run a (statistic, p_value)-returning detector independently per
    subgroup, then apply BH-FDR correction jointly across all subgroups
    before deciding which alerts fire.

    `detector_fn(ref, new) -> (statistic, p_value)` — e.g. wrap
    `mmd_permutation_pvalue` or `concept_drift_logistic` for this signature.

    This is the seam where covariate-shift and concept-drift detectors
    plug into subgroup-aware governance; performative drift uses a
    structurally different comparison and is monitored separately in
    `performativity.py`.
    """
    names = list(subgroup_data.keys())
    statistics = []
    p_values = []

    for name in names:
        ref, new = subgroup_data[name]
        stat, p_val = detector_fn(ref, new)
        statistics.append(stat)
        p_values.append(p_val)

    rejected = benjamini_hochberg(p_values, alpha=alpha)

    # BH-adjusted p-values (for reporting, not for the rejection decision itself)
    p_arr = np.asarray(p_values)
    order = np.argsort(p_arr)
    ranked = p_arr[order]
    n = len(p_arr)
    adjusted_sorted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)
    adjusted = np.empty(n)
    adjusted[order] = adjusted_sorted

    alerts = []
    for i, name in enumerate(names):
        alerts.append(
            DriftAlert(
                subgroup=name,
                drift_type=drift_type,
                statistic=statistics[i],
                threshold=alpha,
                index=-1,
                raw_p_value=p_values[i],
                corrected_p_value=float(adjusted[i]),
                fired=bool(rejected[i]),
            )
        )
    return alerts
