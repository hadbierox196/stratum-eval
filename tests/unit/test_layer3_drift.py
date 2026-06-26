"""
tests/unit/test_layer3_drift.py

Synthetic-data-only unit tests for Layer 3 drift detectors. Per repo
convention, these require no credentials and must be safe on all
supported Python versions. Each detector gets a paired test: fires on an
injected synthetic shift, stays silent on stable (null) data, at the same
pre-specified threshold.

Reference seed locked at 42 throughout, matching the integration suite's
reproducibility convention.
"""

from __future__ import annotations

import numpy as np
import pytest

from stratum.layer3.drift import (
    DriftType,
    cusum_threshold_from_target_arl,
    cusum_gaussian,
    cusum_bernoulli,
    bayesian_online_changepoint,
    mmd_rbf,
    mmd_permutation_pvalue,
    kl_divergence_gaussian,
    concept_drift_logistic,
    benjamini_hochberg,
    monitor_subgroups,
)

SEED = 42


# --------------------------------------------------------------------------- #
# CUSUM (Gaussian)
# --------------------------------------------------------------------------- #

class TestCUSUMGaussian:
    def test_fires_on_injected_mean_shift(self):
        rng = np.random.default_rng(SEED)
        stable = rng.normal(loc=0.0, scale=1.0, size=200)
        shifted = rng.normal(loc=2.5, scale=1.0, size=200)  # large, unambiguous shift
        x = np.concatenate([stable, shifted])

        threshold = cusum_threshold_from_target_arl(target_arl=500, delta=1.0)
        result = cusum_gaussian(x, mu0=0.0, sigma0=1.0, delta=1.0, threshold=threshold)

        assert result.fired
        assert result.alert_index is not None
        # Alert should localize to after the injected shift, not before it
        assert result.alert_index >= 200

    def test_silent_on_stable_null_data(self):
        rng = np.random.default_rng(SEED)
        x = rng.normal(loc=0.0, scale=1.0, size=400)  # no shift at all

        threshold = cusum_threshold_from_target_arl(target_arl=500, delta=1.0)
        result = cusum_gaussian(x, mu0=0.0, sigma0=1.0, delta=1.0, threshold=threshold)

        assert not result.fired
        assert result.alert_index is None

    def test_threshold_must_be_prespecified_positive(self):
        with pytest.raises(ValueError):
            cusum_gaussian([0.1, 0.2], mu0=0.0, sigma0=1.0, delta=1.0, threshold=0.0)

    def test_rejects_invalid_sigma(self):
        with pytest.raises(ValueError):
            cusum_gaussian([0.1], mu0=0.0, sigma0=0.0, delta=1.0, threshold=5.0)

    def test_higher_target_arl_yields_higher_threshold(self):
        low = cusum_threshold_from_target_arl(target_arl=50, delta=1.0)
        high = cusum_threshold_from_target_arl(target_arl=5000, delta=1.0)
        assert high > low


# --------------------------------------------------------------------------- #
# CUSUM (Bernoulli) — e.g. error-rate monitoring
# --------------------------------------------------------------------------- #

class TestCUSUMBernoulli:
    def test_fires_on_injected_rate_increase(self):
        rng = np.random.default_rng(SEED)
        stable = rng.binomial(1, 0.05, size=300)   # 5% baseline error rate
        shifted = rng.binomial(1, 0.30, size=300)  # error rate jumps to 30%
        x = np.concatenate([stable, shifted])

        result = cusum_bernoulli(x, p0=0.05, p1=0.15, threshold=8.0)

        assert result.fired
        assert result.alert_index >= 300 - 50  # allow some detection lag

    def test_silent_on_stable_null_rate(self):
        rng = np.random.default_rng(SEED)
        x = rng.binomial(1, 0.05, size=600)

        result = cusum_bernoulli(x, p0=0.05, p1=0.15, threshold=8.0)

        assert not result.fired

    def test_rejects_invalid_probabilities(self):
        with pytest.raises(ValueError):
            cusum_bernoulli([0, 1], p0=0.0, p1=0.5, threshold=5.0)
        with pytest.raises(ValueError):
            cusum_bernoulli([0, 1], p0=0.5, p1=1.0, threshold=5.0)


# --------------------------------------------------------------------------- #
# Bayesian Online Changepoint Detection
# --------------------------------------------------------------------------- #

class TestBOCPD:
    def test_fires_on_injected_mean_shift(self):
        rng = np.random.default_rng(SEED)
        stable = rng.normal(loc=0.0, scale=0.5, size=80)
        shifted = rng.normal(loc=4.0, scale=0.5, size=80)
        x = np.concatenate([stable, shifted])

        result = bayesian_online_changepoint(x, hazard=1.0 / 100.0, map_threshold=0.4)

        assert len(result.map_changepoints) > 0
        # at least one detected changepoint should land near the true boundary (index 80)
        assert any(70 <= cp <= 95 for cp in result.map_changepoints)

    def test_silent_on_stable_null_data(self):
        rng = np.random.default_rng(SEED)
        x = rng.normal(loc=0.0, scale=0.5, size=160)

        result = bayesian_online_changepoint(x, hazard=1.0 / 100.0, map_threshold=0.4)

        # Some low-probability bookkeeping changepoints near t=0 are expected;
        # the assertion is on a sustained run, not zero probability anywhere.
        sustained_high_prob = sum(1 for p in result.changepoint_probs if p >= 0.4)
        assert sustained_high_prob <= 2


# --------------------------------------------------------------------------- #
# MMD covariate shift
# --------------------------------------------------------------------------- #

class TestMMD:
    def test_fires_on_injected_covariate_shift(self):
        rng = np.random.default_rng(SEED)
        x_ref = rng.normal(loc=0.0, scale=1.0, size=(150, 3))
        x_new = rng.normal(loc=1.5, scale=1.0, size=(150, 3))

        observed, p_value = mmd_permutation_pvalue(x_ref, x_new, n_permutations=200, rng=rng)

        assert observed > 0
        assert p_value < 0.05

    def test_silent_on_identical_distribution(self):
        rng = np.random.default_rng(SEED)
        x_ref = rng.normal(loc=0.0, scale=1.0, size=(150, 3))
        x_new = rng.normal(loc=0.0, scale=1.0, size=(150, 3))

        observed, p_value = mmd_permutation_pvalue(x_ref, x_new, n_permutations=200, rng=rng)

        assert p_value >= 0.05

    def test_mmd_is_nonnegative_in_expectation(self):
        # MMD^2 unbiased estimator can dip slightly negative for identical
        # distributions due to finite-sample noise, but should be small.
        rng = np.random.default_rng(SEED)
        x_ref = rng.normal(size=(200, 2))
        x_new = rng.normal(size=(200, 2))
        observed = mmd_rbf(x_ref, x_new)
        assert observed > -0.05


# --------------------------------------------------------------------------- #
# KL divergence covariate shift
# --------------------------------------------------------------------------- #

class TestKLDivergence:
    def test_fires_on_injected_shift(self):
        rng = np.random.default_rng(SEED)
        x_ref = rng.normal(loc=0.0, scale=1.0, size=300)
        x_new = rng.normal(loc=3.0, scale=1.0, size=300)

        kl = kl_divergence_gaussian(x_ref, x_new)
        assert kl > 1.0  # a 3-sigma mean shift should be clearly nonzero

    def test_near_zero_on_identical_distribution(self):
        rng = np.random.default_rng(SEED)
        x_ref = rng.normal(loc=0.0, scale=1.0, size=300)
        x_new = rng.normal(loc=0.0, scale=1.0, size=300)

        kl = kl_divergence_gaussian(x_ref, x_new)
        assert kl < 0.1


# --------------------------------------------------------------------------- #
# Concept drift (P(Y|X) refit + Wald test)
# --------------------------------------------------------------------------- #

class TestConceptDrift:
    def test_fires_on_injected_coefficient_change(self):
        rng = np.random.default_rng(SEED)
        n = 500
        x_ref = rng.normal(size=(n, 2))
        true_beta_ref = np.array([1.5, -1.0])
        p_ref = 1 / (1 + np.exp(-(x_ref @ true_beta_ref)))
        y_ref = rng.binomial(1, p_ref)

        x_new = rng.normal(size=(n, 2))
        true_beta_new = np.array([-1.5, 1.0])  # sign-flipped relationship: real concept drift
        p_new = 1 / (1 + np.exp(-(x_new @ true_beta_new)))
        y_new = rng.binomial(1, p_new)

        result = concept_drift_logistic(x_ref, y_ref, x_new, y_new, significance_level=0.05)

        assert result.fired
        assert result.p_value < 0.05

    def test_silent_when_relationship_unchanged(self):
        rng = np.random.default_rng(SEED)
        n = 500
        true_beta = np.array([1.5, -1.0])

        x_ref = rng.normal(size=(n, 2))
        p_ref = 1 / (1 + np.exp(-(x_ref @ true_beta)))
        y_ref = rng.binomial(1, p_ref)

        x_new = rng.normal(size=(n, 2))
        p_new = 1 / (1 + np.exp(-(x_new @ true_beta)))  # same coefficients, fresh draw
        y_new = rng.binomial(1, p_new)

        result = concept_drift_logistic(x_ref, y_ref, x_new, y_new, significance_level=0.05)

        assert not result.fired
        assert result.p_value >= 0.05

    def test_stays_silent_under_covariate_shift_alone(self):
        """
        Critical negative control distinguishing concept drift from covariate
        shift: P(X) changes but P(Y|X) does not. This detector must NOT fire,
        since the conceptual point of the layer is that these are different
        drift types requiring different responses.
        """
        rng = np.random.default_rng(SEED)
        n = 500
        true_beta = np.array([1.5, -1.0])

        x_ref = rng.normal(loc=0.0, size=(n, 2))
        p_ref = 1 / (1 + np.exp(-(x_ref @ true_beta)))
        y_ref = rng.binomial(1, p_ref)

        x_new = rng.normal(loc=2.0, size=(n, 2))  # P(X) shifted...
        p_new = 1 / (1 + np.exp(-(x_new @ true_beta)))  # ...but P(Y|X) relationship intact
        y_new = rng.binomial(1, p_new)

        result = concept_drift_logistic(x_ref, y_ref, x_new, y_new, significance_level=0.05)

        assert not result.fired


# --------------------------------------------------------------------------- #
# Benjamini-Hochberg FDR correction
# --------------------------------------------------------------------------- #

class TestBenjaminiHochberg:
    def test_rejects_clearly_significant_pvalues(self):
        p_values = [0.001, 0.002, 0.50, 0.60, 0.70]
        rejected = benjamini_hochberg(p_values, alpha=0.05)
        assert rejected[0] and rejected[1]
        assert not rejected[2] and not rejected[3] and not rejected[4]

    def test_silent_when_all_null(self):
        rng = np.random.default_rng(SEED)
        p_values = rng.uniform(0.0, 1.0, size=20)  # null: uniform p-values
        rejected = benjamini_hochberg(p_values, alpha=0.05)
        # Should reject roughly alpha-fraction at most by chance; allow small slack
        assert rejected.sum() <= 3

    def test_empty_input(self):
        assert len(benjamini_hochberg([])) == 0


# --------------------------------------------------------------------------- #
# monitor_subgroups integration: per-subgroup + FDR
# --------------------------------------------------------------------------- #

class TestMonitorSubgroups:
    def test_fires_only_for_shifted_subgroup(self):
        rng = np.random.default_rng(SEED)

        def detector(ref, new):
            obs, p = mmd_permutation_pvalue(ref, new, n_permutations=100, rng=rng)
            return obs, p

        subgroup_data = {
            "group_a": (rng.normal(size=(100, 1)), rng.normal(loc=2.0, size=(100, 1))),  # shifted
            "group_b": (rng.normal(size=(100, 1)), rng.normal(size=(100, 1))),            # stable
            "group_c": (rng.normal(size=(100, 1)), rng.normal(size=(100, 1))),            # stable
        }

        alerts = monitor_subgroups(
            subgroup_data, detector, drift_type=DriftType.COVARIATE, alpha=0.05
        )

        fired = {a.subgroup: a.fired for a in alerts}
        assert fired["group_a"] is True
        assert fired["group_b"] is False
        assert fired["group_c"] is False

    def test_silent_across_all_subgroups_when_stable(self):
        rng = np.random.default_rng(SEED)

        def detector(ref, new):
            obs, p = mmd_permutation_pvalue(ref, new, n_permutations=100, rng=rng)
            return obs, p

        subgroup_data = {
            f"group_{i}": (rng.normal(size=(80, 1)), rng.normal(size=(80, 1)))
            for i in range(5)
        }

        alerts = monitor_subgroups(
            subgroup_data, detector, drift_type=DriftType.COVARIATE, alpha=0.05
        )

        assert all(not a.fired for a in alerts)
