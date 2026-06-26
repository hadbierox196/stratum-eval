"""
Unit tests for Layer 1 — SCI and invariance primitives.

Each test class maps to one of the six METRIC_STANDARDS.md spec fields:
  1. measures           — SCI detects known spurious signal
  2. cannot_measure     — SCI gives no signal on single environment
  3. assumptions        — assumption_violations populated correctly
  4. graceful_failure   — pathological inputs don't raise, return safe result
  5. falsifiability     — SCI = 0 on a cohort with zero spurious features
  6. complexity         — runtime stays sub-quadratic for realistic n
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from stratum.layer1.spurious_correlation import compute_sci, StratumMetricResult
from tests.fixtures.synthetic_cohort import make_synthetic_cohort


# ── helpers ─────────────────────────────────────────────────────────────────

def _make_invariant_only_cohort(seed: int = 0) -> tuple[dict, dict]:
    """Cohort where all features are invariant — expected SCI ≈ 0."""
    rng = np.random.default_rng(seed)
    reps, labs = {}, {}
    for env in ["env_0", "env_1", "env_2"]:
        X = rng.standard_normal((150, 3)).astype(np.float32)
        log_odds = X @ np.array([1.0, -0.5, 0.8])
        y = (1 / (1 + np.exp(-log_odds)) > 0.5).astype(np.int8)
        reps[env] = X
        labs[env] = y
    return reps, labs


def _make_single_env_cohort() -> tuple[dict, dict]:
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 4)).astype(np.float32)
    y = rng.integers(0, 2, size=100).astype(np.int8)
    return {"only_env": X}, {"only_env": y}


# ── 1. measures ─────────────────────────────────────────────────────────────

class TestSCIMeasures:
    """SCI correctly detects spurious signal in a known-structure cohort."""

    def test_sci_positive_on_spurious_cohort(self):
        cohort = make_synthetic_cohort(seed=42)
        result = compute_sci(cohort.representations, cohort.labels)

        assert isinstance(result, StratumMetricResult)
        assert result.value is not None
        assert result.value > 0.05, (
            f"Expected SCI > 0.05 on cohort with known spurious features, "
            f"got {result.value:.4f}"
        )

    def test_sci_range(self):
        cohort = make_synthetic_cohort(seed=42)
        result = compute_sci(cohort.representations, cohort.labels)
        assert result.value is not None
        assert 0.0 <= result.value <= 1.0


# ── 2. cannot_measure ───────────────────────────────────────────────────────

class TestSCICannotMeasure:
    """SCI returns no value (and correct violation) for single environment."""

    def test_single_env_returns_none_value(self):
        reps, labs = _make_single_env_cohort()
        result = compute_sci(reps, labs)
        assert result.value is None

    def test_single_env_violation_message(self):
        reps, labs = _make_single_env_cohort()
        result = compute_sci(reps, labs)
        assert any(
            "single_environment" in v for v in result.assumption_violations
        ), f"Expected single_environment violation, got: {result.assumption_violations}"

    def test_single_env_no_spurious_warning(self):
        """Should not emit a spurious-feature warning when invariance is untestable."""
        reps, labs = _make_single_env_cohort()
        result = compute_sci(reps, labs)
        assert not any("spurious" in w for w in result.warnings)


# ── 3. assumptions ──────────────────────────────────────────────────────────

class TestSCIAssumptions:
    """assumption_violations populated correctly across scenarios."""

    def test_no_violations_on_valid_multi_env(self):
        cohort = make_synthetic_cohort(seed=7)
        result = compute_sci(cohort.representations, cohort.labels)
        assert result.assumption_violations == []

    def test_single_env_violation_key(self):
        reps, labs = _make_single_env_cohort()
        result = compute_sci(reps, labs)
        assert len(result.assumption_violations) >= 1

    def test_zero_variance_violation(self):
        """Constant predictions → zero_predictive_variance violation."""
        rng = np.random.default_rng(3)
        # All features constant across samples → model will predict constant
        reps = {
            "e0": np.zeros((80, 4), dtype=np.float32),
            "e1": np.zeros((80, 4), dtype=np.float32),
        }
        labs = {
            "e0": rng.integers(0, 2, 80).astype(np.int8),
            "e1": rng.integers(0, 2, 80).astype(np.int8),
        }
        result = compute_sci(reps, labs)
        assert result.value is None
        assert any(
            "zero_predictive_variance" in v for v in result.assumption_violations
        )


# ── 4. graceful_failure ─────────────────────────────────────────────────────

class TestSCIGracefulFailure:
    """Pathological inputs must not raise exceptions."""

    def test_empty_arrays_do_not_raise(self):
        reps = {"e0": np.zeros((0, 4), dtype=np.float32),
                "e1": np.zeros((0, 4), dtype=np.float32)}
        labs = {"e0": np.array([], dtype=np.int8),
                "e1": np.array([], dtype=np.int8)}
        # May raise internally in sklearn — we just require no unhandled crash
        # at the compute_sci boundary; result may have assumption_violations.
        try:
            result = compute_sci(reps, labs)
            assert isinstance(result, StratumMetricResult)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"compute_sci raised unexpectedly: {exc}")

    def test_single_sample_per_env(self):
        reps = {"e0": np.array([[1.0, 0.5, -1.0, 0.2]], dtype=np.float32),
                "e1": np.array([[0.3, -0.2, 0.8, 1.1]], dtype=np.float32)}
        labs = {"e0": np.array([1], dtype=np.int8),
                "e1": np.array([0], dtype=np.int8)}
        result = compute_sci(reps, labs)
        assert isinstance(result, StratumMetricResult)

    def test_nan_in_representations_does_not_crash(self):
        rng = np.random.default_rng(9)
        X0 = rng.standard_normal((60, 4)).astype(np.float32)
        X0[5, 2] = np.nan
        reps = {"e0": X0,
                "e1": rng.standard_normal((60, 4)).astype(np.float32)}
        labs = {"e0": rng.integers(0, 2, 60).astype(np.int8),
                "e1": rng.integers(0, 2, 60).astype(np.int8)}
        # Graceful failure: may return None value but must not raise
        try:
            result = compute_sci(reps, labs)
            assert isinstance(result, StratumMetricResult)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"compute_sci raised on NaN input: {exc}")


# ── 5. falsifiability ───────────────────────────────────────────────────────

class TestSCIFalsifiability:
    """SCI ≈ 0 when no spurious signal exists."""

    def test_sci_near_zero_invariant_only_cohort(self):
        reps, labs = _make_invariant_only_cohort(seed=0)
        result = compute_sci(reps, labs)
        assert result.value is not None
        assert result.value < 0.15, (
            f"Expected SCI near 0 on invariant-only cohort, got {result.value:.4f}"
        )

    def test_sci_deterministic(self):
        """Same input → same output."""
        cohort = make_synthetic_cohort(seed=42)
        r1 = compute_sci(cohort.representations, cohort.labels, random_state=0)
        r2 = compute_sci(cohort.representations, cohort.labels, random_state=0)
        assert r1.value == r2.value


# ── 6. complexity ────────────────────────────────────────────────────────────

class TestSCIComplexity:
    """SCI runtime is acceptable for n=2000, d=50."""

    def test_runtime_under_10_seconds(self):
        rng = np.random.default_rng(99)
        n, d = 2000, 50
        reps, labs = {}, {}
        for e in ["e0", "e1", "e2"]:
            X = rng.standard_normal((n, d)).astype(np.float32)
            y = rng.integers(0, 2, n).astype(np.int8)
            reps[e] = X
            labs[e] = y

        start = time.perf_counter()
        result = compute_sci(reps, labs)
        elapsed = time.perf_counter() - start

        assert result.value is not None
        assert elapsed < 10.0, f"compute_sci took {elapsed:.2f}s on n=2000, d=50"
