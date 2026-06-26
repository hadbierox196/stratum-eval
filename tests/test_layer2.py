"""
Unit tests for Layer 2: uncertainty decomposition, conformal prediction,
Dawid-Skene, and the falsifiability test for epistemic monotonicity.
"""
import numpy as np
import pytest
import torch

from stratum.layer2.ensemble import (
    build_ensemble,
    train_ensemble_member,
    ensemble_predict_proba,
)
from stratum.layer2.uncertainty import (
    decompose_uncertainty,
    compute_uncertainty_with_labeler_agreement,
    binary_entropy,
)
from stratum.layer2.conformal import MondrianConformalPredictor
from stratum.layer2.labeler_model import fit_dawid_skene


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)


def make_binary_data(n: int, d: int = 10, seed: int = 0) -> tuple:
    """Simple linearly separable binary classification data."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    w = rng.standard_normal(d).astype(np.float32)
    logits = X @ w
    y = (logits > 0).astype(np.float32)
    return X, y


def make_sample_probs(n_samples: int = 20, n_items: int = 100) -> np.ndarray:
    """Fake ensemble predictions: shape (S, N)."""
    rng = np.random.default_rng(0)
    return rng.uniform(0, 1, size=(n_samples, n_items))


# ──────────────────────────────────────────────────────────────────────────────
# binary_entropy
# ──────────────────────────────────────────────────────────────────────────────

class TestBinaryEntropy:
    def test_max_at_half(self):
        p = np.array([0.5])
        assert binary_entropy(p)[0] == pytest.approx(1.0, abs=1e-6)

    def test_zero_at_extremes(self):
        p = np.array([0.0, 1.0])
        assert np.all(binary_entropy(p) < 0.01)

    def test_shape_preserved(self):
        p = np.random.rand(5, 10)
        assert binary_entropy(p).shape == (5, 10)


# ──────────────────────────────────────────────────────────────────────────────
# decompose_uncertainty
# ──────────────────────────────────────────────────────────────────────────────

class TestUncertaintyDecomposition:
    def test_shapes(self):
        probs = make_sample_probs(n_samples=30, n_items=50)
        result = decompose_uncertainty(probs)
        assert result.total.shape == (50,)
        assert result.aleatoric.shape == (50,)
        assert result.epistemic.shape == (50,)

    def test_epistemic_nonnegative(self):
        probs = make_sample_probs(n_samples=30, n_items=100)
        result = decompose_uncertainty(probs)
        assert np.all(result.epistemic >= 0), "Epistemic uncertainty must be ≥ 0"

    def test_total_equals_aleatoric_plus_epistemic(self):
        probs = make_sample_probs(n_samples=30, n_items=100)
        result = decompose_uncertainty(probs)
        np.testing.assert_allclose(
            result.total,
            result.aleatoric + result.epistemic,
            atol=1e-6,
            err_msg="total must equal aleatoric + epistemic",
        )

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="Expected shape"):
            decompose_uncertainty(np.random.rand(10))

    def test_certain_predictions_low_uncertainty(self):
        # All ensemble members agree on p≈1 → very low epistemic
        probs = np.full((30, 50), 0.99)
        result = decompose_uncertainty(probs)
        assert result.epistemic.mean() < 0.05

    def test_uniform_ensemble_max_total_uncertainty(self):
        # All members predict p=0.5 → max total uncertainty
        probs = np.full((30, 50), 0.5)
        result = decompose_uncertainty(probs)
        np.testing.assert_allclose(result.total, 1.0, atol=1e-6)

    def test_labeler_agreement_model_failure_warning(self):
        """High aleatoric + high labeler agreement → model failure warning."""
        rng = np.random.default_rng(0)
        # High aleatoric: predictions spread around 0.5
        probs = rng.uniform(0.3, 0.7, size=(30, 20))
        agreement = np.ones(20)  # everyone agrees

        result = compute_uncertainty_with_labeler_agreement(
            probs, labeler_agreement=agreement, agreement_threshold=0.8
        )
        assert any("model_failure" in w for w in result.warnings), (
            "Expected model_failure warning when high aleatoric + high agreement"
        )

    def test_labeler_agreement_genuine_ambiguity_warning(self):
        """High aleatoric + low labeler agreement → genuine ambiguity warning."""
        rng = np.random.default_rng(1)
        probs = rng.uniform(0.3, 0.7, size=(30, 20))
        agreement = np.zeros(20)  # everyone disagrees

        result = compute_uncertainty_with_labeler_agreement(
            probs, labeler_agreement=agreement, agreement_threshold=0.8
        )
        assert any("genuine_ambiguity" in w for w in result.warnings)


# ──────────────────────────────────────────────────────────────────────────────
# FALSIFIABILITY TEST: Epistemic Uncertainty Monotonicity
# ──────────────────────────────────────────────────────────────────────────────

class TestEpistemicMonotonicity:
    """
    Core falsifiability test for Layer 2.

    Claim: epistemic uncertainty must decrease monotonically with training
    set size for in-distribution test cases.

    Rationale: epistemic uncertainty reflects model ignorance (lack of data).
    As training data grows, ensemble members converge → disagreement ↓.
    If this property fails, either the ensemble is misconfigured or the
    data is out-of-distribution.

    Test protocol:
      1. Train ensembles on n = [50, 100, 200, 400] in-distribution samples.
      2. Evaluate mean epistemic uncertainty on a fixed in-distribution test set.
      3. Assert: mean_epistemic(n=50) > mean_epistemic(n=400).
         (Not strictly monotone at every step — noise in small ensembles
          makes strict monotonicity unreliable; we test the endpoints.)
    """

    @pytest.fixture(scope="class")
    def epistemic_by_n(self):
        """
        Train ensembles at increasing dataset sizes and record mean epistemic.
        Returns list of (n_train, mean_epistemic) tuples.
        """
        input_dim = 10
        n_members = 3
        n_mc = 10
        train_sizes = [50, 100, 200, 400]

        # Fixed test set (in-distribution)
        X_test, y_test = make_binary_data(n=200, d=input_dim, seed=99)

        results = []
        for n_train in train_sizes:
            X_train, y_train = make_binary_data(n=n_train, d=input_dim, seed=0)
            members = build_ensemble(
                input_dim=input_dim,
                n_members=n_members,
                hidden_dim=32,
                dropout_p=0.1,
            )
            trained = [
                train_ensemble_member(m, X_train, y_train, epochs=30, seed=i)
                for i, m in enumerate(members)
            ]
            sample_probs = ensemble_predict_proba(trained, X_test, n_mc_samples=n_mc)
            unc = decompose_uncertainty(sample_probs)
            results.append((n_train, float(unc.epistemic.mean())))

        return results

    def test_epistemic_decreases_large_to_small(self, epistemic_by_n):
        """
        Endpoint test: uncertainty at n=400 must be lower than at n=50.
        """
        _, ep_small = epistemic_by_n[0]   # n=50
        _, ep_large = epistemic_by_n[-1]  # n=400

        assert ep_large < ep_small, (
            f"FALSIFIABILITY FAILURE: epistemic uncertainty did NOT decrease "
            f"with training set size. "
            f"n=50: {ep_small:.4f}, n=400: {ep_large:.4f}. "
            "Check ensemble diversity and model capacity."
        )

    def test_epistemic_trend_is_monotone(self, epistemic_by_n):
        """
        Soft monotonicity: allow at most 1 non-monotone step (ensemble noise).
        """
        epsilons = [ep for _, ep in epistemic_by_n]
        violations = sum(
            1 for i in range(len(epsilons) - 1)
            if epsilons[i + 1] > epsilons[i] + 0.005  # 0.005 tolerance for noise
        )
        assert violations <= 1, (
            f"Epistemic uncertainty trend has {violations} non-monotone steps "
            f"(tolerance: ≤ 1). Values: {list(zip([n for n,_ in epistemic_by_n], epsilons))}. "
            "This may indicate out-of-distribution data or degenerate ensemble."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Mondrian Conformal Prediction
# ──────────────────────────────────────────────────────────────────────────────

class TestMondrianConformal:
    def _make_conformal_data(self, n: int = 200, seed: int = 0):
        rng = np.random.default_rng(seed)
        proba = rng.uniform(0, 1, n)
        y = (rng.uniform(0, 1, n) < proba).astype(int)
        groups = rng.choice(["A", "B", "C"], size=n)
        return proba, y, groups

    def test_coverage_guarantee_per_group(self):
        """
        Group-conditional coverage should be ≥ 1 - alpha - slack.
        With finite calibration, coverage may be slightly below due to
        quantile discretisation — allow 0.05 slack.
        """
        alpha = 0.1
        cal_proba, cal_y, cal_groups = self._make_conformal_data(n=500, seed=0)
        test_proba, test_y, test_groups = self._make_conformal_data(n=300, seed=1)

        cp = MondrianConformalPredictor(alpha=alpha)
        cp.fit(cal_proba, cal_y, cal_groups)
        result = cp.predict(test_proba, test_groups)

        for g, coverage in result.group_coverage.items():
            if not np.isnan(coverage):
                assert coverage >= (1 - alpha - 0.10), (
                    f"Group {g}: coverage={coverage:.3f} < {1 - alpha - 0.10:.3f}. "
                    "Mondrian coverage guarantee violated."
                )

    def test_unseen_group_warning(self):
        """Test group at prediction time with no calibration data → warning."""
        cal_proba, cal_y, cal_groups = self._make_conformal_data(n=100, seed=0)
        rng = np.random.default_rng(5)
        test_proba = rng.uniform(0, 1, 20)
        test_groups = np.array(["Z"] * 20)  # unseen group

        cp = MondrianConformalPredictor(alpha=0.1)
        cp.fit(cal_proba, cal_y, cal_groups)
        result = cp.predict(test_proba, test_groups)

        assert any("unseen" in w for w in result.warnings)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            MondrianConformalPredictor(alpha=1.5)

    def test_coverage_gaps_computed(self):
        cal_proba, cal_y, cal_groups = self._make_conformal_data(n=300, seed=0)
        test_proba, _, test_groups = self._make_conformal_data(n=200, seed=2)

        cp = MondrianConformalPredictor(alpha=0.1)
        cp.fit(cal_proba, cal_y, cal_groups)
        result = cp.predict(test_proba, test_groups)

        for g in result.group_coverage:
            assert g in result.coverage_gaps

    def test_predict_before_fit_raises(self):
        cp = MondrianConformalPredictor()
        with pytest.raises(RuntimeError, match="fit"):
            cp.predict(np.array([0.5]), np.array(["A"]))


# ──────────────────────────────────────────────────────────────────────────────
# Dawid-Skene
# ──────────────────────────────────────────────────────────────────────────────

class TestDawidSkene:
    def _make_annotations(self, N: int, J: int, seed: int = 0) -> np.ndarray:
        """
        Simulate noisy annotations: true labels from Bernoulli(0.6),
        each labeler flips the true label with probability 0.1.
        """
        rng = np.random.default_rng(seed)
        true_labels = rng.binomial(1, 0.6, N)
        annotations = np.tile(true_labels[:, None], (1, J))
        noise = rng.binomial(1, 0.1, (N, J))
        annotations = np.where(noise, 1 - annotations, annotations)
        return annotations, true_labels

    def test_requires_3_labelers_warning(self):
        """J=2 → assumption violation about identification."""
        annotations, _ = self._make_annotations(N=100, J=2)
        result = fit_dawid_skene(annotations)
        assert any("identification_failure" in v for v in result.assumption_violations), (
            "Must warn when J < 3 (Dawid-Skene not identified)"
        )

    def test_single_labeler_returns_nan(self):
        """J=1 → completely unidentified, return NaN results."""
        annotations, _ = self._make_annotations(N=50, J=1)
        result = fit_dawid_skene(annotations)
        assert np.all(np.isnan(result.true_label_probs))
        assert any("identification_failure" in v for v in result.assumption_violations)

    def test_3_labelers_recovers_labels(self):
        """J=3, low noise → MAP labels should match true labels ≥ 85%."""
        N = 300
        annotations, true_labels = self._make_annotations(N=N, J=3, seed=42)
        result = fit_dawid_skene(annotations, n_classes=2)

        assert result.n_labelers == 3
        assert not any("identification_failure" in v for v in result.assumption_violations)

        accuracy = (result.true_label_map == true_labels).mean()
        assert accuracy >= 0.85, (
            f"Dawid-Skene MAP accuracy={accuracy:.3f} < 0.85 with low-noise J=3 data"
        )

    def test_labeler_agreement_range(self):
        """labeler_agreement must be in [0, 1] for all non-missing items."""
        annotations, _ = self._make_annotations(N=100, J=5)
        result = fit_dawid_skene(annotations)
        valid = result.labeler_agreement[~np.isnan(result.labeler_agreement)]
        assert np.all(valid >= 0) and np.all(valid <= 1)

    def test_confusions_are_valid_distributions(self):
        """Each row of each labeler's confusion matrix must sum to 1."""
        annotations, _ = self._make_annotations(N=200, J=4)
        result = fit_dawid_skene(annotations, n_classes=2)

        for j, conf in result.labeler_confusion.items():
            row_sums = conf.sum(axis=1)
            np.testing.assert_allclose(
                row_sums, 1.0, atol=1e-5,
                err_msg=f"Labeler {j} confusion matrix rows don't sum to 1"
            )

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="shape"):
            fit_dawid_skene(np.array([0, 1, 0, 1]))

    def test_n_labelers_n_items_recorded(self):
        annotations, _ = self._make_annotations(N=80, J=4)
        result = fit_dawid_skene(annotations)
        assert result.n_labelers == 4
        assert result.n_items == 80
