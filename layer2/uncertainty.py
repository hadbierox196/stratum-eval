"""
Aleatoric / epistemic uncertainty decomposition via deep ensembles.

Total uncertainty  = H[E_θ[p(y|x,θ)]]   (entropy of mean prediction)
Aleatoric          = E_θ[H[p(y|x,θ)]]   (mean entropy across members)
Epistemic          = Total − Aleatoric

For binary classification:
    H(p) = -p*log(p) - (1-p)*log(1-p)

Reference: Depeweg et al. (2018); Kendall & Gal (2017).
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from .ensemble import ensemble_predict_proba


EPS = 1e-8  # numerical stability


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Element-wise binary entropy. Input shape: (..., N)."""
    p = np.clip(p, EPS, 1 - EPS)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


@dataclass
class UncertaintyResult:
    """
    Structured container for per-sample uncertainty estimates.

    Attributes
    ----------
    total       : H[E_θ[p(y|x,θ)]]  — shape (N,)
    aleatoric   : E_θ[H[p(y|x,θ)]] — shape (N,)
    epistemic   : total - aleatoric  — shape (N,)
    mean_proba  : mean predicted probability — shape (N,)
    warnings    : any constraint violations
    """
    total: np.ndarray
    aleatoric: np.ndarray
    epistemic: np.ndarray
    mean_proba: np.ndarray
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Epistemic must be non-negative (numerical noise can push below 0)
        if np.any(self.epistemic < -1e-6):
            self.warnings.append(
                "epistemic_negative: numerical instability — clipping to 0"
            )
        self.epistemic = np.clip(self.epistemic, 0, None)


def decompose_uncertainty(
    sample_probs: np.ndarray,
) -> UncertaintyResult:
    """
    Decompose uncertainty from an ensemble of probability samples.

    Parameters
    ----------
    sample_probs : np.ndarray, shape (S, N)
        S = number of stochastic forward passes (members × MC samples).
        N = number of data points.

    Returns
    -------
    UncertaintyResult
    """
    if sample_probs.ndim != 2:
        raise ValueError(f"Expected shape (S, N), got {sample_probs.shape}")

    # Mean prediction across stochastic samples
    mean_p = sample_probs.mean(axis=0)  # (N,)

    # Total uncertainty: entropy of the mean
    total = binary_entropy(mean_p)  # (N,)

    # Aleatoric: mean of per-sample entropies
    per_sample_entropy = binary_entropy(sample_probs)  # (S, N)
    aleatoric = per_sample_entropy.mean(axis=0)        # (N,)

    # Epistemic: remainder
    epistemic = total - aleatoric  # (N,)

    return UncertaintyResult(
        total=total,
        aleatoric=aleatoric,
        epistemic=epistemic,
        mean_proba=mean_p,
    )


def compute_uncertainty_with_labeler_agreement(
    sample_probs: np.ndarray,
    labeler_agreement: Optional[np.ndarray] = None,
    agreement_threshold: float = 0.8,
) -> UncertaintyResult:
    """
    Decompose uncertainty and flag model-failure cases.

    Novel contribution: condition on labeler agreement.

    Interpretation matrix:
    ┌────────────────────┬──────────────────────────────────────────────────┐
    │ aleatoric / agree  │ Interpretation                                   │
    ├────────────────────┼──────────────────────────────────────────────────┤
    │ High  / High agree │ MODEL FAILURE — labelers agree but model unsure  │
    │ High  / Low agree  │ Genuine ambiguity — normatively contested        │
    │ Low   / High agree │ Model confident, labelers agree — nominal        │
    │ Low   / Low agree  │ Model confident despite labeler disagreement     │
    └────────────────────┴──────────────────────────────────────────────────┘

    Parameters
    ----------
    sample_probs        : shape (S, N)
    labeler_agreement   : shape (N,), values in [0, 1]. None = skip conditioning.
    agreement_threshold : labeler_agreement above this → "labelers agree"

    Returns
    -------
    UncertaintyResult with warnings for model-failure cases.
    """
    result = decompose_uncertainty(sample_probs)

    if labeler_agreement is not None:
        labeler_agreement = np.asarray(labeler_agreement)
        if labeler_agreement.shape != result.total.shape:
            raise ValueError(
                f"labeler_agreement shape {labeler_agreement.shape} "
                f"!= n_samples {result.total.shape}"
            )

        # Aleatoric high = above median
        aleatoric_threshold = np.median(result.aleatoric)
        high_aleatoric = result.aleatoric > aleatoric_threshold
        high_agreement = labeler_agreement >= agreement_threshold

        model_failure_mask = high_aleatoric & high_agreement
        n_failures = model_failure_mask.sum()

        if n_failures > 0:
            result.warnings.append(
                f"model_failure: {n_failures} samples have high aleatoric uncertainty "
                f"despite high labeler agreement (≥{agreement_threshold}). "
                "This is a model failure, not genuine ambiguity."
            )

        genuine_ambiguity_mask = high_aleatoric & ~high_agreement
        n_ambiguous = genuine_ambiguity_mask.sum()
        if n_ambiguous > 0:
            result.warnings.append(
                f"genuine_ambiguity: {n_ambiguous} samples have high aleatoric uncertainty "
                "with low labeler agreement — normatively contested cases."
            )

    return result
