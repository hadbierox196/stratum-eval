"""
Dawid-Skene latent label model for crowdsourced annotations.

Identification conditions (REQUIRED ≥ 3 labelers):
  - With 2 labelers: the model is not identified — you cannot separately
    estimate labeler accuracy and true label prevalence without additional
    assumptions. Warn explicitly, do not silently proceed.
  - With ≥ 3 labelers: identified under local independence assumption:
    P(annotations | true_label) = ∏_j P(annotation_j | true_label, labeler_j)

Outputs:
  - Latent true labels (MAP estimates)
  - Per-labeler confusion matrices (sensitivity, specificity)
  - Per-item labeler agreement score (used by uncertainty.py)

Reference:
    Dawid & Skene (1979). Maximum likelihood estimation of observer
    error-rates using the EM algorithm. JRSS-C 28(1): 20–28.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict


@dataclass
class DawidSkeneResult:
    """
    Output from Dawid-Skene EM estimation.

    Attributes
    ----------
    true_label_probs  : shape (N, K) — posterior P(true_label=k | annotations)
    true_label_map    : shape (N,)   — MAP estimate of true label
    labeler_confusion : dict mapping labeler_id → confusion matrix (K, K)
                        entry [k, j] = P(annotated j | true label k)
    labeler_agreement : shape (N,)   — fraction of labelers agreeing with MAP label
    n_labelers        : int
    n_items           : int
    n_classes         : int
    converged         : bool
    warnings          : list of str
    assumption_violations : list of str
    """
    true_label_probs: np.ndarray
    true_label_map: np.ndarray
    labeler_confusion: Dict[int, np.ndarray]
    labeler_agreement: np.ndarray
    n_labelers: int
    n_items: int
    n_classes: int
    converged: bool
    warnings: List[str] = field(default_factory=list)
    assumption_violations: List[str] = field(default_factory=list)


def fit_dawid_skene(
    annotations: np.ndarray,
    n_classes: int = 2,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> DawidSkeneResult:
    """
    Fit the Dawid-Skene model via EM.

    Parameters
    ----------
    annotations : np.ndarray, shape (N, J)
        N items × J labelers. Missing annotations encoded as -1.
        Values in {0, 1, ..., K-1} for K classes. -1 = no annotation.
    n_classes   : int, number of label classes (default 2 for binary)
    max_iter    : int, maximum EM iterations
    tol         : float, convergence tolerance on log-likelihood

    Returns
    -------
    DawidSkeneResult

    Raises
    ------
    ValueError if annotations is not 2D.
    """
    if annotations.ndim != 2:
        raise ValueError(
            f"annotations must be shape (N, J), got {annotations.shape}"
        )

    N, J = annotations.shape
    K = n_classes
    warnings = []
    assumption_violations = []

    # ── Identification check ──────────────────────────────────────────────────
    if J < 3:
        assumption_violations.append(
            f"identification_failure: Dawid-Skene requires ≥ 3 labelers for "
            f"identification; got J={J}. Labeler accuracy and true label "
            "prevalence are not separately estimable. Results are unreliable."
        )
        if J < 2:
            # Cannot even run EM meaningfully
            return DawidSkeneResult(
                true_label_probs=np.full((N, K), np.nan),
                true_label_map=np.full(N, -1, dtype=int),
                labeler_confusion={},
                labeler_agreement=np.full(N, np.nan),
                n_labelers=J,
                n_items=N,
                n_classes=K,
                converged=False,
                warnings=warnings,
                assumption_violations=assumption_violations,
            )

    # ── Initialise T: posterior over true labels ──────────────────────────────
    # Majority vote initialisation
    T = np.zeros((N, K))  # T[i, k] = P(true_label_i = k)
    for i in range(N):
        obs = annotations[i][annotations[i] >= 0]  # remove missing
        if len(obs) == 0:
            T[i] = np.ones(K) / K  # uniform if no annotations
        else:
            for k in range(K):
                T[i, k] = (obs == k).sum() / len(obs)

    # ── EM ────────────────────────────────────────────────────────────────────
    prev_ll = -np.inf
    converged = False
    pi = np.zeros(K)          # class priors
    error_rates = np.zeros((J, K, K))  # [j, k, l] = P(annotated l | true k, labeler j)

    for iteration in range(max_iter):
        # ── M-step ──────────────────────────────────────────────────────────
        # Class priors
        pi = T.mean(axis=0) + 1e-8
        pi /= pi.sum()

        # Error rates: for each labeler j, class k, annotation class l
        error_rates = np.zeros((J, K, K))
        for j in range(J):
            for k in range(K):
                for l in range(K):
                    mask = annotations[:, j] == l  # items where labeler j said l
                    error_rates[j, k, l] = T[mask, k].sum() + 1e-8
                # Normalise row: ∑_l P(annotated l | true k, j) = 1
                error_rates[j, k] /= error_rates[j, k].sum()

        # ── E-step ──────────────────────────────────────────────────────────
        log_T = np.zeros((N, K))
        for k in range(K):
            log_T[:, k] = np.log(pi[k] + 1e-8)
            for j in range(J):
                obs_j = annotations[:, j]
                for i in range(N):
                    l = obs_j[i]
                    if l >= 0:  # not missing
                        log_T[i, k] += np.log(error_rates[j, k, l] + 1e-8)

        # Normalise in log-space for numerical stability
        log_T -= log_T.max(axis=1, keepdims=True)
        T = np.exp(log_T)
        T /= T.sum(axis=1, keepdims=True)

        # ── Log-likelihood ───────────────────────────────────────────────────
        ll = log_T.max(axis=1).sum()  # approximate
        if abs(ll - prev_ll) < tol:
            converged = True
            break
        prev_ll = ll

    if not converged:
        warnings.append(
            f"em_not_converged: EM did not converge in {max_iter} iterations. "
            "Results may be unstable. Consider increasing max_iter."
        )

    # ── Extract results ───────────────────────────────────────────────────────
    true_label_map = T.argmax(axis=1)

    # Per-item labeler agreement: fraction of present annotations matching MAP
    labeler_agreement = np.zeros(N)
    for i in range(N):
        obs = annotations[i][annotations[i] >= 0]
        if len(obs) == 0:
            labeler_agreement[i] = np.nan
        else:
            labeler_agreement[i] = (obs == true_label_map[i]).mean()

    labeler_confusion = {j: error_rates[j] for j in range(J)}

    # Check local independence warning
    if J < 5:
        warnings.append(
            "local_independence_untestable: local independence assumption "
            f"(P(ann_j | true_k) independent across labelers) cannot be tested "
            f"with J={J} labelers. Treat results with caution."
        )

    return DawidSkeneResult(
        true_label_probs=T,
        true_label_map=true_label_map,
        labeler_confusion=labeler_confusion,
        labeler_agreement=labeler_agreement,
        n_labelers=J,
        n_items=N,
        n_classes=K,
        converged=converged,
        warnings=warnings,
        assumption_violations=assumption_violations,
    )
