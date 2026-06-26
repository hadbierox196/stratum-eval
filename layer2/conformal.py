"""
Mondrian (group-conditional) conformal prediction for binary classification.

Mondrian conformal prediction provides per-subgroup coverage guarantees:
    P(Y ∈ C(X) | group(X) = g) ≥ 1 − α  for each group g.

This is strictly stronger than marginal coverage:
    P(Y ∈ C(X)) ≥ 1 − α  (achieved by standard split conformal).

Reference:
    Venn-Abers / Mondrian: Shafer & Vovk (2008); Johansson et al. (2018).
    Exchangeability assumption: calibration set must be i.i.d. from same
    distribution as test set, WITHIN each Mondrian category (group).
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class MondrianCoverageResult:
    """
    Per-group prediction sets and coverage diagnostics.

    Attributes
    ----------
    prediction_sets  : dict mapping group → boolean mask (N_test,) — True if y=1 included
    group_coverage   : dict mapping group → empirical coverage on calibration set
    group_thresholds : dict mapping group → conformal threshold q̂
    coverage_gaps    : dict mapping group → |empirical_coverage - (1-alpha)|
    warnings         : list of constraint violations
    """
    prediction_sets: Dict[str, np.ndarray]
    group_coverage: Dict[str, float]
    group_thresholds: Dict[str, float]
    coverage_gaps: Dict[str, float]
    warnings: List[str] = field(default_factory=list)


class MondrianConformalPredictor:
    """
    Split Mondrian conformal predictor for binary classification.

    Usage
    -----
    1. Fit on a held-out calibration set with group labels.
    2. Predict on test set; receive group-conditional prediction sets.

    The nonconformity score for binary classification:
        s(x, y) = 1 − p̂(y | x)

    So for label y=1:  score = 1 − p̂(1|x)
       for label y=0:  score = p̂(1|x)

    The threshold q̂_g for group g is the ⌈(n_g + 1)(1 − α)⌉ / n_g quantile
    of calibration nonconformity scores within group g.

    Assumption (Mondrian exchangeability): within each group g, calibration
    and test samples are exchangeable. Violated if groups shift between
    calibration and deployment — flag this explicitly.
    """

    def __init__(self, alpha: float = 0.1):
        """
        Parameters
        ----------
        alpha : float
            Miscoverage level. Coverage guarantee: 1 − alpha per group.
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        self.alpha = alpha
        self.group_thresholds_: Dict[str, float] = {}
        self.group_n_cal_: Dict[str, int] = {}
        self._fitted = False

    def _nonconformity_score(self, proba: np.ndarray, y: np.ndarray) -> np.ndarray:
        """s(x, y) = 1 − p̂(y|x)."""
        scores = np.where(y == 1, 1 - proba, proba)
        return scores

    def fit(
        self,
        cal_proba: np.ndarray,
        cal_y: np.ndarray,
        cal_groups: np.ndarray,
    ) -> "MondrianConformalPredictor":
        """
        Fit per-group conformal thresholds on calibration data.

        Parameters
        ----------
        cal_proba  : shape (N_cal,) — P(y=1|x) from base model
        cal_y      : shape (N_cal,) — true binary labels
        cal_groups : shape (N_cal,) — group identifier per sample
        """
        cal_proba = np.asarray(cal_proba, dtype=float)
        cal_y = np.asarray(cal_y, dtype=int)
        cal_groups = np.asarray(cal_groups)

        groups = np.unique(cal_groups)
        warnings = []

        for g in groups:
            mask = cal_groups == g
            n_g = mask.sum()
            if n_g < 10:
                warnings.append(
                    f"group_{g}_small_calibration: n={n_g} < 10. "
                    "Coverage guarantee may be loose."
                )

            scores_g = self._nonconformity_score(cal_proba[mask], cal_y[mask])

            # Conformal quantile: ceil((n+1)(1-alpha))/n  — finite-sample valid
            level = np.ceil((n_g + 1) * (1 - self.alpha)) / n_g
            level = min(level, 1.0)
            self.group_thresholds_[str(g)] = float(np.quantile(scores_g, level))
            self.group_n_cal_[str(g)] = n_g

        self._fitted = True
        self._fit_warnings = warnings
        return self

    def predict(
        self,
        test_proba: np.ndarray,
        test_groups: np.ndarray,
    ) -> MondrianCoverageResult:
        """
        Produce group-conditional prediction sets for test samples.

        A sample's prediction set includes label y=1 iff its nonconformity
        score for y=1 is ≤ the group threshold (and similarly for y=0).

        For binary problems, we report only whether y=1 is in the set.

        Parameters
        ----------
        test_proba  : shape (N_test,) — P(y=1|x) from base model
        test_groups : shape (N_test,) — group identifier per sample

        Returns
        -------
        MondrianCoverageResult
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict()")

        test_proba = np.asarray(test_proba, dtype=float)
        test_groups = np.asarray(test_groups)

        groups = np.unique(test_groups)
        prediction_sets: Dict[str, np.ndarray] = {}
        group_coverage: Dict[str, float] = {}
        group_thresholds: Dict[str, float] = {}
        coverage_gaps: Dict[str, float] = {}
        warnings = list(self._fit_warnings)

        for g in groups:
            g_str = str(g)
            mask = test_groups == g

            if g_str not in self.group_thresholds_:
                warnings.append(
                    f"group_{g}_unseen: no calibration data for this group. "
                    "Mondrian exchangeability violated — predictions unreliable."
                )
                prediction_sets[g_str] = np.ones(mask.sum(), dtype=bool)
                group_coverage[g_str] = float("nan")
                group_thresholds[g_str] = float("nan")
                coverage_gaps[g_str] = float("nan")
                continue

            q_hat = self.group_thresholds_[g_str]
            proba_g = test_proba[mask]

            # Score for y=1: 1 - proba
            score_y1 = 1 - proba_g
            in_set = score_y1 <= q_hat  # y=1 included in prediction set

            prediction_sets[g_str] = in_set
            group_thresholds[g_str] = q_hat

            # Empirical coverage = fraction of test samples where y=1 included
            group_coverage[g_str] = float(in_set.mean())
            coverage_gaps[g_str] = abs(group_coverage[g_str] - (1 - self.alpha))

        return MondrianCoverageResult(
            prediction_sets=prediction_sets,
            group_coverage=group_coverage,
            group_thresholds=group_thresholds,
            coverage_gaps=coverage_gaps,
            warnings=warnings,
        )
