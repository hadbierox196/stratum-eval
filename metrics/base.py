"""Base class that every stratum-eval metric must inherit from."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


class MetricUndefinedError(ValueError):
    """Raised when a metric cannot be computed for the given inputs.

    This is the canonical error type for pathological inputs.  Callers
    should catch this specifically rather than bare ValueError so that
    pipeline code can distinguish metric failures from data errors.
    """


@dataclass(frozen=True)
class MetricResult:
    """Immutable container returned by every metric.

    Attributes
    ----------
    name:
        Canonical metric name (matches class name).
    value:
        Scalar result, or NaN when the metric is undefined.
    confidence_interval:
        (lower, upper) tuple if bootstrapping was requested, else None.
    n_samples:
        Number of instances used in the computation.
    warnings:
        Human-readable list of non-fatal issues detected during computation.
    metadata:
        Arbitrary extra fields (e.g., threshold used, prevalence seen).
    """

    name: str
    value: float
    confidence_interval: tuple[float, float] | None = None
    n_samples: int = 0
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_defined(self) -> bool:
        """Return True iff the metric value is a real number."""
        return not np.isnan(self.value)


class BaseMetric(ABC):
    """Abstract base for all stratum-eval metrics.

    Subclasses must implement:
        - validate_inputs()
        - compute()

    And must declare the six METRIC_STANDARDS fields as class-level
    docstring sections. The ``stratum-eval metric lint`` command checks
    for their presence.
    """

    #: Minimum number of instances required for the metric to be valid.
    #: Subclasses must override this.
    MINIMUM_SAMPLE_SIZE: int = 2

    def __call__(
        self,
        y_true: NDArray[np.floating[Any]],
        y_pred: NDArray[np.floating[Any]],
        **kwargs: Any,
    ) -> MetricResult:
        """Validate then compute. Never override this method."""
        self.validate_inputs(y_true, y_pred)
        return self.compute(y_true, y_pred, **kwargs)

    @abstractmethod
    def validate_inputs(
        self,
        y_true: NDArray[np.floating[Any]],
        y_pred: NDArray[np.floating[Any]],
    ) -> None:
        """Check all pathological input conditions.

        Must cover at minimum:
        - Empty arrays
        - NaN or Inf in either array
        - Single-class y_true
        - n < MINIMUM_SAMPLE_SIZE

        Raise MetricUndefinedError for unrecoverable conditions.
        Append to self._warnings for recoverable conditions.
        """
        ...

    @abstractmethod
    def compute(
        self,
        y_true: NDArray[np.floating[Any]],
        y_pred: NDArray[np.floating[Any]],
        **kwargs: Any,
    ) -> MetricResult:
        """Compute and return the metric result."""
        ...

    # ------------------------------------------------------------------
    # Shared validation helpers available to all subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _check_empty(
        y_true: NDArray[Any], y_pred: NDArray[Any]
    ) -> None:
        if len(y_true) == 0 or len(y_pred) == 0:
            raise MetricUndefinedError(
                "Cannot compute metric on empty arrays. "
                "Received y_true length={len(y_true)}, y_pred length={len(y_pred)}."
            )

    @staticmethod
    def _check_nan_inf(
        y_true: NDArray[Any], y_pred: NDArray[Any]
    ) -> None:
        for name, arr in (("y_true", y_true), ("y_pred", y_pred)):
            if np.any(~np.isfinite(arr)):
                raise MetricUndefinedError(
                    f"{name} contains NaN or Inf values. "
                    "Impute or filter before calling this metric."
                )

    @staticmethod
    def _check_single_class(y_true: NDArray[Any]) -> None:
        if len(np.unique(y_true)) < 2:
            raise MetricUndefinedError(
                "y_true contains only a single class. "
                "This metric requires both positive and negative instances."
            )

    def _check_min_samples(self, y_true: NDArray[Any]) -> None:
        if len(y_true) < self.MINIMUM_SAMPLE_SIZE:
            raise MetricUndefinedError(
                f"Received {len(y_true)} samples but this metric requires "
                f"at least {self.MINIMUM_SAMPLE_SIZE}."
            )
