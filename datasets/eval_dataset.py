"""Core dataset container for stratum-eval evaluations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class EvalDataset:
    """Container for a stratum-eval evaluation dataset.

    Attributes
    ----------
    y_true:
        Ground-truth labels. Shape (n,) for binary/multiclass,
        (n, c) for multilabel.
    y_pred:
        Model predictions (probabilities or logits). Same shape as y_true.
    metadata:
        DataFrame of per-instance metadata (site, age_group, annotator, etc.).
        Must have the same index as y_true.
    name:
        Human-readable dataset identifier.
    task:
        One of: "binary", "multiclass", "multilabel", "regression".
    prevalence:
        Positive class prevalence in this split. Computed on construction
        for binary tasks; None otherwise.
    """

    y_true: NDArray[np.floating[Any]]
    y_pred: NDArray[np.floating[Any]]
    metadata: pd.DataFrame = field(default_factory=pd.DataFrame)
    name: str = "unnamed"
    task: str = "binary"
    prevalence: float | None = None

    def __post_init__(self) -> None:
        if self.y_true.shape[0] != self.y_pred.shape[0]:
            raise ValueError(
                f"y_true has {self.y_true.shape[0]} rows but "
                f"y_pred has {self.y_pred.shape[0]} rows."
            )
        if self.task == "binary" and self.prevalence is None:
            self.prevalence = float(np.mean(self.y_true))

    @property
    def n(self) -> int:
        """Number of evaluation instances."""
        return int(self.y_true.shape[0])

    def __repr__(self) -> str:
        prev_str = (
            f", prevalence={self.prevalence:.3f}"
            if self.prevalence is not None
            else ""
        )
        return (
            f"EvalDataset(name={self.name!r}, task={self.task!r}, "
            f"n={self.n}{prev_str})"
        )
