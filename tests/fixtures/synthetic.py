"""Synthetic data generators for CI tests.

All functions are deterministic given a seed. No patient data is used anywhere
in this repository. These fixtures exist solely to give CI something to run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stratum_eval.datasets.eval_dataset import EvalDataset


def make_binary_dataset(
    n: int = 200,
    prevalence: float = 0.3,
    signal: float = 0.8,
    seed: int = 42,
    name: str = "synthetic_binary",
) -> EvalDataset:
    """Generate a synthetic binary classification eval dataset.

    Parameters
    ----------
    n:
        Number of instances.
    prevalence:
        Fraction of positive instances.
    signal:
        How informative the predictions are (0 = random, 1 = perfect).
    seed:
        Random seed for reproducibility.
    name:
        Dataset identifier.

    Returns
    -------
    EvalDataset
        Ready for use with any stratum-eval metric.
    """
    rng = np.random.default_rng(seed)
    y_true = (rng.uniform(size=n) < prevalence).astype(float)
    noise = rng.uniform(size=n)
    y_pred = signal * y_true + (1 - signal) * noise
    y_pred = np.clip(y_pred, 0, 1)

    metadata = pd.DataFrame(
        {
            "site": rng.choice(["site_A", "site_B", "site_C"], size=n),
            "age_group": rng.choice(["18-40", "41-65", "65+"], size=n),
            "annotator": rng.choice(["ann_1", "ann_2"], size=n),
        }
    )

    return EvalDataset(
        y_true=y_true,
        y_pred=y_pred,
        metadata=metadata,
        name=name,
        task="binary",
    )
