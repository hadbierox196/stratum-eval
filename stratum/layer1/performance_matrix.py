"""
Performance matrix: task × environment × subgroup.

The matrix is a pandas-free implementation (numpy + plain dicts) to stay
within the Layer 1 dependency budget.  The public API returns a dict-of-dicts
that callers can trivially wrap in a DataFrame if pandas is available.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


MetricFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class PerformanceMatrix:
    """
    Stores per-(task, environment, subgroup) scalar metric values.

    Internal storage: self._data[(task, env, subgroup)] = float
    """
    _data: dict[tuple[str, str, str], float] = field(
        default_factory=dict, init=False, repr=False
    )
    _tasks: list[str] = field(default_factory=list, init=False)
    _envs: list[str] = field(default_factory=list, init=False)
    _subgroups: list[str] = field(default_factory=list, init=False)

    def add(
        self,
        task: str,
        env: str,
        subgroup: str,
        value: float,
    ) -> None:
        self._data[(task, env, subgroup)] = value
        if task not in self._tasks:
            self._tasks.append(task)
        if env not in self._envs:
            self._envs.append(env)
        if subgroup not in self._subgroups:
            self._subgroups.append(subgroup)

    def get(self, task: str, env: str, subgroup: str) -> float | None:
        return self._data.get((task, env, subgroup))

    # ── Invariance gap ──────────────────────────────────────────────────────

    def invariance_gap(self, task: str, subgroup: str) -> float | None:
        """
        max(diagonal) - min(off-diagonal) for a fixed task and subgroup.

        'Diagonal' = performance when the training environment == evaluation
        environment (convention: env name ends with '_train' or is the first
        env seen for this task).

        In practice for Layer 1 we define:
            diagonal   = max metric across environments for (task, subgroup)
            off-diag   = min metric across environments for (task, subgroup)

        This matches the notation in the paper draft: a large gap means the
        model performs well in some environments and poorly in others, which
        is the signature of spurious correlation.

        Returns None if fewer than 2 environments are populated for this cell.
        """
        values = [
            self._data[(task, env, subgroup)]
            for env in self._envs
            if (task, env, subgroup) in self._data
        ]
        if len(values) < 2:
            return None
        return float(max(values) - min(values))

    def invariance_gap_matrix(
        self
    ) -> dict[tuple[str, str], float | None]:
        """Return invariance gap for every (task, subgroup) pair."""
        return {
            (task, sg): self.invariance_gap(task, sg)
            for task in self._tasks
            for sg in self._subgroups
        }

    # ── Convenience ─────────────────────────────────────────────────────────

    def to_nested_dict(self) -> dict:
        """task → env → subgroup → value."""
        out: dict = {}
        for (task, env, sg), val in self._data.items():
            out.setdefault(task, {}).setdefault(env, {})[sg] = val
        return out

    def tasks(self) -> list[str]:
        return list(self._tasks)

    def envs(self) -> list[str]:
        return list(self._envs)

    def subgroups(self) -> list[str]:
        return list(self._subgroups)


def build_performance_matrix(
    *,
    predictions: dict[str, dict[str, np.ndarray]],
    ground_truth: dict[str, dict[str, np.ndarray]],
    subgroup_labels: dict[str, dict[str, np.ndarray]],
    metric_fns: dict[str, MetricFn],
    task_name: str = "default",
) -> PerformanceMatrix:
    """
    Populate a PerformanceMatrix from raw predictions.

    Args:
        predictions:     env → subgroup → array of predicted scores/labels.
        ground_truth:    env → subgroup → array of true labels.
        subgroup_labels: env → subgroup → boolean mask (not used directly
                         here but kept for API symmetry; callers pre-split).
        metric_fns:      name → callable(y_true, y_pred) → float.
        task_name:       string label written into the matrix task axis.

    Returns:
        Populated PerformanceMatrix.
    """
    pm = PerformanceMatrix()
    for env, sg_preds in predictions.items():
        for sg, y_pred in sg_preds.items():
            y_true = ground_truth[env][sg]
            for metric_name, fn in metric_fns.items():
                composite_task = f"{task_name}/{metric_name}"
                val = fn(y_true, y_pred)
                pm.add(composite_task, env, sg, val)
    return pm
