"""Layer 2 — Calibration: ECE per group."""
import numpy as np
import pandas as pd
from stratum.layers.base import BaseLayer, StratumMetricResult

def _ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)

class Layer2(BaseLayer):
    LAYER_INDEX = 2

    def run(self) -> StratumMetricResult:
        df = self.dataset.df
        lc = self.dataset.label_col
        sc = self.dataset.score_col
        gc = self.dataset.group_col

        results = {}
        warnings = []
        for grp, gdf in df.groupby(gc):
            ece = _ece(gdf[lc].values, gdf[sc].values)
            results[str(grp)] = {"ece": ece, "n": len(gdf)}

        return StratumMetricResult(layer_index=2, metrics=results, warnings=warnings)
