"""Layer 3 — Discrimination: AUROC per group."""
import numpy as np
from sklearn.metrics import roc_auc_score
from stratum.layers.base import BaseLayer, StratumMetricResult

class Layer3(BaseLayer):
    LAYER_INDEX = 3

    def run(self) -> StratumMetricResult:
        df = self.dataset.df
        lc = self.dataset.label_col
        sc = self.dataset.score_col
        gc = self.dataset.group_col

        results = {}
        warnings = []
        for grp, gdf in df.groupby(gc):
            if gdf[lc].nunique() < 2:
                warnings.append(f"Group '{grp}': only one class present — AUROC undefined.")
                results[str(grp)] = {"auroc": float("nan"), "n": len(gdf)}
                continue
            auroc = roc_auc_score(gdf[lc].values, gdf[sc].values)
            results[str(grp)] = {"auroc": float(auroc), "n": len(gdf)}

        return StratumMetricResult(layer_index=3, metrics=results, warnings=warnings)
