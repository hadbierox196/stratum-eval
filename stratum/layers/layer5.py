"""Layer 5 — Temporal robustness: AUROC drift across chronological quartiles."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from stratum.layers.base import BaseLayer, StratumMetricResult

class Layer5(BaseLayer):
    LAYER_INDEX = 5

    def run(self) -> StratumMetricResult:
        df = self.dataset.df.copy()
        lc = self.dataset.label_col
        sc = self.dataset.score_col
        tc = self.dataset.time_col

        warnings = []
        results  = {}

        if tc is None or tc not in df.columns:
            warnings.append(
                "No time_col provided — Layer 5 cannot compute temporal robustness. "
                "Pass time_col= when constructing EvalDataset."
            )
            return StratumMetricResult(layer_index=5, metrics=results, warnings=warnings)

        df["_tquartile"] = pd.qcut(df[tc], q=4, labels=["T1","T2","T3","T4"], duplicates="drop")
        aurocs = {}
        for qrt, gdf in df.groupby("_tquartile", observed=True):
            if gdf[lc].nunique() < 2:
                warnings.append(f"Quartile {qrt}: single class — AUROC undefined.")
                aurocs[str(qrt)] = float("nan")
                continue
            aurocs[str(qrt)] = float(roc_auc_score(gdf[lc].values, gdf[sc].values))

        valid = [v for v in aurocs.values() if not np.isnan(v)]
        drift = (max(valid) - min(valid)) if len(valid) >= 2 else float("nan")

        results["auroc_by_quartile"]  = aurocs
        results["auroc_drift"]        = drift
        results["bootstrap_ci_width"] = float("nan")   # placeholder until bootstrap added

        return StratumMetricResult(layer_index=5, metrics=results, warnings=warnings)
