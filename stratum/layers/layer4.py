"""Layer 4 — Intersectional disparity: FNR at group intersections (n >= 30)."""
import numpy as np
import pandas as pd
from stratum.layers.base import BaseLayer, StratumMetricResult

class Layer4(BaseLayer):
    LAYER_INDEX = 4

    def run(self) -> StratumMetricResult:
        df = self.dataset.df.copy()
        lc = self.dataset.label_col
        sc = self.dataset.score_col
        gc = self.dataset.group_col
        threshold = 0.5

        # Build a second axis via score quartile if no explicit intersect_col
        df["_score_quartile"] = pd.qcut(df[sc], q=4, labels=["Q1","Q2","Q3","Q4"])
        pred = (df[sc] >= threshold).astype(int)

        results = {}
        warnings = []
        for (grp, qrt), gdf in df.groupby([gc, "_score_quartile"], observed=True):
            if len(gdf) < 30:
                warnings.append(
                    f"Intersection ({grp}, {qrt}) has < 30 samples — skipped."
                )
                continue
            fn = ((pred[gdf.index] == 0) & (gdf[lc] == 1)).sum()
            tp = ((pred[gdf.index] == 1) & (gdf[lc] == 1)).sum()
            fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
            key = f"{grp}_{qrt}"
            results[key] = {"intersectional_fnr": float(fnr), "n": len(gdf)}

        return StratumMetricResult(layer_index=4, metrics=results, warnings=warnings)
