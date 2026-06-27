"""Layer 1 — Group fairness: FNR / FPR parity."""
import pandas as pd
from stratum.layers.base import BaseLayer, StratumMetricResult

class Layer1(BaseLayer):
    LAYER_INDEX = 1

    def run(self) -> StratumMetricResult:
        df = self.dataset.df
        lc = self.dataset.label_col
        sc = self.dataset.score_col
        gc = self.dataset.group_col
        threshold = 0.5

        results = {}
        warnings = []
        for grp, gdf in df.groupby(gc):
            pred = (gdf[sc] >= threshold).astype(int)
            tp = ((pred == 1) & (gdf[lc] == 1)).sum()
            fn = ((pred == 0) & (gdf[lc] == 1)).sum()
            fp = ((pred == 1) & (gdf[lc] == 0)).sum()
            tn = ((pred == 0) & (gdf[lc] == 0)).sum()
            fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
            fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
            results[str(grp)] = {"fnr": fnr, "fpr": fpr, "n": len(gdf)}
            if len(gdf) < 30:
                warnings.append(f"Group '{grp}' has < 30 samples — estimates unreliable.")

        return StratumMetricResult(layer_index=1, metrics=results, warnings=warnings)
