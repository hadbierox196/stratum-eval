from __future__ import annotations
from typing import Optional
from stratum.layers.base import StratumMetricResult

class StratumReport:
    def __init__(self, results: list[StratumMetricResult], nsd, dataset_metadata: dict):
        self.nsd              = nsd
        self.dataset_metadata = dataset_metadata
        self._results         = {r.layer_index: r for r in results}

        # Named attribute access (report.layer1, report.layer2, …)
        self.layer1: Optional[StratumMetricResult] = self._results.get(1)
        self.layer2: Optional[StratumMetricResult] = self._results.get(2)
        self.layer3: Optional[StratumMetricResult] = self._results.get(3)
        self.layer4: Optional[StratumMetricResult] = self._results.get(4)
        self.layer5: Optional[StratumMetricResult] = self._results.get(5)

    def summary(self) -> str:
        lines = ["=== StratumReport Summary ==="]
        for idx, result in sorted(self._results.items()):
            lines.append(f"  Layer {idx}: {result}")
            if result.warnings:
                for w in result.warnings:
                    lines.append(f"    ⚠  {w}")
        return "\n".join(lines)

    def __repr__(self):
        return f"StratumReport(layers={sorted(self._results.keys())})"
