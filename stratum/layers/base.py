from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class StratumMetricResult:
    layer_index: int
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __repr__(self):
        return f"Layer{self.layer_index}Result(metrics={list(self.metrics.keys())})"

class BaseLayer:
    LAYER_INDEX: int = 0

    def __init__(self, dataset, nsd):
        self.dataset = dataset
        self.nsd     = nsd

    def run(self) -> StratumMetricResult:
        raise NotImplementedError
