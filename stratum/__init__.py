"""stratum-eval: A rigorous evaluation framework for medical AI."""
from stratum.normative.validator import validate_nsd_dict
from stratum.normative.nsd import NormativeSpec
from stratum.data.dataset import EvalDataset
from stratum.layers import LAYER_REGISTRY
from stratum.report.stratum_report import StratumReport
from stratum.exceptions import StratumLayerError

def evaluate(
    data: EvalDataset,
    nsd: NormativeSpec,
    layers: list[int] | None = None,
) -> StratumReport:
    """Run the requested layer evaluations and return a StratumReport."""
    if layers is None:
        layers = [1, 2, 3, 4, 5]

    results = []
    for idx in layers:
        if idx not in LAYER_REGISTRY:
            raise StratumLayerError(
                f"Layer {idx} is not registered. "
                f"Available: {sorted(LAYER_REGISTRY.keys())}"
            )
        layer = LAYER_REGISTRY[idx](dataset=data, nsd=nsd)
        results.append(layer.run())

    return StratumReport(
        results=results,
        nsd=nsd,
        dataset_metadata=data.metadata,
    )

__all__ = [
    "evaluate",
    "EvalDataset",
    "NormativeSpec",
    "StratumReport",
    "LAYER_REGISTRY",
    "validate_nsd_dict",
]
