"""stratum-eval: A rigorous evaluation framework for medical AI."""

__version__ = "0.1.0-dev"

from stratum.metrics.base import BaseMetric, MetricResult, MetricUndefinedError
from stratum.datasets.eval_dataset import EvalDataset
from stratum.normative.nsd import NormativeSpecificationDocument, StakeholderRecord, ExclusionRecord
from stratum.normative.validator import load_nsd, validate_nsd_dict, StratumNSDError

__all__ = [
    "BaseMetric",
    "MetricResult",
    "MetricUndefinedError",
    "EvalDataset",
    "NormativeSpecificationDocument",
    "StakeholderRecord",
    "ExclusionRecord",
    "load_nsd",
    "validate_nsd_dict",
    "StratumNSDError",
]

# NOTE: the previous `evaluate()` / LAYER_REGISTRY / StratumReport orchestration
# has been removed. It only wired together `stratum.layers` (a simpler, separate
# skeleton implementation), not the research-grade layer1/layer2/layer3 modules
# above. `stratum.layers` and `stratum.data` remain importable directly if needed,
# but a real end-to-end evaluate() pipeline needs a deliberate design decision —
# see the "test_full_pipeline.py targets an API that doesn't exist yet" issue.
