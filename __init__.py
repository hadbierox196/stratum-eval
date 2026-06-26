"""
stratum/__init__.py
Public API surface for stratum-eval.
Paper §3.2 "Evaluation Protocol" — implements the five-layer pipeline described there.
"""

from __future__ import annotations

from stratum.normative.validator import validate_nsd_dict
from stratum.normative.nsd import NormativeSpec
from stratum.data.dataset import EvalDataset
from stratum.layers import LAYER_REGISTRY
from stratum.report.stratum_report import StratumReport
from stratum.report.stratum_card import StratumCard
from stratum.exceptions import StratumNSDError, StratumLayerError


def evaluate(
    data: EvalDataset,
    nsd: NormativeSpec,
    layers: list[int],
) -> StratumReport:
    """
    Run the stratum evaluation pipeline.

    Parameters
    ----------
    data : EvalDataset
        Wrapped dataset. Must expose .df (DataFrame), .label_col, .score_col,
        .group_col, and .metadata dict.
    nsd : NormativeSpec
        Validated normative specification. Validation is re-run here as a
        hard gate — the pipeline does not proceed on a stale or expired NSD.
    layers : list[int]
        Ordered list of layer indices to run (1–5). Layers execute in the
        order given; duplicates raise StratumLayerError.

    Returns
    -------
    StratumReport
        Aggregated results with .summary(), .to_stratum_card(),
        and .to_regulatory_export() methods.

    Raises
    ------
    StratumNSDError
        If NSD validation fails at call time (expired horizon,
        missing trade-off language, schema violation).
    StratumLayerError
        If layers list is empty, contains duplicates, or references
        a layer index not in LAYER_REGISTRY.

    Notes
    -----
    Paper §3.2, Figure 2: layer ordering is the caller's responsibility.
    The pipeline does not reorder; it runs layers as given so callers
    can compose partial pipelines for ablation studies.
    """
    # ── Gate 1: NSD hard validation ────────────────────────────────────────
    # Re-validate even if the caller constructed NSD programmatically.
    # Covers expired validity_horizon between NSD construction and evaluate().
    validate_nsd_dict(nsd.model_dump())   # raises StratumNSDError on failure

    # ── Gate 2: layers argument sanity ─────────────────────────────────────
    if not layers:
        raise StratumLayerError("layers must be a non-empty list of layer indices.")
    if len(layers) != len(set(layers)):
        raise StratumLayerError(f"Duplicate layer indices: {layers}. Each layer runs once.")
    unknown = [l for l in layers if l not in LAYER_REGISTRY]
    if unknown:
        raise StratumLayerError(
            f"Unknown layer indices: {unknown}. "
            f"Available: {sorted(LAYER_REGISTRY.keys())}"
        )

    # ── Run layers in caller-specified order ────────────────────────────────
    results: list = []
    for layer_index in layers:
        layer_cls = LAYER_REGISTRY[layer_index]
        layer = layer_cls(data=data, nsd=nsd)
        layer_result = layer.run()          # returns StratumMetricResult
        results.append(layer_result)

    # ── Assemble report ─────────────────────────────────────────────────────
    return StratumReport(
        results=results,
        nsd=nsd,
        dataset_metadata=data.metadata,
    )


__all__ = ["evaluate", "EvalDataset", "NormativeSpec", "StratumReport"]
__version__ = "0.1.0"
