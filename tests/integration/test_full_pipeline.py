"""
tests/integration/test_full_pipeline.py

Full pipeline integration test using synthetic data only (no credentials required).
Runs all five layers. Validates every StratumReport contract.

Paper claim tested: §3.2 "A single call to stratum.evaluate() returns a
StratumReport from which summary statistics, a Stratum Card, and a regulatory
export are all derivable."
"""

from __future__ import annotations

import pytest
from datetime import date, timedelta

import stratum
from stratum.normative.nsd import NormativeSpec, StakeholderRecord
from stratum.data.dataset import EvalDataset
from stratum.connectors.synthetic import build_synthetic_cohort
from stratum.exceptions import StratumNSDError, StratumLayerError


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_data() -> EvalDataset:
    df = build_synthetic_cohort(n=1000, seed=42)
    return EvalDataset(
        df=df,
        label_col="mortality_label",
        score_col="model_score",
        group_col="ethnicity",
        metadata={
            "source": "synthetic",
            "n": 1000,
            "seed": 42,
            "schema": "mimic-iv-compatible",
        },
    )


@pytest.fixture(scope="module")
def valid_nsd() -> NormativeSpec:
    return NormativeSpec(
        use_case="Sepsis-3 early warning in adult ICU",
        deployment_context="Academic medical centre, clinical decision support",
        fairness_criterion="equalised_odds",
        fairness_rationale=(
            "Equalised odds was chosen after stakeholder consultation. "
            "This criterion trades off positive predictive value parity against "
            "equalised error rates; the clinical team accepted higher FPR for "
            "lower-risk groups to avoid under-treatment of higher-risk groups."
        ),
        acceptable_fnr=0.15,
        acceptable_fpr=0.25,
        stakeholders=[
            StakeholderRecord(
                name="ICU Clinical Team", role="deployer", represented=True
            ),
            StakeholderRecord(
                name="Patient Advocacy Group", role="affected", represented=True
            ),
            StakeholderRecord(
                name="Paediatric Patients",
                role="affected",
                represented=False,
                exclusion_reason="Cohort restricted to adults (≥18). Separate NSD required.",
            ),
        ],
        monitoring_frequency="quarterly",
        validity_horizon=(date.today() + timedelta(days=365)).isoformat(),
        sunset_condition="Model retrained or deployment context changes.",
    )


# ── Happy-path tests ─────────────────────────────────────────────────────────

class TestFullPipelineAllLayers:

    def test_evaluate_returns_stratum_report(self, synthetic_data, valid_nsd):
        report = stratum.evaluate(synthetic_data, valid_nsd, layers=[1, 2, 3, 4, 5])
        assert report is not None

    def test_report_has_five_layer_results(self, synthetic_data, valid_nsd):
        report = stratum.evaluate(synthetic_data, valid_nsd, layers=[1, 2, 3, 4, 5])
        assert len(report.results) == 5

    def test_summary_is_non_empty_string(self, synthetic_data, valid_nsd):
        report = stratum.evaluate(synthetic_data, valid_nsd, layers=[1, 2, 3, 4, 5])
        summary = report.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_stratum_card_contains_all_required_headings(self, synthetic_data, valid_nsd):
        report = stratum.evaluate(synthetic_data, valid_nsd, layers=[1, 2, 3, 4, 5])
        card_md = report.to_stratum_card()
        required = [
            "Intended Use and Scope",
            "Fairness Criterion and Rationale",
            "Stakeholders",
            "Layer Results",
            "Dataset Provenance",
            "Monitoring and Review Schedule",
        ]
        for heading in required:
            assert heading in card_md, f"Missing section: {heading}"

    def test_regulatory_export_is_serialisable_dict(self, synthetic_data, valid_nsd):
        import json
        report = stratum.evaluate(synthetic_data, valid_nsd, layers=[1, 2, 3, 4, 5])
        export = report.to_regulatory_export()
        assert isinstance(export, dict)
        # Must be JSON-serialisable — regulatory systems need clean handoff
        serialised = json.dumps(export)
        assert len(serialised) > 0

    def test_regulatory_export_contains_schema_version(self, synthetic_data, valid_nsd):
        report = stratum.evaluate(synthetic_data, valid_nsd, layers=[1, 2, 3, 4, 5])
        export = report.to_regulatory_export()
        assert "schema_version" in export
        assert export["schema_version"].startswith("stratum-card/")

    def test_partial_layer_run(self, synthetic_data, valid_nsd):
        """Ablation: caller can run a subset of layers."""
        report = stratum.evaluate(synthetic_data, valid_nsd, layers=[1, 3])
        assert len(report.results) == 2
        layer_indices = [r.layer_index for r in report.results]
        assert layer_indices == [1, 3]

    def test_metric_values_are_finite(self, synthetic_data, valid_nsd):
        import math
        report = stratum.evaluate(synthetic_data, valid_nsd, layers=[1, 2, 3, 4, 5])
        for r in report.results:
            assert math.isfinite(r.value), f"Layer {r.layer_index} returned non-finite value"

    def test_layer_results_within_3_sig_figs_of_reference(self, synthetic_data, valid_nsd):
        """
        arXiv readiness criterion: figures must reproduce within 3 sig figs.
        Reference values were locked at seed=42 before submission.
        Update these if the synthetic generator changes — bump version too.
        """
        REFERENCE = {
            1: 0.823,   # AUROC — Layer 1 discriminative performance
            2: 0.741,   # ECE-calibrated — Layer 2 calibration
            3: 0.112,   # FNR disparity — Layer 3 group fairness
            4: 0.089,   # Intersectional disparity — Layer 4
            5: 0.031,   # Temporal drift proxy — Layer 5
        }
        report = stratum.evaluate(synthetic_data, valid_nsd, layers=[1, 2, 3, 4, 5])
        for r in report.results:
            ref = REFERENCE.get(r.layer_index)
            if ref is not None:
                assert abs(r.value - ref) / ref < 5e-3, (
                    f"Layer {r.layer_index}: got {r.value:.4f}, "
                    f"reference {ref:.4f} — exceeds 3 sig fig tolerance"
                )


# ── NSD validation gate tests ────────────────────────────────────────────────

class TestNSDGating:

    def test_expired_nsd_raises_before_any_layer_runs(self, synthetic_data):
        import copy
        from stratum.normative.nsd import NormativeSpec, StakeholderRecord
        expired_nsd = NormativeSpec(
            use_case="test",
            deployment_context="test",
            fairness_criterion="equalised_odds",
            fairness_rationale=(
                "This rationale acknowledges the trade-off between FPR and FNR "
                "across groups, accepting higher FPR to equalise error rates."
            ),
            acceptable_fnr=0.15,
            acceptable_fpr=0.25,
            stakeholders=[
                StakeholderRecord(name="Test", role="deployer", represented=True)
            ],
            monitoring_frequency="monthly",
            validity_horizon="2020-01-01",   # past date
            sunset_condition="never",
        )
        with pytest.raises(StratumNSDError, match="expired"):
            stratum.evaluate(synthetic_data, expired_nsd, layers=[1])

    def test_missing_tradeoff_language_raises(self, synthetic_data):
        with pytest.raises((StratumNSDError, ValueError), match="trade"):
            NormativeSpec(
                use_case="test",
                deployment_context="test",
                fairness_criterion="equalised_odds",
                fairness_rationale="Equalised odds was selected.",  # no trade-off language
                acceptable_fnr=0.15,
                acceptable_fpr=0.25,
                stakeholders=[
                    StakeholderRecord(name="Test", role="deployer", represented=True)
                ],
                monitoring_frequency="monthly",
                validity_horizon=(date.today() + timedelta(days=365)).isoformat(),
                sunset_condition="never",
            )


# ── Layer argument validation tests ─────────────────────────────────────────

class TestLayerArguments:

    def test_empty_layers_raises(self, synthetic_data, valid_nsd):
        with pytest.raises(StratumLayerError, match="non-empty"):
            stratum.evaluate(synthetic_data, valid_nsd, layers=[])

    def test_duplicate_layers_raises(self, synthetic_data, valid_nsd):
        with pytest.raises(StratumLayerError, match="Duplicate"):
            stratum.evaluate(synthetic_data, valid_nsd, layers=[1, 1, 2])

    def test_unknown_layer_raises(self, synthetic_data, valid_nsd):
        with pytest.raises(StratumLayerError, match="Unknown"):
            stratum.evaluate(synthetic_data, valid_nsd, layers=[99])
