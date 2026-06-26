"""
tests/unit/test_layer3_shadow_deployment.py

Tests for the shadow deployment harness. Focus areas: correct logging of
the (input, model_output, actual_clinical_decision, outcome) tuple,
non-mutation of resolved records, and — critically — that the harness
exposes no method capable of feeding model output back into a clinical
workflow (the "no intervention" structural constraint).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from stratum.layer3.shadow_deployment import (
    ShadowDeploymentHarness,
    ShadowRecord,
    window_by_time,
)

SEED = 42


def _dummy_score_fn(features: dict) -> float:
    return float(features.get("risk_factor", 0.0))


class TestShadowLogging:
    def test_logs_full_tuple(self):
        harness = ShadowDeploymentHarness(_dummy_score_fn)
        record = harness.log_case(
            case_id="case_1",
            input_features={"risk_factor": 0.7, "age": 55},
            actual_clinical_decision=1,
            outcome=0,
        )
        assert record.case_id == "case_1"
        assert record.model_output == 0.7
        assert record.actual_clinical_decision == 1
        assert record.outcome == 0
        assert record.input_features == {"risk_factor": 0.7, "age": 55}

    def test_outcome_defaults_to_none_when_unresolved(self):
        harness = ShadowDeploymentHarness(_dummy_score_fn)
        record = harness.log_case(
            case_id="case_2",
            input_features={"risk_factor": 0.3},
            actual_clinical_decision=0,
        )
        assert record.outcome is None

    def test_records_are_frozen(self):
        harness = ShadowDeploymentHarness(_dummy_score_fn)
        record = harness.log_case(
            case_id="case_3",
            input_features={"risk_factor": 0.5},
            actual_clinical_decision=1,
            outcome=1,
        )
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            record.outcome = 0


class TestOutcomeResolution:
    def test_resolve_outcome_appends_not_mutates(self):
        harness = ShadowDeploymentHarness(_dummy_score_fn)
        harness.log_case(
            case_id="case_4",
            input_features={"risk_factor": 0.6},
            actual_clinical_decision=1,
        )
        updated = harness.resolve_outcome("case_4", outcome=1)

        assert updated.outcome == 1
        assert updated.supersedes_case_id == "case_4"
        # both the original and the superseding record still exist in raw log
        assert len(harness._records) == 2

    def test_records_returns_only_latest_per_case(self):
        harness = ShadowDeploymentHarness(_dummy_score_fn)
        harness.log_case(
            case_id="case_5",
            input_features={"risk_factor": 0.4},
            actual_clinical_decision=0,
        )
        harness.resolve_outcome("case_5", outcome=0)

        effective = harness.records()
        assert len(effective) == 1
        assert effective[0].outcome == 0

    def test_resolve_outcome_raises_for_unknown_case(self):
        harness = ShadowDeploymentHarness(_dummy_score_fn)
        with pytest.raises(KeyError):
            harness.resolve_outcome("nonexistent_case", outcome=1)


class TestToArrays:
    def test_to_arrays_shapes_match_log_count(self):
        harness = ShadowDeploymentHarness(_dummy_score_fn)
        rng = np.random.default_rng(SEED)
        for i in range(20):
            harness.log_case(
                case_id=f"case_{i}",
                input_features={"risk_factor": float(rng.uniform()), "age": float(rng.integers(20, 80))},
                actual_clinical_decision=int(rng.integers(0, 2)),
                outcome=float(rng.normal()),
                subgroup_labels={"site": "site_a" if i % 2 == 0 else "site_b"},
            )

        arrays = harness.to_arrays(resolved_only=True)
        assert arrays["model_output"].shape[0] == 20
        assert arrays["clinical_decision"].shape[0] == 20
        assert arrays["outcome"].shape[0] == 20
        assert arrays["covariates"].shape == (20, 2)
        assert "site" in arrays["subgroup_labels"]

    def test_unresolved_excluded_when_resolved_only(self):
        harness = ShadowDeploymentHarness(_dummy_score_fn)
        harness.log_case(case_id="resolved", input_features={"risk_factor": 0.5},
                          actual_clinical_decision=1, outcome=1)
        harness.log_case(case_id="pending", input_features={"risk_factor": 0.5},
                          actual_clinical_decision=1)  # no outcome yet

        arrays = harness.to_arrays(resolved_only=True)
        assert arrays["model_output"].shape[0] == 1

    def test_empty_harness_returns_empty_arrays(self):
        harness = ShadowDeploymentHarness(_dummy_score_fn)
        arrays = harness.to_arrays()
        assert arrays["model_output"].shape[0] == 0


class TestNoInterventionConstraint:
    def test_harness_exposes_no_clinician_facing_methods(self):
        """
        Structural check: the harness's public API must not include any
        method whose name suggests it surfaces model output into a
        clinical workflow. This is the closest a unit test can get to
        enforcing "no intervention" — the rest of the guarantee is the
        absence of such a method, which by definition can't be tested by
        calling it. This test guards against someone adding one later
        without updating this assertion (and therefore being forced to
        consciously override it here).
        """
        forbidden_substrings = ["notify", "alert_clinician", "flag_case", "webhook", "push_to_ehr"]
        public_methods = [m for m in dir(ShadowDeploymentHarness) if not m.startswith("_")]
        for method_name in public_methods:
            for forbidden in forbidden_substrings:
                assert forbidden not in method_name.lower(), (
                    f"Method '{method_name}' matches forbidden pattern '{forbidden}' — "
                    "shadow deployment must remain observational-only."
                )

    def test_log_case_returns_record_but_harness_has_no_output_sink(self):
        harness = ShadowDeploymentHarness(_dummy_score_fn)
        record = harness.log_case(
            case_id="case_x", input_features={"risk_factor": 0.9},
            actual_clinical_decision=0,
        )
        # The only way model_output escapes the harness is via the returned
        # record object, which is the caller's responsibility from here on.
        assert isinstance(record, ShadowRecord)
        assert not hasattr(harness, "_clinical_output_sink")


class TestWindowByTime:
    def test_splits_correctly_on_boundary(self):
        harness = ShadowDeploymentHarness(_dummy_score_fn)
        harness.log_case(case_id="a", input_features={"risk_factor": 0.1},
                          actual_clinical_decision=0, outcome=0,
                          timestamp="2025-01-01T00:00:00+00:00")
        harness.log_case(case_id="b", input_features={"risk_factor": 0.2},
                          actual_clinical_decision=1, outcome=1,
                          timestamp="2025-06-01T00:00:00+00:00")

        before, after = window_by_time(
            harness.records(), boundary="2025-03-01T00:00:00+00:00"
        )
        assert len(before) == 1 and before[0].case_id == "a"
        assert len(after) == 1 and after[0].case_id == "b"


class TestJSONLExport:
    def test_to_jsonl_writes_one_record_per_line(self, tmp_path):
        harness = ShadowDeploymentHarness(_dummy_score_fn)
        harness.log_case(case_id="c1", input_features={"risk_factor": 0.5},
                          actual_clinical_decision=1, outcome=1)
        harness.log_case(case_id="c2", input_features={"risk_factor": 0.3},
                          actual_clinical_decision=0, outcome=0)

        path = tmp_path / "shadow_log.jsonl"
        harness.to_jsonl(str(path))

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        parsed = [json.loads(line) for line in lines]
        assert parsed[0]["case_id"] == "c1"
