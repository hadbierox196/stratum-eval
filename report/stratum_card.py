"""
stratum/report/stratum_card.py
Generates structured model cards anchored to NSD commitments.
Paper §4.1 "Transparency Artefacts" — StratumCard is the primary output
artefact discussed there. Every section maps to a paper claim.
"""

from __future__ import annotations

import json
import textwrap
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field

from stratum.normative.nsd import NormativeSpec
from stratum.report.metric_result import StratumMetricResult


class StratumCardSection(BaseModel):
    heading: str
    content: str
    paper_ref: str = Field(
        description="Paper section this card section corresponds to, e.g. '§3.1'."
    )
    code_ref: str = Field(
        description="Repo path that generates or validates this content."
    )


class StratumCard(BaseModel):
    """
    Structured model card generated from a completed StratumReport.

    Paper §4.1: a StratumCard is required output for any evaluation
    submitted to a regulatory pathway. Fields map 1-to-1 to the
    NSD commitments made before evaluation began.
    """

    schema_version: str = "stratum-card/0.1"
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    sections: list[StratumCardSection] = Field(default_factory=list)

    # ── Constructors ────────────────────────────────────────────────────────

    @classmethod
    def from_report(
        cls,
        nsd: NormativeSpec,
        results: list[StratumMetricResult],
        dataset_metadata: dict[str, Any],
    ) -> "StratumCard":
        """
        Build a StratumCard from report components.
        Mirrors the section order of the paper's Appendix B card template.
        """
        card = cls()

        card.sections.append(StratumCardSection(
            heading="Intended Use and Scope",
            content=textwrap.dedent(f"""
                Use case: {nsd.use_case}
                Deployment context: {nsd.deployment_context}
                Validity horizon: {nsd.validity_horizon}
                Sunset condition: {nsd.sunset_condition}
            """).strip(),
            paper_ref="§2.1",
            code_ref="stratum/normative/nsd.py::NormativeSpec.use_case",
        ))

        card.sections.append(StratumCardSection(
            heading="Fairness Criterion and Rationale",
            content=textwrap.dedent(f"""
                Criterion: {nsd.fairness_criterion}
                Rationale (validated for trade-off language):
                {nsd.fairness_rationale}
                Acceptable FNR threshold: {nsd.acceptable_fnr}
                Acceptable FPR threshold: {nsd.acceptable_fpr}
            """).strip(),
            paper_ref="§3.1",
            code_ref="stratum/normative/validator.py::_check_tradeoff_language",
        ))

        card.sections.append(StratumCardSection(
            heading="Stakeholders",
            content=cls._format_stakeholders(nsd),
            paper_ref="§2.2",
            code_ref="stratum/normative/nsd.py::StakeholderRecord",
        ))

        card.sections.append(StratumCardSection(
            heading="Layer Results",
            content=cls._format_layer_results(results),
            paper_ref="§3.2",
            code_ref="stratum/layers/__init__.py::LAYER_REGISTRY",
        ))

        card.sections.append(StratumCardSection(
            heading="Dataset Provenance",
            content=cls._format_dataset(dataset_metadata),
            paper_ref="§3.3",
            code_ref="stratum/data/dataset.py::EvalDataset.metadata",
        ))

        card.sections.append(StratumCardSection(
            heading="Monitoring and Review Schedule",
            content=textwrap.dedent(f"""
                Monitoring frequency: {nsd.monitoring_frequency}
                Next scheduled review: derived from validity_horizon above.
                This card expires with the NSD. Renewal requires re-running
                stratum.evaluate() against an updated NSD.
            """).strip(),
            paper_ref="§4.2",
            code_ref="stratum/normative/nsd.py::NormativeSpec.monitoring_frequency",
        ))

        return card

    # ── Formatters ──────────────────────────────────────────────────────────

    @staticmethod
    def _format_stakeholders(nsd: NormativeSpec) -> str:
        lines = []
        for sh in nsd.stakeholders:
            status = "REPRESENTED" if sh.represented else "EXCLUDED — reason: " + (sh.exclusion_reason or "not provided")
            lines.append(f"  • {sh.name} ({sh.role}): {status}")
        return "Stakeholder map:\n" + "\n".join(lines)

    @staticmethod
    def _format_layer_results(results: list[StratumMetricResult]) -> str:
        lines = []
        for r in results:
            passed = "PASS" if r.passed else "FAIL"
            lines.append(
                f"  Layer {r.layer_index} — {r.metric_name}: "
                f"{r.value:.4f} [{passed}] "
                f"(threshold: {r.threshold})"
            )
        return "\n".join(lines) if lines else "No layer results recorded."

    @staticmethod
    def _format_dataset(meta: dict[str, Any]) -> str:
        lines = [f"  {k}: {v}" for k, v in meta.items()]
        return "\n".join(lines) if lines else "No dataset metadata provided."

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_markdown(self) -> str:
        """Human-readable card. Used by .to_stratum_card() on StratumReport."""
        lines = [
            f"# Stratum Evaluation Card",
            f"Schema: `{self.schema_version}` | Generated: {self.generated_at}",
            "",
        ]
        for sec in self.sections:
            lines += [
                f"## {sec.heading}",
                f"*Paper {sec.paper_ref} · Code: `{sec.code_ref}`*",
                "",
                sec.content,
                "",
            ]
        return "\n".join(lines)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    def to_regulatory_export(self) -> dict[str, Any]:
        """
        Structured dict for downstream regulatory systems.
        Paper §4.3 "Regulatory Pathway Integration" — this dict is the
        artefact handed to audit teams.
        """
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "sections": [
                {
                    "heading": s.heading,
                    "paper_ref": s.paper_ref,
                    "code_ref": s.code_ref,
                    "content": s.content,
                }
                for s in self.sections
            ],
        }
