"""
stratum/layer3/shadow_deployment.py

Layer 3: Sociotechnical and Temporal Validity — Shadow Deployment Harness

Runs a model silently alongside live clinical practice: the model scores
every case that flows through, but its output is never shown to the
clinician and never affects care. This harness only LOGS; it never
intervenes. That property is enforced structurally below, not just by
convention — see ShadowDeploymentHarness's docstring for what that means
and what it does NOT solve (see also the manuscript subsection on what
shadow deployment measures and what it misses).

Logged tuple per case: (input, model_output, actual_clinical_decision, outcome)

METRIC_NAME: drift.shadow_deployment_concordance
METRIC_LAYER: 3
PAPER_REF: sec:temporal-validity
THRESHOLD_CITE: Observational pre-deployment evaluation, c.f. FDA SaMD guidance (2021) "Predetermined Change Control Plans"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class ShadowRecord:
    """One observational tuple. Frozen: a logged record is never mutated
    in place — corrections are appended as new records with a superseding
    flag, preserving an honest audit trail of what was actually observed
    versus corrected after the fact."""
    case_id: str
    timestamp: str
    input_features: dict[str, Any]
    model_output: float
    actual_clinical_decision: int
    outcome: float | int | None   # None if outcome not yet resolved (e.g. censored)
    subgroup_labels: dict[str, str] = field(default_factory=dict)
    supersedes_case_id: str | None = None


class ShadowDeploymentHarness:
    """
    Observational-only evaluation harness.

    Design constraint, stated explicitly because it is the entire point of
    this class: `score()` returns the model's output to the CALLER for
    logging purposes, but the harness itself has no method that writes
    anything back into a clinical system, EHR, or alerting pipeline. There
    is no `notify_clinician()`, no `flag_case()`, no webhook, no side
    channel. If you find yourself wiring this harness's output into
    anything a clinician or downstream system can see or act on, you have
    converted this from a shadow deployment into a live (or quasi-live)
    deployment, and the causal-effect-of-concordance analysis in
    performativity.py is no longer estimating what it claims to estimate
    — your "actual_clinical_decision" column would now be partially caused
    by the thing you're trying to treat as exogenous.

    This is enforced by omission (the method doesn't exist) rather than by
    a runtime check, because a runtime check can be disabled; an absent
    method has to be deliberately added by someone reading this docstring
    first.
    """

    def __init__(self, score_fn: Callable[[dict[str, Any]], float]):
        """
        score_fn : the model's scoring function, input_features -> model_output.
            This harness calls it but never surfaces the result anywhere
            except the returned ShadowRecord.
        """
        self._score_fn = score_fn
        self._records: list[ShadowRecord] = []

    def log_case(
        self,
        case_id: str,
        input_features: dict[str, Any],
        actual_clinical_decision: int,
        outcome: float | int | None = None,
        subgroup_labels: dict[str, str] | None = None,
        timestamp: str | None = None,
    ) -> ShadowRecord:
        """
        Scores a case silently and logs the full observational tuple.

        `actual_clinical_decision` and `outcome` are supplied by the
        caller from the real clinical workflow — this harness does not
        and cannot observe them itself, by design; it only ever touches
        the input features and produces a model_output for logging.
        """
        model_output = float(self._score_fn(input_features))
        record = ShadowRecord(
            case_id=case_id,
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            input_features=dict(input_features),
            model_output=model_output,
            actual_clinical_decision=int(actual_clinical_decision),
            outcome=outcome,
            subgroup_labels=dict(subgroup_labels or {}),
        )
        self._records.append(record)
        return record

    def resolve_outcome(
        self, case_id: str, outcome: float | int, *, timestamp: str | None = None
    ) -> ShadowRecord:
        """
        Appends a superseding record once a previously-logged case's
        outcome resolves (e.g. after a follow-up window closes). Does not
        mutate the original record — see ShadowRecord's frozen contract.
        """
        prior = self._find_latest(case_id)
        if prior is None:
            raise KeyError(f"No prior record found for case_id={case_id!r}")

        updated = ShadowRecord(
            case_id=case_id,
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            input_features=prior.input_features,
            model_output=prior.model_output,
            actual_clinical_decision=prior.actual_clinical_decision,
            outcome=outcome,
            subgroup_labels=prior.subgroup_labels,
            supersedes_case_id=case_id,
        )
        self._records.append(updated)
        return updated

    def _find_latest(self, case_id: str) -> ShadowRecord | None:
        matches = [
            r for r in self._records
            if r.case_id == case_id or r.supersedes_case_id == case_id
        ]
        return matches[-1] if matches else None

    def records(self, *, resolved_only: bool = False) -> list[ShadowRecord]:
        """
        Returns the *effective* record set: for any case_id with multiple
        logged versions (due to resolve_outcome), only the latest survives.
        """
        latest_by_case: dict[str, ShadowRecord] = {}
        for r in self._records:
            key = r.case_id
            latest_by_case[key] = r
        result = list(latest_by_case.values())
        if resolved_only:
            result = [r for r in result if r.outcome is not None]
        return result

    def to_arrays(self, *, resolved_only: bool = True) -> dict[str, np.ndarray]:
        """
        Converts the logged records into the array shapes expected by
        drift.py and performativity.py: model_output, clinical_decision,
        outcome, covariates (stacked input_features, numeric fields only),
        plus subgroup label columns kept separately since they're typically
        categorical and used for slicing rather than as model covariates.
        """
        recs = self.records(resolved_only=resolved_only)
        if not recs:
            return {
                "model_output": np.array([]),
                "clinical_decision": np.array([]),
                "outcome": np.array([]),
                "covariates": np.array([]).reshape(0, 0),
                "subgroup_labels": {},
            }

        numeric_keys = sorted(
            k for k, v in recs[0].input_features.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        )
        covariates = np.array([[r.input_features.get(k, np.nan) for k in numeric_keys] for r in recs])

        subgroup_keys = sorted(recs[0].subgroup_labels.keys())
        subgroup_labels = {
            k: np.array([r.subgroup_labels.get(k, "unknown") for r in recs])
            for k in subgroup_keys
        }

        return {
            "model_output": np.array([r.model_output for r in recs]),
            "clinical_decision": np.array([r.actual_clinical_decision for r in recs]),
            "outcome": np.array([r.outcome for r in recs], dtype=float),
            "covariates": covariates,
            "covariate_names": numeric_keys,
            "subgroup_labels": subgroup_labels,
        }

    def to_jsonl(self, path: str, *, resolved_only: bool = False) -> None:
        """Appends-friendly export: one JSON object per logged record
        (including superseded ones), for audit trails that need every
        version of a case, not just the latest."""
        recs = self._records if not resolved_only else [
            r for r in self._records if r.outcome is not None
        ]
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(asdict(r)) + "\n")

    def summary(self) -> dict[str, Any]:
        """Quick non-identifying summary for sanity checks — counts only,
        no PHI surfaces here regardless of what's in input_features."""
        recs = self.records()
        n_resolved = sum(1 for r in recs if r.outcome is not None)
        concordance = None
        resolved = self.records(resolved_only=False)
        if resolved:
            model_decisions = np.array(
                [1 if r.model_output >= 0.5 else 0 for r in resolved]
            )
            clinical_decisions = np.array([r.actual_clinical_decision for r in resolved])
            concordance = float(np.mean(model_decisions == clinical_decisions))

        return {
            "n_cases_logged": len(recs),
            "n_outcomes_resolved": n_resolved,
            "n_outcomes_pending": len(recs) - n_resolved,
            "model_clinician_concordance_rate": concordance,
        }


def window_by_time(
    records: Iterable[ShadowRecord],
    *,
    boundary: str,
) -> tuple[list[ShadowRecord], list[ShadowRecord]]:
    """
    Splits records into (before, after) a timestamp boundary (ISO 8601
    string). The natural way to produce ref_window / new_window inputs for
    performativity.run_performativity_check from a single harness's log.
    """
    before, after = [], []
    for r in records:
        (before if r.timestamp < boundary else after).append(r)
    return before, after
