# stratum-eval

**A normative-first evaluation framework for clinical machine learning.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![Status: v0.1.0-dev](https://img.shields.io/badge/status-v0.1.0--dev-orange)](pyproject.toml)

stratum-eval exists because a metric that cannot state what it cannot measure should not be used to make clinical deployment decisions. Most medical AI evaluation pipelines compute discriminative performance first and defer questions of *which errors matter, to whom, and under what conditions* until after a model is already built. stratum-eval inverts that order: normative specification is a prerequisite, not a postscript.

This repository accompanies the paper *"stratum-eval: A Normative-First Evaluation Framework for Clinical Machine Learning"* and implements the framework described in it.

---

## Why stratum-eval

Standard evaluation practice in medical AI tends to fail quietly. stratum-eval is built around a taxonomy of eight recurring failure modes, spanning issues in the evaluation **data**, the **metrics** used, the **deployment** context, and the underlying **normative** assumptions — most of which produce plausible-looking numbers that conceal the underlying problem rather than obviously broken ones. Two of these failure modes are normative rather than technical: fairness criteria that are mathematically incompatible given the observed base rates in the data, and evaluations that measure a model in isolation while missing the failure modes that only emerge from human–AI interaction in practice.

Rather than treating fairness as something a metric can optimize away, stratum-eval requires evaluators to complete a **Normative Specification Document (NSD)** before any metric is computed. The NSD records the use case, the chosen fairness criterion together with an explicit account of its trade-offs, a stakeholder map with documented exclusions, and a time-bounded validity horizon. The specification is hard-validated: an incomplete, unjustified, or expired NSD halts the pipeline with an error rather than allowing evaluation to proceed on undocumented assumptions.

## Core concepts

| Concept | What it does |
|---|---|
| **NSD (Normative Specification Document)** | Encodes the use case, fairness criterion, fairness rationale, acceptable FNR/FPR thresholds, stakeholder map, and validity horizon. Validated by `validate_nsd_dict` before any data is touched; missing or expired fields raise `StratumNSDError`. |
| **SVSP (Stakeholder Verification and Sign-off Protocol)** | The four-stage process — stakeholder mapping, consultation, normative deliberation, sign-off and horizon-setting — that governs how an NSD is produced. |
| **Chouldechova impossibility visualiser** | Plots the jointly achievable (FNR, FPR) region for each demographic group *before* modelling begins, surfacing cases where equalized odds and calibration cannot both be satisfied given unequal base rates. |
| **Five-layer evaluation protocol** | Discriminative performance → calibration/label quality → group fairness → intersectional disparity → temporal robustness, run in that order against the committed NSD. |
| **StratumReport / StratumCard** | The structured output of a run: a model card built from NSD commitments, with every section traceable back to a paper section and a line of code, plus a machine-readable export for regulatory submission. |

## The five-layer evaluation protocol

1. **Layer 1 — Invariance-tested benchmarking.** Uses a variance-based Risk Extrapolation approximation to estimate a Spurious Correlation Index, the fraction of predictive signal that fails to generalize across environments. Undefined (and reported as such, not as a misleading number) when only one deployment environment is available.
2. **Layer 2 — Label quality, uncertainty, and fairness bounds.** Dawid–Skene label-quality estimation (requires ≥3 independent labellers), conformal prediction under the exchangeability assumption, and the formal fairness-impossibility check.
3. **Layer 3 — Group fairness.** Per-group FNR/FPR compared directly against the thresholds committed to in the NSD, with violations surfaced in the report rather than averaged away.
4. **Layer 4 — Intersectional disparity.** FNR at the intersection of two grouping variables. Cells with fewer than 30 observations stay visible in the output but are marked underpowered instead of being silently dropped or silently reported as if reliable.
5. **Layer 5 — Temporal robustness.** Chronological-quartile AUROC drift for time-indexed datasets, or a bootstrap stability check when no temporal index exists.

Every layer follows the same design rule: on a pathological or underpowered input, the framework returns an explicit `assumption_violations` flag and withholds the point estimate rather than emitting a silent, misleading number.

## Installation

```bash
pip install git+https://github.com/hadbierox196/stratum-eval.git
```

Requires Python ≥ 3.10. Core dependencies: NumPy ≥ 1.26, pandas ≥ 2.1, scikit-learn ≥ 1.4, pydantic ≥ 2.5, rich ≥ 13.7.

For development (tests, linting, type checking):

```bash
pip install "stratum-eval[dev] @ git+https://github.com/hadbierox196/stratum-eval.git"
```

## Quickstart

```python
from tests.fixtures.synthetic import make_binary_dataset

dataset = make_binary_dataset(n=500, prevalence=0.25, signal=0.75)
print(dataset)
# EvalDataset(name='quickstart_synthetic', task='binary', n=500, prevalence=0.250)
```

The synthetic connector requires no credentials and generates a MIMIC-IV-schema-compatible dataset constructed to exhibit a Chouldechova base-rate gap across groups, so the impossibility checker has something real to surface. A full evaluation run additionally requires a completed NSD; `evaluate()` validates it as its first step and will refuse to run against an incomplete or expired specification.


## Data connectors

Two connectors are provided, both returning an `EvalDataset` with identical schema so downstream code is connector-agnostic:

- **MIMIC-IV connector** — builds a Sepsis-3 cohort via BigQuery, requires PhysioNet credentialing.
- **Synthetic connector** — no credentials required; produces a schema-compatible dataset designed to exhibit fairness impossibility across groups with differing base rates. Useful for testing and for the quickstart above.

## Metric governance

Every metric shipped in stratum-eval must have a corresponding entry in [`METRIC_STANDARDS.md`](METRIC_STANDARDS.md) declaring its layer assignment, paper section reference, threshold source, and what it explicitly cannot measure. `scripts/check_metric_specs.py` enforces this as a CI gate — a metric without a standards entry cannot be merged. See [`METRIC_STANDARDS.md`](METRIC_STANDARDS.md) for the full specification and the design principles behind it (metrics must declare their blind spots, fail gracefully rather than return a silent incorrect value, and ship with a falsifiability scenario).

## Validation on MIMIC-IV

The framework was validated on the MIMIC-IV Clinical Database Demo (v2.2): 140 ICU stays, Sepsis-3 outcome, overall prevalence 41.4%. An XGBoost classifier reached a held-out AUROC of 0.905, but per-group results diverged sharply — male patients (Group A) showed a false-negative rate of 0.429 against an NSD-committed threshold of 0.15, a violation the framework surfaced explicitly rather than hiding behind the aggregate AUROC. The impossibility visualiser had already flagged the underlying 14.8-percentage-point base-rate gap between groups before model training began. Full results, including intersectional and temporal-robustness analysis, are in the accompanying paper (`docs/paper/`); note that the Demo cohort's small size (n=100 patients) means intersectional and temporal-quartile estimates are explicitly flagged as underpowered proof-of-concept results, not deployment-ready findings.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). New metrics require a completed Metric Standards Declaration in the pull request description.

## Citation

If you use stratum-eval in your research, please cite:

```bibtex
@software{stratum_eval_2025,
  title   = {stratum-eval: A Rigorous Evaluation Framework for Medical AI},
  author  = {Farooq, Hassan and Jawad, Abdullah and Butt, Muhammad Salman},
  year    = {2025},
  url     = {https://github.com/hadbierox196/stratum-eval},
  license = {Apache-2.0}
}
```

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
