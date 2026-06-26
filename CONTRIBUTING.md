# Contributing to stratum-eval

Thank you for contributing. stratum-eval is a research-grade framework and
holds its metrics to a higher standard than most ML libraries. Please read
this document fully before opening a PR.

---

## Code of Conduct

Be precise. Be honest about limitations. Do not overstate what a metric
measures. These are not soft norms — they are enforced in review.

---

## Types of Contributions

| Type | Branch prefix | Review SLA |
|------|--------------|------------|
| New metric | `metric/` | 2 reviewers, 7 days |
| Bug fix | `fix/` | 1 reviewer, 3 days |
| Documentation | `docs/` | 1 reviewer, 5 days |
| New dataset loader | `data/` | 1 reviewer + data steward, 7 days |
| Infrastructure / CI | `infra/` | 1 reviewer, 3 days |

---

## Pull Request Checklist

### All PRs

- [ ] Branch is up to date with `main`
- [ ] `ruff check .` passes with zero warnings
- [ ] `mypy stratum_eval/` passes with zero errors
- [ ] `pytest` passes (all existing tests green)
- [ ] Docstrings follow NumPy style
- [ ] No new dependencies added without discussion in an issue first

### New Metric PRs (additional requirements)

Every metric PR must include a completed `METRIC_STANDARDS.md` block in the
PR description. Copy the template below and fill every field. Incomplete
fields are grounds for immediate return without review.
Metric Standards Declaration
Metric name: 
Field 1 — What It Measures
MEASURES: ...
Field 2 — What It Cannot Measure
CANNOT_MEASURE_CALIBRATION: yes/no — ...
CANNOT_MEASURE_SUBGROUP: yes/no — ...
CANNOT_MEASURE_DISTRIBUTION_SHIFT: yes/no — ...
OTHER: ...
Field 3 — Required Assumptions
| ID | Assumption | Testable? | Consequence if violated |
|----|-----------|-----------|------------------------|
| A1 | | | |
Field 4 — Graceful Failure Behavior
| Condition | Expected behavior | Error type |
|-----------|------------------|------------|
| All predictions identical | | |
| All labels identical | | |
| NaN / Inf in predictions | | |
| Empty arrays | | |
| Prevalence < 0.01 or > 0.99 | | |
| n < minimum_sample_size | | |
Field 5 — Falsifiability Condition
FALSE_NEGATIVE: ...
FALSE_POSITIVE: ...
Field 6 — Computational Complexity
TIME_COMPLEXITY: O(...)
SPACE_COMPLEXITY: O(...)
PARALLELIZABLE: yes/no/partial
MINIMUM_SAMPLE_SIZE: ...
RECOMMENDED_SAMPLE_SIZE: ...
### New Metric PRs — Code Requirements

- [ ] Metric inherits from `BaseMetric` in `stratum_eval/metrics/base.py`
- [ ] `validate_inputs()` covers all six pathological cases in Field 4
- [ ] `compute()` returns a `MetricResult` dataclass (not a raw float)
- [ ] Unit tests cover: normal operation, each graceful failure case,
      boundary values, and at least one known-answer numerical test
- [ ] `stratum-eval metric lint <YourMetricName>` exits 0
- [ ] The anti-pattern checklist in `METRIC_STANDARDS.md` has been reviewed
      and each item is either N/A or explicitly addressed

### Dataset Loader PRs (additional requirements)

- [ ] No raw patient data committed to the repository under any circumstances
- [ ] Loader is a reproducible recipe, not a data dump
- [ ] Provenance documented: source, access date, license, IRB status
- [ ] Synthetic fixture committed to `tests/fixtures/` for CI use

---

## Local Development Setup

git clone https://github.com/<your-org>/stratum-eval.git
cd stratum-eval
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install

Running the Full Check Suite
ruff check .
mypy stratum_eval/
pytest --tb=short -v
stratum-eval metric lint --all   # once CLI is implemented

Review Philosophy
Reviewers are expected to push back on Field 2 and Field 5. A limitation
section that reads as a disclaimer rather than a genuine blind-spot analysis
will be returned. A falsifiability section that only constructs toy adversarial
examples without clinical grounding will be returned.
We would rather have ten well-specified metrics than fifty optimistic ones.


---

## .github/workflows/ci.yml

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint-and-type-check:
    name: Ruff + Mypy
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Ruff lint
        run: ruff check .

      - name: Mypy type check
        run: mypy stratum_eval/

  test:
    name: Pytest
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Run tests
        run: pytest tests/ --tb=short -v --cov=stratum_eval --cov-report=term-missing
