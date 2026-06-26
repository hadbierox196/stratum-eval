# stratum-eval

**A rigorous evaluation framework for medical AI.**

stratum-eval exists because a metric that cannot state what it cannot measure
should not be used to make clinical deployment decisions.

---

## Quickstart

```bash
pip install git+https://github.com/hadbierox196/stratum-eval.git
from tests.fixtures.synthetic import make_binary_dataset

dataset = make_binary_dataset(n=500, prevalence=0.25, signal=0.75)
print(dataset)
# EvalDataset(name='quickstart_synthetic', task='binary', n=500, prevalence=0.250)
Design Philosophy
Metrics must declare their blind spots. Every metric in this framework
specifies what it cannot measure (Field 2 of the Metric Standards).
Graceful failure is mandatory. No metric returns a silent incorrect
value on pathological inputs.
Falsifiability is enforced. Every metric must ship with a concrete
scenario in which a good model scores poorly and a bad model scores well.
See METRIC_STANDARDS.md for the full specification.
Contributing
See CONTRIBUTING.md. New metrics require a completed
Metric Standards Declaration in the PR description.
License
Apache 2.0. See LICENSE.
Citation
If you use stratum-eval in your research, please cite:
@software{stratum_eval_2025,
  title  = {stratum-eval: A Rigorous Evaluation Framework for Medical AI},
  year   = {2025},
  url    = {https://github.com/<your-org>/stratum-eval},
  license = {Apache-2.0}
}
---

That's all eight deliverables. Here's a summary of what maps where:

| Task | File |
|---|---|
| ⬡ Directory structure | Tree above — copy as-is |
| ⬡ METRIC_STANDARDS.md | Full 6-field spec + anti-pattern catalogue |
| ⬡ CONTRIBUTING.md | PR checklist + metric declaration template |
| ⬡ CI (ruff + mypy + pytest) | `.github/workflows/ci.yml` |
| ◈ Failure taxonomy table | `docs/paper/section_01_introduction.md` — Table 1 with all 8 columns |
| ◈ Identification strategy outline | `docs/paper/section_01_identification_strategy_outline.md` — 7 steps |
| ◎ Colab 00 | `notebooks/00_quickstart.ipynb` |
| Supporting code | `base.py`, `eval_dataset.py`, `test_base_metric.py`, `synthetic.py` |

A few things worth flagging before you push: replace `<your-org>` in 3 places (pyproject, README, CI), confirm the `tests/fixtures/synthetic.py` import path matches your actual package structure in Colab 00, and the `setuptools` backend string in pyproject.toml should be `"setuptools.build_meta"` for modern setuptools — I wrote the legacy form which still works but the new form is cleaner.
