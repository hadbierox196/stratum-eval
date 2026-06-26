# Section 2: Identification Strategy — Outline

*Purpose of this section:* Give practitioners a step-by-step procedure for
identifying which failure modes (F1–F8) are active in a given evaluation
pipeline before results are reported.

---

## Step 1 — Establish Provenance (targets F1, F3)

*Topic sentence:* Before any metric is computed, the evaluation dataset must
be audited for label provenance and distributional fidelity, because both
label endogeneity and label distribution collapse are invisible to metric-level
analysis.

Sub-steps:
- 1a. Label provenance audit: who generated the labels, when, under what
  protocol, and with access to what information?
- 1b. Annotator-model independence check: confirm no causal path from model
  outputs to annotation decisions.
- 1c. Prevalence documentation: record observed prevalence in the evaluation
  set and compare to stated clinical prevalence; flag if ratio > 1.5×.
- 1d. Inclusion criteria audit: identify any selection rule that could
  artificially inflate or deflate prevalence.

---

## Step 2 — Select a Metric Bundle, Not a Single Metric (targets F2, F4)

*Topic sentence:* No single metric is sufficient for a clinical evaluation,
because the failure modes introduced by any given metric are not covered by
that same metric — coverage requires deliberate bundling across complementary
blind spots.

Sub-steps:
- 2a. Apply the stratum-eval metric linter to every proposed metric; confirm
  all six METRIC_STANDARDS fields are complete.
- 2b. Map each metric's Field 4 (blind spots) against the other metrics in
  the bundle; identify any blind spot not covered by any other metric.
- 2c. Include at minimum: one discrimination metric, one calibration metric,
  one subgroup metric, and one threshold-behavior metric.
- 2d. Verify that the bundle is not redundant: two metrics that measure the
  same quantity under different names add overhead without coverage.

---

## Step 3 — Audit for Spurious Invariance (targets F4)

*Topic sentence:* After selecting metrics, each metric must be tested against
the specific invariances it is known to exhibit, because a metric that is
invariant to a clinically important property cannot detect failure on that
property regardless of the sample size.

Sub-steps:
- 3a. For rank-based metrics (AUROC, AUPRC): run calibration diagnostics
  (ECE, reliability diagram); confirm the metric bundle includes at least
  one calibration-sensitive measure.
- 3b. For threshold-based metrics (sensitivity, specificity, F1): document
  the threshold selection procedure; test metric stability across a ±10%
  threshold perturbation.
- 3c. For aggregate metrics (macro-average, micro-average): decompose to
  subgroup level; confirm no subgroup AUROC below the clinical acceptability
  threshold is hidden by aggregation.

---

## Step 4 — Probe Reasoning and Output Consistency (targets F5)

*Topic sentence:* For generative and multi-step models, output accuracy is
necessary but not sufficient evidence of clinical safety, because models can
produce correct outputs via incorrect reasoning chains that will fail
systematically on out-of-distribution inputs.

Sub-steps:
- 4a. Define the set of reasoning steps required for the task (clinical
  reasoning graph).
- 4b. For each reasoning step, construct a minimal perturbation that should
  change the intermediate reasoning but not the final answer; verify the
  model's reasoning chain responds appropriately.
- 4c. For LLM-based evaluators: pre-register the rubric; compute inter-rater
  agreement against a human panel on a random 10% sample (target κ ≥ 0.6).

---

## Step 5 — Evaluate Distribution Shift Robustness (targets F6)

*Topic sentence:* Evaluation on a static held-out set measures retrospective
performance; prospective robustness requires explicit testing under the
distribution shifts most likely to occur after deployment.

Sub-steps:
- 5a. Define the three most clinically plausible shift scenarios (e.g.,
  site shift, temporal shift, demographic shift).
- 5b. For each shift: hold out a stratum, train/evaluate on the complement,
  report the performance gap.
- 5c. For temporal shift specifically: if the evaluation set is not time-ordered,
  flag this as an untestable risk rather than reporting a number.

---

## Step 6 — Check Fairness Criterion Compatibility (targets F7)

*Topic sentence:* Before reporting fairness metrics, the evaluation design
must verify that the specified fairness criteria are mutually satisfiable
given the observed base rate differences between groups, because specifying
incompatible criteria produces an evaluation that no model can pass.

Sub-steps:
- 6a. Compute base rates for all protected groups in the evaluation set.
- 6b. Run the stratum-eval fairness compatibility checker; it will flag
  criterion pairs that are mathematically incompatible at the observed
  base rates.
- 6c. If incompatibilities are detected: document the incompatibility,
  select a prioritized criterion set with explicit normative justification,
  and report the trade-off rather than suppressing it.

---

## Step 7 — Assess Sociotechnical Context (targets F8)

*Topic sentence:* The final identification step is the hardest to operationalize
and the most frequently omitted: evaluating whether the model's performance
in isolation is a valid proxy for the performance of the human-AI system
in which it will be embedded.

Sub-steps:
- 7a. Map the clinical workflow: where does the model's output enter the
  decision process, and who acts on it?
- 7b. Identify automation bias risks: what decisions will clinicians defer
  to the model, and what are the consequences if the model is wrong?
- 7c. If a human-in-the-loop simulation is feasible: conduct it; report
  team performance, not just model performance.
- 7d. If simulation is not feasible: explicitly state this as an
  uncharacterized risk in the evaluation report rather than omitting it.
