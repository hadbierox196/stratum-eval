# Metric Standards for stratum-eval

Every metric submitted to stratum-eval must fully specify the six fields below
before a pull request can be merged. A metric that cannot answer one of these
fields honestly should not exist in this framework.

The goal is not bureaucracy. It is epistemic discipline: a metric that cannot
state what it cannot measure is more dangerous than no metric at all.

---

## The Six Required Fields

### Field 1 — What It Measures

State, in one or two precise sentences, the quantity this metric computes.
Avoid hedges like "captures" or "reflects." Name the mathematical object.

**Template**
MEASURES:  computes  over ,
returning a value in  where .
**Example (AUROC)**
MEASURES: AUROC computes the probability that a randomly chosen positive
instance is ranked higher than a randomly chosen negative instance by the
model score, returning a value in [0, 1] where 0.5 denotes random performance
and 1.0 denotes perfect rank separation.
---

### Field 2 — What It Cannot Measure

State at least three explicit limitations. These must be falsifiable claims
about the metric's blind spots, not generic disclaimers.

**Required sub-fields**
- `CANNOT_MEASURE_CALIBRATION`: yes/no + one sentence explanation
- `CANNOT_MEASURE_SUBGROUP`: yes/no + which subgroup failures are invisible
- `CANNOT_MEASURE_DISTRIBUTION_SHIFT`: yes/no + explanation
- `OTHER`: at least one domain-specific limitation

**Anti-patterns to avoid**
- ❌ "May not generalize to all settings" (too vague)
- ❌ "Should be used alongside other metrics" (not a limitation statement)
- ✅ "Cannot detect a model that achieves 0.95 AUROC by exploiting a
     spurious correlation with patient age that disappears under site shift"

---

### Field 3 — Required Assumptions

List every assumption that must hold for the metric to be valid. Violations
must be detectable or at minimum acknowledgeably untestable.

**Required entries**
| ID | Assumption | Testable? | Consequence if violated |
|----|-----------|-----------|------------------------|
| A1 | | yes / no / partial | |
| A2 | | | |
| …  | | | |

**Common assumptions to evaluate explicitly**
- IID sampling between train and eval set
- Label noise is non-differential (errors are not correlated with the feature
  being predicted)
- The positive class is clinically homogeneous
- Annotator agreement is sufficient to treat labels as ground truth
- Metric is invariant to the prevalence in the evaluation cohort (or: it is not,
  and the prevalence must be stated)

---

### Field 4 — Graceful Failure Behavior

Specify what the metric returns, logs, or raises under each pathological input.
The implementation must match this specification exactly.

**Required cases**

| Condition | Expected behavior | Error type |
|-----------|------------------|------------|
| All predictions identical (zero variance) | | |
| All labels identical (single class) | | |
| NaN or Inf in predictions | | |
| Empty input arrays | | |
| Prevalence < 0.01 or > 0.99 | | |
| n < minimum_sample_size | | |

Acceptable behaviors: return `NaN` with a logged warning, raise a typed
`MetricUndefinedError` with a human-readable message, or return a sentinel
value documented in the class docstring. Silent incorrect values are never
acceptable.

**Implementation requirement**

Every metric must implement `validate_inputs()` that is called before
`compute()`. Any condition listed above that is not covered by
`validate_inputs()` is a blocking defect.

---

### Field 5 — Falsifiability Condition

State a concrete empirical scenario in which a model that is genuinely good
would score poorly on this metric, AND a scenario in which a model that is
genuinely bad would score well. If you cannot construct both scenarios, the
metric is not falsifiable and must not be merged.

**Template**
FALSE_NEGATIVE: A model that  will score <poorly / below threshold>
on this metric when .
FALSE_POSITIVE: A model that  will score <well / above threshold>
on this metric when .
**Example (macro-F1 on a 3-class triage task)**

FALSE_NEGATIVE: A model with near-perfect sensitivity on the rarest, most
critical class will score below 0.5 macro-F1 if it performs at chance on the
two common classes, even though its clinical failure mode is on low-severity cases.
FALSE_POSITIVE: A model that always predicts the two common classes correctly
and always misclassifies the critical class achieves macro-F1 ≈ 0.67 while
missing every high-acuity case.

---

### Field 6 — Computational Complexity

State time and space complexity in Big-O notation as a function of the
parameters that actually matter for the use case (n = number of instances,
c = number of classes, b = number of bootstrap replicates, etc.).

**Required entries**

TIME_COMPLEXITY: O(...)
SPACE_COMPLEXITY: O(...)
PARALLELIZABLE: yes / no / partial
BOOTSTRAP_MULTIPLIER: 
MINIMUM_SAMPLE_SIZE: <integer, with citation or derivation>
RECOMMENDED_SAMPLE_SIZE: <integer for 80% power at clinically meaningful delta>

If the metric depends on an external model call (e.g., LLM-as-judge), state:
- Number of API calls per evaluation instance
- Whether calls can be batched
- Estimated cost per 1,000 instances at current pricing (note the pricing date)

---

## Anti-Pattern Catalogue

The following patterns have appeared in prior metric PRs and are grounds for
immediate request-for-changes.

### AP-1: The Unmarked Aggregate

A metric that averages over a demographic or site dimension without disclosing
it. Example: reporting mean AUROC across sites when site-level variance is
high. **Requirement:** if a metric aggregates, the aggregation function and
grouping variable must be named in Field 1.

### AP-2: The Optimistic Undefined

Returning 1.0 or 0.0 instead of NaN when the metric is undefined (e.g.,
precision when no positives are predicted). **Requirement:** covered by
Field 4 — must be NaN + warning.

### AP-3: The Floating Threshold

A binary metric reported without specifying how the operating threshold was
chosen. If the threshold is tuned on the evaluation set, the metric is
measuring optimization, not performance. **Requirement:** threshold selection
procedure must be in Field 3 assumptions.

### AP-4: The Unnamed Annotator Model

A reference-based NLG metric (e.g., BERTScore, ROUGE) used without stating
which model version produced the reference embeddings or reference texts.
**Requirement:** model name, version, and access date must appear in Field 3.

### AP-5: The Unfalsified Rubric

An LLM-as-judge rubric where the rubric designer is also the model developer.
**Requirement:** rubric must be pre-registered before evaluation data is seen,
or evaluated for inter-rater agreement against a blinded human panel (κ ≥ 0.6).

### AP-6: Complexity Laundering

Wrapping an O(n²) operation in a utility function and reporting the metric as
O(n). **Requirement:** complexity must reflect the full call stack.

---

## Enforcement

These fields are checked by the `stratum-eval metric lint` CLI command and by
the `MetricStandardsChecker` class in `stratum_eval/utils/standards.py`.
CI will fail on any metric that does not pass the linter. Human reviewers are
expected to read Field 2 and Field 5 with particular skepticism — these are
the fields most likely to be filled in optimistically rather than honestly.
