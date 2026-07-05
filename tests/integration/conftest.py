"""
Collection-time exclusion for test_full_pipeline.py.

This test targets an API surface that doesn't exist on either the
simple or research-grade implementation (stratum.connectors.synthetic.build_synthetic_cohort,
stratum.exceptions.StratumNSDError, stratum.normative.nsd.NormativeSpec,
and an NSD field shape matching neither implementation). A
pytest.mark.skip on the module is NOT sufficient here because the
file's top-level imports fail before any skip marker takes effect —
collection itself must be excluded.

See issue: test_full_pipeline.py targets an API that doesn't exist yet.
Remove this exclusion once that API is deliberately designed and the
test file is rewritten to match it.
"""

collect_ignore = ["test_full_pipeline.py"]
