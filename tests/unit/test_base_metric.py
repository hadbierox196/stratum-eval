"""Tests for BaseMetric and MetricResult."""

import numpy as np
import pytest

from stratum_eval.metrics.base import BaseMetric, MetricResult, MetricUndefinedError


class _DummyMetric(BaseMetric):
    """Minimal concrete metric for testing the base contract."""

    MINIMUM_SAMPLE_SIZE = 5

    def validate_inputs(self, y_true, y_pred):
        self._check_empty(y_true, y_pred)
        self._check_nan_inf(y_true, y_pred)
        self._check_single_class(y_true)
        self._check_min_samples(y_true)

    def compute(self, y_true, y_pred, **kwargs):
        return MetricResult(
            name="dummy",
            value=float(np.mean(y_pred[y_true == 1])),
            n_samples=len(y_true),
        )


@pytest.fixture()
def metric():
    return _DummyMetric()


@pytest.fixture()
def valid_inputs():
    rng = np.random.default_rng(42)
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1], dtype=float)
    y_pred = rng.uniform(0, 1, size=10)
    return y_true, y_pred


class TestMetricResult:
    def test_is_defined_with_real_value(self):
        r = MetricResult(name="x", value=0.75, n_samples=100)
        assert r.is_defined()

    def test_is_defined_with_nan(self):
        r = MetricResult(name="x", value=float("nan"), n_samples=0)
        assert not r.is_defined()


class TestBaseMetricValidation:
    def test_valid_inputs_pass(self, metric, valid_inputs):
        y_true, y_pred = valid_inputs
        result = metric(y_true, y_pred)
        assert result.is_defined()

    def test_empty_arrays_raise(self, metric):
        with pytest.raises(MetricUndefinedError, match="empty"):
            metric(np.array([]), np.array([]))

    def test_nan_in_y_pred_raises(self, metric):
        y_true = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        y_pred = np.array([0.1, float("nan"), 0.3, 0.9, 0.2, 0.8])
        with pytest.raises(MetricUndefinedError, match="NaN"):
            metric(y_true, y_pred)

    def test_single_class_raises(self, metric):
        y_true = np.zeros(10)
        y_pred = np.random.default_rng(0).uniform(size=10)
        with pytest.raises(MetricUndefinedError, match="single class"):
            metric(y_true, y_pred)

    def test_below_minimum_samples_raises(self, metric):
        y_true = np.array([0.0, 1.0, 0.0])
        y_pred = np.array([0.1, 0.9, 0.2])
        with pytest.raises(MetricUndefinedError, match="requires at least"):
            metric(y_true, y_pred)
