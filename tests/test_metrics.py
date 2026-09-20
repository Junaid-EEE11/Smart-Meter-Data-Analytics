"""Unit tests for reported evaluation metrics."""

import numpy as np

from smart_meter_analytics.modeling import regression_metrics


def test_regression_metrics_are_zero_for_perfect_predictions() -> None:
    actual = np.array([0.2, 0.5, 1.1, 0.0])
    metrics = regression_metrics(actual, actual.copy())

    assert metrics["mae_kwh"] == 0
    assert metrics["rmse_kwh"] == 0
    assert metrics["r2"] == 1
    assert metrics["smape_percent"] == 0
