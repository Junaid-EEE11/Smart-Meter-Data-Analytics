"""Regression tests for the leakage controls in the experiment design."""

import numpy as np
import pandas as pd

from smart_meter_analytics.config import ExperimentConfig
from smart_meter_analytics.modeling import (
    create_forecast_features,
    partition_labels,
    temporal_boundaries,
)
from smart_meter_analytics.synthetic import generate_data


def test_global_split_is_strictly_chronological_for_every_household() -> None:
    data = generate_data(n_households=4, days=10, seed=3)
    boundaries = temporal_boundaries(data, ExperimentConfig())
    labels = partition_labels(data["timestamp"], boundaries)

    train_times = data.loc[labels.eq("train"), "timestamp"]
    calibration_times = data.loc[labels.eq("calibration"), "timestamp"]
    test_times = data.loc[labels.eq("test"), "timestamp"]
    assert train_times.max() < calibration_times.min()
    assert calibration_times.max() < test_times.min()
    assert data.loc[labels.eq("test"), "household_id"].nunique() == 4


def test_rolling_features_never_cross_household_boundaries() -> None:
    timestamps = pd.date_range("2025-01-01", periods=48, freq="h")
    frame = pd.DataFrame(
        {
            "timestamp": list(timestamps) * 2,
            "household_id": ["A"] * 48 + ["B"] * 48,
            "hour": list(timestamps.hour) * 2,
            "day_of_week": list(timestamps.dayofweek) * 2,
            "is_weekend": [0] * 96,
            "temperature_c": [20.0] * 96,
            "energy_kwh": list(np.arange(1, 49, dtype=float))
            + list(np.arange(1001, 1049, dtype=float)),
        }
    )

    featured = create_forecast_features(frame)
    first_b = featured.loc[featured["household_id"].eq("B")].iloc[0]
    assert first_b["timestamp"] == timestamps[24]
    assert first_b["lag_1h"] == 1024
    assert first_b["rolling_mean_24h"] == np.mean(np.arange(1001, 1025))
