"""Tests for deterministic generation and panel-data invariants."""

import pandas as pd
import pytest

from smart_meter_analytics.synthetic import generate_data, validate_data


def test_generation_is_deterministic_and_schema_is_valid() -> None:
    first = generate_data(n_households=5, days=3, seed=7)
    second = generate_data(n_households=5, days=3, seed=7)

    pd.testing.assert_frame_equal(first, second)
    quality = validate_data(first, expected_households=5, expected_periods=72)
    assert quality["status"] == "passed"
    assert quality["rows"] == 360
    assert (first["energy_kwh"] >= 0).all()


def test_validation_rejects_duplicate_meter_hours() -> None:
    data = generate_data(n_households=3, days=2, seed=11)
    malformed = pd.concat([data, data.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate"):
        validate_data(malformed)
