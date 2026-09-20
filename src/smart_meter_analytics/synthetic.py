"""Transparent synthetic data-generating process for smart-meter experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HOUSEHOLD_TYPES = ("low_usage", "medium_usage", "high_usage", "solar_home")
TYPE_PROBABILITIES = (0.30, 0.40, 0.20, 0.10)


def generate_data(
    n_households: int = 80,
    days: int = 60,
    seed: int = 42,
    start: str = "2025-01-01",
) -> pd.DataFrame:
    """Generate hourly household demand with known regimes and injected events.

    The generator deliberately retains latent labels (household type, clean expected
    demand, and anomaly type) so that unsupervised methods can be audited without
    claiming that synthetic performance transfers directly to a real power system.
    """

    if n_households < 2:
        raise ValueError("n_households must be at least 2")
    if days < 2:
        raise ValueError("days must be at least 2")

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start, periods=days * 24, freq="h")
    n_periods = len(timestamps)
    hour = timestamps.hour.to_numpy()
    day_of_week = timestamps.dayofweek.to_numpy()
    is_weekend = (day_of_week >= 5).astype(int)
    elapsed_days = np.arange(n_periods) / 24

    # A shared, forecast-available weather signal. It is identical for every meter.
    temperature_c = (
        19
        + 4.5 * np.sin(2 * np.pi * (hour - 14) / 24)
        + 0.035 * elapsed_days
        + rng.normal(0, 0.55, n_periods)
    )

    frames: list[pd.DataFrame] = []
    base_by_type = {
        "low_usage": 0.30,
        "medium_usage": 0.58,
        "high_usage": 0.95,
        "solar_home": 0.66,
    }
    peak_by_type = {
        "low_usage": 0.42,
        "medium_usage": 0.78,
        "high_usage": 1.28,
        "solar_home": 0.73,
    }

    for household_number in range(1, n_households + 1):
        household_type = str(rng.choice(HOUSEHOLD_TYPES, p=TYPE_PROBABILITIES))
        scale = rng.lognormal(mean=0.0, sigma=0.09)
        base = base_by_type[household_type] * scale
        evening_amplitude = peak_by_type[household_type] * scale
        morning_amplitude = evening_amplitude * rng.uniform(0.35, 0.65)
        phase_jitter = rng.normal(0, 0.30)

        morning = morning_amplitude * np.exp(
            -0.5 * ((hour - (7 + phase_jitter)) / 1.9) ** 2
        )
        evening = evening_amplitude * np.exp(
            -0.5 * ((hour - (20 + phase_jitter)) / 2.6) ** 2
        )
        circadian = 0.07 * np.sin(2 * np.pi * (hour - 5) / 24)
        weekend_multiplier = np.where(is_weekend == 1, rng.uniform(1.05, 1.12), 1.0)
        heating_load = 0.018 * np.maximum(18 - temperature_c, 0) * scale

        solar_offset = np.zeros(n_periods)
        if household_type == "solar_home":
            daylight = np.maximum(np.sin(np.pi * (hour - 7) / 12), 0)
            solar_offset = rng.uniform(0.30, 0.48) * daylight

        deterministic_load = (
            (base + morning + evening + circadian + heating_load)
            * weekend_multiplier
            - solar_offset
        )
        clean_load = np.maximum(
            deterministic_load + rng.normal(0, 0.065, n_periods), 0.02
        )
        observed_load = clean_load.copy()

        event_draw = rng.random(n_periods)
        spike_mask = event_draw < 0.006
        drop_mask = (event_draw >= 0.006) & (event_draw < 0.009)
        observed_load[spike_mask] *= rng.uniform(2.2, 3.8, spike_mask.sum())
        observed_load[drop_mask] *= rng.uniform(0.02, 0.15, drop_mask.sum())
        observed_load = np.maximum(observed_load, 0.02)

        anomaly_type = np.full(n_periods, "none", dtype=object)
        anomaly_type[spike_mask] = "spike"
        anomaly_type[drop_mask] = "drop"

        frames.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "household_id": f"H{household_number:03d}",
                    "household_type": household_type,
                    "hour": hour,
                    "day_of_week": day_of_week,
                    "is_weekend": is_weekend,
                    "temperature_c": np.round(temperature_c, 3),
                    "expected_kwh": np.round(clean_load, 6),
                    "energy_kwh": np.round(observed_load, 6),
                    "true_anomaly": (spike_mask | drop_mask).astype(int),
                    "anomaly_type": anomaly_type,
                }
            )
        )

    data = pd.concat(frames, ignore_index=True)
    validate_data(data, expected_households=n_households, expected_periods=n_periods)
    return data


def validate_data(
    data: pd.DataFrame,
    expected_households: int | None = None,
    expected_periods: int | None = None,
) -> dict[str, int | float | str]:
    """Validate schema and panel integrity; return an auditable quality summary."""

    required = {
        "timestamp",
        "household_id",
        "household_type",
        "hour",
        "day_of_week",
        "is_weekend",
        "energy_kwh",
        "true_anomaly",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data.empty:
        raise ValueError("Dataset is empty")
    if data[list(required)].isna().any().any():
        raise ValueError("Required columns contain missing values")
    if (data["energy_kwh"] < 0).any():
        raise ValueError("energy_kwh must be non-negative")
    if data.duplicated(["household_id", "timestamp"]).any():
        raise ValueError("Duplicate household/timestamp records found")
    if not set(data["true_anomaly"].unique()).issubset({0, 1}):
        raise ValueError("true_anomaly must be binary")

    parsed_time = pd.to_datetime(data["timestamp"], errors="raise")
    counts = data.assign(timestamp=parsed_time).groupby("household_id")["timestamp"].size()
    timestamp_counts = data.assign(timestamp=parsed_time).groupby("household_id")[
        "timestamp"
    ].nunique()
    if counts.nunique() != 1 or not counts.equals(timestamp_counts):
        raise ValueError("Household panels are unbalanced or contain repeated timestamps")
    if expected_households is not None and len(counts) != expected_households:
        raise ValueError("Unexpected number of households")
    if expected_periods is not None and counts.iloc[0] != expected_periods:
        raise ValueError("Unexpected number of periods per household")

    ordered = data.assign(timestamp=parsed_time).sort_values(["household_id", "timestamp"])
    gaps = ordered.groupby("household_id")["timestamp"].diff().dropna()
    if not gaps.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Each household must have a complete hourly time index")

    return {
        "status": "passed",
        "rows": int(len(data)),
        "households": int(data["household_id"].nunique()),
        "timestamps": int(parsed_time.nunique()),
        "missing_values": int(data.isna().sum().sum()),
        "duplicate_meter_hours": int(data.duplicated(["household_id", "timestamp"]).sum()),
        "anomalies": int(data["true_anomaly"].sum()),
        "anomaly_rate": float(data["true_anomaly"].mean()),
        "start": parsed_time.min().isoformat(),
        "end": parsed_time.max().isoformat(),
    }


def save_data(data: pd.DataFrame, path: str | Path) -> Path:
    """Validate and save data using a stable column order."""

    validate_data(data)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False, lineterminator="\n")
    return output_path
