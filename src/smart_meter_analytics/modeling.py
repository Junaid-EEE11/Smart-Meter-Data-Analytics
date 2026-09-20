"""Leakage-aware modeling and evaluation for smart-meter panel data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    normalized_mutual_info_score,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from .config import ExperimentConfig


@dataclass(frozen=True)
class TemporalBoundaries:
    """Shared chronological boundaries used by every predictive task."""

    calibration_start: pd.Timestamp
    test_start: pd.Timestamp
    end: pd.Timestamp


def temporal_boundaries(
    data: pd.DataFrame, config: ExperimentConfig
) -> TemporalBoundaries:
    """Create global time boundaries, never row-order or household boundaries."""

    config.validate()
    timestamps = pd.DatetimeIndex(pd.to_datetime(data["timestamp"]).unique()).sort_values()
    if len(timestamps) < 72:
        raise ValueError("At least 72 hourly timestamps are required for temporal evaluation")
    calibration_index = int(np.floor(len(timestamps) * config.train_fraction))
    test_index = int(
        np.floor(
            len(timestamps)
            * (config.train_fraction + config.calibration_fraction)
        )
    )
    if not 0 < calibration_index < test_index < len(timestamps):
        raise ValueError("Temporal split produced an empty partition")
    return TemporalBoundaries(
        calibration_start=pd.Timestamp(timestamps[calibration_index]),
        test_start=pd.Timestamp(timestamps[test_index]),
        end=pd.Timestamp(timestamps[-1]),
    )


def partition_labels(
    timestamps: pd.Series, boundaries: TemporalBoundaries
) -> pd.Series:
    """Label rows train/calibration/test from their timestamp alone."""

    timestamps = pd.to_datetime(timestamps)
    labels = np.where(
        timestamps < boundaries.calibration_start,
        "train",
        np.where(timestamps < boundaries.test_start, "calibration", "test"),
    )
    return pd.Series(labels, index=timestamps.index, dtype="string")


def create_customer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create interpretable, robust load-shape features for each household."""

    ordered = data.sort_values(["household_id", "timestamp"]).copy()
    grouped = ordered.groupby("household_id", sort=True)
    features = grouped["energy_kwh"].agg(
        mean_kwh="mean",
        std_kwh="std",
        median_kwh="median",
        p95_kwh=lambda values: values.quantile(0.95),
    )

    windows = {
        "night_kwh": ordered["hour"].between(0, 5),
        "morning_kwh": ordered["hour"].between(6, 9),
        "midday_kwh": ordered["hour"].between(10, 15),
        "evening_kwh": ordered["hour"].between(18, 22),
    }
    for name, mask in windows.items():
        features[name] = ordered.loc[mask].groupby("household_id")["energy_kwh"].mean()

    weekend_mean = ordered.loc[ordered["is_weekend"].eq(1)].groupby("household_id")[
        "energy_kwh"
    ].mean()
    weekday_mean = ordered.loc[ordered["is_weekend"].eq(0)].groupby("household_id")[
        "energy_kwh"
    ].mean()
    ramp = grouped["energy_kwh"].diff().abs().groupby(ordered["household_id"]).mean()

    features["load_factor"] = features["mean_kwh"] / features["p95_kwh"].clip(1e-6)
    features["weekend_ratio"] = weekend_mean / weekday_mean.clip(1e-6)
    features["mean_abs_ramp"] = ramp
    features["midday_to_night"] = (
        features["midday_kwh"] / features["night_kwh"].clip(1e-6)
    )
    return features.replace([np.inf, -np.inf], np.nan).fillna(0).reset_index()


def run_clustering(data: pd.DataFrame, seed: int = 42) -> dict[str, object]:
    """Select K by silhouette score and audit clusters against latent types."""

    profiles = create_customer_features(data)
    type_labels = (
        data.groupby("household_id", sort=True)["household_type"].first().reindex(
            profiles["household_id"]
        )
    )
    feature_columns = [column for column in profiles.columns if column != "household_id"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(profiles[feature_columns])

    max_k = min(6, len(profiles) - 1)
    if max_k < 2:
        raise ValueError("At least three households are required for clustering")
    selection_rows: list[dict[str, float | int]] = []
    fitted_models: dict[int, KMeans] = {}
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, n_init=25, random_state=seed)
        labels = model.fit_predict(scaled)
        fitted_models[k] = model
        selection_rows.append(
            {
                "k": k,
                "silhouette_score": float(silhouette_score(scaled, labels)),
                "adjusted_rand_index": float(adjusted_rand_score(type_labels, labels)),
                "normalized_mutual_information": float(
                    normalized_mutual_info_score(type_labels, labels)
                ),
            }
        )

    selection = pd.DataFrame(selection_rows)
    selected_k = int(selection.loc[selection["silhouette_score"].idxmax(), "k"])
    labels = fitted_models[selected_k].labels_
    profiles["cluster"] = labels

    profiles["household_type"] = type_labels.to_numpy()
    projection = PCA(n_components=2, random_state=seed).fit_transform(scaled)
    profiles["pc1"] = projection[:, 0]
    profiles["pc2"] = projection[:, 1]

    latent_k = int(type_labels.nunique())
    latent_row = selection.loc[selection["k"].eq(latent_k)]
    metrics = {
        "selected_k": selected_k,
        "silhouette_score": float(selection["silhouette_score"].max()),
        "adjusted_rand_index_vs_latent_type": float(
            adjusted_rand_score(type_labels, labels)
        ),
        "normalized_mutual_information_vs_latent_type": float(
            normalized_mutual_info_score(type_labels, labels)
        ),
        "latent_regime_count": latent_k,
        "ari_at_latent_regime_count": (
            float(latent_row["adjusted_rand_index"].iloc[0]) if not latent_row.empty else None
        ),
        "silhouette_at_latent_regime_count": (
            float(latent_row["silhouette_score"].iloc[0]) if not latent_row.empty else None
        ),
        "fit_scope": "training period only",
    }
    return {"profiles": profiles, "selection": selection, "metrics": metrics}


def _anomaly_features(
    data: pd.DataFrame, reference: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """Construct context-normalized features using training-period references."""

    ordered = data.sort_values(["household_id", "timestamp"]).copy()
    medians = (
        reference.groupby(["household_id", "hour"])["energy_kwh"]
        .median()
        .rename("typical_kwh")
        .reset_index()
    )
    household_fallback = reference.groupby("household_id")["energy_kwh"].median()
    ordered = ordered.merge(medians, how="left", on=["household_id", "hour"])
    ordered["typical_kwh"] = ordered["typical_kwh"].fillna(
        ordered["household_id"].map(household_fallback)
    )
    ordered["typical_kwh"] = ordered["typical_kwh"].fillna(
        reference["energy_kwh"].median()
    )

    previous = ordered.groupby("household_id")["energy_kwh"].shift(1)
    ordered["relative_level"] = ordered["energy_kwh"] / ordered["typical_kwh"].clip(1e-3)
    ordered["absolute_deviation"] = (
        ordered["energy_kwh"] - ordered["typical_kwh"]
    ).abs()
    ordered["relative_change"] = (
        (ordered["energy_kwh"] - previous).abs() / previous.clip(1e-3)
    ).fillna(0).clip(upper=20)
    ordered["hour_sin"] = np.sin(2 * np.pi * ordered["hour"] / 24)
    ordered["hour_cos"] = np.cos(2 * np.pi * ordered["hour"] / 24)
    feature_columns = [
        "energy_kwh",
        "relative_level",
        "absolute_deviation",
        "relative_change",
        "hour_sin",
        "hour_cos",
        "is_weekend",
    ]
    return ordered, feature_columns


def run_anomaly_detection(
    data: pd.DataFrame,
    boundaries: TemporalBoundaries,
    config: ExperimentConfig,
) -> dict[str, object]:
    """Fit without labels on training data and score only the held-out test period."""

    labels = partition_labels(data["timestamp"], boundaries)
    reference = data.loc[labels.eq("train")]
    featured, feature_columns = _anomaly_features(data, reference)
    featured["partition"] = partition_labels(featured["timestamp"], boundaries).to_numpy()

    train = featured.loc[featured["partition"].eq("train")]
    test = featured.loc[featured["partition"].eq("test")].copy()
    model = make_pipeline(
        RobustScaler(),
        IsolationForest(
            n_estimators=250,
            contamination=config.anomaly_contamination,
            max_samples="auto",
            random_state=config.seed,
            n_jobs=-1,
        ),
    )
    model.fit(train[feature_columns])
    test["predicted_anomaly"] = (model.predict(test[feature_columns]) == -1).astype(int)
    test["anomaly_score"] = -model.decision_function(test[feature_columns])

    truth = test["true_anomaly"].astype(int)
    prediction = test["predicted_anomaly"]
    metrics = {
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "average_precision": float(average_precision_score(truth, test["anomaly_score"])),
        "roc_auc": float(roc_auc_score(truth, test["anomaly_score"])),
        "test_anomalies": int(truth.sum()),
        "flagged_observations": int(prediction.sum()),
        "evaluation_scope": "held-out test period only",
        "labels_used_for_training": False,
    }
    matrix = confusion_matrix(truth, prediction, labels=[0, 1])
    return {
        "model": model,
        "test_predictions": test,
        "metrics": metrics,
        "confusion_matrix": matrix,
    }


FORECAST_FEATURES = [
    "lag_1h",
    "lag_24h",
    "rolling_mean_24h",
    "rolling_std_24h",
    "hour_sin",
    "hour_cos",
    "week_sin",
    "week_cos",
    "is_weekend",
    "temperature_c",
]


def create_forecast_features(data: pd.DataFrame) -> pd.DataFrame:
    """Construct causal features independently within each household panel."""

    frame = data.sort_values(["household_id", "timestamp"]).copy()
    grouped = frame.groupby("household_id", sort=False)["energy_kwh"]
    frame["lag_1h"] = grouped.shift(1)
    frame["lag_24h"] = grouped.shift(24)
    frame["rolling_mean_24h"] = grouped.transform(
        lambda values: values.shift(1).rolling(24, min_periods=24).mean()
    )
    frame["rolling_std_24h"] = grouped.transform(
        lambda values: values.shift(1).rolling(24, min_periods=24).std()
    )
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
    frame["week_sin"] = np.sin(2 * np.pi * frame["day_of_week"] / 7)
    frame["week_cos"] = np.cos(2 * np.pi * frame["day_of_week"] / 7)
    frame["target_kwh"] = frame["energy_kwh"]
    return frame.dropna(subset=FORECAST_FEATURES + ["target_kwh"]).reset_index(drop=True)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return complementary point-forecast metrics."""

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = np.abs(actual) + np.abs(predicted)
    smape = 100 * np.mean(np.divide(2 * np.abs(predicted - actual), denominator + 1e-8))
    return {
        "mae_kwh": float(mean_absolute_error(actual, predicted)),
        "rmse_kwh": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "smape_percent": float(smape),
    }


def _daily_block_mae_interval(
    timestamps: pd.Series,
    actual: np.ndarray,
    predicted: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    """Estimate a 95% MAE interval by resampling whole test days."""

    errors = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps).dt.date.to_numpy(),
            "absolute_error": np.abs(np.asarray(actual) - np.asarray(predicted)),
        }
    )
    daily = errors.groupby("date")["absolute_error"].mean().to_numpy()
    rng = np.random.default_rng(seed)
    samples = rng.choice(daily, size=(iterations, len(daily)), replace=True).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return float(lower), float(upper)


def run_forecasting(
    data: pd.DataFrame,
    boundaries: TemporalBoundaries,
    config: ExperimentConfig,
) -> dict[str, object]:
    """Benchmark a global one-hour-ahead model against two strong naive baselines."""

    frame = create_forecast_features(data)
    frame["partition"] = partition_labels(frame["timestamp"], boundaries).to_numpy()
    train = frame.loc[frame["partition"].eq("train")]
    calibration = frame.loc[frame["partition"].eq("calibration")]
    test = frame.loc[frame["partition"].eq("test")].copy()
    if train.empty or calibration.empty or test.empty:
        raise ValueError("Forecast split contains an empty partition")

    model = RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=config.seed,
        n_jobs=-1,
    )
    model.fit(train[FORECAST_FEATURES], train["target_kwh"])
    calibration_prediction = model.predict(calibration[FORECAST_FEATURES])
    test["random_forest"] = model.predict(test[FORECAST_FEATURES])
    test["persistence"] = test["lag_1h"]
    test["seasonal_naive"] = test["lag_24h"]

    metrics_rows: list[dict[str, float | str]] = []
    for name in ("persistence", "seasonal_naive", "random_forest"):
        row: dict[str, float | str] = {"model": name}
        row.update(regression_metrics(test["target_kwh"].to_numpy(), test[name].to_numpy()))
        if name == "random_forest":
            lower, upper = _daily_block_mae_interval(
                test["timestamp"],
                test["target_kwh"].to_numpy(),
                test[name].to_numpy(),
                config.bootstrap_iterations,
                config.seed,
            )
            row["mae_ci95_lower_kwh"] = lower
            row["mae_ci95_upper_kwh"] = upper
        metrics_rows.append(row)
    metrics = pd.DataFrame(metrics_rows)

    calibration_error = np.abs(
        calibration["target_kwh"].to_numpy() - calibration_prediction
    )
    quantile_level = min(
        1.0,
        np.ceil((len(calibration_error) + 1) * (1 - config.interval_alpha))
        / len(calibration_error),
    )
    radius = float(np.quantile(calibration_error, quantile_level, method="higher"))
    test["interval_lower"] = np.maximum(test["random_forest"] - radius, 0)
    test["interval_upper"] = test["random_forest"] + radius
    interval_coverage = float(
        (
            test["target_kwh"].between(
                test["interval_lower"], test["interval_upper"], inclusive="both"
            )
        ).mean()
    )
    interval_metrics = {
        "nominal_coverage": float(1 - config.interval_alpha),
        "empirical_coverage": interval_coverage,
        "mean_interval_width_kwh": float(
            (test["interval_upper"] - test["interval_lower"]).mean()
        ),
        "calibration_residual_quantile_kwh": radius,
        "method": "split conformal absolute residual",
    }
    importance = pd.DataFrame(
        {"feature": FORECAST_FEATURES, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False, ignore_index=True)
    return {
        "model": model,
        "test_predictions": test,
        "metrics": metrics,
        "interval_metrics": interval_metrics,
        "feature_importance": importance,
    }
