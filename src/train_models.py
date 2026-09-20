"""Command-line entry point for the reproducible analytics experiment."""

from __future__ import annotations

import argparse
import json

from smart_meter_analytics.config import DEFAULT_DATA_PATH, ExperimentConfig
from smart_meter_analytics.modeling import (
    create_customer_features,
    create_forecast_features,
    run_anomaly_detection,
    run_clustering,
    run_forecasting,
    temporal_boundaries,
)
from smart_meter_analytics.pipeline import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-aware segmentation, anomaly detection, and forecasting."
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate the default synthetic dataset before analysis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_experiment(
        data_path=args.data,
        config=ExperimentConfig(seed=args.seed),
        regenerate=args.regenerate,
    )
    summary = {
        "clustering": results["clustering"],
        "anomaly_detection": results["anomaly_detection"],
        "forecasting": results["forecasting"],
    }
    print(json.dumps(summary, indent=2))
    print("\nArtifacts written to figures/ and reports/.")


if __name__ == "__main__":
    main()


__all__ = [
    "create_customer_features",
    "create_forecast_features",
    "run_anomaly_detection",
    "run_clustering",
    "run_forecasting",
    "temporal_boundaries",
]
