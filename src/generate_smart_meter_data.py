"""Command-line entry point for deterministic synthetic data generation."""

from __future__ import annotations

import argparse

from smart_meter_analytics.config import DEFAULT_DATA_PATH
from smart_meter_analytics.synthetic import generate_data, save_data, validate_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an auditable hourly smart-meter panel."
    )
    parser.add_argument("--households", type=int, default=80)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--output", default=str(DEFAULT_DATA_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = generate_data(
        n_households=args.households,
        days=args.days,
        seed=args.seed,
        start=args.start,
    )
    output = save_data(data, args.output)
    quality = validate_data(data)
    print(
        f"Saved {quality['rows']:,} rows from {quality['households']} households "
        f"to {output} (anomaly rate={quality['anomaly_rate']:.3%})."
    )


if __name__ == "__main__":
    main()
