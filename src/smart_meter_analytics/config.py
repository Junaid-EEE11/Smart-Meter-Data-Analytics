"""Shared configuration for the research pipeline."""

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "reports"
DEFAULT_DATA_PATH = DATA_DIR / "synthetic_smart_meter_data.csv"


@dataclass(frozen=True)
class ExperimentConfig:
    """Parameters that define one reproducible experiment."""

    seed: int = 42
    train_fraction: float = 0.70
    calibration_fraction: float = 0.10
    interval_alpha: float = 0.05
    anomaly_contamination: float = 0.01
    bootstrap_iterations: int = 1_000

    def validate(self) -> None:
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1")
        if not 0 < self.calibration_fraction < 1:
            raise ValueError("calibration_fraction must be between 0 and 1")
        if self.train_fraction + self.calibration_fraction >= 1:
            raise ValueError("train and calibration fractions must leave a test set")
        if not 0 < self.interval_alpha < 1:
            raise ValueError("interval_alpha must be between 0 and 1")
        if not 0 < self.anomaly_contamination < 0.5:
            raise ValueError("anomaly_contamination must be between 0 and 0.5")
