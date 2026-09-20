# Smart Meter Data Analytics

A reproducible Python project for leakage-aware smart-meter segmentation, anomaly detection, and probabilistic forecasting.

This repository builds a synthetic hourly smart-meter dataset, evaluates customer load behavior, trains forecasting models, and measures anomaly detection performance under a strict chronological train/calibration/test split.

## Highlights

- Deterministic synthetic smart-meter data generation
- Leakage-aware temporal partitioning for train/calibration/test evaluation
- Customer segmentation with clustering analysis
- Label-free anomaly detection for injected events
- Forecasting comparison with persistence and random-forest baselines
- Artifacts saved to `figures/` and `reports/`
- Configurable seed and experiment settings via Python dataclasses

## Project structure

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── CITATION.cff
├── src/
│   ├── generate_smart_meter_data.py
│   ├── train_models.py
│   └── smart_meter_analytics/
│       ├── __init__.py
│       ├── config.py
│       ├── modeling.py
│       ├── pipeline.py
│       ├── synthetic.py
│       └── visualization.py
├── tests/
│   ├── test_metrics.py
│   ├── test_synthetic.py
│   └── test_temporal_integrity.py
├── data/
├── figures/
├── reports/
└── notebooks/
```

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/Junaid-EEE11/Smart-Meter-Data-Analytics.git
cd Smart-Meter-Data-Analytics
python -m pip install -U pip
python -m pip install -e .
```

Optional development tools:

```bash
python -m pip install -e .[dev]
```

## Quick start

### 1) Generate synthetic smart-meter data

```bash
smart-meter-generate --households 80 --days 60 --seed 42
```

This creates the default dataset at `data/synthetic_smart_meter_data.csv`.

### 2) Run the analytics experiment

```bash
smart-meter-analyze --seed 42
```

This runs the full pipeline and writes:

- figures in `figures/`
- summary metrics in `reports/metrics.json`
- quality diagnostics in `reports/data_quality.json`
- result summaries in `reports/results.md`
- CSV exports for clustering and forecasting output

## Python entry points

The project exposes two command-line scripts from `pyproject.toml`:

- `smart-meter-generate` → generate synthetic meter data
- `smart-meter-analyze` → run the analysis pipeline

## Model and analysis workflow

The pipeline is organized around a reproducible experiment configuration and end-to-end orchestration:

- `smart_meter_analytics/config.py` defines experiment parameters and dataset paths
- `smart_meter_analytics/synthetic.py` creates the synthetic panel data
- `smart_meter_analytics/modeling.py` contains clustering, anomaly detection, and forecasting logic
- `smart_meter_analytics/pipeline.py` runs the full workflow and writes reports/plots
- `smart_meter_analytics/visualization.py` produces diagnostic charts

## Testing

Run the test suite with:

```bash
pytest
```

The repository includes tests for:

- metric conventions and key outputs
- synthetic data validity
- temporal integrity and leakage-safe evaluation behavior

## Notes

This project is designed for research and reproducibility using synthetic data. Results are useful for validating methodology and software behavior, but they should not be treated as evidence for operational deployment on real-world meter fleets without additional validation using real customer and contextual data.

## License

This repository does not currently declare a custom license in the project metadata. Please check the repository contents and the upstream project policy before reuse in a production or commercial context.

## Citation

If you use this project in academic or research work, the repository includes a `CITATION.cff` file for citation metadata.
