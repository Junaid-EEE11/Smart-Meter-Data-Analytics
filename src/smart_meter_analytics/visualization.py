"""Publication-style figures for the experiment outputs."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve


COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#5B6573",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "legend.frameon": False,
        }
    )


def plot_load_profiles(data: pd.DataFrame, output_path) -> None:
    """Plot mean diurnal profiles for the known simulation regimes."""

    _style()
    profile = data.groupby(["household_type", "hour"])["energy_kwh"].mean().unstack(0)
    color_cycle = [COLORS["blue"], COLORS["orange"], COLORS["red"], COLORS["green"]]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for color, household_type in zip(color_cycle, profile.columns, strict=False):
        ax.plot(profile.index, profile[household_type], label=household_type, color=color, lw=2)
    ax.set(
        title="Mean diurnal electricity demand by latent household regime",
        xlabel="Hour of day",
        ylabel="Mean energy (kWh per hour)",
        xticks=np.arange(0, 24, 3),
    )
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_diagnostics(result: dict[str, object], output_path) -> None:
    """Show model selection and the final two-dimensional PCA projection."""

    _style()
    selection = result["selection"]
    profiles = result["profiles"]
    metrics = result["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(
        selection["k"], selection["silhouette_score"], marker="o", color=COLORS["blue"]
    )
    selected_k = metrics["selected_k"]
    chosen = selection.loc[selection["k"].eq(selected_k), "silhouette_score"].iloc[0]
    axes[0].scatter([selected_k], [chosen], s=90, color=COLORS["orange"], zorder=3)
    latent_k = metrics["latent_regime_count"]
    axes[0].axvline(
        latent_k,
        color=COLORS["green"],
        ls="--",
        lw=1.4,
        label=f"Known simulation regimes (K={latent_k})",
    )
    axes[0].set(
        title="Cluster-count selection",
        xlabel="Number of clusters (K)",
        ylabel="Silhouette score",
        xticks=selection["k"],
    )
    axes[0].legend(loc="lower left")

    scatter = axes[1].scatter(
        profiles["pc1"],
        profiles["pc2"],
        c=profiles["cluster"],
        cmap="viridis",
        s=55,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.4,
    )
    axes[1].set(
        title=f"Customer segments (K={selected_k}, PCA display only)",
        xlabel="Principal component 1",
        ylabel="Principal component 2",
    )
    legend = axes[1].legend(*scatter.legend_elements(), title="Cluster", loc="best")
    axes[1].add_artist(legend)
    fig.suptitle("Unsupervised customer segmentation", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_anomaly_diagnostics(result: dict[str, object], confusion_path, curve_path) -> None:
    """Plot held-out confusion matrix and precision-recall curve."""

    _style()
    predictions = result["test_predictions"]
    matrix = result["confusion_matrix"]
    metrics = result["metrics"]

    fig, ax = plt.subplots(figsize=(5.4, 4.7))
    display = ConfusionMatrixDisplay(matrix, display_labels=["Normal", "Injected event"])
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format=",d")
    ax.grid(False)
    ax.set_title(f"Held-out anomaly detection (F1={metrics['f1']:.3f})")
    fig.tight_layout()
    fig.savefig(confusion_path, bbox_inches="tight")
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(
        predictions["true_anomaly"], predictions["anomaly_score"]
    )
    prevalence = predictions["true_anomaly"].mean()
    fig, ax = plt.subplots(figsize=(6.2, 4.7))
    ax.plot(recall, precision, color=COLORS["blue"], lw=2, label="Isolation Forest")
    ax.axhline(
        prevalence,
        color=COLORS["gray"],
        ls="--",
        label=f"Random baseline ({prevalence:.3f})",
    )
    ax.set(
        title=f"Held-out precision-recall curve (AP={metrics['average_precision']:.3f})",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(curve_path, bbox_inches="tight")
    plt.close(fig)


def plot_forecast_diagnostics(result: dict[str, object], comparison_path, trace_path) -> None:
    """Plot baseline comparison and a chronological household forecast trace."""

    _style()
    metrics = result["metrics"]
    predictions = result["test_predictions"]
    labels = {
        "persistence": "Persistence",
        "seasonal_naive": "Seasonal naive",
        "random_forest": "Random forest",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ordered = metrics.sort_values("mae_kwh", ascending=False)
    colors = [
        COLORS["gray"] if name != "random_forest" else COLORS["blue"]
        for name in ordered["model"]
    ]
    bars = ax.barh(
        [labels[name] for name in ordered["model"]], ordered["mae_kwh"], color=colors
    )
    ax.bar_label(bars, fmt="%.3f", padding=4)
    ax.set(
        title="One-hour-ahead forecast benchmark on the held-out period",
        xlabel="Mean absolute error (kWh; lower is better)",
        ylabel="",
    )
    ax.set_xlim(0, ordered["mae_kwh"].max() * 1.18)
    fig.tight_layout()
    fig.savefig(comparison_path, bbox_inches="tight")
    plt.close(fig)

    sample_household = str(predictions["household_id"].min())
    sample = (
        predictions.loc[predictions["household_id"].eq(sample_household)]
        .sort_values("timestamp")
        .head(24 * 7)
    )
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.fill_between(
        sample["timestamp"],
        sample["interval_lower"],
        sample["interval_upper"],
        color=COLORS["blue"],
        alpha=0.16,
        label="95% conformal interval",
    )
    ax.plot(
        sample["timestamp"], sample["target_kwh"], color="black", lw=1.2, label="Observed"
    )
    ax.plot(
        sample["timestamp"],
        sample["random_forest"],
        color=COLORS["blue"],
        lw=1.5,
        label="Random forest",
    )
    ax.set(
        title=f"Seven-day held-out forecast trace — household {sample_household}",
        xlabel="Timestamp",
        ylabel="Energy (kWh per hour)",
    )
    ax.legend(ncol=3, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(trace_path, bbox_inches="tight")
    plt.close(fig)
