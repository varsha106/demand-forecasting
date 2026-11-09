"""Forecast plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_forecast_comparison(
    actual: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    series_label: str,
    out_path: str | Path,
) -> Path:
    """Actual vs. predicted units for one series over the holdout window.

    `predictions` maps model name -> frame with columns date, y_pred.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(actual["date"], actual["units"], label="Actual", color="black", linewidth=1.6)
    for name, pred in predictions.items():
        pred = pred.sort_values("date")
        ax.plot(pred["date"], pred["y_pred"], label=name, linewidth=1.2, alpha=0.85)
    ax.set_title(f"Holdout forecast — {series_label}")
    ax.set_ylabel("Units sold")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_metrics_bar(metrics: pd.DataFrame, metric: str, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(metrics.index, metrics[metric], color=["#4C78A8", "#F58518"][: len(metrics)])
    ax.set_title(f"Model comparison — {metric} (lower is better)")
    ax.set_ylabel(metric)
    for i, v in enumerate(metrics[metric]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
