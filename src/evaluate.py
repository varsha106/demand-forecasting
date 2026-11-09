"""Time-based validation and forecast metrics.

Random splits leak future information into training for time series, so this
project validates the way a production forecasting system is validated:
expanding-window walk-forward folds, plus a final untouched holdout.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    y_true = np.asarray(y_true, dtype=float)
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE_%": mape(y_true, y_pred),
        "sMAPE_%": smape(y_true, y_pred),
    }


@dataclass
class Fold:
    """One walk-forward fold: train on [start, cutoff), test on [cutoff, cutoff+horizon)."""

    train_end: pd.Timestamp  # exclusive
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # exclusive


def walk_forward_folds(
    dates: pd.Series, n_folds: int = 3, horizon: int = 28, min_train_days: int = 365
) -> list[Fold]:
    """Expanding-window folds anchored to the most recent data."""
    unique_dates = np.sort(dates.unique())
    last = pd.Timestamp(unique_dates[-1]) + pd.Timedelta(days=1)
    folds = []
    for i in range(n_folds):
        test_end = last - pd.Timedelta(days=horizon * i)
        test_start = test_end - pd.Timedelta(days=horizon)
        train_days = (test_start - pd.Timestamp(unique_dates[0])).days
        if train_days < min_train_days:
            break
        folds.append(Fold(train_end=test_start, test_start=test_start, test_end=test_end))
    return list(reversed(folds))
