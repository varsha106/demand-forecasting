"""Feature engineering for tabular (gradient boosting) forecasting.

All features are computed per store/product series and use only information
available at forecast time: lagged sales, rolling statistics of lagged sales
(shifted so the target day never leaks into its own features), calendar
attributes, price and promotion flags.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 28]
GROUP_COLS = ["store", "product"]

FEATURE_COLUMNS = (
    [f"lag_{lag}" for lag in LAGS]
    + [f"roll_mean_{w}" for w in ROLLING_WINDOWS]
    + [f"roll_std_{w}" for w in ROLLING_WINDOWS]
    + [
        "dayofweek",
        "dayofmonth",
        "month",
        "weekofyear",
        "is_weekend",
        "dow_sin",
        "dow_cos",
        "price",
        "promo",
    ]
)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dayofweek"] = df["date"].dt.day_of_week
    df["dayofmonth"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a modeling frame with lag/rolling features and the target."""
    df = add_calendar_features(df)
    grouped = df.groupby(GROUP_COLS)["units"]
    for lag in LAGS:
        df[f"lag_{lag}"] = grouped.shift(lag)
    for w in ROLLING_WINDOWS:
        shifted = grouped.shift(1)
        df[f"roll_mean_{w}"] = shifted.groupby([df["store"], df["product"]]).transform(
            lambda s: s.rolling(w, min_periods=1).mean()
        )
        df[f"roll_std_{w}"] = shifted.groupby([df["store"], df["product"]]).transform(
            lambda s: s.rolling(w, min_periods=1).std()
        )
    df = df.dropna(subset=[f"lag_{max(LAGS)}"]).reset_index(drop=True)
    return df
