"""Sample data generation and loading.

The bundled dataset is *synthetic* retail sales data generated with a fixed
seed. It reproduces the structure of the production data this project is
modeled on (daily unit sales per store/product, with price, promotions and
calendar effects) without containing any real company data.

Each series combines:
  * a base level and slow trend per store/product
  * weekly seasonality (weekend uplift) and yearly seasonality
  * random promotions (price cut -> demand lift)
  * holiday spikes and multiplicative noise
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

STORES = ["store_a", "store_b", "store_c"]
PRODUCTS = ["product_1", "product_2"]
BASE_LEVEL = {
    ("store_a", "product_1"): 120.0,
    ("store_a", "product_2"): 65.0,
    ("store_b", "product_1"): 95.0,
    ("store_b", "product_2"): 48.0,
    ("store_c", "product_1"): 80.0,
    ("store_c", "product_2"): 55.0,
}
BASE_PRICE = {"product_1": 24.99, "product_2": 14.49}


def generate_sample_data(
    start: str = "2023-01-01",
    periods: int = 730,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic daily sales for every store/product pair."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=periods, freq="D")
    t = np.arange(periods)

    frames = []
    for store in STORES:
        for product in PRODUCTS:
            base = BASE_LEVEL[(store, product)]
            trend = 1.0 + 0.0004 * t * rng.uniform(0.6, 1.4)
            weekly = 1.0 + 0.28 * (dates.dayofweek >= 5) - 0.08 * (dates.dayofweek == 0)
            yearly = 1.0 + 0.18 * np.sin(2 * np.pi * (dates.dayofyear - 30) / 365.25)
            promo = (rng.random(periods) < 0.10).astype(int)
            promo_lift = 1.0 + promo * rng.uniform(0.25, 0.45)
            price = BASE_PRICE[product] * (1.0 - 0.15 * promo) * (1.0 + 0.01 * np.sin(t / 53.0))
            holiday = ((dates.month == 12) & (dates.day >= 18)).astype(float) * 0.35
            noise = rng.normal(1.0, 0.09, periods)

            units = base * trend * weekly * yearly * promo_lift * (1.0 + holiday) * noise
            units = np.clip(np.round(units), 0, None).astype(int)

            frames.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "store": store,
                        "product": product,
                        "units": units,
                        "price": np.round(price, 2),
                        "promo": promo,
                    }
                )
            )
    df = pd.concat(frames, ignore_index=True).sort_values(["store", "product", "date"])
    return df.reset_index(drop=True)


def load_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load sales data from CSV if given, otherwise generate the sample data."""
    if csv_path is not None:
        df = pd.read_csv(csv_path, parse_dates=["date"])
    else:
        df = generate_sample_data()
    return df.sort_values(["store", "product", "date"]).reset_index(drop=True)


def save_sample_data(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_sample_data().to_csv(path, index=False)
    return path
