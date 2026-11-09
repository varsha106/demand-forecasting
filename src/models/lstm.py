"""PyTorch LSTM forecaster.

One global sequence model across all series. Each training example is a
window of the past `lookback` days of per-series features
[units, price, promo, day-of-week one-hots, is_weekend] and the target is the
next day's units. Series are scaled per series (mean/std from the training
window only) so a single model can serve stores of very different volumes.

Evaluation uses rolling one-step-ahead prediction: for each test day the
model consumes the *actual* history up to the previous day. This matches how
the model is used operationally for next-day replenishment decisions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEQ_FEATURES = ["units", "price", "promo", "is_weekend"]
GROUP_COLS = ["store", "product"]
LOOKBACK = 56


def _series_arrays(df: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    out = {}
    for key, g in df.groupby(GROUP_COLS):
        g = g.sort_values("date").copy()
        g["dow"] = g["date"].dt.day_of_week
        g["is_weekend"] = (g["dow"] >= 5).astype(float)
        out[key] = g
    return out


def _window_matrix(g: pd.DataFrame) -> np.ndarray:
    dow_onehot = np.eye(7, dtype=np.float32)[g["dow"].to_numpy()]
    base = g[SEQ_FEATURES].to_numpy(dtype=np.float32)
    return np.hstack([base, dow_onehot])


class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class LSTMForecaster:
    def __init__(
        self,
        lookback: int = LOOKBACK,
        hidden_size: int = 64,
        epochs: int = 25,
        lr: float = 1e-3,
        batch_size: int = 64,
        seed: int = 42,
    ):
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.scalers: dict[tuple[str, str], tuple[float, float]] = {}
        self.model: _LSTMNet | None = None

    # -- data prep ---------------------------------------------------------
    def _build_windows(
        self, df: pd.DataFrame, fit_scalers: bool
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
        series = _series_arrays(df)
        X_list, y_list, keys = [], [], []
        for key, g in series.items():
            mat = _window_matrix(g)
            units = mat[:, 0].copy()
            if fit_scalers:
                mu, sigma = float(units.mean()), float(units.std() + 1e-6)
                self.scalers[key] = (mu, sigma)
            mu, sigma = self.scalers[key]
            mat[:, 0] = (units - mu) / sigma
            for i in range(self.lookback, len(mat)):
                X_list.append(mat[i - self.lookback : i])
                y_list.append((units[i] - mu) / sigma)
                keys.append(key)
        return np.asarray(X_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32), keys

    # -- training ----------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "LSTMForecaster":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X, y, _ = self._build_windows(df, fit_scalers=True)
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model = _LSTMNet(input_size=X.shape[2], hidden_size=self.hidden_size)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
        return self

    # -- rolling one-step-ahead prediction ---------------------------------
    @torch.no_grad()
    def predict_period(self, history: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
        """Predict each day in `test` using actual history (history + earlier test days)."""
        if self.model is None:
            raise RuntimeError("Model must be fit before predicting.")
        self.model.eval()
        results = []
        for key, g_test in test.groupby(GROUP_COLS):
            g_hist = history[history["store"] == key[0]]
            g_hist = g_hist[g_hist["product"] == key[1]].sort_values("date")
            combined = pd.concat([g_hist, g_test.sort_values("date")], ignore_index=True)
            combined["dow"] = combined["date"].dt.day_of_week
            combined["is_weekend"] = (combined["dow"] >= 5).astype(float)
            mat = _window_matrix(combined)
            mu, sigma = self.scalers[key]
            mat_scaled = mat.copy()
            mat_scaled[:, 0] = (mat[:, 0] - mu) / sigma
            n_hist = len(g_hist)
            for j in range(len(g_test)):
                idx = n_hist + j
                window = torch.from_numpy(mat_scaled[idx - self.lookback : idx]).unsqueeze(0)
                pred_scaled = self.model(window).item()
                results.append(
                    {
                        "date": g_test.sort_values("date").iloc[j]["date"],
                        "store": key[0],
                        "product": key[1],
                        "y_true": float(g_test.sort_values("date").iloc[j]["units"]),
                        "y_pred": max(pred_scaled * sigma + mu, 0.0),
                    }
                )
        return pd.DataFrame(results)
