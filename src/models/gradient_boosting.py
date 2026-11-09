"""Gradient boosting forecaster (sklearn HistGradientBoostingRegressor).

A single global model is trained across all store/product series, with
series identity supplied as categorical features — the same "one model, many
series" pattern used for large retail assortments. Swap in LightGBM or
XGBoost by replacing the estimator; the feature pipeline is unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.features import FEATURE_COLUMNS

CATEGORICAL = ["store", "product"]
MODEL_FEATURES = FEATURE_COLUMNS + CATEGORICAL


class GradientBoostingForecaster:
    def __init__(self, **kwargs):
        params = dict(
            max_iter=400,
            learning_rate=0.06,
            max_leaf_nodes=63,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=42,
        )
        params.update(kwargs)
        self.model = HistGradientBoostingRegressor(
            categorical_features=[MODEL_FEATURES.index(c) for c in CATEGORICAL],
            **params,
        )
        self._fitted = False

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[MODEL_FEATURES].copy()
        for col in CATEGORICAL:
            X[col] = X[col].astype("category").cat.codes
        return X

    def fit(self, df: pd.DataFrame) -> "GradientBoostingForecaster":
        self.model.fit(self._prepare(df), df["units"].to_numpy(dtype=float))
        self._fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fit before predicting.")
        return np.clip(self.model.predict(self._prepare(df)), 0, None)

    def feature_importance(self) -> pd.Series | None:
        """Permutation-free gain importance is not exposed by sklearn's HGBT;
        use permutation importance externally if needed. Returns None."""
        return None
