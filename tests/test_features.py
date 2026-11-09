"""Smoke tests: feature leakage guards and metric sanity."""
import numpy as np
import pandas as pd

from src.data import generate_sample_data
from src.evaluate import compute_metrics, walk_forward_folds
from src.features import FEATURE_COLUMNS, build_features


def test_features_have_no_target_leakage():
    df = generate_sample_data(periods=120)
    feats = build_features(df)
    assert set(FEATURE_COLUMNS).issubset(feats.columns)
    one = feats[(feats["store"] == "store_a") & (feats["product"] == "product_1")]
    raw = df[(df["store"] == "store_a") & (df["product"] == "product_1")].reset_index(drop=True)
    row = one.iloc[10]
    pos = raw.index[raw["date"] == row["date"]][0]
    assert row["lag_1"] == raw.loc[pos - 1, "units"]
    assert row["lag_7"] == raw.loc[pos - 7, "units"]


def test_walk_forward_folds_are_time_ordered():
    df = generate_sample_data(periods=730)
    folds = walk_forward_folds(df["date"], n_folds=3, horizon=28)
    assert len(folds) == 3
    for fold in folds:
        assert fold.train_end == fold.test_start < fold.test_end


def test_metrics_perfect_prediction():
    y = np.array([10.0, 20.0, 30.0])
    m = compute_metrics(y, y)
    assert m["MAE"] == 0 and m["RMSE"] == 0
