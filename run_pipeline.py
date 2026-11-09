"""End-to-end demand forecasting pipeline.

Usage:
    python run_pipeline.py [--csv data/sample_sales.csv] [--epochs 25]

Steps:
    1. Load (or generate) daily store/product sales data.
    2. Walk-forward validation (expanding window, 28-day horizon) for a
       seasonal-naive baseline, a gradient boosting model, and an LSTM.
    3. Retrain both models on all pre-holdout data and evaluate on the
       final 28-day holdout.
    4. Write metrics (CSV + Markdown) and forecast plots to outputs/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_data, save_sample_data
from src.evaluate import compute_metrics, walk_forward_folds
from src.features import build_features
from src.models.gradient_boosting import GradientBoostingForecaster
from src.models.lstm import LSTMForecaster
from src.visualize import plot_forecast_comparison, plot_metrics_bar

ROOT = Path(__file__).resolve().parent
HOLDOUT_DAYS = 28


def evaluate_gbm(features: pd.DataFrame, train_mask, test_mask) -> np.ndarray:
    model = GradientBoostingForecaster().fit(features[train_mask])
    return model.predict(features[test_mask])


def run(csv: str | None, epochs: int) -> None:
    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    sample_csv = ROOT / "data" / "sample_sales.csv"
    if csv is None and not sample_csv.exists():
        save_sample_data(sample_csv)
        print(f"Generated sample data -> {sample_csv}")

    raw = load_data(csv or sample_csv)
    features = build_features(raw)
    all_dates = pd.Series(raw["date"].unique()).sort_values()
    holdout_start = all_dates.iloc[-HOLDOUT_DAYS]
    print(f"Data: {len(raw):,} rows, {raw['date'].min().date()} to {raw['date'].max().date()}, "
          f"holdout from {pd.Timestamp(holdout_start).date()}")

    rows: list[dict] = []

    # ---- walk-forward validation on pre-holdout data --------------------
    pre = raw[raw["date"] < holdout_start]
    pre_feat = features[features["date"] < holdout_start]
    folds = walk_forward_folds(pre["date"], n_folds=3, horizon=28)
    print(f"Walk-forward folds: {len(folds)}")
    for fold in folds:
        tr = pre_feat["date"] < fold.train_end
        te = (pre_feat["date"] >= fold.test_start) & (pre_feat["date"] < fold.test_end)
        test_feat = pre_feat[te]
        y_true = test_feat["units"].to_numpy(dtype=float)

        gbm_pred = evaluate_gbm(pre_feat, tr, te)
        rows.append({"phase": "walk_forward", "model": "Gradient Boosting",
                     **compute_metrics(y_true, gbm_pred)})
        rows.append({"phase": "walk_forward", "model": "Seasonal Naive (lag-7)",
                     **compute_metrics(y_true, test_feat["lag_7"].to_numpy(dtype=float))})

        lstm = LSTMForecaster(epochs=epochs).fit(pre[pre["date"] < fold.train_end])
        lstm_pred_df = lstm.predict_period(
            pre[pre["date"] < fold.train_end],
            pre[(pre["date"] >= fold.test_start) & (pre["date"] < fold.test_end)],
        )
        merged = test_feat[["date", "store", "product"]].merge(
            lstm_pred_df, on=["date", "store", "product"], how="inner"
        )
        rows.append({"phase": "walk_forward", "model": "LSTM",
                     **compute_metrics(merged["y_true"].to_numpy(), merged["y_pred"].to_numpy())})
        print(f"  fold ending {fold.test_end.date()}: done")

    # ---- final holdout ---------------------------------------------------
    tr_mask = features["date"] < holdout_start
    te_mask = features["date"] >= holdout_start
    test_feat = features[te_mask]
    y_true = test_feat["units"].to_numpy(dtype=float)

    gbm = GradientBoostingForecaster().fit(features[tr_mask])
    gbm_pred = gbm.predict(test_feat)
    rows.append({"phase": "holdout", "model": "Gradient Boosting", **compute_metrics(y_true, gbm_pred)})
    rows.append({"phase": "holdout", "model": "Seasonal Naive (lag-7)",
                 **compute_metrics(y_true, test_feat["lag_7"].to_numpy(dtype=float))})

    train_raw = raw[raw["date"] < holdout_start]
    test_raw = raw[raw["date"] >= holdout_start]
    lstm = LSTMForecaster(epochs=epochs).fit(train_raw)
    lstm_pred_df = lstm.predict_period(train_raw, test_raw)
    merged = test_feat[["date", "store", "product"]].merge(
        lstm_pred_df, on=["date", "store", "product"], how="inner"
    )
    rows.append({"phase": "holdout", "model": "LSTM",
                 **compute_metrics(merged["y_true"].to_numpy(), merged["y_pred"].to_numpy())})

    metrics = pd.DataFrame(rows)
    metrics.to_csv(outputs / "metrics.csv", index=False)

    summary = (
        metrics.groupby(["phase", "model"])[["MAE", "RMSE", "MAPE_%", "sMAPE_%"]]
        .mean()
        .round(2)
        .reset_index()
    )
    md = ["# Model results (synthetic sample data)", "",
          "| Phase | Model | MAE | RMSE | MAPE % | sMAPE % |",
          "|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        md.append(f"| {r['phase']} | {r['model']} | {r['MAE']:.2f} | {r['RMSE']:.2f} "
                  f"| {r['MAPE_%']:.2f} | {r['sMAPE_%']:.2f} |")
    (outputs / "metrics.md").write_text("\n".join(md) + "\n")

    # ---- plots ------------------------------------------------------------
    key = ("store_a", "product_1")
    actual = test_raw[(test_raw["store"] == key[0]) & (test_raw["product"] == key[1])]
    gbm_frame = test_feat[["date", "store", "product"]].copy()
    gbm_frame["y_pred"] = gbm_pred
    gbm_series = gbm_frame[(gbm_frame["store"] == key[0]) & (gbm_frame["product"] == key[1])]
    lstm_series = lstm_pred_df[
        (lstm_pred_df["store"] == key[0]) & (lstm_pred_df["product"] == key[1])
    ]
    plot_forecast_comparison(
        actual,
        {"Gradient Boosting": gbm_series, "LSTM": lstm_series},
        f"{key[0]} / {key[1]}",
        outputs / "forecast_holdout.png",
    )
    holdout_summary = summary[summary["phase"] == "holdout"].set_index("model")
    plot_metrics_bar(holdout_summary, "RMSE", outputs / "metrics_rmse.png")

    print("\n=== Summary (mean over folds / holdout) ===")
    print(summary.to_string(index=False))
    print(f"\nArtifacts written to {outputs}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=None, help="Path to sales CSV (default: bundled sample)")
    parser.add_argument("--epochs", type=int, default=25, help="LSTM training epochs")
    args = parser.parse_args()
    run(args.csv, args.epochs)
