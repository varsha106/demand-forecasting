# Retail Demand Forecasting — Gradient Boosting vs. LSTM

End-to-end demand forecasting for daily store/product sales: feature
engineering (lags, rolling statistics, calendar, price/promotion), a global
gradient boosting model, a PyTorch LSTM sequence model, and a seasonal-naive
baseline — all compared with **time-based validation** (expanding-window
walk-forward folds plus a final untouched holdout), never random splits.

> **Honest scope.** This repository is a faithful, self-contained
> demonstration of the forecasting methods I used in production retail
> demand-planning work — gradient boosting and LSTM models over engineered
> time-series features, validated with walk-forward backtesting. The bundled
> dataset is **synthetic** (generated with a fixed seed, with trend, weekly
> and yearly seasonality, promotions and holiday effects) and the metrics
> below are results on that sample data, not production results.

## Problem

Retail replenishment and labor planning depend on knowing, for every
store × product, how many units will sell tomorrow and over the next few
weeks. Forecast too high and inventory expires on the shelf; too low and the
shelf is empty. This project forecasts **daily unit sales** for 6
store/product series over a 28-day horizon, one day at a time.

## Approach

1. **Data** — daily sales per store/product with price and promotion flags
   (`src/data.py`). Use the generated sample data or drop in your own CSV
   with the same columns.
2. **Features** (`src/features.py`) — lagged sales (1/7/14/28 days), rolling
   mean/std of lagged sales, calendar attributes (day-of-week cyclical
   encoding, weekend, month, week-of-year), price and promotion. All rolling
   features are shifted so a day's features never see that day's sales.
3. **Models**
   - **Gradient Boosting** (`src/models/gradient_boosting.py`) — one global
     `HistGradientBoostingRegressor` across all series, series identity as
     categorical features. (LightGBM/XGBoost can be swapped in behind the
     same interface.)
   - **LSTM** (`src/models/lstm.py`) — one global PyTorch LSTM over 56-day
     windows, per-series scaling, rolling one-step-ahead prediction at
     evaluation time.
   - **Seasonal Naive** — last week's same-day sales (lag-7), the baseline
     any real model must beat.
4. **Validation** (`src/evaluate.py`) — 3 expanding-window walk-forward
   folds (28-day horizon) plus a final 28-day holdout. Metrics: MAE, RMSE,
   MAPE, sMAPE.

## Results (synthetic sample data)

| Phase | Model | MAE | RMSE | MAPE % | sMAPE % |
|---|---|---|---|---|---|
| Walk-forward (mean of 3 folds) | Gradient Boosting | 7.59 | 10.29 | 8.36 | 8.25 |
| Walk-forward (mean of 3 folds) | LSTM | 8.72 | 11.85 | 9.57 | 9.53 |
| Walk-forward (mean of 3 folds) | Seasonal Naive (lag-7) | 11.43 | 15.64 | 12.54 | 12.31 |
| Holdout (final 28 days) | Gradient Boosting | 13.48 | 19.80 | 11.53 | 11.44 |
| Holdout (final 28 days) | LSTM | 17.29 | 26.39 | 13.54 | 14.67 |
| Holdout (final 28 days) | Seasonal Naive (lag-7) | 21.36 | 31.29 | 17.21 | 18.77 |

Artifacts from a run: `outputs/metrics.csv`, `outputs/metrics.md`,
`outputs/forecast_holdout.png` (actual vs. both models, store_a/product_1),
`outputs/metrics_rmse.png`.

Reading the results: gradient boosting is the strongest overall — fast to
train, and it exploits the promotion/price interactions through the tabular
features. The LSTM is a clear second, ahead of the baseline on every metric,
and is the better tool when raw sequence shape matters more than exogenous
features. The holdout window (Dec 3–30) contains the pre-Christmas demand
spike; both models under-forecast it, which is exactly the regime where a
production system would add explicit holiday/event features and human
review of the plan.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_pipeline.py            # generates sample data on first run
python run_pipeline.py --csv path/to/your_sales.csv --epochs 25
pytest tests/                     # leakage + validation smoke tests
```

Expected input CSV columns: `date, store, product, units, price, promo`.

## Project structure

```
demand-forecasting/
├── run_pipeline.py            # end-to-end entry point
├── requirements.txt
├── data/
│   └── sample_sales.csv       # generated synthetic dataset (seeded)
├── src/
│   ├── data.py                # sample data generation / loading
│   ├── features.py            # lag, rolling, calendar features
│   ├── evaluate.py            # walk-forward folds + metrics
│   ├── visualize.py           # forecast and metrics plots
│   └── models/
│       ├── gradient_boosting.py
│       └── lstm.py            # PyTorch LSTM forecaster
├── outputs/                   # metrics + plots from the latest run
└── tests/
    └── test_features.py       # leakage guard, fold ordering, metric sanity
```

## Tech stack

Python · pandas · NumPy · scikit-learn (HistGradientBoosting) · PyTorch ·
matplotlib

## Extensions

- Swap the boosting estimator for LightGBM/XGBoost, or add quantile losses
  for prediction intervals.
- Add exogenous regressors (weather, local events) to both pipelines.
- Multi-step direct forecasting (one model per horizon day) for the LSTM.
- Hierarchical reconciliation (store → region → chain) on top of the
  per-series forecasts.

## Author

Varsha Puli — AI/ML Engineer · [GitHub](https://github.com/varsha106) ·
[LinkedIn](https://www.linkedin.com/in/pvarsha3468)
