# 📈 Market Signal Dashboard

An end-to-end pipeline that pulls live market data, trains a next-day-direction
model with walk-forward validation, backtests it honestly (out-of-sample only),
and serves everything through an interactive dashboard — with a scheduled
GitHub Action that retrains the model daily.

**Live demo:** _add your Streamlit Community Cloud / Hugging Face Space link here after deploying_

![Tests](https://github.com/Samarty-1/market-signal-dashboard/actions/workflows/tests.yml/badge.svg)
![Retrain](https://github.com/Samarty-1/market-signal-dashboard/actions/workflows/retrain.yml/badge.svg)

## What this project demonstrates

Most portfolio ML projects stop at "trained a model, got a metric." This one
chains the three pieces that actually make up a real system:

1. **Data ingestion** — live OHLCV pulls from Yahoo Finance (`src/data_ingestion.py`), not a static Kaggle CSV.
2. **Model training with proper validation** — `src/model.py` compares Logistic Regression vs Random Forest using **walk-forward (expanding-window) validation**, splitting on calendar date so no ticker's future ever leaks into an earlier fold's training set.
3. **An honest backtest** — `src/backtest.py` scores the strategy using *only* each fold's out-of-sample predictions, never the final full-data-fit model on its own training history (a common and easy-to-miss backtest overfitting trap).
4. **An interactive dashboard** — `app.py` (Streamlit) visualizes all of the above, plus one genuinely live piece: a same-day model score computed from data fetched at page-load time.
5. **A scheduled retrain** — `.github/workflows/retrain.yml` runs the whole pipeline on a cron schedule and commits the refreshed model/metrics back to the repo, so the commit history itself is evidence this runs on a real cadence, not just once locally.

## Honest results (read this before the dashboard)

Predicting tomorrow's stock direction from today's technical indicators is
close to a coin flip — the walk-forward ROC AUC here sits right around **0.48**,
i.e. no better than random. **This is expected, not a failure of the code.**
Daily price direction is close to the textbook definition of weak-form market
efficiency: if a signal this simple reliably beat the market, it would already
be arbitraged away. The backtest reflects that honestly too — the strategy
underperforms plain buy-and-hold on most tickers in the current bull-market
window, which is the correct outcome for a near-random signal that occasionally
sits in cash instead of being fully invested.

The point of this project isn't claiming alpha — it's the pipeline: real data
in, properly-validated model, leakage-free backtest, live dashboard, scheduled
retraining. That combination is the part worth demonstrating.

## Architecture

```
                    ┌─────────────────────────────┐
   yfinance ───────▶│  src/data_ingestion.py       │
                    └──────────────┬───────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │  src/features.py             │  technical indicators +
                    │                              │  next-day-direction label
                    └──────────────┬───────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │  src/model.py                │  walk-forward validation,
                    │                              │  Logistic Regression vs
                    │                              │  Random Forest, saves the
                    │                              │  better one
                    └──────────────┬───────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │  src/backtest.py             │  out-of-sample strategy
                    │                              │  vs buy-and-hold
                    └──────────────┬───────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │  app.py (Streamlit)          │  reads the artifacts above
                    │                              │  + one live inference call
                    └─────────────────────────────┘

.github/workflows/retrain.yml runs the ingestion → model → backtest chain on a
daily cron schedule and commits models/ + reports/ back to the repo.
```

## Project structure

```
market-signal-dashboard/
├── app.py                        # Streamlit dashboard (entry point)
├── src/
│   ├── data_ingestion.py         # yfinance OHLCV puller, CLI-runnable
│   ├── features.py               # technical indicators + labeling
│   ├── model.py                  # walk-forward validation + model selection
│   └── backtest.py               # out-of-sample strategy vs buy-and-hold
├── tests/
│   ├── test_features.py          # label correctness, no-lookahead checks
│   └── test_backtest.py          # performance-metric math, signal logic
├── models/                       # committed: model.joblib, metrics.json
├── reports/                      # committed: backtest_metrics.json, predictions.csv
├── data/                         # gitignored: raw prices are fetched fresh, not cached
├── .github/workflows/
│   ├── tests.yml                 # pytest on every push/PR
│   └── retrain.yml               # daily scheduled retrain + commit
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

## Usage

Run the full pipeline (fetches live data — needs internet):

```bash
python -m src.model      # trains + walk-forward validates, saves models/
python -m src.backtest   # backtests the chosen model, saves reports/
```

Launch the dashboard:

```bash
streamlit run app.py
```

Run the tests (no network required — synthetic data):

```bash
pytest -v
```

## Configuration

Tickers default to `AAPL, MSFT, GOOGL, AMZN, SPY`. Override on any pipeline command:

```bash
python -m src.model --tickers TSLA,NVDA,SPY --period 3y
python -m src.backtest --tickers TSLA,NVDA,SPY --threshold 0.55
```

## Scheduled retraining

`.github/workflows/retrain.yml` runs weekdays at 21:30 UTC (after the US
market closes), re-pulls data, re-runs walk-forward validation, re-backtests,
and commits `models/` and `reports/` back to the repo if anything changed.
Trigger it manually from the **Actions** tab (`workflow_dispatch`) to see it
run immediately rather than waiting for the schedule.

## Possible next steps

- Add more tickers / an actual sector or index universe instead of five mega-caps
- Try gradient-boosted trees (LightGBM/XGBoost) or a simple LSTM as additional candidates
- Add position sizing and transaction costs to the backtest (currently binary long/flat, zero costs)
- Add a regime filter (e.g. VIX level) so the model only trades when its historical edge held
- Swap the scheduled-commit pattern for a proper model registry if this ever needed to scale past one repo

## Disclaimer

This is an educational portfolio project. Nothing here is investment advice,
and the model's near-random accuracy on daily direction should make that
obvious on its own.
