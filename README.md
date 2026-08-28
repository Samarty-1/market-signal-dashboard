# 📈 Market Signal Dashboard

An end-to-end pipeline that pulls live market data, trains a next-day-direction
model with walk-forward validation, backtests it honestly (out-of-sample only),
and serves everything through an interactive dashboard — with a scheduled
GitHub Action that retrains the model daily.

**Live demo:** _add your GitHub Pages / static hosting link here after deploying `frontend/`_

![Tests](https://github.com/Samarty-1/market-signal-dashboard/actions/workflows/tests.yml/badge.svg)
![Retrain](https://github.com/Samarty-1/market-signal-dashboard/actions/workflows/retrain.yml/badge.svg)

## What this project demonstrates

Most portfolio ML projects stop at "trained a model, got a metric." This one
chains the pieces that actually make up a real system:

1. **Data ingestion** — live OHLCV pulls from Yahoo Finance (`src/data_ingestion.py`) across a ~30-ticker universe spanning tech, semis, financials, healthcare, consumer, energy, industrials, comms, and broad ETFs, not a static Kaggle CSV or a handful of mega-caps.
2. **Model training with proper validation** — `src/model.py` compares Logistic Regression, Random Forest, LightGBM, and XGBoost using **walk-forward (expanding-window) validation**, splitting on calendar date so no ticker's future ever leaks into an earlier fold's training set.
3. **An honest, cost-aware backtest** — `src/backtest.py` scores the strategy using *only* each fold's out-of-sample predictions (never the final full-data-fit model on its own training history — a common and easy-to-miss overfitting trap), with configurable position sizing (binary or confidence-scaled), per-trade transaction costs, and an optional VIX regime filter that only trades a regime once it's shown positive historical edge.
4. **An interactive dashboard** — a static, Nocturne-styled front end in `frontend/` (plain HTML/CSS/JS, no build step, no framework server) visualizes all of the above: a ticker compare overlay, candlestick charts, a real confusion matrix/ROC curve computed from the out-of-sample predictions, CSV export, and a live-as-of-last-export price/probability reading per ticker.
5. **A model registry** — `src/registry.py` versions every trained model under `models/registry/<timestamp>/` with a `registry.json` pointer/history (instead of overwriting one file in place), so there's a rollback-able audit trail; old artifacts beyond the most recent 30 versions are pruned to keep the repo bounded while their metadata stays in history.
6. **A scheduled retrain** — `.github/workflows/retrain.yml` runs the whole pipeline on a cron schedule — ingestion → model → registry → backtest → frontend data export — and commits the refreshed artifacts back to the repo, so the commit history itself is evidence this runs on a real cadence, not just once locally.
7. **A live sentiment signal from a model trained on real Hugging Face data** — `src/sentiment.py` trains a TF-IDF + Logistic Regression classifier on Hugging Face's [`zeroshot/twitter-financial-news-sentiment`](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) dataset (9,543 train / 2,388 validation rows), then applies it live to each ticker's current Yahoo Finance headlines. It's deliberately **not** folded into the backtested price model above or its historical numbers — that dataset has no publish dates, so there's no honest way to align a headline to a specific historical trading day, and fabricating that alignment would be exactly the kind of overfitting/leakage this project's backtest section (#3) is written to avoid. See the panel at the bottom of the dashboard.

## Sentiment classifier: honest results

Held out on the dataset's own validation split (never seen during training):
**83.0% accuracy, 0.773 macro F1** across 3 classes (bearish / bullish / neutral).
That's a real, respectable number for 3-class financial sentiment — no cherry-picking
here since it's the dataset's standard train/validation split. Applied live, it's a
sentiment *reading*, not a validated trading signal: because the training data isn't
dated, this hasn't been (and can't honestly be) backtested the way the price model
was. Treat it as informational context, not an input you should assume has predictive
value the way the walk-forward-validated model does.

## Honest results (read this before the dashboard)

Predicting tomorrow's stock direction from today's technical indicators is
close to a coin flip — the walk-forward ROC AUC here sits right around **0.50**
across all four candidate models (Logistic Regression, Random Forest, LightGBM,
XGBoost), i.e. no better than random. **This is expected, not a failure of the code.**
Daily price direction is close to the textbook definition of weak-form market
efficiency: if a signal this simple reliably beat the market, it would already
be arbitraged away. The backtest reflects that honestly too — the strategy
underperforms plain buy-and-hold on most tickers in the current bull-market
window, which is the correct outcome for a near-random signal that occasionally
sits in cash instead of being fully invested.

The point of this project isn't claiming alpha — it's the pipeline: real data
in, properly-validated model, leakage-free backtest, live dashboard, scheduled
retraining. That combination is the part worth demonstrating.

## Why the AUC is stuck at ~0.50, and the one target that actually moved it

The theoretical reason next-day direction sits at 0.50 isn't a code bug —
it's the target itself. "Will ticker X go up tomorrow" is dominated by
whatever the *whole market* does tomorrow (SPY's own next-day direction is
close to a coin flip too), and every ticker in the universe shares that same
market-wide move. A model trained only on one ticker's own lagged technical
indicators has no way to see that common component coming, so it's trying
to predict something that's mostly noise from its vantage point.

`src/features.add_cross_sectional_label` tests a different, real target
instead: not "will this ticker go up," but **"will this ticker beat the
day's cross-sectional median return"** (across the same ~30-ticker
universe). This nets out the shared market-wide move and targets the
smaller, better-documented cross-sectional relative-strength effect instead
— the same category of effect behind short-term reversal / relative-strength
research in the academic literature. Tested through the exact same
leakage-free walk-forward harness (`src/model.py`), same features, same
models, only the label changed:

| Target | Best model | Mean walk-forward ROC AUC |
|---|---|---|
| Next-day direction (baseline) | XGBoost | 0.5014 |
| Beat cross-sectional median (new) | XGBoost | **0.5157** |

Small, but real and consistent — the cross-sectional target beat the
baseline in **every one of the 5 walk-forward folds**, not just on average
(fold AUCs 0.51–0.54 vs. 0.49–0.53 for baseline over the same folds). For
context: a 5-day-ahead direction target was also tested as a hypothesis (the
theory being that a longer horizon has less noise) and it **didn't hold up**
— AUC came back lower (~0.49), not higher. Reporting that too, since a
hypothesis that turns out wrong is still a result, not something to bury.

This is trained and registered as a genuinely separate model
(`models/cross_sectional/registry/`, via `python -m src.model`) alongside
the original next-day model, shown on the dashboard's "Baseline vs.
cross-sectional target" panel. It is **not** turned into its own backtested
trading strategy here — predicting relative outperformance implies a
long/short, market-neutral construction (long the top-ranked names, short
or avoid the bottom-ranked ones) with its own cost/hedging assumptions,
which is a different, larger scope than this repo's existing long/flat
single-ticker backtest. Shipping that without validating it would repeat the
exact mistake this section is trying to avoid — an unvalidated result
dressed up as more than it is. Flagged here as the obvious next step for
whoever picks this up.

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
                    │                              │  Logistic Regression /
                    │                              │  Random Forest / LightGBM /
                    │                              │  XGBoost, saves the best one
                    └──────────────┬───────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │  src/registry.py             │  versions the model under
                    │                              │  models/registry/<ts>/,
                    │                              │  prunes old artifacts
                    └──────────────┬───────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │  src/backtest.py             │  out-of-sample strategy vs
                    │                              │  buy-and-hold, with position
                    │                              │  sizing, transaction costs,
                    │                              │  and a VIX regime filter
                    └──────────────┬───────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │  src/export_frontend_data.py │  bundles real OHLCV+signal
                    │                              │  rows, model metrics, and a
                    │                              │  live snapshot per ticker
                    │                              │  into frontend/data/*.json
                    └──────────────┬───────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │  frontend/ (static site)     │  fetch()es the JSON above —
                    │                              │  plain HTML/CSS/JS, no
                    │                              │  backend server required
                    └─────────────────────────────┘

.github/workflows/retrain.yml runs the ingestion → model → registry → backtest
→ export chain on a daily cron schedule and commits models/registry/,
reports/, and frontend/data/ back to the repo.
```

## Project structure

```
market-signal-dashboard/
├── src/
│   ├── data_ingestion.py         # yfinance OHLCV + VIX puller, CLI-runnable
│   ├── features.py               # technical indicators + labeling
│   ├── model.py                  # walk-forward validation + model selection
│   ├── registry.py               # versioned model registry (save/load/prune)
│   ├── backtest.py               # out-of-sample strategy vs buy-and-hold
│   └── export_frontend_data.py   # bundles real data for frontend/ (CLI-runnable)
├── tests/
│   ├── test_features.py          # label correctness, no-lookahead checks
│   ├── test_backtest.py          # performance-metric math, sizing/cost/regime logic
│   └── test_registry.py          # registry save/load/prune round-trips
├── frontend/                     # static dashboard (Nocturne design system) — no backend
│   ├── index.html                # the dashboard itself; fetch()es data/dashboard_data.json
│   ├── support.js                # the template runtime the comp was built with (self-contained)
│   ├── _ds/…/styles.css          # Nocturne design tokens + component CSS
│   └── data/dashboard_data.json  # committed: written by src/export_frontend_data.py
├── models/registry/              # committed: <timestamp>/{model.joblib,metrics.json} + registry.json pointer
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
python -m src.model                  # trains + walk-forward validates, saves a new models/registry/<version>/
python -m src.backtest               # backtests the chosen model, saves reports/
python -m src.export_frontend_data   # bundles real data (+ one live snapshot per ticker) into frontend/data/
```

Launch the dashboard — it's a static site, so any HTTP server works (it must be served over HTTP, not opened as a `file://` path, since it loads its data with `fetch()`):

```bash
cd frontend
python -m http.server 8000
# then open http://localhost:8000
```

Use the sidebar's **Refresh data** button to re-fetch `dashboard_data.json` after re-running the export step, without reloading the page.

Run the tests (no network required — synthetic data):

```bash
pytest -v
```

## Configuration

Tickers default to a ~30-name universe across sectors (see `src/data_ingestion.py`).
Override on any pipeline command:

```bash
python -m src.model --tickers TSLA,NVDA,SPY --period 3y
python -m src.backtest --tickers TSLA,NVDA,SPY --threshold 0.55
```

`src/backtest.py` also accepts:

```bash
python -m src.backtest \
  --position-sizing confidence \  # "binary" (default) or "confidence" (scales with P(up) - threshold)
  --cost-bps 5 \                  # per-unit-of-turnover transaction cost, in basis points (default 0)
  --regime-filter                 # only trade when the current VIX tercile has shown positive prior-fold edge
```

The dashboard's Backtest tab recomputes a binary long/flat equity curve
client-side from the exported out-of-sample predictions as you move the
threshold slider — the position-sizing/cost/regime-filter options above are
CLI/backtest-report-level controls for now (porting all three into the
static frontend's JS would be the next step if that interactivity is wanted
client-side too).

## Scheduled retraining

`.github/workflows/retrain.yml` runs weekdays at 21:30 UTC (after the US
market closes), re-pulls data, re-runs walk-forward validation, re-backtests,
re-exports `frontend/data/dashboard_data.json`, and commits `models/`,
`reports/`, and `frontend/data/` back to the repo if anything changed.
Trigger it manually from the **Actions** tab (`workflow_dispatch`) to see it
run immediately rather than waiting for the schedule.

## Deploying the frontend

`frontend/` is a self-contained static site (HTML/CSS/JS, one JSON data file)
— any static host works. For GitHub Pages: Settings → Pages → deploy from
branch, folder `/frontend`. The scheduled retrain keeps
`frontend/data/dashboard_data.json` fresh in the same repo, so once Pages is
pointed at this folder no further wiring is needed.

## Possible next steps

- Add a simple LSTM as a further candidate — needs a separate sequence-windowing data path from the current flat-feature pipeline, plus a DL framework dependency; a bigger lift than the tree-based candidates above.
- Wire up an external model registry (e.g. MLflow) if this ever needed to scale past one repo — the current `src/registry.py` is a lightweight in-repo version, deliberately scoped to avoid provisioning external infrastructure for a single-repo educational project.
- Portfolio-level (not just per-ticker) position sizing that accounts for cross-ticker correlation.
- Port position sizing / transaction costs / the VIX regime filter into the frontend's client-side backtest JS, so those controls are explorable in the dashboard itself instead of only via the `src/backtest.py` CLI flags.
- A true live feed would need a small backend (or a scheduled function) serving fresh inference on request — the current frontend is deliberately backend-free, so "live" means "as of the last export," refreshable on demand but not push-updated.

## Disclaimer

This is an educational portfolio project. Nothing here is investment advice,
and the model's near-random accuracy on daily direction should make that
obvious on its own.
