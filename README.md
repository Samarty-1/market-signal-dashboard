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

## Cross-sectional long-short: the full investigation

That flagged next step was picked up. The motivating evidence: the ML
asset-pricing literature (Gu, Kelly & Xiu, and successors) finds individual-
stock predictive R² is tiny (~0.3–0.4%) — but pooling that tiny edge across
hundreds of stocks in a **long-short decile portfolio** (long the top decile
by predicted score, short the bottom decile) turns it into a real portfolio
Sharpe in their study (~1.35 out-of-sample). That's a structurally different
claim from "can I predict one stock" — it's "does ranking hundreds of stocks
against each other carry information, even if each individual prediction is
almost worthless." This section is the honest record of testing that claim
against this repo's own data and models, including the two dead ends before
the real (small) answer.

**Infrastructure added**: `src/data_ingestion.fetch_sp500_tickers()` /
`fetch_universe_prices()` widen the universe from ~30 tickers to the full
S&P 500 (503 names, batched via `yf.download()` — 9 seconds for 3 years,
not 500 sequential requests). `src/features.py` adds momentum (21/63/126-day
and classic "12-1 month" Jegadeesh-Titman momentum), volatility (60-day),
and liquidity (log dollar volume, Amihud illiquidity) feature families —
the literature's three dominant predictive families, not any single exotic
indicator. `src/long_short_backtest.py` implements the actual portfolio
construction and honest evaluation against an equal-weighted universe
benchmark.

### Attempt 1: naive daily rebalancing (large-cap, short window) — failed loudly

First pass: full S&P 500, 3 years of data, hard top/bottom-decile
membership recomputed from scratch every day. Result: **Sharpe -1.70**,
losing money, while the equal-weighted universe benchmark scored **Sharpe
2.03** over the same (unusually strong, uninterrupted bull) window.
Decomposing why: at **zero transaction cost** the same portfolio scored
**Sharpe +0.57** — a real signal was there. The problem was cost: daily
full-decile rebalancing on ~50 names per leg measured **~20bps/day**
turnover cost against a **~5bps/day** raw spread — paying away 4x the
signal just to chase noisy day-to-day rank reshuffling.

### Attempt 2: lower rebalancing frequency — looked great, failed confirmation

Rebalancing less often (weekly/monthly/quarterly instead of daily) fixed
the cost problem directly: Sharpe went daily -1.70 → weekly 0.15 → biweekly
0.33 → monthly 0.69 → (XGBoost specifically) quarterly 1.09–1.68. That
last number is a real trap, and it's worth showing exactly how it broke:
searching across 3 models × several rebalance frequencies on the *same*
~300-day test window and reporting whichever combination scored best is
just curve-fitting with extra steps — the same mistake this repo's own
"Honest results" section above was written to avoid, and the same one
`dual-ma-rsi-strategy-r`'s README documents catching in its own parameter
search. Applying that repo's own discipline here: the ~300 days of
out-of-sample predictions were split into a **selection set** (used to pick
model + rebalance frequency) and a **confirmation set** (never touched
until the final check). XGBoost + monthly rebalancing scored **Sharpe 2.25**
on the selection set. On confirmation: **Sharpe -0.63**. The "great" number
was noise from a limited, single-regime (one uninterrupted bull run) test
window, not a real edge — exactly what the selection/confirmation split
exists to catch, and did.

### Regime diversity, and a second confirmation failure

The 3-year window was also too short to be trustworthy on its own — it
never included a real drawdown, so the equal-weighted benchmark's Sharpe
(~2.0) was itself an artifact of one abnormally smooth bull run. Extended
to 10 years / 500 tickers (9 usable years after the 252-day momentum
warm-up, spanning the 2022 bear market and 2020's aftermath, not just one
regime) with 10 walk-forward folds, split at the midpoint into selection
(folds 0-5) and confirmation (folds 6-9). Selection-set best (rebalance
every 10 days): Sharpe 0.38. Confirmation: **Sharpe -0.16**. Failed again
— a second, independent disconfirmation on a differently-constructed
test window.

### Small-cap universe: a different, more definitive failure

The literature specifically flags small/micro-cap as having a *structural*
(not just statistical) illiquidity premium — institutions can't deploy
meaningful capital there without moving the price, so the mechanism is
different from "large-cap noise." Tested the identical pipeline on the
S&P 600 SmallCap universe (602 tickers, same 10-year window, more
realistic 20bps cost for wider small-cap spreads). Result: **negative
raw spread even at zero transaction cost**, at every rebalance frequency
tested. This isn't a cost problem like the large-cap case — the ranking
itself doesn't work on this universe with this feature set. A clean,
different kind of failure, reported as such rather than glossed over.

### The actual finding: alphalens confirms a real, small, un-tradeable edge

Before concluding "no edge, full stop," one more check: **alphalens-reloaded**
(industry-standard factor analysis, not a metric built for this repo) on the
large-cap XGBoost predictions. Its Information Coefficient (Spearman rank
correlation between predicted score and forward return) came back
**positive at every horizon tested (1D/5D/10D/21D/63D), on both the
selection set (0.012–0.026) and the untouched confirmation set
(0.019–0.026)** — positive in 18 of 24 months in the confirmation window.
This is the real result: **the ranking signal is genuine and reproduces
out-of-sample** — it just isn't large enough to survive being turned into
an actual costed, tradeable portfolio using a hard-decile construction.

One more honest attempt followed from that diagnosis directly (not a blind
re-search): `build_hysteresis_long_short_returns` replaces the hard decile
cutoff with **asymmetric entry/exit thresholds** (enter the long leg only
in the top 10%, exit only once a name falls out of the top 25% — same
principle as `quantpulse`'s `MIN_HOLD_DAYS` fix for RL churn, applied here
to noisy rank reshuffling instead) combined with less frequent rebalancing.
At zero cost this scored Sharpe 0.95 on the selection set — genuinely good
— but even hysteresis alone, checked daily, still cost ~13bps/day against
the same ~5bps/day signal. Combined with 10-day rebalancing: selection-set
Sharpe 0.40, and on the **untouched confirmation set: Sharpe 0.04** — not
the outright loss of the earlier attempts, but not a real edge either.
Essentially flat, exactly consistent with what a genuine-but-small IC
(~0.02–0.03) should produce once realistic costs are applied.

### Honest conclusion

A real, statistically-confirmed cross-sectional signal exists in this
feature set on large-cap US equities (positive IC, reproduced out-of-sample
on data that had zero influence on any modeling decision). It is **too
small to be a profitable market-neutral trading strategy** after realistic
transaction costs — every construction tested (hard decile, various
rebalance frequencies, hysteresis) that looked good on a selection set
failed or went flat on honest confirmation, except the final hysteresis
version, which converged to breakeven rather than a loss. That's a
different, more informative conclusion than "found nothing" — the effect
is real and measurable, it just isn't tradeable at this magnitude without
either far lower transaction costs than a real account would pay, or a
genuinely stronger source of signal than price/volume technicals alone can
provide (see "Next steps" below).

Reproduce this yourself: `python -m src.cross_sectional_long_short_pipeline
[--universe sp500|smallcap] [--period 10y] [--model xgboost]`.

### Correction: every number above was computed on a survivorship-biased universe

The investigation above was run on a universe built by
`fetch_sp500_tickers()`, which scrapes the S&P 500 constituent list **as it
looks today** and then pulls years of history for those names. That is the
textbook survivorship-bias bug, and the effect here is not marginal:
comparing index membership in 2016-08 against 2026-08, **175 of the 505 names
then in the index (34.7%) are no longer in it.** Companies leave the S&P 500
by going bankrupt, collapsing to a small-cap, or being acquired — so the names
silently excluded are heavily skewed toward the losers. Ranking the survivors
against each other assumes you knew in advance which firms would still exist.

The same line of code carried a second, opposite bias: a company that only
joined the index in 2024 still had its 2016–2023 history fed into the
cross-section for those years. Stocks are *added* to the S&P 500 after a
strong run, so their pre-inclusion history is selected on past performance
too.

`src/universe.py` fixes what can be fixed. It reconstructs membership as of
any past date from the Wikipedia page's own **revision history** — free, no
API key, and reproducible, since each answer is pinned to a specific revision
id — so a stock is only ranked on dates it was genuinely an index member, and
the names that later *left* the index are pulled too.

**Measured on 5 years (`python -m scripts.measure_survivorship_bias
--period 5y`), holding model, features, and portfolio construction fixed and
changing only the universe:**

| Confirmation-set metric | Survivorship-biased | Point-in-time | Inflation |
|---|---|---|---|
| Sharpe | 1.33 | **0.58** | +0.75 |
| Annualized return | 30.7% | **11.3%** | +19.4pp |
| Total return | 37.5% | **13.6%** | +23.9pp |
| Max drawdown | -19.9% | **-23.9%** | understated by 4.0pp |

**The bias more than doubled the reported Sharpe.** For scale, 610 names were
index members at some point in that 5-year window versus 503 today — 107
companies left the index and were silently dropped from every earlier number
in this section. The equal-weighted benchmark is inflated too (Sharpe 2.07 →
1.80), but far less, because the *long leg* is where the survivors
concentrate. Note the honest conclusion above is unchanged in direction and
gets stronger: the strategy underperforms simply holding the universe in
both arms, and by more once the bias is removed.

#### What this still doesn't fix, and the trap in fixing it naively

Survivorship bias **cannot be fully removed with free data**, and this repo
does not pretend otherwise. Yahoo Finance serves no history for most delisted
tickers — 50 of the 610 point-in-time members (8.2%) return nothing at all
(CHK, SIVB, FRC, ANTM, CTXS, ATVI, CELG, …), and those are disproportionately
the failures. So the bias is **bounded and reported** rather than eliminated:
`universe.coverage_report()` prints what fraction of the true universe is
missing, and that caveat belongs next to any Sharpe computed here.

Fixing this naively is worse than not fixing it, because **exchanges recycle
ticker symbols**. `BBBY` was Bed Bath & Beyond, an index member until 2022; it
went bankrupt in 2023, and Yahoo now serves ~30 rows for that symbol starting
2026-07-17 belonging to an unrelated company. Simply widening the download
list to include removed names silently injects one company's prices under
another company's identity — a data-integrity failure, not just a bias.
`universe.drop_recycled_tickers()` catches it by requiring a ticker's price
history to actually overlap its membership window (6 dropped in the 5-year
run: FB, FERG, INFO, RDDT, SBNY, VMRK).

### Also fixed: label leakage at the walk-forward fold boundary

`model.py` claimed "no ticker ever leaks future dates into an earlier fold's
training set." That claim was false at the boundary. Training used
`df[df.date <= train_end]`, but every label here is built from the **next**
day's close — so a training row dated exactly `train_end` had a label
determined by the first day of the test fold. The model was trained on the
answer to the first question it was about to be asked.

`PURGE_DAYS` (default 1, matching the 1-day label horizon) now drops the last
training day before each test fold — the standard purging treatment from
López de Prado, *Advances in Financial Machine Learning*, ch. 7. The effect on
headline numbers is small (one day × universe size, out of hundreds of
thousands of training rows); the point is that the correctness claim the
module makes is now actually true. `tests/test_purging.py` pins it down.

### Also fixed: the benchmark was scored over a different date range

`long_short_backtest._evaluate_daily` compared the strategy against an
equal-weighted universe benchmark, but computed the strategy's returns over
only the days it actually traded while computing the benchmark over **every**
prediction date. Both portfolio builders skip days (too few names to fill a
decile, an empty leg after hysteresis), so the two series covered different
holding periods — and `performance_metrics` annualizes by dividing by each
series' own `n_days`. The benchmark is now restricted to the strategy's traded
dates, which is what makes "does the ranking beat just holding the universe"
a real comparison. The reported `n_benchmark_days` makes the alignment visible.

### Next steps (flagged, not yet attempted)

Two evidence-backed levers were identified but not built, given the time
this investigation already took and the honest-conclusion discipline of not
shipping another unvalidated result:

- **A real, dated sentiment dataset**: [`FNSPID`](https://github.com/Zdong104/FNSPID_Financial_News_Dataset)
  (`Zihan1004/FNSPID` on Hugging Face) has 15.7M financial news records with
  real per-article dates for 4,775 S&P 500 companies, 1999–2023 — unlike the
  live-only sentiment classifier in this repo's "Live headline sentiment"
  panel (see `src/sentiment.py`), this could actually be backtested and
  added as a genuine historical feature, since it has the dates the earlier
  dataset lacked.
- **Real fundamental data**: [`edgartools`](https://github.com/dgunning/edgartools)
  (free, no API key beyond an email identifier) pulls actual SEC filings —
  10-K/10-Q financials, insider trades, institutional holdings — back to
  1994. Every feature tested in this investigation is price/volume-derived;
  genuine value/quality/earnings-surprise factors are a structurally
  different data source this repo has never touched.

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
