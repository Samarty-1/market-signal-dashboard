"""Reproduces the cross-sectional long-short investigation end to end: fetch
a broad universe (S&P 500 by default), build the extended feature set,
walk-forward predict, evaluate both portfolio constructions, and print the
honest selection-vs-confirmation split that this repo's README documents.

This is not a "run this to make money" script -- see README "Cross-sectional
long-short: the full investigation" for why the honest conclusion is a real
but small (IC ~0.02-0.03), not robustly tradeable, edge. It exists so the
investigation is reproducible, not just described.

Usage:
    python -m src.cross_sectional_long_short_pipeline
    python -m src.cross_sectional_long_short_pipeline --universe smallcap --period 10y
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from src import model as model_mod
from src import universe as universe_mod
from src.data_ingestion import fetch_sp500_tickers, fetch_universe_prices
from src.features import build_feature_dataset
from src.long_short_backtest import evaluate_hysteresis_long_short, evaluate_long_short

SP600_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"


def _fetch_smallcap_tickers() -> list[str]:
    import io
    import urllib.request

    req = urllib.request.Request(SP600_URL, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=30).read()
    table = pd.read_html(io.BytesIO(html))[0]
    return table["Symbol"].str.replace(".", "-", regex=False).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", choices=["sp500", "smallcap"], default="sp500")
    parser.add_argument("--period", default="10y", help="yfinance period, e.g. 3y, 10y")
    parser.add_argument("--model", default="xgboost", choices=list(model_mod.CANDIDATE_MODELS))
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument(
        "--survivorship",
        choices=["point-in-time", "current-members"],
        default="point-in-time",
        help=(
            "point-in-time (default): reconstruct historical index membership so a "
            "stock is only ranked on dates it was actually a member, and pull the "
            "names that later LEFT the index too. current-members: the old, "
            "survivorship-biased behaviour -- today's members only. Kept as an "
            "option purely so the size of the bias can be measured, not because "
            "it is ever the right choice."
        ),
    )
    args = parser.parse_args()

    membership = None
    if args.universe == "sp500" and args.survivorship == "point-in-time":
        # Look-back window for membership must match the price window, so
        # "10y" of prices is ranked against 10y of membership rather than
        # against today's list. See src/universe.py for why this matters and
        # what it can't fix.
        years = int(args.period.rstrip("y") or 10) if args.period.endswith("y") else 10
        start = pd.Timestamp.today() - pd.DateOffset(years=years)
        print(f"Reconstructing point-in-time S&P 500 membership from {start.date()} (quarterly samples)...")
        t0 = time.time()
        membership = universe_mod.membership_history(start)
        tickers = universe_mod.ever_members(membership)
        current = set(universe_mod.members_asof(pd.Timestamp.today()))
        print(
            f"  {len(tickers)} names were index members at some point "
            f"({len(set(tickers) - current)} of them have since left the index) in {time.time() - t0:.1f}s"
        )
    else:
        print(f"Fetching {args.universe} tickers (current members only -- survivorship-biased)...")
        tickers = fetch_sp500_tickers() if args.universe == "sp500" else _fetch_smallcap_tickers()
        print(f"  {len(tickers)} tickers")

    print(f"Fetching {args.period} of price history...")
    t0 = time.time()
    prices = fetch_universe_prices(tickers, period=args.period)
    print(f"  {len(prices)} rows in {time.time() - t0:.1f}s")

    if membership is not None:
        coverage = universe_mod.coverage_report(membership, prices)
        print(
            f"  Price coverage of the true universe: {coverage['n_with_price_data']}/"
            f"{coverage['n_ever_members']} names "
            f"({coverage['pct_missing']}% MISSING -- residual survivorship bias, see src/universe.py)"
        )

        prices, recycled = universe_mod.drop_recycled_tickers(prices, membership)
        if recycled:
            print(
                f"  Dropped {len(recycled)} recycled ticker(s) whose price history doesn't overlap "
                f"their index membership: {', '.join(recycled[:12])}"
                + (" ..." if len(recycled) > 12 else "")
            )

    print("Building features...")
    feats = build_feature_dataset(prices)
    print(f"  {len(feats)} rows, {feats['ticker'].nunique()} tickers, {feats['date'].min().date()} to {feats['date'].max().date()}")

    if membership is not None:
        before = len(feats)
        feats = universe_mod.apply_point_in_time_membership(feats, membership)
        print(
            f"  Point-in-time membership mask: {before} -> {len(feats)} rows "
            f"({100 * (before - len(feats)) / max(before, 1):.1f}% dropped as not-yet/no-longer index members)"
        )

    print(f"Walk-forward predicting ({args.model}, {args.n_folds} folds)...")
    t0 = time.time()
    preds = model_mod.walk_forward_predictions(
        feats, args.model, label_column="label_beat_median_next_day", n_folds=args.n_folds
    )
    print(f"  {len(preds)} predictions in {time.time() - t0:.1f}s")

    # Selection/confirmation split at the halfway fold -- picking a portfolio
    # construction using the SAME data you then report results on is exactly
    # the overfitting trap this whole investigation was built to avoid (see
    # README). Whatever you tune, tune only on `selection`; only ever look
    # at `confirmation` once, at the end, without going back to retune.
    fold_dates = preds.groupby("fold")["date"].max()
    mid_fold = args.n_folds // 2 - 1
    selection_end = fold_dates.loc[mid_fold]
    selection = preds[preds["date"] <= selection_end]
    confirmation = preds[preds["date"] > selection_end]
    print(f"\nSelection set:    {selection['date'].min().date()} to {selection['date'].max().date()} ({selection['date'].nunique()} days)")
    print(f"Confirmation set: {confirmation['date'].min().date()} to {confirmation['date'].max().date()} ({confirmation['date'].nunique()} days)")

    print("\n=== Hard-decile construction, selection set ===")
    for n in (1, 10, 21, 63):
        r = evaluate_long_short(selection, cost_bps=args.cost_bps, rebalance_every=n)
        print(f"  rebalance_every={n:3d}: sharpe={r['long_short']['sharpe']:7.3f} return={r['long_short']['total_return']:7.3f}")

    print("\n=== Hysteresis construction, selection set ===")
    for n in (1, 10, 21):
        r = evaluate_hysteresis_long_short(selection, cost_bps=args.cost_bps, entry_pct=0.10, exit_pct=0.25, rebalance_every=n)
        print(f"  rebalance_every={n:3d}: sharpe={r['long_short']['sharpe']:7.3f} return={r['long_short']['total_return']:7.3f}")

    print("\n=== Honest confirmation (hysteresis, rebalance_every=10 -- the README's selection-set winner) ===")
    r = evaluate_hysteresis_long_short(confirmation, cost_bps=args.cost_bps, entry_pct=0.10, exit_pct=0.25, rebalance_every=10)
    print(f"  long_short:         {r['long_short']}")
    print(f"  universe_eq_weight: {r['universe_equal_weight']}")


if __name__ == "__main__":
    main()
