"""Measures how much the survivorship-biased universe inflated this repo's
cross-sectional long-short results.

Runs the SAME model, features, and portfolio construction twice -- once on
today's S&P 500 members (the old behaviour), once on point-in-time membership
with the names that later left the index included (src/universe.py) -- and
prints both side by side. Everything except the universe is held fixed, so any
difference is attributable to the universe construction alone.

    python -m scripts.measure_survivorship_bias --period 5y

See src/universe.py for why the point-in-time arm still can't fully remove the
bias (Yahoo doesn't serve most delisted names) and what it reports instead.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import pandas as pd

from src import model as model_mod
from src import universe as universe_mod
from src.data_ingestion import fetch_sp500_tickers, fetch_universe_prices
from src.features import build_feature_dataset
from src.long_short_backtest import evaluate_hysteresis_long_short

warnings.filterwarnings("ignore")

LABEL = "label_beat_median_next_day"


def run_arm(name: str, prices: pd.DataFrame, membership: pd.DataFrame | None, args: dict) -> dict:
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")

    feats = build_feature_dataset(prices)
    print(f"  features: {len(feats):,} rows, {feats['ticker'].nunique()} tickers")

    if membership is not None:
        before = len(feats)
        feats = universe_mod.apply_point_in_time_membership(feats, membership)
        print(f"  point-in-time mask: {before:,} -> {len(feats):,} rows")

    t0 = time.time()
    preds = model_mod.walk_forward_predictions(
        feats, args["model"], label_column=LABEL, n_folds=args["n_folds"]
    )
    print(f"  {len(preds):,} out-of-sample predictions in {time.time() - t0:.0f}s")

    fold_dates = preds.groupby("fold")["date"].max()
    mid_fold = args["n_folds"] // 2 - 1
    selection_end = fold_dates.loc[mid_fold]
    confirmation = preds[preds["date"] > selection_end]

    result = evaluate_hysteresis_long_short(
        confirmation, cost_bps=args["cost_bps"], entry_pct=0.10, exit_pct=0.25, rebalance_every=10
    )
    print(f"  confirmation-set long/short: {json.dumps(result['long_short'])}")
    print(f"  confirmation-set benchmark : {json.dumps(result['universe_equal_weight'])}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="5y")
    parser.add_argument("--model", default="xgboost", choices=list(model_mod.CANDIDATE_MODELS))
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    args = parser.parse_args()
    cfg = {"model": args.model, "n_folds": args.n_folds, "cost_bps": args.cost_bps}

    years = int(args.period.rstrip("y"))
    start = pd.Timestamp.today() - pd.DateOffset(years=years)

    print(f"Reconstructing point-in-time membership from {start.date()} (cached after first run)...")
    membership = universe_mod.membership_history(start)
    ever = universe_mod.ever_members(membership)
    current = sorted(fetch_sp500_tickers())
    left_index = sorted(set(ever) - set(current))
    print(f"  ever-members: {len(ever)}  |  current members: {len(current)}  |  left the index: {len(left_index)}")

    print(f"\nFetching {args.period} of prices for the full ever-member universe...")
    t0 = time.time()
    prices_all = fetch_universe_prices(ever, period=args.period)
    print(f"  {len(prices_all):,} rows in {time.time() - t0:.0f}s")

    coverage = universe_mod.coverage_report(membership, prices_all)
    print(f"  coverage: {coverage['n_with_price_data']}/{coverage['n_ever_members']} names have data "
          f"({coverage['pct_missing']}% missing -> residual, unfixable-with-free-data survivorship bias)")

    prices_all, recycled = universe_mod.drop_recycled_tickers(prices_all, membership)
    print(f"  recycled tickers dropped: {len(recycled)} {recycled[:15]}")

    biased = run_arm(
        "ARM A -- current members only (the old, survivorship-biased universe)",
        prices_all[prices_all["ticker"].isin(current)].reset_index(drop=True),
        None,
        cfg,
    )
    honest = run_arm(
        "ARM B -- point-in-time membership (includes names that later left the index)",
        prices_all,
        membership,
        cfg,
    )

    print(f"\n{'=' * 70}\nSURVIVORSHIP BIAS, MEASURED\n{'=' * 70}")
    print(f"{'metric':<22}{'biased':>14}{'point-in-time':>16}{'inflation':>14}")
    for key in ("total_return", "annualized_return", "sharpe", "max_drawdown"):
        b, h = biased["long_short"][key], honest["long_short"][key]
        print(f"{key:<22}{b:>14.4f}{h:>16.4f}{b - h:>14.4f}")
    print(f"\n{'benchmark sharpe':<22}{biased['universe_equal_weight']['sharpe']:>14.4f}"
          f"{honest['universe_equal_weight']['sharpe']:>16.4f}"
          f"{biased['universe_equal_weight']['sharpe'] - honest['universe_equal_weight']['sharpe']:>14.4f}")

    with open("data/survivorship_comparison.json", "w") as fh:
        json.dump({"biased": biased, "point_in_time": honest, "coverage": coverage,
                   "recycled_dropped": recycled, "config": cfg}, fh, indent=2, default=str)
    print("\nSaved data/survivorship_comparison.json")


if __name__ == "__main__":
    main()
