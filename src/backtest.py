"""Turns the model's out-of-sample predictions into a simple long/flat strategy
and compares it against buy-and-hold.

Only out-of-sample predictions (from src.model.walk_forward_predictions) are used —
scoring the final, full-data-fit model on its own training history would leak the
future into the past and produce an unrealistically good backtest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_ingestion import fetch_prices
from src.features import build_feature_dataset
from src.model import train_and_select_best, walk_forward_predictions

TRADING_DAYS_PER_YEAR = 252


def add_strategy_returns(predictions: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    df = predictions.dropna(subset=["next_day_return"]).copy()
    df["signal"] = (df["proba_up"] > threshold).astype(int)
    df["strategy_return"] = np.where(df["signal"] == 1, df["next_day_return"], 0.0)
    df["buy_hold_return"] = df["next_day_return"]
    return df


def performance_metrics(returns: pd.Series) -> dict:
    returns = returns.dropna()
    if returns.empty:
        return {"total_return": 0.0, "annualized_return": 0.0, "annualized_vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}

    cumulative = (1 + returns).cumprod()
    n_days = len(returns)
    total_return = cumulative.iloc[-1] - 1
    annualized_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / n_days) - 1
    annualized_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    # A near-zero (rather than exactly-zero) std from floating-point noise would
    # otherwise blow the Sharpe ratio up to an enormous, meaningless number.
    sharpe = (returns.mean() * TRADING_DAYS_PER_YEAR) / annualized_vol if annualized_vol > 1e-8 else 0.0
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    max_drawdown = drawdown.min()

    return {
        "total_return": round(float(total_return), 4),
        "annualized_return": round(float(annualized_return), 4),
        "annualized_vol": round(float(annualized_vol), 4),
        "sharpe": round(float(sharpe), 4),
        "max_drawdown": round(float(max_drawdown), 4),
    }


def backtest_by_ticker(predictions: pd.DataFrame, threshold: float = 0.5) -> dict:
    scored = add_strategy_returns(predictions, threshold)
    results = {}
    for ticker, group in scored.groupby("ticker"):
        group = group.sort_values("date")
        results[ticker] = {
            "n_days": int(len(group)),
            "strategy": performance_metrics(group["strategy_return"]),
            "buy_hold": performance_metrics(group["buy_hold_return"]),
        }
    return results


def backtest_portfolio(predictions: pd.DataFrame, threshold: float = 0.5) -> dict:
    """Equal-weight across all tickers each day (rebalanced daily)."""
    scored = add_strategy_returns(predictions, threshold)
    daily = scored.groupby("date").agg(strategy_return=("strategy_return", "mean"), buy_hold_return=("buy_hold_return", "mean"))
    return {
        "n_days": int(len(daily)),
        "strategy": performance_metrics(daily["strategy_return"]),
        "buy_hold": performance_metrics(daily["buy_hold_return"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers")
    parser.add_argument("--threshold", type=float, default=0.5, help="Long if predicted P(up) exceeds this")
    parser.add_argument("--predictions-out", default="reports/backtest_predictions.csv")
    parser.add_argument("--metrics-out", default="reports/backtest_metrics.json")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    prices = fetch_prices(tickers) if tickers else fetch_prices()
    df = build_feature_dataset(prices)

    best_name, _, _ = train_and_select_best(df)
    predictions = walk_forward_predictions(df, best_name)

    by_ticker = backtest_by_ticker(predictions, args.threshold)
    portfolio = backtest_portfolio(predictions, args.threshold)

    Path(args.predictions_out).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.predictions_out, index=False)
    Path(args.metrics_out).write_text(
        json.dumps({"model": best_name, "threshold": args.threshold, "by_ticker": by_ticker, "portfolio": portfolio}, indent=2)
    )

    print(f"Model used: {best_name}")
    print(f"Portfolio strategy Sharpe: {portfolio['strategy']['sharpe']} vs buy-hold Sharpe: {portfolio['buy_hold']['sharpe']}")
    print(f"Saved predictions to {args.predictions_out}, metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
