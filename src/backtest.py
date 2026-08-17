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

from src.data_ingestion import fetch_prices, fetch_vix
from src.features import build_feature_dataset
from src.model import train_and_select_best, walk_forward_predictions

TRADING_DAYS_PER_YEAR = 252


def compute_regime_gate(scored: pd.DataFrame, vix: pd.DataFrame) -> pd.Series:
    """Whether each row's VIX regime has shown positive historical trading edge.

    `scored` must carry date, fold, position (the model's raw threshold decision,
    before any gating) and next_day_return. For each fold, VIX terciles (low/mid/
    high) and the mean return earned on days the model would have traded in each
    tercile are estimated using only STRICTLY PRIOR folds — never the fold being
    scored — so this stays leakage-free the same way the walk-forward split
    already is. Rows with no prior evidence (fold 0, or a regime never seen
    before) pass through ungated rather than being blocked on nothing.
    """
    df = scored.merge(vix, on="date", how="left")
    df["vix_close"] = df["vix_close"].ffill()
    gate = pd.Series(True, index=scored.index)

    if "fold" not in df.columns or df["vix_close"].isna().all():
        return gate

    df["regime"] = pd.Series(pd.NA, index=df.index, dtype="object")
    for fold in sorted(df["fold"].dropna().unique()):
        prior_mask = df["fold"] < fold
        current_mask = df["fold"] == fold
        prior_vix = df.loc[prior_mask, "vix_close"]
        if len(prior_vix) < 10:
            continue  # not enough history yet to define regimes for this fold

        q1, q3 = prior_vix.quantile([1 / 3, 2 / 3])
        bins = [-np.inf, q1, q3, np.inf]
        labels = ["low", "mid", "high"]
        df.loc[prior_mask, "regime"] = pd.cut(prior_vix, bins=bins, labels=labels).astype(str)
        df.loc[current_mask, "regime"] = pd.cut(df.loc[current_mask, "vix_close"], bins=bins, labels=labels).astype(str)

        traded_prior = df.loc[prior_mask & (df["position"] > 0)]
        edge_by_regime = traded_prior.groupby("regime")["next_day_return"].mean()
        good_regimes = set(edge_by_regime[edge_by_regime > 0].index)

        current_regimes = df.loc[current_mask, "regime"]
        seen_regimes = set(edge_by_regime.index)
        fold_gate = current_regimes.isin(good_regimes) | ~current_regimes.isin(seen_regimes)
        gate.loc[df.index[current_mask]] = fold_gate.values

    return gate


def add_strategy_returns(
    predictions: pd.DataFrame,
    threshold: float = 0.5,
    position_sizing: str = "binary",
    cost_bps: float = 0.0,
    regime_filter: bool = False,
    vix: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Turns model predictions into a strategy return series.

    position_sizing: "binary" (full size whenever proba_up > threshold, the
    original behavior) or "confidence" (size scales from 0 to 1 with how far
    proba_up sits above threshold).
    cost_bps: per-unit-of-turnover transaction cost in basis points, charged
    whenever position size changes (opening, closing, or resizing a position).
    regime_filter: if True, positions are zeroed out on days whose VIX regime
    hasn't shown positive historical edge (see compute_regime_gate) — requires
    `vix` (see src.data_ingestion.fetch_vix).
    """
    df = predictions.dropna(subset=["next_day_return"]).copy()

    if position_sizing == "binary":
        raw_position = (df["proba_up"] > threshold).astype(float)
    elif position_sizing == "confidence":
        raw_position = ((df["proba_up"] - threshold) / (1 - threshold)).clip(lower=0.0, upper=1.0)
    else:
        raise ValueError(f"Unknown position_sizing: {position_sizing!r}")
    df["position"] = raw_position

    if regime_filter:
        if vix is None:
            raise ValueError("regime_filter=True requires a vix DataFrame (see fetch_vix())")
        gate = compute_regime_gate(df, vix)
        df["position"] = df["position"].where(gate.reindex(df.index, fill_value=True), 0.0)

    df["signal"] = (df["position"] > 0).astype(int)

    df = df.sort_values(["ticker", "date"])
    prev_position = df.groupby("ticker")["position"].shift(1).fillna(0.0)
    turnover = (df["position"] - prev_position).abs()
    df["transaction_cost"] = turnover * (cost_bps / 10_000.0)

    df["strategy_return"] = df["position"] * df["next_day_return"] - df["transaction_cost"]
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


def backtest_by_ticker(
    predictions: pd.DataFrame,
    threshold: float = 0.5,
    position_sizing: str = "binary",
    cost_bps: float = 0.0,
    regime_filter: bool = False,
    vix: pd.DataFrame | None = None,
) -> dict:
    scored = add_strategy_returns(predictions, threshold, position_sizing, cost_bps, regime_filter, vix)
    results = {}
    for ticker, group in scored.groupby("ticker"):
        group = group.sort_values("date")
        results[ticker] = {
            "n_days": int(len(group)),
            "strategy": performance_metrics(group["strategy_return"]),
            "buy_hold": performance_metrics(group["buy_hold_return"]),
        }
    return results


def backtest_portfolio(
    predictions: pd.DataFrame,
    threshold: float = 0.5,
    position_sizing: str = "binary",
    cost_bps: float = 0.0,
    regime_filter: bool = False,
    vix: pd.DataFrame | None = None,
) -> dict:
    """Equal-weight across all tickers each day (rebalanced daily)."""
    scored = add_strategy_returns(predictions, threshold, position_sizing, cost_bps, regime_filter, vix)
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
    parser.add_argument("--position-sizing", choices=["binary", "confidence"], default="binary")
    parser.add_argument("--cost-bps", type=float, default=0.0, help="Per-unit-of-turnover transaction cost, in basis points")
    parser.add_argument(
        "--regime-filter", action="store_true",
        help="Only trade when the current VIX regime has shown positive historical edge",
    )
    parser.add_argument("--predictions-out", default="reports/backtest_predictions.csv")
    parser.add_argument("--metrics-out", default="reports/backtest_metrics.json")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    prices = fetch_prices(tickers) if tickers else fetch_prices()
    df = build_feature_dataset(prices)

    best_name, _, _ = train_and_select_best(df)
    predictions = walk_forward_predictions(df, best_name)
    vix = fetch_vix() if args.regime_filter else None

    kwargs = dict(
        position_sizing=args.position_sizing, cost_bps=args.cost_bps,
        regime_filter=args.regime_filter, vix=vix,
    )
    by_ticker = backtest_by_ticker(predictions, args.threshold, **kwargs)
    portfolio = backtest_portfolio(predictions, args.threshold, **kwargs)

    Path(args.predictions_out).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.predictions_out, index=False)
    Path(args.metrics_out).write_text(
        json.dumps(
            {
                "model": best_name,
                "threshold": args.threshold,
                "position_sizing": args.position_sizing,
                "cost_bps": args.cost_bps,
                "regime_filter": args.regime_filter,
                "by_ticker": by_ticker,
                "portfolio": portfolio,
            },
            indent=2,
        )
    )

    print(f"Model used: {best_name}")
    print(f"Portfolio strategy Sharpe: {portfolio['strategy']['sharpe']} vs buy-hold Sharpe: {portfolio['buy_hold']['sharpe']}")
    print(f"Saved predictions to {args.predictions_out}, metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
