"""Cross-sectional long-short decile portfolio.

This is the construction the ML asset-pricing literature (Gu/Kelly/Xiu and
successors) actually uses to turn a tiny individual-stock edge (out-of-sample
R^2 ~0.3-0.4% per stock) into a real portfolio-level Sharpe -- by pooling
that edge across hundreds of stocks each day, not by trying to predict any
single stock well. See README "Long-short decile portfolio" for the numbers
this produced.

Each day: rank every ticker with an out-of-sample prediction that day by
proba_up (from src.model's cross-sectional walk-forward classifier), go long
the top decile (equal-weighted) and short the bottom decile (equal-weighted).
Equal notional on both legs makes this dollar-neutral -- the portfolio
return each day is mean(top-decile next_day_return) - mean(bottom-decile
next_day_return), which isolates the cross-sectional ranking signal from
whatever the overall market did that day, unlike a long-only strategy.

Only ever uses walk-forward OUT-OF-SAMPLE predictions (never a full-data-fit
model scored on its own training history) -- same discipline as
src.backtest, for the same reason: scoring a model on data it trained on
leaks the future into what's supposed to be a "past" prediction and produces
backtest numbers that are fiction.
"""

from __future__ import annotations

import pandas as pd

from src.backtest import performance_metrics

N_DECILES = 10
MIN_UNIVERSE_PER_DECILE = 3  # need at least this many names per decile bucket
# for "decile" to mean anything -- fewer and a single name's idiosyncratic
# move dominates the whole leg's return.


def build_long_short_returns(
    predictions: pd.DataFrame,
    cost_bps: float = 10.0,
    n_deciles: int = N_DECILES,
    rebalance_every: int = 1,
) -> pd.DataFrame:
    """predictions must have columns: date, ticker, proba_up, next_day_return
    (exactly what src.model.walk_forward_predictions produces). Returns one
    row per date with the long-short portfolio's return and diagnostics.

    rebalance_every: re-rank and re-pick the long/short names only every
    this many trading days; membership (and turnover cost) is held fixed
    between rebalances, though the return is still marked to market every
    day using whichever names are currently held. Default 1 = re-rank daily.
    Daily rebalancing measured ~20bps/day turnover cost against a ~5bps/day
    raw spread -- costs 4x the signal -- see README "Why daily rebalancing
    doesn't work" for the honest before/after. rebalance_every=5/21 (weekly/
    monthly) is the standard institutional fix for exactly this problem:
    the ranking signal persists across days even though noisy day-to-day
    reshuffling at the margin doesn't carry real information, so paying to
    chase it is pure cost with no benefit.
    """
    df = predictions.dropna(subset=["proba_up", "next_day_return"]).copy()
    dates = sorted(df["date"].unique())

    rows = []
    prev_long: set[str] = set()
    prev_short: set[str] = set()
    current_long: set[str] = set()
    current_short: set[str] = set()
    for i, date in enumerate(dates):
        group = df[df["date"] == date]
        is_rebalance_day = i % rebalance_every == 0

        if is_rebalance_day:
            if len(group) < n_deciles * MIN_UNIVERSE_PER_DECILE:
                continue
            # Rank first, then qcut on the rank -- qcut directly on
            # proba_up can fail (or silently merge buckets) when a tree
            # model outputs the same probability for many tickers on the
            # same day, which happens often in practice.
            ranks = group["proba_up"].rank(method="first")
            decile = pd.qcut(ranks, n_deciles, labels=False)
            current_long = set(group.loc[decile == n_deciles - 1, "ticker"])
            current_short = set(group.loc[decile == 0, "ticker"])

        if not current_long and not current_short:
            continue

        long_names, short_names = current_long, current_short
        long_ret = group.loc[group["ticker"].isin(long_names), "next_day_return"].mean()
        short_ret = group.loc[group["ticker"].isin(short_names), "next_day_return"].mean()

        # Turnover-based cost: only charged on rebalance days (see docstring)
        # -- fraction of each leg that changed since the last rebalance.
        if is_rebalance_day:
            long_turnover = len(long_names.symmetric_difference(prev_long)) / max(len(long_names), 1)
            short_turnover = len(short_names.symmetric_difference(prev_short)) / max(len(short_names), 1)
            cost = (long_turnover + short_turnover) * (cost_bps / 10_000.0)
            prev_long, prev_short = long_names, short_names
        else:
            cost = 0.0

        rows.append(
            {
                "date": date,
                "n_universe": len(group),
                "n_long": len(long_names),
                "n_short": len(short_names),
                "long_return": long_ret,
                "short_return": short_ret,
                "spread_return": long_ret - short_ret,
                "transaction_cost": cost,
                "long_short_return": (long_ret - short_ret) - cost,
            }
        )

    columns = [
        "date", "n_universe", "n_long", "n_short",
        "long_return", "short_return", "spread_return", "transaction_cost", "long_short_return",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("date").reset_index(drop=True)


def build_hysteresis_long_short_returns(
    predictions: pd.DataFrame,
    cost_bps: float = 10.0,
    entry_pct: float = 0.10,
    exit_pct: float = 0.25,
    rebalance_every: int = 1,
) -> pd.DataFrame:
    """Like build_long_short_returns, but with asymmetric entry/exit
    thresholds instead of a rigid decile re-picked from scratch every
    rebalance day: a name enters the long leg only once it ranks in the top
    entry_pct, but stays in the long leg until it falls out of the wider
    top exit_pct (mirrored for the short leg on the bottom).

    Motivation: an alphalens IC check (see README "Why hard deciles were
    the wrong construction") confirmed the underlying ranking signal is
    real and holds up out-of-sample (positive IC on a genuinely untouched
    confirmation set) -- but build_long_short_returns's hard top/bottom
    decile cutoff, re-picked from scratch every day, was paying to chase
    noisy day-to-day reshuffling at the margin (a name ranked 51st vs. 50th
    is not a meaningful difference, but a hard cutoff treats it as one).
    This is the same hysteresis principle already used to fix the RL
    trading agent's churn problem (see quantpulse's MIN_HOLD_DAYS) --
    applied here to a different kind of "stop switching on noise."

    rebalance_every: hysteresis alone (checked every day) still left
    turnover cost (~13bps/day) above the raw signal (~5bps/day) -- daily
    re-checking, even with wide bands, still churns on noisy day-to-day rank
    wobble near the bands. Combining hysteresis with periodic rebalancing
    (only re-check membership every N days, same idea as
    build_long_short_returns's rebalance_every) is what actually gets
    turnover below the signal -- see README for the honest before/after.
    """
    df = predictions.dropna(subset=["proba_up", "next_day_return"]).copy()
    dates = sorted(df["date"].unique())

    rows = []
    current_long: set[str] = set()
    current_short: set[str] = set()
    for i, date in enumerate(dates):
        group = df[df["date"] == date]
        n = len(group)
        if n < 10:
            continue
        is_rebalance_day = i % rebalance_every == 0

        if is_rebalance_day:
            ranked = group.sort_values("proba_up", ascending=False).reset_index(drop=True)
            ranked["pct_rank_from_top"] = (ranked.index + 1) / n  # 1/n = best, 1.0 = worst

            entry_long = set(ranked.loc[ranked["pct_rank_from_top"] <= entry_pct, "ticker"])
            keep_long_eligible = set(ranked.loc[ranked["pct_rank_from_top"] <= exit_pct, "ticker"])
            new_long = (current_long & keep_long_eligible) | entry_long

            entry_short = set(ranked.loc[ranked["pct_rank_from_top"] > 1 - entry_pct, "ticker"])
            keep_short_eligible = set(ranked.loc[ranked["pct_rank_from_top"] > 1 - exit_pct, "ticker"])
            new_short = (current_short & keep_short_eligible) | entry_short
        else:
            new_long, new_short = current_long, current_short

        if not new_long or not new_short:
            continue

        long_ret = group.loc[group["ticker"].isin(new_long), "next_day_return"].mean()
        short_ret = group.loc[group["ticker"].isin(new_short), "next_day_return"].mean()

        if is_rebalance_day:
            long_turnover = len(new_long.symmetric_difference(current_long)) / max(len(new_long), 1)
            short_turnover = len(new_short.symmetric_difference(current_short)) / max(len(new_short), 1)
            cost = (long_turnover + short_turnover) * (cost_bps / 10_000.0)
        else:
            cost = 0.0

        rows.append(
            {
                "date": date,
                "n_universe": n,
                "n_long": len(new_long),
                "n_short": len(new_short),
                "long_return": long_ret,
                "short_return": short_ret,
                "spread_return": long_ret - short_ret,
                "transaction_cost": cost,
                "long_short_return": (long_ret - short_ret) - cost,
            }
        )
        current_long, current_short = new_long, new_short

    columns = [
        "date", "n_universe", "n_long", "n_short",
        "long_return", "short_return", "spread_return", "transaction_cost", "long_short_return",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("date").reset_index(drop=True)


def _evaluate_daily(daily: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """Shared honest-evaluation tail for both portfolio constructions: the
    long-short portfolio's performance vs. an equal-weighted buy-and-hold of
    the same universe (not the S&P index -- keeps this an apples-to-apples
    "does the ranking add anything beyond just holding the universe"
    comparison).

    The benchmark is restricted to the dates the strategy actually traded.
    Both builders skip days (too few names to fill a decile, an empty leg
    after hysteresis), so scoring the strategy over its own subset of dates
    while scoring the benchmark over *every* prediction date compares two
    different holding periods -- total_return over 2,300 benchmark days
    against 1,900 strategy days is not a comparison, and the annualization in
    performance_metrics divides by a different n_days for each. Aligning the
    date sets is what makes the headline "does the ranking beat just holding
    the universe" claim mean anything.
    """
    if daily.empty:
        return {
            "n_days": 0,
            "long_short": performance_metrics(pd.Series(dtype=float)),
            "universe_equal_weight": performance_metrics(pd.Series(dtype=float)),
        }

    traded_dates = set(daily["date"])
    universe = (
        predictions.dropna(subset=["next_day_return"])
        .groupby("date")["next_day_return"]
        .mean()
    )
    universe = universe[universe.index.isin(traded_dates)]

    return {
        "n_days": int(len(daily)),
        "n_benchmark_days": int(len(universe)),
        "long_short": performance_metrics(daily.set_index("date")["long_short_return"]),
        "universe_equal_weight": performance_metrics(universe),
        "mean_n_universe": round(float(daily["n_universe"].mean()), 1),
        "mean_n_long": round(float(daily["n_long"].mean()), 1),
        "mean_n_short": round(float(daily["n_short"].mean()), 1),
    }


def evaluate_long_short(
    predictions: pd.DataFrame, cost_bps: float = 10.0, n_deciles: int = N_DECILES, rebalance_every: int = 1
) -> dict:
    """Hard-decile construction (see build_long_short_returns)."""
    daily = build_long_short_returns(predictions, cost_bps, n_deciles, rebalance_every)
    return _evaluate_daily(daily, predictions)


def evaluate_hysteresis_long_short(
    predictions: pd.DataFrame,
    cost_bps: float = 10.0,
    entry_pct: float = 0.10,
    exit_pct: float = 0.25,
    rebalance_every: int = 1,
) -> dict:
    """Hysteresis construction (see build_hysteresis_long_short_returns)."""
    daily = build_hysteresis_long_short_returns(predictions, cost_bps, entry_pct, exit_pct, rebalance_every)
    return _evaluate_daily(daily, predictions)
