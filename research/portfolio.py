"""Dollar-neutral portfolio simulator with honest turnover accounting.

Why this replaces the repo's set-membership construction
--------------------------------------------------------
`long_short_backtest.build_long_short_returns` picks a hard top/bottom decile
each day and charges cost on how many NAMES changed. Two problems:

1. A hard cutoff turns an infinitesimal rank change into a full position.
   A name that drifts from 50th to 51st place goes from fully-held to
   fully-sold, paying a round trip for a difference that carries no
   information. With a noisy daily signal that is most of the turnover.

2. Name-count turnover ignores position SIZE. Real cost is paid on the
   dollars traded, i.e. sum |w_t - w_{t-1}|, not on how many tickers
   appeared or disappeared from a set.

This module holds CONTINUOUS weights derived from the signal's cross-sectional
rank, so a small change in rank moves a small amount of money, and charges cost
on actual weight deltas. It adds the two levers the repo never tried:

  * signal smoothing  -- an EMA over the score. Turnover falls roughly with the
    smoothing window while alpha is retained to the extent the signal is
    persistent, so it is close to a free trade of cost against a little IC.
  * a no-trade band   -- leave a position alone until it drifts far enough from
    target to be worth the spread. Standard practice, and the direct fix for
    paying to chase noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def signal_to_weights(score: pd.Series, gross: float = 1.0) -> pd.DataFrame:
    """Cross-sectional rank -> dollar-neutral weights, as a wide date x ticker frame.

    Rank within the date, centre it, and scale so the book is dollar-neutral
    (sum w = 0) with gross exposure `gross` (sum |w| = gross). Every name gets a
    weight proportional to how far from median it ranks, so the portfolio
    expresses the whole ranking rather than just its two tails.
    """
    wide = score.unstack("ticker").sort_index()
    r = wide.rank(axis=1, pct=True) - 0.5          # [-0.5, 0.5], NaNs preserved
    r = r.sub(r.mean(axis=1), axis=0)              # exact dollar neutrality
    denom = r.abs().sum(axis=1).replace(0, np.nan)
    return r.div(denom, axis=0) * gross


def decile_weights(score: pd.Series, n_q: int = 10, gross: float = 1.0) -> pd.DataFrame:
    """The repo's construction: equal-weight the top decile long and the bottom
    decile short, nothing in between. Kept so the before/after comparison runs
    both books through the identical cost model and P&L accounting.
    """
    wide = score.unstack("ticker").sort_index()
    r = wide.rank(axis=1, pct=True)
    long_leg = (r > 1 - 1 / n_q).astype(float)
    short_leg = (r <= 1 / n_q).astype(float)
    n_long = long_leg.sum(axis=1).replace(0, np.nan)
    n_short = short_leg.sum(axis=1).replace(0, np.nan)
    w = long_leg.div(n_long, axis=0) * (gross / 2) - short_leg.div(n_short, axis=0) * (gross / 2)
    return w.where(wide.notna())


def smooth(weights: pd.DataFrame, halflife: float | None) -> pd.DataFrame:
    """EMA the target weights over time, then re-neutralise and re-scale.

    This is the highest-leverage cost lever available: turnover falls roughly
    in proportion to the smoothing window, while the alpha retained depends on
    how persistent the underlying signal is. Re-normalising after smoothing
    keeps gross exposure constant so the Sharpe comparison stays like-for-like
    (otherwise smoothing would quietly shrink the book and flatter the vol).
    """
    if halflife is None or halflife <= 0:
        return weights
    sm = weights.ewm(halflife=halflife, min_periods=1, ignore_na=True).mean()
    sm = sm.where(weights.notna())
    sm = sm.sub(sm.mean(axis=1), axis=0)
    denom = sm.abs().sum(axis=1).replace(0, np.nan)
    gross = weights.abs().sum(axis=1)
    return sm.div(denom, axis=0).mul(gross, axis=0)


def simulate(
    target_w: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    cost_bps: float = 10.0,
    band: float = 0.0,
    rebalance_every: int = 1,
) -> pd.DataFrame:
    """Walk the book forward one day at a time.

    target_w / fwd_ret : wide date x ticker, aligned.
    band               : no-trade threshold as a fraction of the average
                         absolute target weight. A position is left alone until
                         it drifts more than this from its target.
    Returns per-date: gross_return, cost, net_return, turnover.
    """
    dates = target_w.index
    cols = target_w.columns
    tgt = target_w.to_numpy(dtype=float)
    ret = fwd_ret.reindex(index=dates, columns=cols).to_numpy(dtype=float)

    held = np.zeros(len(cols))
    rows = []
    cost_rate = cost_bps / 10_000.0

    for i in range(len(dates)):
        t = tgt[i]
        tradable = ~np.isnan(t)
        t = np.where(tradable, t, 0.0)

        if i % rebalance_every == 0:
            if band > 0:
                scale = np.nanmean(np.abs(t)) if tradable.any() else 0.0
                move = np.abs(t - held) > band * scale
                new = np.where(move, t, held)
            else:
                new = t
            # A name that has left the universe must be closed regardless of band.
            new = np.where(tradable, new, 0.0)
        else:
            new = np.where(tradable, held, 0.0)

        turnover = np.abs(new - held).sum()
        cost = turnover * cost_rate
        held = new

        r = ret[i]
        gross = float(np.nansum(held * np.where(np.isnan(r), 0.0, r)))
        rows.append((dates[i], gross, cost, gross - cost, turnover))

    return pd.DataFrame(rows, columns=["date", "gross_return", "cost", "net_return", "turnover"]).set_index("date")


def metrics(sim: pd.DataFrame) -> dict:
    net, gross = sim["net_return"], sim["gross_return"]
    if len(net) == 0:
        return {}
    def _sharpe(x):
        v = x.std() * np.sqrt(252)
        return round(float(x.mean() * 252 / v), 3) if v > 0 else np.nan
    curve = (1 + net).cumprod()
    return {
        "n_days": int(len(net)),
        "gross_sharpe": _sharpe(gross),
        "net_sharpe": _sharpe(net),
        "gross_ann_ret": round(float(gross.mean() * 252), 4),
        "net_ann_ret": round(float(net.mean() * 252), 4),
        "ann_vol": round(float(net.std() * np.sqrt(252)), 4),
        "max_dd": round(float((curve / curve.cummax() - 1).min()), 4),
        "turnover_daily": round(float(sim["turnover"].mean()), 4),
        "cost_bps_day": round(float(sim["cost"].mean() * 10_000), 2),
        "gross_bps_day": round(float(gross.mean() * 10_000), 2),
    }
