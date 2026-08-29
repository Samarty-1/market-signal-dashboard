"""Cost-aware dollar-neutral portfolio construction.

Replaces the hard-decile construction in `long_short_backtest.py`, which loses
money not because the signal is bad but because of how the signal is traded.
Measured on the S&P 500, 2015-2026, 10bps one-way cost, same predictions in both
cases: hard decile re-picked daily earns ~4.5bps/day of spread and pays
~12bps/day in turnover. The construction here pays under 1bp/day.

Three changes, each fixing a specific defect:

1. CONTINUOUS WEIGHTS instead of decile membership.
   A hard cutoff turns an infinitesimal rank change into a full round trip -- a
   name drifting from 50th to 51st place is sold outright, though the
   difference carries no information. Weighting every name by how far from
   median it ranks means a small rank change moves a small amount of money.

2. SIGNAL SMOOTHING (EMA on the target weights).
   Turnover falls roughly in proportion to the smoothing window while the alpha
   retained depends on how persistent the signal is. For a signal this
   persistent it is close to a free trade of cost against a little edge.

3. A NO-TRADE BAND.
   Leave a position alone until it drifts far enough from target to be worth
   the spread, rather than topping up every position every day.

Cost is charged on actual dollars traded, sum |w_t - w_{t-1}|, not on how many
tickers entered or left a set -- name-count turnover ignores position size and
understates the cost of a book that reshuffles its weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_HALFLIFE = 42.0
DEFAULT_BAND = 0.5
DEFAULT_COST_BPS = 10.0


def signal_to_weights(score: pd.Series, gross: float = 1.0) -> pd.DataFrame:
    """Cross-sectional rank -> dollar-neutral weights (wide date x ticker).

    `score` is a long Series indexed by (date, ticker). Ranking within the date
    makes the weights depend only on the ORDER of that day's scores, so a
    model whose raw output drifts in level over time cannot change the book's
    exposure -- only its ranking can.
    """
    wide = score.unstack("ticker").sort_index()
    r = wide.rank(axis=1, pct=True) - 0.5
    r = r.sub(r.mean(axis=1), axis=0)                       # sum w = 0
    denom = r.abs().sum(axis=1).replace(0, np.nan)
    return r.div(denom, axis=0) * gross                     # sum |w| = gross


def smooth_weights(weights: pd.DataFrame, halflife: float | None = DEFAULT_HALFLIFE) -> pd.DataFrame:
    """EMA the target weights, then re-neutralise and restore gross exposure.

    Re-normalising afterwards matters: without it, smoothing quietly shrinks the
    book and flatters the volatility, so a Sharpe comparison against the
    unsmoothed version would not be like-for-like.
    """
    if not halflife or halflife <= 0:
        return weights
    sm = weights.ewm(halflife=halflife, min_periods=1, ignore_na=True).mean()
    sm = sm.where(weights.notna())
    sm = sm.sub(sm.mean(axis=1), axis=0)
    denom = sm.abs().sum(axis=1).replace(0, np.nan)
    return sm.div(denom, axis=0).mul(weights.abs().sum(axis=1), axis=0)


def simulate(
    target_weights: pd.DataFrame,
    forward_returns: pd.DataFrame,
    cost_bps: float = DEFAULT_COST_BPS,
    band: float = DEFAULT_BAND,
    return_positions: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Walk the book forward a day at a time. Returns per-date columns:
    gross_return, cost, net_return, turnover.

    With return_positions=True also returns the actually-held weights per date,
    which is what you want to inspect when checking exposures or confirming a
    delisted name really was closed.
    """
    dates, cols = target_weights.index, target_weights.columns
    tgt = target_weights.to_numpy(dtype=float)
    ret = forward_returns.reindex(index=dates, columns=cols).to_numpy(dtype=float)

    held = np.zeros(len(cols))
    cost_rate = cost_bps / 10_000.0
    rows = []
    positions = np.empty_like(tgt) if return_positions else None

    for i in range(len(dates)):
        t = tgt[i]
        tradable = ~np.isnan(t)
        t = np.where(tradable, t, 0.0)

        if band > 0:
            scale = np.abs(t).mean() if tradable.any() else 0.0
            new = np.where(np.abs(t - held) > band * scale, t, held)
        else:
            new = t
        # A name that has left the universe must be closed regardless of band.
        new = np.where(tradable, new, 0.0)

        turnover = float(np.abs(new - held).sum())
        held = new
        if positions is not None:
            positions[i] = held
        r = ret[i]
        gross = float(np.nansum(held * np.where(np.isnan(r), 0.0, r)))
        cost = turnover * cost_rate
        rows.append((dates[i], gross, cost, gross - cost, turnover))

    sim = pd.DataFrame(
        rows, columns=["date", "gross_return", "cost", "net_return", "turnover"]
    ).set_index("date")
    if positions is not None:
        return sim, pd.DataFrame(positions, index=dates, columns=cols)
    return sim


def performance(sim: pd.DataFrame) -> dict:
    """Annualised summary of a simulated book."""
    if len(sim) == 0:
        return {}
    net, gross = sim["net_return"], sim["gross_return"]

    def _sharpe(x: pd.Series) -> float | None:
        v = float(x.std() * np.sqrt(252))
        return round(float(x.mean() * 252) / v, 3) if v > 0 else None

    curve = (1 + net).cumprod()
    return {
        "n_days": int(len(net)),
        "gross_sharpe": _sharpe(gross),
        "net_sharpe": _sharpe(net),
        "gross_ann_return": round(float(gross.mean() * 252), 4),
        "net_ann_return": round(float(net.mean() * 252), 4),
        "ann_vol": round(float(net.std() * np.sqrt(252)), 4),
        "max_drawdown": round(float((curve / curve.cummax() - 1).min()), 4),
        "turnover_daily": round(float(sim["turnover"].mean()), 4),
        "cost_bps_per_day": round(float(sim["cost"].mean() * 10_000), 2),
        "gross_bps_per_day": round(float(gross.mean() * 10_000), 2),
    }


def evaluate_signal(
    score: pd.Series,
    forward_returns: pd.DataFrame,
    halflife: float | None = DEFAULT_HALFLIFE,
    band: float = DEFAULT_BAND,
    cost_bps: float = DEFAULT_COST_BPS,
) -> dict:
    """Convenience: score -> weights -> smoothed -> simulated -> metrics."""
    w = smooth_weights(signal_to_weights(score), halflife)
    fwd = forward_returns.reindex(index=w.index, columns=w.columns)
    return performance(simulate(w, fwd, cost_bps=cost_bps, band=band))


def decompose_signal(
    score: pd.Series, min_periods: int = 120
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Split a score into its persistent and live parts, causally.

        static(t, i) = mean(score(s, i) for s < t)   persistent per-name tilt
        timing(t, i) = score(t, i) - static(t, i)    live deviation from normal

    Worth reporting separately: a book on `static` alone collects a size /
    liquidity / volatility risk premium while barely trading, which is a real
    return but is NOT evidence the model can time anything. Only the `timing`
    book measures stock-selection skill, and it is tilt-free by construction.
    """
    wide = score.unstack("ticker").sort_index()
    static = wide.expanding(min_periods=min_periods).mean().shift(1)
    timing = wide - static
    keep = static.notna() & wide.notna()
    return (
        wide.where(keep).stack().dropna(),
        static.where(keep).stack().dropna(),
        timing.where(keep).stack().dropna(),
    )
