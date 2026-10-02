"""Builds the long-format modelling panel once and caches it.

For every (date, ticker):
  raw_*      features as the current repo computes them (absolute levels)
  cs_*       same features rank-normalised within the date to [-1, 1]
  csz_*      same features z-scored within the date (keeps relative MAGNITUDE,
             which rank throws away -- at a 1-day horizon the size of a move is
             most of the signal, so this matters)
  fwd_ret    next-day close-to-close return (what the portfolio actually earns)
  fwd_ret_K  cumulative K-day forward return (training targets at longer,
             more persistent horizons)
  y_binary   repo's target: beat the date's cross-sectional median
  y_rank_K   fwd_ret_K rank-normalised within the date to [-1, 1]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.harness import (
    FEATURES, build_features, cs_rank_norm, forward_return, load_wide,
)

PANEL = Path("cache/panel.parquet")
HORIZONS = (5, 10, 21)


def cs_zscore(df: pd.DataFrame, clip: float = 5.0) -> pd.DataFrame:
    """Z-score within each date, clipped. Unlike a rank this preserves how far
    apart two names are, not just their order."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0).clip(-clip, clip)


def build_panel() -> pd.DataFrame:
    close, volume = load_wide()
    feats = build_features(close, volume)

    # Investable-universe mask: a name only counts on a date where it has a real
    # price and real volume. Ranking against peers that are not actually there
    # would be a subtle leak.
    valid = close.notna() & volume.notna() & (volume > 0)
    for name in FEATURES:
        feats[name] = feats[name].where(valid)

    cols: dict[str, pd.Series] = {}
    for name in FEATURES:
        f = feats[name]
        cols[f"raw_{name}"] = f.stack()
        cols[f"cs_{name}"] = cs_rank_norm(f).stack()
        cols[f"csz_{name}"] = cs_zscore(f).stack()

    fwd1 = forward_return(close).where(valid)
    cols["fwd_ret"] = fwd1.stack()
    med = fwd1.median(axis=1)
    cols["y_binary"] = fwd1.gt(med, axis=0).where(fwd1.notna()).stack()
    cols["y_rank"] = cs_rank_norm(fwd1).stack()

    for k in HORIZONS:
        fk = (close.shift(-k) / close - 1).where(valid)
        cols[f"fwd_ret_{k}"] = fk.stack()
        cols[f"y_rank_{k}"] = cs_rank_norm(fk).stack()

    panel = pd.DataFrame(cols)
    panel.index.names = ["date", "ticker"]

    required = (
        [f"raw_{n}" for n in FEATURES]
        + [f"cs_{n}" for n in FEATURES]
        + [f"csz_{n}" for n in FEATURES]
        + ["fwd_ret", "y_binary", "y_rank"]
    )
    panel = panel.dropna(subset=required)

    # Keep only dates with a cross-section wide enough for ranking to mean
    # anything (deciles of 12 names are noise, not deciles).
    counts = panel.groupby(level="date").size()
    panel = panel[panel.index.get_level_values("date").isin(counts[counts >= 100].index)]

    panel["y_binary"] = panel["y_binary"].astype("int8")
    for c in panel.columns:
        if c != "y_binary":
            panel[c] = panel[c].astype("float32")
    return panel.sort_index()


def load_panel(rebuild: bool = False) -> pd.DataFrame:
    if PANEL.exists() and not rebuild:
        return pd.read_parquet(PANEL)
    p = build_panel()
    PANEL.parent.mkdir(parents=True, exist_ok=True)
    p.to_parquet(PANEL)
    return p


if __name__ == "__main__":
    p = load_panel(rebuild=True)
    dates = p.index.get_level_values("date")
    print(f"rows      : {len(p):,}")
    print(f"tickers   : {p.index.get_level_values('ticker').nunique()}")
    print(f"dates     : {dates.nunique():,}  ({dates.min().date()} -> {dates.max().date()})")
    print(f"names/date: median {int(p.groupby(level='date').size().median())}")
    print(f"columns   : {len(p.columns)}")
