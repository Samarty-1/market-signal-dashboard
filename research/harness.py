"""Research harness for diagnosing the cross-sectional signal.

Everything is computed in WIDE format (date x ticker) because the entire
question here is cross-sectional: "how does this ticker rank against its peers
today". Wide format makes the per-date operations (rank, demean, median) one
vectorized call instead of a groupby over 3,000 dates, and it makes the
leakage-safety of each transform visually obvious -- every cross-sectional op
is `axis=1` (within one date, no time travel), every time-series op is a
shift/rolling down a column (backward-looking only).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CACHE = Path("cache/prices_sp500_12y.parquet")

# Feature families the ML asset-pricing literature finds dominant (Gu/Kelly/Xiu):
# momentum, liquidity, volatility -- plus the short-horizon reversal terms that
# matter specifically at a 1-day prediction horizon.
FEATURES = [
    "return_1d", "return_5d",
    "sma_10_ratio", "sma_20_ratio", "sma_50_ratio",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "volatility_10d", "volatility_20d", "volatility_60d",
    "volume_change", "log_dollar_volume", "amihud_illiquidity",
    "mom_21d", "mom_63d", "mom_126d", "mom_12m_skip1m",
]


# --------------------------------------------------------------------------
# wide panel construction
# --------------------------------------------------------------------------
def load_wide() -> tuple[pd.DataFrame, pd.DataFrame]:
    px = pd.read_parquet(CACHE)
    close = px.pivot(index="date", columns="ticker", values="close").sort_index()
    volume = px.pivot(index="date", columns="ticker", values="volume").sort_index()
    return close, volume


def _ewm_wide(df: pd.DataFrame, span=None, alpha=None, min_periods=0) -> pd.DataFrame:
    return df.ewm(span=span, alpha=alpha, min_periods=min_periods, adjust=False).mean()


def build_features(close: pd.DataFrame, volume: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Each returned frame is date x ticker. Every op is backward-looking in time."""
    f: dict[str, pd.DataFrame] = {}
    ret1 = close.pct_change(1)

    f["return_1d"] = ret1
    f["return_5d"] = close.pct_change(5)

    for w in (10, 20, 50):
        f[f"sma_{w}_ratio"] = close / close.rolling(w).mean() - 1

    # RSI(14) via Wilder smoothing
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = _ewm_wide(gain, alpha=1 / 14, min_periods=14)
    avg_loss = _ewm_wide(loss, alpha=1 / 14, min_periods=14)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    f["rsi_14"] = (100 - 100 / (1 + rs)).fillna(50)

    ema_f, ema_s = _ewm_wide(close, span=12), _ewm_wide(close, span=26)
    macd = ema_f - ema_s
    signal = _ewm_wide(macd, span=9)
    # MACD is a raw price difference: a $5 MACD on a $900 stock and a $5 MACD on
    # a $30 stock are wildly different signals. Scale by price so it is
    # comparable across the cross-section at all.
    f["macd"] = macd / close
    f["macd_signal"] = signal / close
    f["macd_hist"] = (macd - signal) / close

    for w in (10, 20, 60):
        f[f"volatility_{w}d"] = ret1.rolling(w).std()

    f["volume_change"] = volume.pct_change(1)
    dollar_vol = (close * volume).replace(0, np.nan)
    f["log_dollar_volume"] = np.log(dollar_vol)
    f["amihud_illiquidity"] = (ret1.abs() / dollar_vol).rolling(21).mean()

    for w in (21, 63, 126):
        f[f"mom_{w}d"] = close.pct_change(w)
    f["mom_12m_skip1m"] = close.shift(21) / close.shift(252) - 1

    # A single +/-inf (volume_change on a day whose prior volume was 0) makes
    # that date's cross-sectional mean and std NaN, which silently deletes the
    # entire date from the panel. Rank-based transforms tolerate it and z-score
    # ones do not, so sanitise here rather than in one transform.
    for k in f:
        f[k] = f[k].replace([np.inf, -np.inf], np.nan)

    return f


def forward_return(close: pd.DataFrame) -> pd.DataFrame:
    """Return earned holding from today's close to tomorrow's close."""
    return close.shift(-1) / close - 1


# --------------------------------------------------------------------------
# cross-sectional transforms  (THE core fix -- all axis=1, within one date)
# --------------------------------------------------------------------------
def cs_rank_norm(df: pd.DataFrame) -> pd.DataFrame:
    """Rank each ticker against its same-day peers, scaled to [-1, 1].

    This is the Gu/Kelly/Xiu standard preprocessing. It matters because the
    TARGET is relative ("beat the day's median") while a raw feature is
    absolute. On a day the whole market rips +3%, every ticker's raw momentum
    shifts together but the relative target does not -- so the mapping from
    raw feature to relative outcome is different every single day, and a model
    fed raw levels has to waste capacity re-learning it per regime. A rank is
    stationary by construction: 0.9 means "top decile today" on every date.
    Also kills outliers and any need for a fitted scaler (nothing to leak).
    """
    return df.rank(axis=1, pct=True) * 2 - 1


def cs_demean(df: pd.DataFrame) -> pd.DataFrame:
    return df.sub(df.mean(axis=1), axis=0)


def to_long(frames: dict[str, pd.DataFrame], names: list[str]) -> pd.DataFrame:
    out = pd.DataFrame({n: frames[n].stack() for n in names})
    out.index.names = ["date", "ticker"]
    return out


# --------------------------------------------------------------------------
# metrics -- the professional ones
# --------------------------------------------------------------------------
def daily_rank_ic(pred: pd.Series, fwd: pd.Series) -> pd.Series:
    """Spearman rank correlation between prediction and forward return,
    computed WITHIN each date then returned as a series over dates.

    This is the metric a cross-sectional signal actually lives or dies by.
    Pooled ROC AUC (what the repo reports) mixes observations across dates, so
    a model that merely knows "high-vol days have more winners" scores well on
    it while carrying zero ability to rank names against each other on any
    given day -- which is the only thing a market-neutral book can monetize.
    """
    df = pd.DataFrame({"p": pred, "f": fwd}).dropna()
    def _ic(g):
        if len(g) < 20 or g["p"].nunique() < 5:
            return np.nan
        return stats.spearmanr(g["p"], g["f"]).statistic
    return df.groupby(level="date").apply(_ic).dropna()


def ic_summary(ic: pd.Series) -> dict:
    if len(ic) == 0:
        return {"n_days": 0, "mean_ic": np.nan, "icir_ann": np.nan, "t_stat": np.nan, "hit_rate": np.nan}
    mean, sd = float(ic.mean()), float(ic.std())
    return {
        "n_days": int(len(ic)),
        "mean_ic": round(mean, 5),
        # ICIR annualized: mean/std of the daily IC series, scaled by sqrt(252).
        # This is the signal's own information ratio and upper-bounds the
        # Sharpe any portfolio built on it can achieve.
        "icir_ann": round(mean / sd * np.sqrt(252), 3) if sd > 0 else np.nan,
        "t_stat": round(mean / sd * np.sqrt(len(ic)), 2) if sd > 0 else np.nan,
        "hit_rate": round(float((ic > 0).mean()), 4),
    }


def quantile_spread(pred: pd.Series, fwd: pd.Series, n_q: int = 5) -> pd.Series:
    """Daily return of a dollar-neutral top-quantile minus bottom-quantile book."""
    df = pd.DataFrame({"p": pred, "f": fwd}).dropna()

    def _spread(g):
        if len(g) < n_q * 5:
            return np.nan
        q = pd.qcut(g["p"].rank(method="first"), n_q, labels=False)
        return g.loc[q == n_q - 1, "f"].mean() - g.loc[q == 0, "f"].mean()

    return df.groupby(level="date").apply(_spread).dropna()


def perf(daily: pd.Series) -> dict:
    if len(daily) == 0:
        return {"n_days": 0, "ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    ann_r = float(daily.mean() * 252)
    ann_v = float(daily.std() * np.sqrt(252))
    curve = (1 + daily).cumprod()
    dd = float((curve / curve.cummax() - 1).min())
    return {
        "n_days": int(len(daily)),
        "ann_return": round(ann_r, 4),
        "ann_vol": round(ann_v, 4),
        "sharpe": round(ann_r / ann_v, 3) if ann_v > 0 else np.nan,
        "max_dd": round(dd, 4),
    }
