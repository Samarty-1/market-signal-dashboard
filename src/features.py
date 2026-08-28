"""Technical-indicator feature engineering and next-day direction labeling."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "sma_10_ratio",
    "sma_20_ratio",
    "sma_50_ratio",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "volatility_10d",
    "volatility_20d",
    "volume_change",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def add_features_for_ticker(group: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators for a single ticker's price history (sorted by date)."""
    group = group.sort_values("date").copy()
    close = group["close"]

    group["return_1d"] = close.pct_change(1)
    group["return_5d"] = close.pct_change(5)

    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    group["sma_10_ratio"] = close / sma_10 - 1
    group["sma_20_ratio"] = close / sma_20 - 1
    group["sma_50_ratio"] = close / sma_50 - 1

    group["rsi_14"] = _rsi(close, 14)
    macd_line, signal_line, hist = _macd(close)
    group["macd"] = macd_line
    group["macd_signal"] = signal_line
    group["macd_hist"] = hist

    group["volatility_10d"] = group["return_1d"].rolling(10).std()
    group["volatility_20d"] = group["return_1d"].rolling(20).std()
    group["volume_change"] = group["volume"].pct_change(1)

    # Label: did next day's close finish higher than today's close? The last row has
    # no "next day" yet, so it must stay NA rather than silently becoming 0 — a plain
    # `(close.shift(-1) > close)` comparison would turn that NaN into False.
    next_close = close.shift(-1)
    label = pd.Series(pd.NA, index=group.index, dtype="Int64")
    known = next_close.notna()
    label[known] = (next_close[known] > close[known]).astype(int)
    group["label_next_day_up"] = label
    # The actual return an investor earns by holding from today's close to tomorrow's
    # close — used by the backtest to score a signal made "as of" today's data.
    group["next_day_return"] = next_close / close - 1

    return group


def add_cross_sectional_label(df: pd.DataFrame) -> pd.DataFrame:
    """Label: did this ticker's next-day return beat the day's cross-sectional
    median return (across the whole ticker universe), rather than just "up or down"?

    This needs the full multi-ticker frame (not a single ticker's group like
    add_features_for_ticker), since the median is computed across tickers on
    each date. Motivation: next-day direction alone is dominated by market-wide
    moves that hit every ticker together (~coin flip, see model.py docstring);
    asking "did this one outperform its peers today" nets out that common
    market move and targets the smaller, better-documented cross-sectional
    relative-strength effect instead.
    """
    df = df.copy()
    cs_median = df.groupby("date")["next_day_return"].transform("median")
    label = pd.Series(pd.NA, index=df.index, dtype="Int64")
    known = df["next_day_return"].notna() & cs_median.notna()
    label[known] = (df.loc[known, "next_day_return"] > cs_median[known]).astype(int)
    df["label_beat_median_next_day"] = label
    return df


def build_feature_dataset(prices: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering per ticker and drop indicator warm-up / label NaN rows."""
    enriched = pd.concat(
        [add_features_for_ticker(group) for _, group in prices.groupby("ticker", sort=False)],
        ignore_index=True,
    )
    enriched = add_cross_sectional_label(enriched)
    required = FEATURE_COLUMNS + ["label_next_day_up", "label_beat_median_next_day"]
    return enriched.dropna(subset=required).reset_index(drop=True)
