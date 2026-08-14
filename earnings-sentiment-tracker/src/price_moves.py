"""Compute the price move around each earnings date."""
from __future__ import annotations

import pandas as pd
import yfinance as yf


def get_price_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Return daily OHLCV history with a tz-aware UTC datetime index."""
    hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    hist.index = pd.to_datetime(hist.index, utc=True)
    return hist


def price_move_after(price_history: pd.DataFrame, event_date: pd.Timestamp, horizon_days: int = 1) -> float | None:
    """% close-to-close move from the first trading day on/after `event_date`
    to `horizon_days` trading sessions later. Returns None if out of range.
    """
    if price_history.empty:
        return None
    idx = price_history.index
    on_or_after = idx[idx >= event_date]
    if on_or_after.empty:
        return None
    start_pos = idx.get_loc(on_or_after[0])
    end_pos = start_pos + horizon_days
    if end_pos >= len(idx):
        return None
    start_close = price_history["Close"].iloc[start_pos]
    end_close = price_history["Close"].iloc[end_pos]
    if start_close == 0:
        return None
    return (end_close - start_close) / start_close * 100
