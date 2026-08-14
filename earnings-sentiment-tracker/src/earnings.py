"""Fetch historical + upcoming earnings dates for a ticker via yfinance."""
from __future__ import annotations

import pandas as pd
import yfinance as yf


def get_earnings_dates(ticker: str, limit: int = 12) -> pd.DataFrame:
    """Return a DataFrame of earnings dates with columns: date, eps_estimate, eps_actual, surprise_pct.

    yfinance's `get_earnings_dates` returns both past and upcoming reports;
    rows with no actual EPS yet are still-upcoming events.
    """
    raw = yf.Ticker(ticker).get_earnings_dates(limit=limit)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "eps_estimate", "eps_actual", "surprise_pct"])

    df = raw.reset_index().rename(
        columns={
            "Earnings Date": "date",
            "EPS Estimate": "eps_estimate",
            "Reported EPS": "eps_actual",
            "Surprise(%)": "surprise_pct",
        }
    )
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df[["date", "eps_estimate", "eps_actual", "surprise_pct"]].sort_values("date").reset_index(drop=True)
