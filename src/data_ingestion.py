"""Pulls daily OHLCV price data from Yahoo Finance for a set of tickers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"]
DEFAULT_PERIOD = "2y"
DEFAULT_INTERVAL = "1d"


def fetch_prices(
    tickers: list[str] = DEFAULT_TICKERS,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """Fetch OHLCV history for each ticker and return one long-format DataFrame.

    Columns: date, ticker, open, high, low, close, volume
    """
    frames = []
    for ticker in tickers:
        history = yf.Ticker(ticker).history(period=period, interval=interval)
        if history.empty:
            continue
        history = history.reset_index()
        history["ticker"] = ticker
        history = history.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        frames.append(history[["date", "ticker", "open", "high", "low", "close", "volume"]])

    if not frames:
        raise RuntimeError(f"No data returned for any of: {tickers}")

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.tz_localize(None)
    return combined.sort_values(["ticker", "date"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS), help="Comma-separated tickers")
    parser.add_argument("--period", default=DEFAULT_PERIOD, help="yfinance period, e.g. 2y, 5y, max")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, help="yfinance interval, e.g. 1d, 1h")
    parser.add_argument("--out", default="data/prices.csv", help="Output CSV path")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    df = fetch_prices(tickers, period=args.period, interval=args.interval)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows for {len(tickers)} tickers to {out_path}")


if __name__ == "__main__":
    main()
