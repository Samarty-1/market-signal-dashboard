"""Pulls daily OHLCV price data from Yahoo Finance for a set of tickers."""

from __future__ import annotations

import argparse
import io
import urllib.request
from pathlib import Path

import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    # Semiconductors
    "AMD", "AVGO",
    # Financials
    "JPM", "BAC", "GS",
    # Healthcare
    "JNJ", "UNH", "PFE",
    # Consumer
    "WMT", "PG", "KO", "MCD",
    # Energy
    "XOM", "CVX",
    # Industrials
    "CAT", "BA",
    # Communications
    "NFLX", "DIS",
    # Broad ETFs
    "SPY", "QQQ", "DIA", "IWM",
]
DEFAULT_PERIOD = "2y"
DEFAULT_INTERVAL = "1d"
VIX_TICKER = "^VIX"


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


def fetch_sp500_tickers() -> list[str]:
    """S&P 500 constituent list from Wikipedia (needs a browser User-Agent,
    Wikipedia 403s the default urllib one). Cross-sectional ML asset-pricing
    edges come from ranking hundreds of stocks against each other (see
    README) -- this is what widens the universe from DEFAULT_TICKERS'
    ~30 names to that scale, still using only free data.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=30).read()
    table = pd.read_html(io.BytesIO(html))[0]
    # yfinance uses '-' where Wikipedia uses '.' for share classes (e.g. BRK.B -> BRK-B)
    return table["Symbol"].str.replace(".", "-", regex=False).tolist()


def fetch_universe_prices(tickers: list[str], period: str = "3y", interval: str = "1d") -> pd.DataFrame:
    """Batched equivalent of fetch_prices() for large ticker lists (hundreds+).

    fetch_prices() fetches one ticker at a time (yf.Ticker(t).history()),
    which is fine for ~30 tickers but doesn't scale -- yf.download() with a
    ticker list batches the requests (500 S&P tickers x 3y in ~10s locally,
    vs. hundreds of sequential HTTP round-trips one-ticker-at-a-time).
    Returns the same long-format columns as fetch_prices: date, ticker,
    open, high, low, close, volume.
    """
    raw = yf.download(tickers, period=period, interval=interval, group_by="ticker", threads=True, progress=False, auto_adjust=True)

    frames = []
    for ticker in tickers:
        if ticker not in raw.columns.get_level_values(0):
            continue
        history = raw[ticker].dropna(how="all").reset_index()
        if history.empty:
            continue
        history["ticker"] = ticker
        history = history.rename(
            columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
        )
        frames.append(history[["date", "ticker", "open", "high", "low", "close", "volume"]])

    if not frames:
        raise RuntimeError(f"No data returned for any of: {tickers}")

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.tz_localize(None)
    return combined.sort_values(["ticker", "date"]).reset_index(drop=True)


def fetch_vix(period: str = DEFAULT_PERIOD) -> pd.DataFrame:
    """Fetch daily VIX closes — a macro regime signal, not a tradable ticker in the model.

    Returns columns: date, vix_close.
    """
    vix = fetch_prices([VIX_TICKER], period=period)
    return vix[["date", "close"]].rename(columns={"close": "vix_close"})


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
