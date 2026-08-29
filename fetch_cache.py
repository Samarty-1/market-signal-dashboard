"""Fetch the S&P 500 universe once and cache to parquet so every experiment
below reads the exact same price panel (no re-download, no drift between runs)."""
import io
import time
import urllib.request
from pathlib import Path

import pandas as pd
import yfinance as yf

OUT = Path("cache/prices_sp500_12y.parquet")


def sp500_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=60).read()
    table = pd.read_html(io.BytesIO(html))[0]
    return table["Symbol"].str.replace(".", "-", regex=False).tolist()


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tickers = sp500_tickers()
    print(f"{len(tickers)} tickers", flush=True)

    t0 = time.time()
    raw = yf.download(
        tickers, period="12y", interval="1d", group_by="ticker",
        threads=True, progress=False, auto_adjust=True,
    )
    print(f"downloaded in {time.time()-t0:.1f}s", flush=True)

    frames = []
    lvl0 = set(raw.columns.get_level_values(0))
    for t in tickers:
        if t not in lvl0:
            continue
        h = raw[t].dropna(how="all").reset_index()
        if h.empty:
            continue
        h["ticker"] = t
        h = h.rename(columns={"Date": "date", "Open": "open", "High": "high",
                              "Low": "low", "Close": "close", "Volume": "volume"})
        frames.append(h[["date", "ticker", "open", "high", "low", "close", "volume"]])

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df.to_parquet(OUT, index=False)
    print(f"saved {len(df)} rows, {df['ticker'].nunique()} tickers, "
          f"{df['date'].min().date()} -> {df['date'].max().date()}", flush=True)


if __name__ == "__main__":
    main()
