"""Fetch the S&P 500 universe once and cache to parquet so every experiment
below reads the exact same price panel (no re-download, no drift between runs).

    python fetch_cache.py                    # today's members (survivorship-biased)
    python fetch_cache.py --universe pit     # every name that was a member in the window

The point-in-time mode also writes the membership history the research panel
masks with (MSD_UNIVERSE=pit), drops recycled ticker symbols, and records how
much of the true universe Yahoo could not supply -- the residual bias.
"""
import argparse
import io
import json
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["current", "pit"], default="current")
    args = ap.parse_args()
    out = OUT if args.universe == "current" else Path("cache/prices_sp500pit_12y.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)

    history = None
    if args.universe == "pit":
        from src import universe as universe_mod
        start = pd.Timestamp.today().normalize() - pd.DateOffset(years=12)
        history = universe_mod.membership_history(start)
        tickers = universe_mod.ever_members(history)
        history.to_parquet("cache/membership_pit.parquet", index=False)
        print(f"{len(tickers)} names were members at some point since {start.date()}", flush=True)
    else:
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
    if history is not None:
        df, recycled = universe_mod.drop_recycled_tickers(df, history)
        cov = universe_mod.coverage_report(history, df)
        cov["recycled_dropped"] = recycled
        Path("cache/pit_coverage.json").write_text(json.dumps(cov, indent=2, default=str))
        print(f"coverage {cov['n_with_price_data']}/{cov['n_ever_members']} names "
              f"({cov['pct_missing']}% missing); {len(recycled)} recycled symbols dropped", flush=True)
    df.to_parquet(out, index=False)
    print(f"saved {out}: {len(df)} rows, {df['ticker'].nunique()} tickers, "
          f"{df['date'].min().date()} -> {df['date'].max().date()}", flush=True)


if __name__ == "__main__":
    main()
