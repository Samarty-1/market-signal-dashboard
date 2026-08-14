"""Fetch recent news headlines for a ticker via yfinance."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


def fetch_news(ticker: str) -> pd.DataFrame:
    """Return a DataFrame of recent headlines for `ticker`.

    Columns: published_at (UTC datetime), title, publisher, link.
    yfinance only exposes a rolling window of recent news (no deep history),
    so this is best-effort and typically covers the last few weeks.
    """
    raw = yf.Ticker(ticker).news or []
    rows = []
    for item in raw:
        content = item.get("content", item)
        title = content.get("title") or item.get("title")
        if not title:
            continue
        pub_date = content.get("pubDate") or item.get("providerPublishTime")
        published_at = _parse_published(pub_date)
        publisher = (content.get("provider") or {}).get("displayName") or item.get("publisher", "")
        link = (content.get("canonicalUrl") or {}).get("url") or item.get("link", "")
        rows.append(
            {
                "published_at": published_at,
                "title": title,
                "publisher": publisher,
                "link": link,
            }
        )
    df = pd.DataFrame(rows, columns=["published_at", "title", "publisher", "link"])
    if not df.empty:
        df = df.dropna(subset=["published_at"]).sort_values("published_at").reset_index(drop=True)
    return df


def _parse_published(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    try:
        return pd.to_datetime(value, utc=True).to_pydatetime()
    except (ValueError, TypeError):
        return None
