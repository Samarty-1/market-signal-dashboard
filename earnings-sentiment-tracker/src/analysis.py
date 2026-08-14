"""Join pre-earnings news sentiment with the subsequent price move."""
from __future__ import annotations

import pandas as pd

from .earnings import get_earnings_dates
from .news_fetch import fetch_news
from .price_moves import get_price_history, price_move_after
from .sentiment import score_headlines


def avg_sentiment_before(news_df: pd.DataFrame, event_date: pd.Timestamp, window_days: int = 5) -> float | None:
    """Mean VADER compound score for headlines published in the `window_days`
    before `event_date`. Returns None if no headlines fall in that window.
    """
    if news_df.empty:
        return None
    window_start = event_date - pd.Timedelta(days=window_days)
    mask = (news_df["published_at"] >= window_start) & (news_df["published_at"] < event_date)
    windowed = news_df.loc[mask]
    if windowed.empty:
        return None
    return windowed["sentiment"].mean()


def build_sentiment_earnings_table(
    ticker: str, sentiment_window_days: int = 5, price_horizon_days: int = 1
) -> pd.DataFrame:
    """For each earnings date with a reported (not upcoming) result, compute
    the average pre-earnings headline sentiment and the price move following
    the report. One row per earnings event; NaN where data is unavailable.
    """
    earnings = get_earnings_dates(ticker)
    reported = earnings.dropna(subset=["eps_actual"])
    if reported.empty:
        return pd.DataFrame(
            columns=["date", "eps_actual", "surprise_pct", "avg_sentiment", "price_move_pct", "n_headlines"]
        )

    news = score_headlines(fetch_news(ticker))
    price_history = get_price_history(ticker)

    rows = []
    for _, event in reported.iterrows():
        event_date = event["date"]
        sentiment = avg_sentiment_before(news, event_date, sentiment_window_days)
        n_headlines = 0
        if not news.empty:
            window_start = event_date - pd.Timedelta(days=sentiment_window_days)
            n_headlines = int(
                ((news["published_at"] >= window_start) & (news["published_at"] < event_date)).sum()
            )
        move = price_move_after(price_history, event_date, price_horizon_days)
        rows.append(
            {
                "date": event_date,
                "eps_actual": event["eps_actual"],
                "surprise_pct": event["surprise_pct"],
                "avg_sentiment": sentiment,
                "price_move_pct": move,
                "n_headlines": n_headlines,
            }
        )
    return pd.DataFrame(rows)


def sentiment_price_correlation(table: pd.DataFrame) -> float | None:
    """Pearson correlation between avg_sentiment and price_move_pct.
    Returns None if fewer than 3 complete pairs are available (yfinance's
    news feed is shallow, so most tickers won't have enough overlap).
    """
    valid = table.dropna(subset=["avg_sentiment", "price_move_pct"])
    if len(valid) < 3:
        return None
    return valid["avg_sentiment"].corr(valid["price_move_pct"])
