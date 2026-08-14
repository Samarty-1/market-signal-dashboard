"""Earnings Sentiment Tracker — Streamlit dashboard.

Fetches recent news headlines for a ticker, scores them with VADER, and
checks whether average pre-earnings sentiment lines up with the price move
that follows each earnings report.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis import build_sentiment_earnings_table, sentiment_price_correlation
from src.news_fetch import fetch_news
from src.sentiment import score_headlines

st.set_page_config(page_title="Earnings Sentiment Tracker", page_icon="📰", layout="wide")

st.title("📰 Earnings Sentiment Tracker")
st.caption(
    "News-headline sentiment vs. the price move around each earnings report. "
    "Built on yfinance's free news + earnings-date feeds — no API key required."
)

ticker = st.sidebar.text_input("Ticker", value="AAPL").strip().upper()
sentiment_window = st.sidebar.slider("Pre-earnings sentiment window (days)", 1, 14, 5)
price_horizon = st.sidebar.slider("Price-move horizon after earnings (trading days)", 1, 5, 1)

if not ticker:
    st.stop()

st.header(f"Recent headline sentiment — {ticker}")
with st.spinner("Fetching news..."):
    news = fetch_news(ticker)

if news.empty:
    st.warning("No recent headlines found for this ticker via yfinance's news feed.")
else:
    news = score_headlines(news)
    fig = px.scatter(
        news,
        x="published_at",
        y="sentiment",
        hover_data=["title", "publisher"],
        color="sentiment",
        color_continuous_scale="RdYlGn",
        range_color=[-1, 1],
        title=f"Headline sentiment over time ({len(news)} headlines)",
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Show headlines"):
        st.dataframe(
            news[["published_at", "sentiment", "title", "publisher"]].sort_values(
                "published_at", ascending=False
            ),
            use_container_width=True,
        )

st.header("Sentiment vs. post-earnings price move")
with st.spinner("Building earnings/sentiment table..."):
    table = build_sentiment_earnings_table(ticker, sentiment_window, price_horizon)

if table.empty:
    st.warning("No reported earnings events found for this ticker.")
else:
    display = table.copy()
    display["date"] = display["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(display, use_container_width=True)

    valid = table.dropna(subset=["avg_sentiment", "price_move_pct"])
    if len(valid) >= 2:
        fig2 = px.scatter(
            valid,
            x="avg_sentiment",
            y="price_move_pct",
            trendline="ols" if len(valid) >= 3 else None,
            labels={
                "avg_sentiment": f"Avg. sentiment ({sentiment_window}d before earnings)",
                "price_move_pct": f"Price move (%, {price_horizon}d after earnings)",
            },
            title="Sentiment vs. price move",
        )
        st.plotly_chart(fig2, use_container_width=True)

    corr = sentiment_price_correlation(table)
    if corr is None:
        st.info(
            "Not enough overlapping headline/earnings data to compute a correlation "
            "(yfinance's news feed only covers the last few weeks, so older earnings "
            "events typically have zero matching headlines)."
        )
    else:
        st.metric("Pearson correlation (sentiment vs. price move)", f"{corr:.2f}")

st.divider()
st.caption(
    "**Read this before trusting the correlation number.** yfinance exposes only a shallow, "
    "rolling window of recent news — most tickers will have headline coverage for at most the "
    "last 1-2 earnings events, so the correlation above is usually computed on a handful of "
    "points and is not statistically meaningful on its own. This tool is a starting point for "
    "exploring the relationship, not a validated signal — treat any correlation here as "
    "illustrative, not predictive, until it's been checked against a much larger, properly "
    "backtested sample."
)
