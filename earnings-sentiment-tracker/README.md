# 📰 Earnings Sentiment Tracker

Fetches recent news headlines for a stock ticker, scores them with rule-based
sentiment analysis (VADER), and checks whether the average sentiment in the
days before an earnings report lines up with the price move that follows it.

## What this does

1. **News ingestion** (`src/news_fetch.py`) — pulls recent headlines for a
   ticker from yfinance's free news feed. No API key required.
2. **Sentiment scoring** (`src/sentiment.py`) — scores each headline with
   [VADER](https://github.com/cjhutto/vaderSentiment), a lexicon-based
   sentiment analyzer tuned for short, informal text like headlines. Fast,
   deterministic, no model download.
3. **Earnings calendar** (`src/earnings.py`) — pulls historical and upcoming
   earnings dates + reported EPS via yfinance.
4. **Price-move calculation** (`src/price_moves.py`) — computes the
   close-to-close % move starting from each earnings date.
5. **Join + correlate** (`src/analysis.py`) — averages headline sentiment in
   an N-day window before each earnings date and correlates it against the
   subsequent price move.
6. **Dashboard** (`app.py`, Streamlit) — visualizes headline sentiment over
   time and the sentiment-vs-price-move scatter plot for any ticker.

## Honest limitation (read before trusting the numbers)

yfinance's news endpoint only exposes a **shallow, rolling window** of recent
headlines (typically the last few weeks) — it is not a historical news
archive. That means:

- Most tickers will only have headline coverage overlapping their **most
  recent** earnings event or two, not a deep history.
- The sentiment/price-move correlation is usually computed on a handful of
  points, which is not statistically meaningful on its own.
- This project demonstrates the *pipeline* (ingest → score → align with
  events → correlate → visualize), not a validated trading signal. Treat any
  correlation shown as illustrative, not predictive.

A more rigorous version would swap in a paid news archive (e.g. a financial
news API with full history) to get enough headline density per earnings
event for the correlation to mean something statistically.

## Running it

```bash
cd earnings-sentiment-tracker
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
pytest
```

Tests cover sentiment scoring and the sentiment/price-move join logic with
synthetic data — no network calls required.
