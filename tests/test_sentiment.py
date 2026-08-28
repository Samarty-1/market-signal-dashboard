import numpy as np
import pytest

from src import sentiment


class _FakeModel:
    """Deterministic stand-in for the trained pipeline: maps headline text
    to fixed probabilities so the aggregation/labeling logic in
    score_ticker_sentiment can be tested without downloading the real
    Hugging Face dataset or hitting the network on every test run."""

    def __init__(self, probs_by_text):
        self._probs_by_text = probs_by_text

    def predict_proba(self, texts):
        return np.array([self._probs_by_text[t] for t in texts])

    def predict(self, texts):
        return np.array([int(np.argmax(self._probs_by_text[t])) for t in texts])


def test_score_ticker_sentiment_aggregates_bullish(monkeypatch):
    headlines = ["Great quarter, stock soars", "Analysts raise price target"]
    probs = {
        headlines[0]: [0.05, 0.90, 0.05],  # bearish, bullish, neutral
        headlines[1]: [0.10, 0.80, 0.10],
    }
    monkeypatch.setattr(sentiment, "fetch_live_headlines", lambda ticker, limit=10: headlines)

    result = sentiment.score_ticker_sentiment("AAPL", model=_FakeModel(probs))

    assert result["label"] == "bullish"
    assert result["n_headlines"] == 2
    assert result["sentiment_score"] > 0.15


def test_score_ticker_sentiment_aggregates_bearish(monkeypatch):
    headlines = ["Shares tumble on guidance cut"]
    probs = {headlines[0]: [0.85, 0.05, 0.10]}
    monkeypatch.setattr(sentiment, "fetch_live_headlines", lambda ticker, limit=10: headlines)

    result = sentiment.score_ticker_sentiment("XYZ", model=_FakeModel(probs))

    assert result["label"] == "bearish"
    assert result["sentiment_score"] < -0.15


def test_score_ticker_sentiment_handles_no_headlines(monkeypatch):
    monkeypatch.setattr(sentiment, "fetch_live_headlines", lambda ticker, limit=10: [])

    result = sentiment.score_ticker_sentiment("QQQ", model=_FakeModel({}))

    assert result["label"] == "no_data"
    assert result["sentiment_score"] is None
    assert result["headlines"] == []


def test_fetch_live_headlines_parses_content_title(monkeypatch):
    class _FakeTicker:
        news = [
            {"content": {"title": "Headline one"}},
            {"content": {"title": "Headline two"}},
            {"content": {}},  # missing title should be skipped, not raise
        ]

    monkeypatch.setattr(sentiment.yf, "Ticker", lambda ticker: _FakeTicker())

    headlines = sentiment.fetch_live_headlines("AAPL", limit=10)

    assert headlines == ["Headline one", "Headline two"]


def test_fetch_live_headlines_returns_empty_on_error(monkeypatch):
    class _BrokenTicker:
        @property
        def news(self):
            raise RuntimeError("network error")

    monkeypatch.setattr(sentiment.yf, "Ticker", lambda ticker: _BrokenTicker())

    assert sentiment.fetch_live_headlines("AAPL") == []
