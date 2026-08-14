import pandas as pd
import pytest

from src.analysis import avg_sentiment_before, sentiment_price_correlation


def test_avg_sentiment_before_window():
    event_date = pd.Timestamp("2026-01-10", tz="UTC")
    news = pd.DataFrame(
        {
            "published_at": [
                pd.Timestamp("2026-01-06", tz="UTC"),  # inside 5-day window
                pd.Timestamp("2026-01-09", tz="UTC"),  # inside window
                pd.Timestamp("2026-01-01", tz="UTC"),  # outside window
                pd.Timestamp("2026-01-11", tz="UTC"),  # after event, excluded
            ],
            "sentiment": [0.5, -0.3, 0.9, -0.9],
        }
    )
    result = avg_sentiment_before(news, event_date, window_days=5)
    assert result == pytest.approx(0.1)


def test_avg_sentiment_before_no_headlines_returns_none():
    event_date = pd.Timestamp("2026-01-10", tz="UTC")
    news = pd.DataFrame({"published_at": [], "sentiment": []})
    assert avg_sentiment_before(news, event_date) is None


def test_correlation_requires_min_points():
    table = pd.DataFrame(
        {
            "avg_sentiment": [0.1, 0.2, None],
            "price_move_pct": [1.0, -1.0, 2.0],
        }
    )
    assert sentiment_price_correlation(table) is None


def test_correlation_computed_with_enough_points():
    table = pd.DataFrame(
        {
            "avg_sentiment": [0.1, 0.2, 0.3, -0.1],
            "price_move_pct": [1.0, 2.0, 3.0, -1.0],
        }
    )
    corr = sentiment_price_correlation(table)
    assert corr is not None
    assert corr > 0.9
