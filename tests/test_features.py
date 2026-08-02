import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, _rsi, add_features_for_ticker, build_feature_dataset


def _make_prices(n: int = 80, ticker: str = "TEST", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    close = np.maximum(close, 1.0)  # keep prices positive
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n),
        }
    )


def test_rsi_bounded_between_0_and_100():
    prices = _make_prices()
    rsi = _rsi(prices["close"])
    assert rsi.min() >= 0
    assert rsi.max() <= 100


def test_add_features_for_ticker_has_all_columns():
    prices = _make_prices()
    enriched = add_features_for_ticker(prices)
    for col in FEATURE_COLUMNS:
        assert col in enriched.columns
    assert "label_next_day_up" in enriched.columns
    assert "next_day_return" in enriched.columns


def test_label_matches_actual_next_day_direction():
    prices = _make_prices()
    enriched = add_features_for_ticker(prices)
    non_null = enriched.dropna(subset=["label_next_day_up"])
    for _, row in non_null.iterrows():
        idx = enriched.index.get_loc(row.name)
        if idx + 1 < len(enriched):
            expected_up = enriched["close"].iloc[idx + 1] > enriched["close"].iloc[idx]
            assert bool(row["label_next_day_up"]) == expected_up


def test_last_row_has_no_label_or_next_day_return():
    prices = _make_prices()
    enriched = add_features_for_ticker(prices)
    last = enriched.iloc[-1]
    assert pd.isna(last["label_next_day_up"])
    assert pd.isna(last["next_day_return"])


def test_build_feature_dataset_drops_warmup_and_last_row():
    prices = pd.concat([_make_prices(ticker="AAA", seed=1), _make_prices(ticker="BBB", seed=2)], ignore_index=True)
    feats = build_feature_dataset(prices)

    # No NaNs remain in any required column.
    required = FEATURE_COLUMNS + ["label_next_day_up"]
    assert not feats[required].isna().any().any()

    # Warm-up rows (needed for the 50-day SMA) and the final row (no label yet) are gone.
    assert len(feats[feats["ticker"] == "AAA"]) < len(prices[prices["ticker"] == "AAA"])

    # Both tickers survive.
    assert set(feats["ticker"].unique()) == {"AAA", "BBB"}


def test_build_feature_dataset_raises_helpfully_on_too_little_history():
    tiny = _make_prices(n=5)
    feats = build_feature_dataset(tiny)
    # Not enough history for a 50-day SMA -> everything gets dropped, not an error.
    assert feats.empty
