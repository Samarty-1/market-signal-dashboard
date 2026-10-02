"""Tests for walk-forward fold purging.

model.py's docstring claims "no ticker ever leaks future dates into an earlier
fold's training set". Before PURGE_DAYS existed that claim was false at the
fold boundary: the label is built from the NEXT day's close, so a training row
dated exactly at train_end had a label determined by the first day of the test
fold. These tests pin the claim down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.model import _purged_train_test


@pytest.fixture
def panel() -> pd.DataFrame:
    """Two tickers over 10 trading days."""
    dates = pd.bdate_range("2024-01-01", periods=10)
    return pd.DataFrame(
        [{"date": d, "ticker": t} for d in dates for t in ("AAA", "BBB")]
    )


def test_boundary_day_is_purged_from_training(panel):
    """The last training day must be dropped, because its label is realised on
    the first test day."""
    dates = sorted(panel["date"].unique())
    train_end, test_end = dates[5], dates[9]

    train, test = _purged_train_test(panel, train_end, test_end, purge_days=1)

    assert train["date"].max() < train_end, "row at the boundary still leaks its label into the test fold"
    assert train["date"].max() == dates[4]
    assert test["date"].min() == dates[6]


def test_purge_removes_the_whole_boundary_day_across_tickers(panel):
    """Purging is per trading day, not per row -- every ticker's boundary
    observation goes, not just the first one encountered."""
    dates = sorted(panel["date"].unique())
    train, _ = _purged_train_test(panel, dates[5], dates[9], purge_days=1)

    assert dates[5] not in set(train["date"])
    assert len(train) == 10  # 5 days x 2 tickers


def test_no_gap_between_train_and_test_without_purging(panel):
    """Documents the old behaviour so the difference is explicit: with
    purge_days=0 the training set runs right up to the boundary."""
    dates = sorted(panel["date"].unique())
    train, test = _purged_train_test(panel, dates[5], dates[9], purge_days=0)

    assert train["date"].max() == dates[5]
    assert test["date"].min() == dates[6]


def test_multi_day_purge(panel):
    dates = sorted(panel["date"].unique())
    train, _ = _purged_train_test(panel, dates[5], dates[9], purge_days=3)

    assert train["date"].max() == dates[2]


def test_purge_larger_than_training_window_yields_empty_train(panel):
    """Degenerate case must produce an empty frame the caller can skip, not a
    partially-purged one that silently still leaks."""
    dates = sorted(panel["date"].unique())
    train, _ = _purged_train_test(panel, dates[1], dates[9], purge_days=5)

    assert train.empty


def test_label_horizon_matches_purge_window():
    """The concrete leak, shown end to end: build the real label and confirm
    the boundary training row's label is determined by a test-fold price."""
    from src.features import add_features_for_ticker

    dates = pd.bdate_range("2024-01-01", periods=60)
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAA",
            "close": 100 + np.cumsum(rng.normal(0, 1, len(dates))),
            "volume": 1_000_000,
        }
    )
    enriched = add_features_for_ticker(prices)

    boundary = dates[40]
    row = enriched[enriched["date"] == boundary].iloc[0]
    next_close = enriched[enriched["date"] == dates[41]]["close"].iloc[0]

    # The boundary row's label is a fact about the NEXT day, which is in the
    # test fold -- hence the purge.
    assert row["label_next_day_up"] == int(next_close > row["close"])
