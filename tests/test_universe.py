"""Tests for point-in-time universe construction.

Each test here corresponds to a bug that was live in the cross-sectional
pipeline: survivorship bias, look-ahead index inclusion, recycled tickers, and
a benchmark scored over a different date range than the strategy.

The membership fixtures are hand-built rather than fetched so the suite stays
offline and deterministic; the network paths (revision_asof / members_asof)
are exercised by the pipeline itself, not by CI.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.universe import (
    apply_point_in_time_membership,
    coverage_report,
    drop_recycled_tickers,
    ever_members,
    membership_windows,
    sampling_interval,
)


@pytest.fixture
def history() -> pd.DataFrame:
    """Three quarterly samples. GOOD is a member throughout. LEAVER is a
    member for the first two samples then drops out (the survivorship case).
    JOINER only appears at the last sample (the look-ahead inclusion case)."""
    rows = [
        ("2020-03-31", "GOOD"), ("2020-03-31", "LEAVER"),
        ("2020-06-30", "GOOD"), ("2020-06-30", "LEAVER"),
        ("2020-09-30", "GOOD"), ("2020-09-30", "JOINER"),
    ]
    return pd.DataFrame(
        [{"sample_date": pd.Timestamp(d), "ticker": t} for d, t in rows],
        columns=["sample_date", "ticker"],
    )


def test_ever_members_includes_names_that_left_the_index(history):
    """The download list must contain LEAVER. Using only the final sample --
    what fetch_sp500_tickers() does -- silently drops every company that went
    bankrupt or was acquired during the window, which is the survivorship bug."""
    assert ever_members(history) == ["GOOD", "JOINER", "LEAVER"]


def test_membership_mask_excludes_a_stock_before_it_joined(history):
    """JOINER's pre-inclusion history must not enter the cross-section.

    Stocks are added to the S&P 500 after a strong run, so ranking a future
    member's earlier history alongside genuine members leaks the knowledge
    that it was about to be promoted.
    """
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-04-15", "2020-04-15", "2020-10-15", "2020-10-15"]),
            "ticker": ["GOOD", "JOINER", "GOOD", "JOINER"],
        }
    )
    kept = apply_point_in_time_membership(df, history)
    april = kept[kept["date"] == pd.Timestamp("2020-04-15")]["ticker"].tolist()
    october = kept[kept["date"] == pd.Timestamp("2020-10-15")]["ticker"].tolist()

    assert april == ["GOOD"], "JOINER was not an index member in April 2020"
    assert sorted(october) == ["GOOD", "JOINER"]


def test_membership_mask_excludes_a_stock_after_it_left(history):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-07-15", "2020-10-15"]),
            "ticker": ["LEAVER", "LEAVER"],
        }
    )
    kept = apply_point_in_time_membership(df, history)
    assert kept["date"].tolist() == [pd.Timestamp("2020-07-15")], (
        "LEAVER should be ranked while it was a member, and only then"
    )


def test_membership_is_held_forward_never_backward(history):
    """A date between samples inherits the LAST sample at or before it.

    Carrying membership backward from a later sample would reintroduce exactly
    the look-ahead this module removes.
    """
    df = pd.DataFrame({"date": pd.to_datetime(["2020-08-01"]), "ticker": ["JOINER"]})
    assert apply_point_in_time_membership(df, history).empty


def test_recycled_ticker_is_dropped(history):
    """The BBBY case: symbol reassigned to a different company after the
    original was delisted, so the price history doesn't overlap membership."""
    prices = pd.DataFrame(
        {
            "date": list(pd.date_range("2020-01-01", periods=200))
            + list(pd.date_range("2026-07-17", periods=30)),
            "ticker": ["GOOD"] * 200 + ["LEAVER"] * 30,
            "close": [1.0] * 230,
        }
    )
    filtered, dropped = drop_recycled_tickers(prices, history)

    assert dropped == ["LEAVER"]
    assert set(filtered["ticker"]) == {"GOOD"}


def test_overlapping_ticker_is_kept(history):
    prices = pd.DataFrame(
        {
            "date": list(pd.date_range("2020-01-01", periods=200)) * 2,
            "ticker": ["GOOD"] * 200 + ["LEAVER"] * 200,
            "close": [1.0] * 400,
        }
    )
    filtered, dropped = drop_recycled_tickers(prices, history)

    assert dropped == []
    assert set(filtered["ticker"]) == {"GOOD", "LEAVER"}


def test_single_sample_membership_is_not_mistaken_for_recycling(history):
    """Regression: a ticker that was an index member in only ONE sample used
    to be dropped as "recycled" no matter how good its price history was.

    Its window was [d, d] -- zero length -- so it could never show the 20 days
    of overlap the check demanded. Real names were being deleted (a stock that
    joined at the final sample, or left just after the first one) by the very
    filter meant to catch data-integrity failures.
    """
    single = pd.DataFrame(
        [{"sample_date": pd.Timestamp("2020-09-30"), "ticker": "LATEJOINER"}],
        columns=["sample_date", "ticker"],
    )
    history = pd.concat([history, single], ignore_index=True)
    prices = pd.DataFrame(
        {
            "date": list(pd.date_range("2019-01-01", periods=700)),
            "ticker": ["LATEJOINER"] * 700,
            "close": [1.0] * 700,
        }
    )
    filtered, dropped = drop_recycled_tickers(prices, history)

    assert dropped == [], "a genuine late joiner with full price history must be kept"
    assert len(filtered) == 700


def test_disjoint_history_still_dropped_for_short_window_ticker(history):
    """The relaxation above must not let the actual recycled case through."""
    single = pd.DataFrame(
        [{"sample_date": pd.Timestamp("2020-03-31"), "ticker": "DEADCO"}],
        columns=["sample_date", "ticker"],
    )
    history = pd.concat([history, single], ignore_index=True)
    prices = pd.DataFrame(
        {
            "date": list(pd.date_range("2026-07-17", periods=30)),
            "ticker": ["DEADCO"] * 30,
            "close": [1.0] * 30,
        }
    )
    _, dropped = drop_recycled_tickers(prices, history)

    assert dropped == ["DEADCO"]


def test_coverage_report_counts_missing_price_data(history):
    """The residual bias that can't be fixed with free data must be reported,
    not silently absorbed."""
    prices = pd.DataFrame(
        {"date": pd.to_datetime(["2020-04-15", "2020-04-16"]), "ticker": ["GOOD", "GOOD"]}
    )
    report = coverage_report(history, prices)

    assert report["n_ever_members"] == 3
    assert report["n_with_price_data"] == 1
    assert report["n_missing_price_data"] == 2
    assert report["missing_tickers"] == ["JOINER", "LEAVER"]
    assert report["n_left_index_during_window"] == 1


def test_membership_windows_bound_each_ticker(history):
    """`last_member` is the final sample PLUS one sampling interval, because a
    sample stands for the interval it opens rather than a single instant --
    see membership_windows. `first_member` is the raw first sample: membership
    is never carried backward, which would be look-ahead."""
    windows = membership_windows(history).set_index("ticker")
    interval = sampling_interval(history)

    assert windows.loc["LEAVER", "last_member"] == pd.Timestamp("2020-06-30") + interval
    assert windows.loc["JOINER", "first_member"] == pd.Timestamp("2020-09-30")
    # Quarterly samples in the fixture -> roughly a quarter.
    assert pd.Timedelta(80, unit="D") < interval < pd.Timedelta(100, unit="D")
