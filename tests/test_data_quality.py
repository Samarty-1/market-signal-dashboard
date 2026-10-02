"""The data-quality gate: what it removes, what it only flags, and what stops the run."""

import numpy as np
import pandas as pd
import pytest

from src import data_quality as dq
from src.features import add_features_for_ticker

TODAY = pd.Timestamp("2026-10-02")


def _prices(tickers=("AAA", "BBB"), n=30, end=TODAY) -> pd.DataFrame:
    dates = pd.bdate_range(end=end, periods=n)
    frames = []
    for i, t in enumerate(tickers):
        close = 100 + i + np.arange(n) * 0.1
        frames.append(pd.DataFrame({"date": dates, "ticker": t, "open": close, "high": close * 1.01,
                                    "low": close * 0.99, "close": close, "volume": 1_000_000}))
    return pd.concat(frames, ignore_index=True)


def test_clean_data_passes_untouched():
    df, rep = dq.validate_and_clean(_prices(), expected_tickers=["AAA", "BBB"], today=TODAY)
    assert rep.ok and rep.rows_out == rep.rows_in == len(df) and not rep.warnings


def test_unusable_rows_are_removed_and_counted():
    p = _prices()
    p = pd.concat([p, p.iloc[[0]]], ignore_index=True)        # duplicate (date, ticker)
    p.loc[3, "close"] = np.nan                                  # missing price
    p.loc[4, "low"] = 0.0                                       # non-positive price
    p.loc[5, "volume"] = -10                                    # negative volume
    df, rep = dq.validate_and_clean(p, today=TODAY)
    assert rep.removed["duplicate (date, ticker)"] == 1
    assert rep.removed["missing or non-finite price"] == 1
    assert rep.removed["zero or negative price"] == 1
    assert rep.removed["negative volume"] == 1
    assert not df.duplicated(["date", "ticker"]).any() and (df[dq.PRICE_COLUMNS] > 0).all().all()


def test_suspicious_but_possibly_real_rows_are_kept_and_flagged():
    p = _prices()
    p.loc[10, ["open", "high", "low", "close"]] = [300, 303, 297, 300]   # +190% jump
    p.loc[12, "volume"] = 0
    p.loc[14, "high"] = p.loc[14, "low"] * 0.9                             # high below low
    df, rep = dq.validate_and_clean(p, today=TODAY)
    assert rep.ok and len(df) == len(p), "a real crash day must never be deleted for looking extreme"
    assert {"zero volume", "OHLC inconsistent"} <= set(rep.flagged)
    assert any("one-day move" in k for k in rep.flagged)


def test_stale_feed_is_an_error():
    _, rep = dq.validate_and_clean(_prices(end=TODAY - pd.Timedelta(days=14)), today=TODAY)
    assert not rep.ok and "stale" in rep.errors[0]
    with pytest.raises(dq.DataQualityError):
        dq.require_ok(rep)


def test_weekend_is_not_stale():
    _, rep = dq.validate_and_clean(_prices(end=pd.Timestamp("2026-10-02")),  # Friday
                                   today=pd.Timestamp("2026-10-05"))            # Monday
    assert rep.ok


def test_missing_tickers_warn_when_few_and_fail_when_many():
    asked = [f"T{i}" for i in range(20)]
    _, few = dq.validate_and_clean(_prices(tickers=asked[:19]), expected_tickers=asked, today=TODAY)
    assert few.ok and any("T19" in w for w in few.warnings), "a silently dropped ticker must be reported"
    _, many = dq.validate_and_clean(_prices(tickers=asked[:10]), expected_tickers=asked, today=TODAY)
    assert not many.ok


def test_ticker_that_stopped_updating_is_reported():
    p = _prices()
    p = p[~((p["ticker"] == "BBB") & (p["date"] > TODAY - pd.Timedelta(days=12)))]
    _, rep = dq.validate_and_clean(p, today=TODAY)
    assert any("stopped updating" in w and "BBB" in w for w in rep.warnings)


def test_missing_columns_fail_fast():
    _, rep = dq.validate_and_clean(_prices().drop(columns=["volume"]), today=TODAY)
    assert not rep.ok and "volume" in rep.errors[0]


def test_zero_volume_day_never_produces_an_infinite_feature():
    p = _prices(tickers=("AAA",), n=60)
    p.loc[40, "volume"] = 0
    vc = add_features_for_ticker(p)["volume_change"]
    assert not np.isinf(vc).any(), "inf here makes StandardScaler reject the whole training matrix"


def test_report_round_trips_to_json(tmp_path):
    _, rep = dq.validate_and_clean(_prices(), today=TODAY)
    rep.write(tmp_path / "dq.json")
    assert (tmp_path / "dq.json").read_text().count('"ok": true') == 1


def test_unfinished_session_is_dropped_while_market_is_open():
    from datetime import datetime, timezone
    p = _prices()                                                    # last bar = 2026-10-02 (Fri)
    during = datetime(2026, 10, 2, 15, 0, tzinfo=timezone.utc)      # 11:00 New York
    after = datetime(2026, 10, 2, 21, 30, tzinfo=timezone.utc)      # 17:30 New York
    df, rep = dq.validate_and_clean(p, today=TODAY, now=during)
    assert df["date"].max() < TODAY and rep.removed["in-progress session (market not closed yet)"] == 2
    df, rep = dq.validate_and_clean(p, today=TODAY, now=after)
    assert df["date"].max() == TODAY, "a finished session must be kept"
