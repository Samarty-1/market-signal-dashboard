"""Data-quality gate: validate and clean raw prices before anything trains on them.

Yahoo Finance is a free feed and behaves like one. Things this has to survive:

* a ticker silently returning nothing (delisted, renamed, rate-limited) -- the
  ingestion loop used to `continue` past it, so the universe shrank with no record
* duplicate (date, ticker) rows, which double-weight a day in training
* zero / negative / missing prices, which turn returns into inf or NaN
* zero volume, which turns `volume.pct_change()` into inf -- StandardScaler then
  refuses the whole training matrix
* bars where high < low, or the close sits outside the day's range (a bad print)
* a one-day move too large to be real (an unadjusted split, a bad tick)
* today's bar while the market is still open -- Yahoo serves the in-progress
  session as if it were a finished day, so a run at 11am trains on a "close"
  that is really a mid-morning price
* a feed that stopped updating -- the model retrains "successfully" on old data
  and the dashboard shows a stale signal as if it were today's

Policy: rows that are unusable (duplicates, non-positive or missing prices,
negative volume) are REMOVED, and the count is reported. Rows that are
suspicious but might be real (big moves, inconsistent OHLC, zero volume) are KEPT
and FLAGGED -- deleting a genuine crash day because it looked extreme would bias
the data toward calm markets. Problems that make the whole run untrustworthy
(missing columns, a stale feed, too much of the universe missing) are ERRORS and
stop the pipeline before any artifact is written.

The report is written to reports/data_quality.json on every run, so the
committed history shows what the data looked like each day, not just the model.
"""

from __future__ import annotations

import json
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close"]

# A >50% one-day move in a large-cap or ETF is far more often a data error
# (unadjusted split, bad print) than a real move. Flagged, never deleted.
MAX_ABS_DAILY_RETURN = 0.5
# Tolerance for OHLC consistency -- Yahoo's adjusted bars are rounded, so tiny
# violations are rounding, not errors.
OHLC_TOLERANCE = 0.005
# The feed is stale if its newest bar is older than this many business days.
# 3 covers a weekend plus one exchange holiday.
MAX_STALE_BUSINESS_DAYS = 3
# A ticker whose own last bar trails the panel's last bar by more than this
# has stopped updating (halted, delisted, symbol change).
MAX_TICKER_LAG_DAYS = 5
# Losing more than this share of the requested universe is an error, not a warning.
MAX_MISSING_TICKER_FRACTION = 0.10
DEFAULT_REPORT_PATH = Path("reports/data_quality.json")
EXCHANGE_TZ = ZoneInfo("America/New_York")
# 16:00 close plus a margin for Yahoo to publish the final print.
SESSION_FINAL_AFTER = time(16, 30)


@dataclass
class QualityReport:
    rows_in: int = 0
    rows_out: int = 0
    tickers_requested: int = 0
    tickers_received: int = 0
    first_date: str = ""
    last_date: str = ""
    removed: dict[str, int] = field(default_factory=dict)       # check -> rows dropped
    flagged: dict[str, list[str]] = field(default_factory=dict)  # check -> examples kept
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        head = (f"data quality: {'OK' if self.ok else 'FAILED'} -- {self.rows_out}/{self.rows_in} rows kept, "
                f"{self.tickers_received}/{self.tickers_requested} tickers, {self.first_date} -> {self.last_date}")
        lines = [head]
        lines += [f"  removed {n} row(s): {check}" for check, n in self.removed.items() if n]
        lines += [f"  ERROR   {e}" for e in self.errors]
        lines += [f"  warning {w}" for w in self.warnings]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self) | {"ok": self.ok}

    def write(self, path: Path | str = DEFAULT_REPORT_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


class DataQualityError(RuntimeError):
    def __init__(self, report: QualityReport):
        super().__init__(report.summary())
        self.report = report


def _examples(frame: pd.DataFrame, limit: int = 8) -> list[str]:
    return [f"{t} {pd.Timestamp(d).date()}" for t, d in zip(frame["ticker"], frame["date"])][:limit]


def validate_and_clean(
    prices: pd.DataFrame,
    expected_tickers: list[str] | None = None,
    today: pd.Timestamp | None = None,
    now: datetime | None = None,
) -> tuple[pd.DataFrame, QualityReport]:
    """Return (cleaned prices, report). Never raises -- the caller decides what an
    error means (see :func:`require_ok`), so tests and notebooks can inspect a
    failing report instead of getting an exception."""
    rep = QualityReport(rows_in=len(prices))
    expected = sorted(set(expected_tickers)) if expected_tickers else None
    rep.tickers_requested = len(expected) if expected else int(prices["ticker"].nunique()) if "ticker" in prices else 0

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in prices.columns]
    if missing_cols:
        rep.errors.append(f"missing required columns: {missing_cols}")
        return prices, rep

    df = prices[REQUIRED_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in PRICE_COLUMNS + ["volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    def drop(mask: pd.Series, check: str) -> None:
        nonlocal df
        rep.removed[check] = int(mask.sum())
        df = df[~mask]

    drop(df.duplicated(["date", "ticker"], keep="last"), "duplicate (date, ticker)")
    drop(df[PRICE_COLUMNS].isna().any(axis=1) | ~np.isfinite(df[PRICE_COLUMNS]).all(axis=1),
         "missing or non-finite price")
    drop((df[PRICE_COLUMNS] <= 0).any(axis=1), "zero or negative price")
    drop(df["volume"] < 0, "negative volume")
    # An unfinished session is not a daily bar. Only checked when the caller is
    # running live (`now`), so historical/test data is never affected.
    if now is not None:
        ny = now.astimezone(EXCHANGE_TZ)
        session = pd.Timestamp(ny.date())
        open_now = ny.weekday() < 5 and ny.time() < SESSION_FINAL_AFTER
        drop((df["date"].dt.normalize() == session) & open_now, "in-progress session (market not closed yet)")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # --- kept but flagged: plausibly real, so not ours to delete
    tol = 1 + OHLC_TOLERANCE
    bad_ohlc = (df["high"] * tol < df[["open", "close", "low"]].max(axis=1)) | \
               (df["low"] > df[["open", "close", "high"]].min(axis=1) * tol)
    ret = df.groupby("ticker")["close"].pct_change()
    big_move = ret.abs() > MAX_ABS_DAILY_RETURN
    zero_vol = df["volume"].fillna(0) == 0
    for check, mask in (("OHLC inconsistent", bad_ohlc),
                        (f"one-day move above {MAX_ABS_DAILY_RETURN:.0%}", big_move),
                        ("zero volume", zero_vol)):
        if mask.any():
            rep.flagged[check] = _examples(df[mask])
            rep.warnings.append(f"{int(mask.sum())} row(s) {check} (kept), e.g. {', '.join(rep.flagged[check][:3])}")

    rep.rows_out = len(df)
    if df.empty:
        rep.errors.append("no usable rows after cleaning")
        return df, rep

    received = sorted(df["ticker"].unique())
    rep.tickers_received = len(received)
    rep.first_date = str(df["date"].min().date())
    rep.last_date = str(df["date"].max().date())

    if expected:
        missing = sorted(set(expected) - set(received))
        if missing:
            frac = len(missing) / len(expected)
            msg = f"{len(missing)}/{len(expected)} requested tickers returned no usable data: {', '.join(missing[:10])}"
            (rep.errors if frac > MAX_MISSING_TICKER_FRACTION else rep.warnings).append(msg)

    # --- freshness: of the feed as a whole, and of each ticker within it
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    last = df["date"].max().normalize()
    behind = len(pd.bdate_range(last, today)) - 1
    if behind > MAX_STALE_BUSINESS_DAYS:
        rep.errors.append(f"feed is stale: newest bar {last.date()} is {behind} business days old")
    lag = df.groupby("ticker")["date"].max()
    stopped = lag[(last - lag).dt.days > MAX_TICKER_LAG_DAYS]
    if len(stopped):
        rep.warnings.append(f"{len(stopped)} ticker(s) stopped updating: "
                            + ", ".join(f"{t} (last {d.date()})" for t, d in stopped.head(8).items()))
    return df, rep


def validate_live(prices: pd.DataFrame, expected_tickers: list[str] | None = None):
    """validate_and_clean for a pipeline run happening now: drops an unfinished
    session and checks freshness against today."""
    now = datetime.now(timezone.utc)
    return validate_and_clean(prices, expected_tickers, today=pd.Timestamp(now.date()), now=now)


def require_ok(report: QualityReport) -> None:
    if not report.ok:
        raise DataQualityError(report)
