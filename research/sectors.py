"""Point-in-time GICS sector for every index member, from the same pinned
Wikipedia revisions src/universe.py uses for membership.

    python -m research.sectors     # writes cache/sectors_pit.parquet

A name's sector on date t is read from the latest sampled revision at or before t
(held forward, never back-filled), so a 2018 GICS reshuffle is seen only once
it happened.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from src import universe as U

OUT = Path("cache/sectors_pit.parquet")
_SECTOR_COLUMNS = {"gics sector", "gics\xa0sector", "sector"}


def _parse_sectors(html: bytes) -> dict[str, str]:
    for table in pd.read_html(io.BytesIO(html)):
        cols = {c.strip().lower(): c for c in table.columns if isinstance(c, str)}
        tcol = next((cols[c] for c in cols if c in U._TICKER_COLUMNS), None)
        scol = next((cols[c] for c in cols if c in _SECTOR_COLUMNS), None)
        if tcol is None or scol is None or len(table) < U._MIN_CONSTITUENTS:
            continue
        t = table[tcol].astype(str).str.strip().str.replace(".", "-", regex=False)
        return dict(zip(t, table[scol].astype(str).str.strip()))
    raise RuntimeError("no constituents table with a sector column in this revision")


def build(start: pd.Timestamp) -> pd.DataFrame:
    end = pd.Timestamp.today().normalize()
    dates = pd.date_range(start, end, freq="QE").union(pd.DatetimeIndex([start, end]))
    rows = []
    for d in dates:
        revid, _ = U.revision_asof(d)
        cache = U.CACHE_DIR / f"sectors_{revid}.json"
        if cache.exists():
            sect = pd.read_json(cache, typ="series").to_dict()
        else:
            sect = _parse_sectors(U._http_get(f"https://en.wikipedia.org/w/index.php?oldid={revid}"))
            pd.Series(sect).to_json(cache)
        rows += [{"sample_date": d, "ticker": t, "sector": s} for t, s in sect.items()]
        print(f"{d.date()} rev {revid}: {len(sect)} names, {len(set(sect.values()))} sectors", flush=True)
    return pd.DataFrame(rows)


def sector_wide(index: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    """date x ticker sector label, held forward from the latest sample <= date."""
    hist = pd.read_parquet(OUT)
    wide = (hist.pivot_table(index="sample_date", columns="ticker", values="sector", aggfunc="first")
            .reindex(columns=columns).sort_index())
    # a name absent from a revision keeps its last known sector (it left the index;
    # membership masking already decides whether it is tradable)
    wide = wide.ffill()
    pos = np.clip(wide.index.searchsorted(index, side="right") - 1, 0, len(wide) - 1)
    return pd.DataFrame(wide.iloc[pos].to_numpy(), index=index, columns=columns)


if __name__ == "__main__":
    df = build(pd.Timestamp.today().normalize() - pd.DateOffset(years=12))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(df.groupby("sample_date").sector.nunique().describe())
    print(df.sector.value_counts())
