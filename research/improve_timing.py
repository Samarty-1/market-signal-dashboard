"""Addendum C of PREREGISTRATION_v2.md (H8): alpha-aware execution. Slow targets
from the ML blend; the fast reversal composite only decides WHEN a needed trade
is executed. DEV decides; --confirm scores once.

    MSD_UNIVERSE=pit python -m research.improve_timing [--confirm]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.blend_study import HORIZONS
from research.cost_study import fwd_wide
from research.experiment import DEV_END
from research.harness import SUFFIX
from research.horizon_study import get_predictions
from research.improve import BAND, BORROW_BPS_YR, COST_BPS, HALFLIFE, blend, line, market_and_beta, score
from research.improve_composite import composite
from research.improve_sectors import add_sector_features, book, sector_long
from research.panel import load_panel
from research.portfolio import signal_to_weights, smooth

FORCE = 2.0  # multiple of the band beyond which a trade executes regardless of timing


def simulate_timed(target_w: pd.DataFrame, fwd_ret: pd.DataFrame, timing: pd.DataFrame,
                   cost_bps: float, band: float, borrow_bps_yr: float) -> pd.DataFrame:
    """portfolio.simulate with one change: inside (band, FORCE*band] a trade is
    executed only if `timing` (known at today's close) agrees with its direction."""
    dates, cols = target_w.index, target_w.columns
    tgt = target_w.to_numpy(dtype=float)
    ret = fwd_ret.reindex(index=dates, columns=cols).to_numpy(dtype=float)
    tim = timing.reindex(index=dates, columns=cols).to_numpy(dtype=float)
    held = np.zeros(len(cols))
    cost_rate, borrow_rate = cost_bps / 1e4, borrow_bps_yr / 1e4 / 252
    rows = []
    for i in range(len(dates)):
        t = tgt[i]
        tradable = ~np.isnan(t)
        t = np.where(tradable, t, 0.0)
        scale = np.nanmean(np.abs(t)) if tradable.any() else 0.0
        gap = t - held
        size = np.abs(gap)
        agree = np.sign(gap) == np.sign(np.nan_to_num(tim[i]))
        move = (size > FORCE * band * scale) | ((size > band * scale) & agree)
        new = np.where(tradable, np.where(move, t, held), 0.0)
        turnover = np.abs(new - held).sum()
        held = new
        cost = turnover * cost_rate + np.clip(-held, 0.0, None).sum() * borrow_rate
        r = ret[i]
        gross = float(np.nansum(held * np.where(np.isnan(r), 0.0, r)))
        rows.append((dates[i], gross, cost, gross - cost, turnover))
    return pd.DataFrame(rows, columns=["date", "gross_return", "cost", "net_return", "turnover"]).set_index("date")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args()

    panel = load_panel()
    _, mkt, _ = market_and_beta()
    panel = add_sector_features(panel, sector_long(panel))
    ml = blend({h: get_predictions(panel, "raw_", h, "full") for h in HORIZONS})
    timing = composite(panel).unstack("ticker")

    w = smooth(signal_to_weights(ml), HALFLIFE)
    fwd = fwd_wide(panel, ml.index).reindex(index=w.index, columns=w.columns)
    sims = {
        "H0 baseline + borrow": book(ml, panel),
        "H8 reversal-timed execution": simulate_timed(w, fwd, timing, COST_BPS, BAND, BORROW_BPS_YR),
    }
    out = {}
    periods = [("DEV", None, DEV_END)] + ([("CONFIRM", DEV_END, None)] if args.confirm else [])
    for period, lo, hi in periods:
        print(f"================ {period} ================")
        for name, sim in sims.items():
            m = score(sim, mkt, lo, hi)
            out.setdefault(period, {})[name] = m
            print(line(name, m), flush=True)
    dest = Path(f"research/results_v2_timing{'_confirm' if args.confirm else '_dev'}{SUFFIX}.json")
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
