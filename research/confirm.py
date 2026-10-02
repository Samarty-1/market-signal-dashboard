"""FINAL CONFIRMATION -- run once, on data no design decision ever touched.

Everything is frozen before this runs:
  signal       blend of 4 horizons (1/5/10/21d), raw features, regularised
               LightGBM, 3 seeds, purged walk-forward (embargo = h + 2)
  construction continuous cross-sectional rank weights, dollar-neutral,
               EMA halflife 42, no-trade band 0.5, 10bps one-way cost
  baseline     the repo's own construction: hard top/bottom decile from the
               1-day signal, re-picked daily, same cost model

DEV     (<= 2022-12-31) chose all of the above.
CONFIRM (>  2022-12-31) is scored here and nothing is tuned against it.

The portfolio is simulated across the whole period and only the confirmation
slice is reported, so the EMA arrives warm and positions carried in from the
development period are real -- exactly how it would have traded live.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.blend_study import cs_rank
from research.cost_study import fwd_wide
from research.decompose import split
from research.experiment import DEV_END
from research.horizon_study import HORIZONS, get_predictions
from research.panel import load_panel
from research.portfolio import decile_weights, metrics, signal_to_weights, simulate, smooth

HALFLIFE = 42
BAND = 0.5
COST_BPS = 10.0


def run_book(sig: pd.Series, panel: pd.DataFrame, construction: str,
             halflife: float | None, band: float) -> pd.DataFrame:
    w = decile_weights(sig) if construction == "decile" else signal_to_weights(sig)
    w = smooth(w, halflife)
    fwd = fwd_wide(panel, sig.index).reindex(index=w.index, columns=w.columns)
    return simulate(w, fwd, cost_bps=COST_BPS, band=band)


def slice_metrics(sim: pd.DataFrame, lo: pd.Timestamp | None, hi: pd.Timestamp | None) -> dict:
    s = sim
    if lo is not None:
        s = s[s.index > lo]
    if hi is not None:
        s = s[s.index <= hi]
    return metrics(s)


def main() -> None:
    panel = load_panel()
    preds = {h: get_predictions(panel, "raw_", h, "full") for h in HORIZONS}
    idx = preds[1].index
    for h in HORIZONS:
        idx = idx.intersection(preds[h].index)
    blend = sum(cs_rank(preds[h].loc[idx]) for h in HORIZONS) / len(HORIZONS)
    full, static, timing = split(blend)

    books = {
        "REPO baseline (h1, hard decile, daily)": (preds[1], "decile", None, 0.0),
        "FIXED full (blend, rank, hl42, band)":   (full,     "rank",   HALFLIFE, BAND),
        "  ...static/tilt component":            (static,   "rank",   HALFLIFE, BAND),
        "  ...timing component":                 (timing,   "rank",   HALFLIFE, BAND),
    }

    out = {}
    for period, (lo, hi) in (("DEV", (None, DEV_END)), ("CONFIRM", (DEV_END, None))):
        print(f"\n================ {period} ================")
        hdr = (f"{'book':<40} {'gross_Sh':>9} {'net_Sh':>8} {'net_ret':>8} "
               f"{'ann_vol':>8} {'max_dd':>8} {'turn/d':>7} {'days':>6}")
        print(hdr)
        print("-" * len(hdr))
        for name, (sig, cons, hl, band) in books.items():
            m = slice_metrics(run_book(sig, panel, cons, hl, band), lo, hi)
            out.setdefault(period, {})[name] = m
            print(f"{name:<40} {m['gross_sharpe']:>+9.2f} {m['net_sharpe']:>+8.2f} "
                  f"{m['net_ann_ret']:>+8.3f} {m['ann_vol']:>8.3f} {m['max_dd']:>+8.3f} "
                  f"{m['turnover_daily']:>7.4f} {m['n_days']:>6}", flush=True)

    Path("research/results_confirm.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nwrote research/results_confirm.json")


if __name__ == "__main__":
    main()
