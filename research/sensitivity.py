"""Robustness checks on the confirmed result.

A fix that only works at exactly the cost assumption it was tuned under is not a
fix. Two checks:

  1. cost sensitivity -- 5 / 10 / 20 / 30 bps one-way. The repo's own smallcap
     experiment used 20bps for wider spreads, so the strategy should be scored
     there too rather than only at the friendliest assumption.
  2. year-by-year -- a single aggregate Sharpe can hide a strategy that made
     everything in one year and bled the rest.
"""
from __future__ import annotations

import pandas as pd

from research.blend_study import cs_rank
from research.confirm import BAND, HALFLIFE, run_book
from research.cost_study import fwd_wide
from research.decompose import split
from research.experiment import DEV_END
from research.horizon_study import HORIZONS, get_predictions
from research.panel import load_panel
from research.portfolio import (
    decile_weights, metrics, signal_to_weights, simulate, smooth,
)


def main() -> None:
    panel = load_panel()
    preds = {h: get_predictions(panel, "raw_", h, "full") for h in HORIZONS}
    idx = preds[1].index
    for h in HORIZONS:
        idx = idx.intersection(preds[h].index)
    blend = sum(cs_rank(preds[h].loc[idx]) for h in HORIZONS) / len(HORIZONS)
    full, static, timing = split(blend)

    print("=== 1. cost sensitivity (CONFIRM period only) ===")
    hdr = f"{'book':<22} {'cost_bps':>9} {'gross_Sh':>9} {'net_Sh':>8} {'net_ret':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name, (sig, cons, hl, band) in (
        ("repo baseline", (preds[1], "decile", None, 0.0)),
        ("fixed full", (full, "rank", HALFLIFE, BAND)),
        ("timing only", (timing, "rank", HALFLIFE, BAND)),
    ):
        for cb in (5.0, 10.0, 20.0, 30.0):
            w = smooth(signal_to_weights(sig) if cons == "rank" else decile_weights(sig), hl)
            fwd = fwd_wide(panel, sig.index).reindex(index=w.index, columns=w.columns)
            sim = simulate(w, fwd, cost_bps=cb, band=band)
            sim = sim[sim.index > DEV_END]
            m = metrics(sim)
            print(f"{name:<22} {cb:>9.0f} {m['gross_sharpe']:>+9.2f} "
                  f"{m['net_sharpe']:>+8.2f} {m['net_ann_ret']:>+8.3f}", flush=True)
        print()

    print("=== 2. year by year, net Sharpe (10bps) ===")
    books = {
        "repo baseline": (preds[1], "decile", None, 0.0),
        "fixed full": (full, "rank", HALFLIFE, BAND),
        "tilt": (static, "rank", HALFLIFE, BAND),
        "timing": (timing, "rank", HALFLIFE, BAND),
    }
    sims = {n: run_book(s, panel, c, h, b) for n, (s, c, h, b) in books.items()}
    years = sorted({d.year for d in next(iter(sims.values())).index})
    print(f"{'year':<6} " + " ".join(f"{n:>15}" for n in books))
    print("-" * (6 + 16 * len(books)))
    for y in years:
        cells = []
        for n in books:
            s = sims[n]
            m = metrics(s[s.index.year == y])
            cells.append(f"{m['net_sharpe']:>+15.2f}" if m else f"{'-':>15}")
        tag = "*" if y > DEV_END.year else " "
        print(f"{y}{tag} " + " ".join(cells))
    print("\n(* = confirmation period, never used for any decision)")


if __name__ == "__main__":
    main()
