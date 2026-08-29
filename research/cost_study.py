"""Turns a signal into a book that survives its own transaction costs.

The signal is not the problem (ICIR ~2.3 gross). The problem is that the repo's
construction pays more in turnover than the signal produces. This sweeps the
three levers that fix that, on DEV only:

  construction : hard decile (repo) vs continuous rank weights
  smoothing    : EMA halflife on the target weights
  band         : no-trade threshold

Everything is scored net of a 10bps one-way cost on actual dollars traded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from research.horizon_study import get_predictions
from research.panel import load_panel
from research.portfolio import decile_weights, metrics, signal_to_weights, simulate

COST_BPS = 10.0


def fwd_wide(panel: pd.DataFrame, index) -> pd.DataFrame:
    return panel.loc[index, "fwd_ret"].unstack("ticker").sort_index()


def evaluate(pred: pd.Series, panel: pd.DataFrame, construction: str,
             halflife: float | None, band: float, cost_bps: float = COST_BPS) -> dict:
    from research.portfolio import smooth
    w = decile_weights(pred) if construction == "decile" else signal_to_weights(pred)
    w = smooth(w, halflife)
    fwd = fwd_wide(panel, pred.index).reindex(index=w.index, columns=w.columns)
    return metrics(simulate(w, fwd, cost_bps=cost_bps, band=band))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="raw_")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--out", default="research/results_cost.json")
    args = ap.parse_args()

    panel = load_panel()
    pred = get_predictions(panel, args.prefix, args.horizon, "dev")
    print(f"signal: {args.prefix}h{args.horizon}  ({len(pred):,} predictions)\n", flush=True)

    rows = []
    hdr = (f"{'construction':<12} {'halflife':>8} {'band':>5} "
           f"{'gross_Sh':>9} {'net_Sh':>8} {'net_ret':>8} {'turn/d':>7} "
           f"{'cost_bps':>9} {'gross_bps':>10}")
    print(hdr)
    print("-" * len(hdr))

    for construction in ("decile", "rank"):
        for hl in (None, 5, 10, 21, 42):
            for band in (0.0, 0.5):
                m = evaluate(pred, panel, construction, hl, band)
                rows.append({"construction": construction, "halflife": hl, "band": band, **m})
                print(f"{construction:<12} {str(hl):>8} {band:>5.1f} "
                      f"{m['gross_sharpe']:>+9.2f} {m['net_sharpe']:>+8.2f} "
                      f"{m['net_ann_ret']:>+8.3f} {m['turnover_daily']:>7.3f} "
                      f"{m['cost_bps_day']:>9.2f} {m['gross_bps_day']:>10.2f}", flush=True)

    Path(args.out).write_text(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
