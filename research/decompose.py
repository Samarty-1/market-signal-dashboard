"""Clean three-way decomposition of the signal.

For each ticker, split its score into the part that is always there and the part
that moves, using only that ticker's own past:

    static(t, i) = mean(pred(s, i) for s < t)      persistent per-name tilt
    timing(t, i) = pred(t, i) - static(t, i)       live deviation from normal
    full(t, i)   = pred(t, i)                      = static + timing

Building a separate book on each answers the question the earlier frozen-weights
test asked badly: how much of the performance is a risk premium the portfolio
would collect by standing still, and how much is the model actually timing names.
Both component books are well-posed here (each is a real cross-sectional score),
unlike an expanding mean of already-normalised weights, which collapses toward
zero and then gets amplified by renormalisation.
"""
from __future__ import annotations

import pandas as pd

from research.blend_study import HORIZONS, cs_rank
from research.cost_study import evaluate
from research.horizon_study import get_predictions
from research.panel import load_panel

MIN_HISTORY = 120


def split(pred: pd.Series, min_periods: int = MIN_HISTORY) -> tuple[pd.Series, pd.Series, pd.Series]:
    wide = pred.unstack("ticker").sort_index()
    static = wide.expanding(min_periods=min_periods).mean().shift(1)
    timing = wide - static
    keep = static.notna() & wide.notna()
    return (
        wide.where(keep).stack().dropna(),
        static.where(keep).stack().dropna(),
        timing.where(keep).stack().dropna(),
    )


def main() -> None:
    panel = load_panel()
    preds = {h: get_predictions(panel, "raw_", h, "dev") for h in HORIZONS}
    idx = preds[1].index
    for h in HORIZONS:
        idx = idx.intersection(preds[h].index)
    blend = sum(cs_rank(preds[h].loc[idx]) for h in HORIZONS) / len(HORIZONS)

    for name, base in (("h1", preds[1]), ("blend", blend)):
        full, static, timing = split(base)
        print(f"\n########## {name} ##########")
        hdr = (f"{'component':<12} {'halflife':>8} {'gross_Sh':>9} {'net_Sh':>8} "
               f"{'net_ret':>8} {'ann_vol':>8} {'max_dd':>8} {'turn/d':>7}")
        print(hdr)
        print("-" * len(hdr))
        for comp_name, sig in (("full", full), ("static/tilt", static), ("timing", timing)):
            for hl in (10, 21, 42):
                m = evaluate(sig, panel, "rank", hl, 0.5)
                print(f"{comp_name:<12} {hl:>8} {m['gross_sharpe']:>+9.2f} "
                      f"{m['net_sharpe']:>+8.2f} {m['net_ann_ret']:>+8.3f} "
                      f"{m['ann_vol']:>8.3f} {m['max_dd']:>+8.3f} "
                      f"{m['turnover_daily']:>7.4f}", flush=True)
            print()


if __name__ == "__main__":
    main()
