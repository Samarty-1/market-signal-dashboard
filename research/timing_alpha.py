"""Separate genuine stock-TIMING alpha from the static tilt underneath it.

Neutralising against size/liquidity/volatility was not enough: a frozen book
still beat the live one, so the prediction carries a persistent per-name
component (some tickers simply score high every day) that no small factor set
fully captures.

The clean separation is to demean each ticker's score against its OWN history:

    timing(t, i) = pred(t, i) - mean(pred(s, i) for s < t)

The subtracted term uses only that ticker's past, so this stays causal. What is
left answers "is this name unusually attractive versus how it normally scores",
which is pure timing -- a constant tilt cannot survive it, by construction.

Reporting both books side by side is the honest decomposition:
  tilt   = what a frozen portfolio would have earned anyway (a risk premium)
  timing = what the model's live updating actually adds (skill)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.blend_study import HORIZONS, cs_rank
from research.cost_study import evaluate, fwd_wide
from research.horizon_study import get_predictions
from research.panel import load_panel
from research.portfolio import metrics, signal_to_weights, simulate

MIN_HISTORY = 120  # days of a ticker's own history before its mean is trusted


def timing_signal(pred: pd.Series, min_periods: int = MIN_HISTORY) -> pd.Series:
    """Remove each ticker's causal expanding-mean score."""
    wide = pred.unstack("ticker").sort_index()
    # shift(1) so date t's baseline is built strictly from dates before t.
    mu = wide.expanding(min_periods=min_periods).mean().shift(1)
    return (wide - mu).stack().dropna()


def frozen_book(sig: pd.Series, panel: pd.DataFrame) -> dict:
    """Expanding-mean weights: keeps the average tilt, removes live timing."""
    w = signal_to_weights(sig)
    fwd = fwd_wide(panel, sig.index).reindex(index=w.index, columns=w.columns)
    fz = w.expanding().mean()
    fz = fz.sub(fz.mean(axis=1), axis=0)
    fz = fz.div(fz.abs().sum(axis=1).replace(0, np.nan), axis=0)
    return metrics(simulate(fz, fwd, cost_bps=10.0, band=0.5))


def main() -> None:
    panel = load_panel()
    preds = {h: get_predictions(panel, "raw_", h, "dev") for h in HORIZONS}
    idx = preds[1].index
    for h in HORIZONS:
        idx = idx.intersection(preds[h].index)

    signals = {
        "h1": preds[1],
        "blend": sum(cs_rank(preds[h].loc[idx]) for h in HORIZONS) / len(HORIZONS),
    }

    hdr = (f"{'signal':<18} {'halflife':>8} {'gross_Sh':>9} {'net_Sh':>8} "
           f"{'net_ret':>8} {'ann_vol':>8} {'turn/d':>7}")
    for name, base in signals.items():
        tim = timing_signal(base)
        print(f"\n########## {name}  (timing rows: {len(tim):,}) ##########")
        print(f"  frozen-tilt benchmark net_Sh = {frozen_book(tim, panel)['net_sharpe']:+.2f}"
              "   <- should collapse if the tilt is gone")
        print()
        print(hdr)
        print("-" * len(hdr))
        for hl in (5, 10, 21, 42):
            m = evaluate(tim, panel, "rank", hl, 0.5)
            print(f"{name + ' timing':<18} {hl:>8} {m['gross_sharpe']:>+9.2f} "
                  f"{m['net_sharpe']:>+8.2f} {m['net_ann_ret']:>+8.3f} "
                  f"{m['ann_vol']:>8.3f} {m['turnover_daily']:>7.4f}", flush=True)


if __name__ == "__main__":
    main()
