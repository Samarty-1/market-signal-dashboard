"""Is the smoothed book actually using the model, or has it degenerated into a
slow static factor tilt that any constant portfolio would have earned?

Net Sharpe improved monotonically as the smoothing halflife grew, with no
interior optimum inside the grid. That is the signature of a book converging on
a near-constant tilt (long low-vol / short high-vol, say) rather than one
trading a live prediction. If a causally-frozen version of the same weights
scores as well, the "improvement" is a factor exposure, not model alpha, and
must be reported as such.

Three checks:
  1. push the halflife far past the grid (126, 252, ~expanding mean)
  2. score a FROZEN book: weights fixed at their expanding-mean, which uses only
     past information but contains no live signal update
  3. correlate the smoothed weights with individual raw features to see what
     exposure the book actually carries
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.cost_study import evaluate, fwd_wide
from research.horizon_study import get_predictions
from research.harness import FEATURES
from research.panel import load_panel
from research.portfolio import metrics, signal_to_weights, simulate, smooth
from research.blend_study import cs_rank, HORIZONS


def main() -> None:
    panel = load_panel()
    preds = {h: get_predictions(panel, "raw_", h, "dev") for h in HORIZONS}
    idx = preds[1].index
    for h in HORIZONS:
        idx = idx.intersection(preds[h].index)
    blend = sum(cs_rank(preds[h].loc[idx]) for h in HORIZONS) / len(HORIZONS)

    print("=== 1. halflife pushed past the grid ===")
    for hl in (84, 126, 252, 504):
        m = evaluate(blend, panel, "rank", hl, 0.5)
        print(f"  halflife={hl:>4}: gross_Sh={m['gross_sharpe']:+.2f} net_Sh={m['net_sharpe']:+.2f} "
              f"turn/d={m['turnover_daily']:.4f}", flush=True)

    print("\n=== 2. frozen book (expanding mean of weights, no live signal) ===")
    w = signal_to_weights(blend)
    fwd = fwd_wide(panel, blend.index).reindex(index=w.index, columns=w.columns)
    # Expanding mean is causal: weights on date t use only signal up to t. It
    # keeps the average tilt while removing essentially all timing information.
    frozen = w.expanding().mean()
    frozen = frozen.sub(frozen.mean(axis=1), axis=0)
    frozen = frozen.div(frozen.abs().sum(axis=1).replace(0, np.nan), axis=0)
    m_frozen = metrics(simulate(frozen, fwd, cost_bps=10.0, band=0.5))
    m_live = evaluate(blend, panel, "rank", 84, 0.5)
    print(f"  frozen (expanding-mean tilt): net_Sh={m_frozen['net_sharpe']:+.2f} "
          f"net_ret={m_frozen['net_ann_ret']:+.3f}")
    print(f"  live   (halflife 84)        : net_Sh={m_live['net_sharpe']:+.2f} "
          f"net_ret={m_live['net_ann_ret']:+.3f}")
    print("  -> if these are close, the book is a tilt, not a live signal")

    print("\n=== 3. what exposure does the smoothed book carry? ===")
    sm = smooth(w, 84)
    # Average cross-sectional correlation between held weight and each feature's
    # within-date rank: what the book is systematically long and short.
    corrs = {}
    for f in FEATURES:
        feat = panel.loc[blend.index, f"cs_{f}"].unstack("ticker").reindex(
            index=sm.index, columns=sm.columns)
        num = (sm * feat).sum(axis=1)
        den = np.sqrt((sm ** 2).sum(axis=1) * (feat ** 2).sum(axis=1)).replace(0, np.nan)
        corrs[f] = float((num / den).mean())
    for f, c in sorted(corrs.items(), key=lambda kv: -abs(kv[1]))[:8]:
        print(f"  {f:<22} {c:+.3f}")


if __name__ == "__main__":
    main()
