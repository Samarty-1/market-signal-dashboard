"""Two remaining DEV questions before the confirmation run.

1. Is halflife=42 a real optimum or just the edge of the grid I happened to try?
2. Does blending the horizons beat any single one?

Blending is standard practice: h1 and h21 are near-independent views of the same
names (their 1-day ICs barely overlap), so averaging them should cancel some
model noise and raise the information ratio without adding turnover -- the
closest thing to a free lunch in signal construction.
"""
from __future__ import annotations

import pandas as pd

from research.cost_study import evaluate
from research.horizon_study import get_predictions
from research.panel import load_panel

HORIZONS = [1, 5, 10, 21]


def cs_rank(s: pd.Series) -> pd.Series:
    """Put every signal on the same within-date scale before averaging, so a
    horizon with a wider raw spread cannot dominate the blend by accident."""
    return s.groupby(level="date").rank(pct=True) * 2 - 1


def main() -> None:
    panel = load_panel()
    preds = {h: get_predictions(panel, "raw_", h, "dev") for h in HORIZONS}

    idx = preds[1].index
    for h in HORIZONS:
        idx = idx.intersection(preds[h].index)
    blend = sum(cs_rank(preds[h].loc[idx]) for h in HORIZONS) / len(HORIZONS)

    signals = {f"h{h}": preds[h] for h in HORIZONS}
    signals["blend"] = blend

    hdr = (f"{'signal':<8} {'halflife':>8} {'band':>5} {'gross_Sh':>9} {'net_Sh':>8} "
           f"{'net_ret':>8} {'ann_vol':>8} {'max_dd':>8} {'turn/d':>7}")
    print(hdr)
    print("-" * len(hdr))

    best = None
    for name, sig in signals.items():
        for hl in (21, 42, 63, 84):
            m = evaluate(sig, panel, "rank", hl, 0.5)
            print(f"{name:<8} {hl:>8} {0.5:>5.1f} {m['gross_sharpe']:>+9.2f} "
                  f"{m['net_sharpe']:>+8.2f} {m['net_ann_ret']:>+8.3f} "
                  f"{m['ann_vol']:>8.3f} {m['max_dd']:>+8.3f} {m['turnover_daily']:>7.3f}",
                  flush=True)
            if best is None or m["net_sharpe"] > best[0]:
                best = (m["net_sharpe"], name, hl)
        print()

    print(f"DEV best: signal={best[1]} halflife={best[2]} net_sharpe={best[0]:+.2f}")


if __name__ == "__main__":
    main()
