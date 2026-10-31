"""Reconcile improve.py's baseline with confirm.py's published number.

confirm.py routes the blend through decompose.split(), whose `keep` mask drops
any (date, name) with fewer than MIN_HISTORY=120 prior predictions -- in practice,
names recently added to the index. improve.py fed the raw blend. This isolates
that one difference, on both periods, with no borrow (as published).
"""
from __future__ import annotations

from research.blend_study import HORIZONS
from research.decompose import split
from research.experiment import DEV_END
from research.horizon_study import get_predictions
from research.improve import blend, book, market_and_beta, score
from research.panel import load_panel


def main() -> None:
    panel = load_panel()
    _, mkt, _ = market_and_beta()
    raw = blend({h: get_predictions(panel, "raw_", h, "full") for h in HORIZONS})
    full, _, _ = split(raw)
    dropped = raw.index.difference(full.index)
    print(f"blend rows {len(raw):,}; dropped by the 120-day history mask {len(dropped):,}")
    for name, sig in (("raw blend (improve.py)", raw), ("masked blend (confirm.py)", full)):
        sim = book(sig, panel, None, 0.0)
        for period, lo, hi in (("DEV", None, DEV_END), ("CONFIRM", DEV_END, None)):
            m = score(sim, mkt, lo, hi)
            print(f"{period:<8} {name:<26} net_Sh {m['net_sharpe']:+.2f}  ret {m['net_ann_ret']:+.3f}  "
                  f"vol {m['ann_vol']:.3f}  dd {m['max_dd']:+.3f}")


if __name__ == "__main__":
    main()
