"""Addendum B of PREREGISTRATION_v2.md: unfitted reversal composite (H7a) and its
50/50 blend with the ML signal (H7b). DEV decides; --confirm scores the winner once.

    MSD_UNIVERSE=pit python -m research.improve_composite [--confirm]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.blend_study import HORIZONS, cs_rank
from research.experiment import DEV_END
from research.harness import SUFFIX, daily_rank_ic, ic_summary
from research.horizon_study import get_predictions
from research.improve import blend, line, market_and_beta, score
from research.improve_sectors import add_sector_features, book, sector_long
from research.panel import load_panel

# (feature, sign) -- the literature's short-term-reversal direction, never fitted
COMPOSITE = [("return_1d", -1), ("return_5d", -1), ("mom_21d", -1), ("sector_mom_21d", -1)]


def composite(panel):
    parts = [cs_rank(sign * panel[f"raw_{f}"].dropna()) for f, sign in COMPOSITE]
    idx = parts[0].index
    for p in parts[1:]:
        idx = idx.intersection(p.index)
    return sum(p.loc[idx] for p in parts) / len(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args()

    panel = load_panel()
    _, mkt, _ = market_and_beta()
    panel = add_sector_features(panel, sector_long(panel))

    ml = blend({h: get_predictions(panel, "raw_", h, "full") for h in HORIZONS})
    comp = composite(panel).loc[lambda s: s.index.isin(ml.index)]
    idx = ml.index.intersection(comp.index)
    mixed = (cs_rank(ml.loc[idx]) + cs_rank(comp.loc[idx])) / 2

    for name, sig in (("ML blend", ml), ("H7a composite", comp), ("H7b 50/50", mixed)):
        d = sig[sig.index.get_level_values("date") <= DEV_END]
        ic = ic_summary(daily_rank_ic(d, panel.loc[d.index, "fwd_ret_5"]))
        print(f"DEV IC vs 5d fwd  {name:<14} IC {ic['mean_ic']:+.4f}  ICIR {ic['icir_ann']:+.2f}  t {ic['t_stat']:+.1f}")

    configs = {"H0 baseline + borrow": ml, "H7a composite": comp, "H7b 50/50": mixed}
    if args.confirm:
        win = json.loads(Path("research/adopted_v2_composite.json").read_text())["winner"]
        configs = {"H0 baseline + borrow": ml, f"ADOPTED {win}": configs[win]}

    out = {}
    periods = [("DEV", None, DEV_END)] + ([("CONFIRM", DEV_END, None)] if args.confirm else [])
    for period, lo, hi in periods:
        print(f"================ {period} ================")
        for name, sig in configs.items():
            m = score(book(sig, panel), mkt, lo, hi)
            out.setdefault(period, {})[name] = m
            print(line(name, m), flush=True)
    dest = Path(f"research/results_v2_composite{'_confirm' if args.confirm else '_dev'}{SUFFIX}.json")
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
