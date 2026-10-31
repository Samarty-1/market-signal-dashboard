"""Addendum A of PREREGISTRATION_v2.md: sector-neutral construction (H5) and
sector-relative features (H6). DEV decides; --confirm scores the adopted stack once.

    MSD_UNIVERSE=pit python -m research.improve_sectors [--confirm]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.blend_study import HORIZONS
from research.decompose import split  # noqa: F401  (kept importable for reconciliation)
from research.experiment import DEV_END, lgbm_regressor_lowsnr, run
from research.harness import FEATURES, SUFFIX, daily_rank_ic, ic_summary
from research.horizon_study import PRED_DIR, get_predictions, target_for
from research.improve import BAND, BORROW_BPS_YR, COST_BPS, HALFLIFE, blend, line, market_and_beta, score
from research.cost_study import fwd_wide
from research.panel import load_panel
from research.portfolio import signal_to_weights, simulate, smooth
from research.sectors import sector_wide

REL_BASE = ["return_1d", "return_5d", "mom_21d", "mom_12m_skip1m"]
SECTOR_FEATURES = [f"{f}_secrel" for f in REL_BASE] + ["sector_mom_21d", "sector_mom_126d"]
_ALIASES = {"Telecommunications Services": "Telecommunication Services"}


def sector_long(panel: pd.DataFrame) -> pd.Series:
    dates = panel.index.get_level_values("date").unique().sort_values()
    tick = panel.index.get_level_values("ticker").unique()
    wide = sector_wide(dates, tick).replace(_ALIASES)
    s = wide.stack()
    s.index.names = ["date", "ticker"]
    return s.reindex(panel.index)


def add_sector_features(panel: pd.DataFrame, sector: pd.Series) -> pd.DataFrame:
    """Same-day sector means over index members only (the panel holds members only),
    so every feature uses information available at that day's close."""
    key = [panel.index.get_level_values("date"), sector.fillna("UNKNOWN").to_numpy()]
    for f in REL_BASE:
        col = panel[f"raw_{f}"]
        panel[f"raw_{f}_secrel"] = (col - col.groupby(key).transform("mean")).astype("float32")
    for f, out in (("mom_21d", "sector_mom_21d"), ("mom_126d", "sector_mom_126d")):
        panel[f"raw_{out}"] = panel[f"raw_{f}"].groupby(key).transform("mean").astype("float32")
    return panel


def sector_predictions(panel: pd.DataFrame, h: int) -> pd.Series:
    p = PRED_DIR / f"sector_h{h}_full.parquet"
    if p.exists():
        return pd.read_parquet(p)["pred"]
    t0 = time.time()
    pred = run(panel, "raw_", target_for(h), lgbm_regressor_lowsnr, use_ticker=False,
               embargo=h + 2, n_seeds=3, date_end=None, neutralize=True,
               features=FEATURES + SECTOR_FEATURES)
    print(f"    trained sector h{h} in {time.time()-t0:.0f}s", flush=True)
    pred.to_frame("pred").to_parquet(p)
    return pred


def sector_neutral(w: pd.DataFrame, sector: pd.Series) -> pd.DataFrame:
    """Demean live target weights within each (date, sector), then restore gross.
    Sector-neutral implies dollar-neutral, since the sectors partition the book."""
    live = w.notna()
    long = w.stack()
    long.index.names = ["date", "ticker"]
    sec = sector.reindex(long.index).fillna("UNKNOWN")
    key = [long.index.get_level_values("date"), sec.to_numpy()]
    demeaned = (long - long.groupby(key).transform("mean")).unstack("ticker")
    demeaned = demeaned.reindex(index=w.index, columns=w.columns)
    gross = w.abs().sum(axis=1)
    out = demeaned.div(demeaned.abs().sum(axis=1).replace(0, np.nan), axis=0).mul(gross, axis=0)
    return out.where(live)


def book(sig, panel, sector=None):
    w = smooth(signal_to_weights(sig), HALFLIFE)
    if sector is not None:
        w = sector_neutral(w, sector)
    fwd = fwd_wide(panel, sig.index).reindex(index=w.index, columns=w.columns)
    return simulate(w, fwd, cost_bps=COST_BPS, band=BAND, borrow_bps_yr=BORROW_BPS_YR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args()

    panel = load_panel()
    _, mkt, _ = market_and_beta()
    sector = sector_long(panel)
    print(f"sector coverage {sector.notna().mean():.1%} of panel rows", flush=True)
    panel = add_sector_features(panel, sector)

    base = blend({h: get_predictions(panel, "raw_", h, "full") for h in HORIZONS})
    secf = blend({h: sector_predictions(panel, h) for h in HORIZONS})
    for name, sig in (("baseline blend", base), ("sector-feature blend", secf)):
        d = sig[sig.index.get_level_values("date") <= DEV_END]
        ic = ic_summary(daily_rank_ic(d, panel.loc[d.index, "fwd_ret_5"]))
        print(f"DEV IC vs 5d fwd  {name:<22} IC {ic['mean_ic']:+.4f}  ICIR {ic['icir_ann']:+.2f}  t {ic['t_stat']:+.1f}")

    configs = {
        "H0 baseline + borrow": (base, None),
        "H5 + sector-neutral": (base, sector),
        "H6 + sector features": (secf, None),
        "H5+H6": (secf, sector),
    }
    if args.confirm:
        a = json.loads(Path("research/adopted_v2_sectors.json").read_text())
        configs = {"H0 baseline + borrow": configs["H0 baseline + borrow"],
                   "ADOPTED STACK": (secf if a["H6"] else base, sector if a["H5"] else None)}

    out = {}
    periods = [("DEV", None, DEV_END)] + ([("CONFIRM", DEV_END, None)] if args.confirm else [])
    for period, lo, hi in periods:
        print(f"================ {period} ================")
        for name, (sig, sec) in configs.items():
            m = score(book(sig, panel, sec), mkt, lo, hi)
            out.setdefault(period, {})[name] = m
            print(line(name, m), flush=True)
    dest = Path(f"research/results_v2_sectors{'_confirm' if args.confirm else '_dev'}{SUFFIX}.json")
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
