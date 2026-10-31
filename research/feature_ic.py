"""Diagnostic, DEV only: is the information in the new features weak, or is the
model mis-using it? Daily rank IC of each single raw feature vs 5-day forward
return. Adopts nothing."""
from __future__ import annotations

import pandas as pd

from research.experiment import DEV_END
from research.harness import daily_rank_ic, ic_summary
from research.improve import augment, market_and_beta, residual_features
from research.improve_sectors import add_sector_features, sector_long
from research.panel import load_panel

COLS = ["return_1d", "return_1d_secrel", "res_1d",
        "return_5d", "return_5d_secrel", "res_5d",
        "mom_21d", "mom_21d_secrel", "res_21d",
        "mom_12m_skip1m", "mom_12m_skip1m_secrel", "res_mom_12_1",
        "sector_mom_21d", "sector_mom_126d", "volatility_60d", "beta_252d", "log_dollar_volume"]


def main() -> None:
    panel = load_panel()
    ret, mkt, beta = market_and_beta()
    panel = augment(panel, residual_features(ret, mkt, beta))
    panel = add_sector_features(panel, sector_long(panel))
    dev = panel[panel.index.get_level_values("date") <= DEV_END]
    # thin to every 5th date: 5-day forward returns overlap, so daily ICs are not independent
    keep = dev.index.get_level_values("date").unique()[::5]
    dev = dev[dev.index.get_level_values("date").isin(keep)]
    rows = []
    for c in COLS:
        s = ic_summary(daily_rank_ic(dev[f"raw_{c}"], dev["fwd_ret_5"]))
        rows.append({"feature": c, "IC": s["mean_ic"], "t": s["t_stat"], "hit": s["hit_rate"]})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
