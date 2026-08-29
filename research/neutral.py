"""Factor-neutralise the prediction, then ask what alpha is left.

`tilt_check` showed the smoothed long-short book was mostly a static exposure to
size / liquidity / volatility (long illiquid, high-vol, small; short liquid,
low-vol, large) -- a frozen version of the weights scored within 0.03 Sharpe of
the live signal. That is a factor bet, not stock selection, and it is doubly
misleading in a cost study because the illiquid names it loads on are exactly
the ones that cost the most to trade.

Standard fix: every day, regress the prediction on the factor exposures the book
is drifting into and keep the RESIDUAL. What remains is the part of the ranking
orthogonal to those factors -- genuine relative stock selection. If it still
pays after costs, that is a real result; if it does not, that is the honest
answer and the tilt was the whole story.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The exposures the unneutralised book actually carried, per tilt_check.
NEUTRAL_FACTORS = [
    "cs_log_dollar_volume",
    "cs_amihud_illiquidity",
    "cs_volatility_60d",
    "cs_volatility_20d",
]


def neutralize(pred: pd.Series, panel: pd.DataFrame,
               factors: list[str] = NEUTRAL_FACTORS) -> pd.Series:
    """Per-date OLS of the prediction on the factor exposures; return residuals.

    Done within each date (never pooled across dates), so no information moves
    backward or forward in time -- this is a pure cross-sectional projection
    using only that day's own data.
    """
    F = panel.loc[pred.index, factors]
    df = pd.concat([pred.rename("_p"), F], axis=1).dropna()

    def _resid(g: pd.DataFrame) -> pd.Series:
        X = np.column_stack([np.ones(len(g)), g[factors].to_numpy(dtype=float)])
        y = g["_p"].to_numpy(dtype=float)
        if len(g) <= X.shape[1] + 5:
            return pd.Series(np.nan, index=g.index)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return pd.Series(y - X @ beta, index=g.index)

    out = df.groupby(level="date", group_keys=False).apply(_resid)
    return out.dropna()


if __name__ == "__main__":
    from research.blend_study import HORIZONS, cs_rank
    from research.cost_study import evaluate, fwd_wide
    from research.horizon_study import get_predictions
    from research.panel import load_panel
    from research.portfolio import metrics, signal_to_weights, simulate

    panel = load_panel()
    preds = {h: get_predictions(panel, "raw_", h, "dev") for h in HORIZONS}
    idx = preds[1].index
    for h in HORIZONS:
        idx = idx.intersection(preds[h].index)
    blend = sum(cs_rank(preds[h].loc[idx]) for h in HORIZONS) / len(HORIZONS)

    neut = neutralize(blend, panel)
    print(f"neutralised {len(neut):,} of {len(blend):,} predictions\n")

    hdr = f"{'signal':<14} {'halflife':>8} {'gross_Sh':>9} {'net_Sh':>8} {'net_ret':>8} {'turn/d':>7}"
    print(hdr)
    print("-" * len(hdr))
    for name, sig in (("raw blend", blend), ("neutralised", neut)):
        for hl in (10, 21, 42, 84):
            m = evaluate(sig, panel, "rank", hl, 0.5)
            print(f"{name:<14} {hl:>8} {m['gross_sharpe']:>+9.2f} {m['net_sharpe']:>+8.2f} "
                  f"{m['net_ann_ret']:>+8.3f} {m['turnover_daily']:>7.4f}", flush=True)
        print()

    # Re-run the frozen-vs-live test on the neutralised signal: if the tilt is
    # gone, freezing the weights should now destroy the performance.
    w = signal_to_weights(neut)
    fwd = fwd_wide(panel, neut.index).reindex(index=w.index, columns=w.columns)
    frozen = w.expanding().mean()
    frozen = frozen.sub(frozen.mean(axis=1), axis=0)
    frozen = frozen.div(frozen.abs().sum(axis=1).replace(0, np.nan), axis=0)
    mf = metrics(simulate(frozen, fwd, cost_bps=10.0, band=0.5))
    ml = evaluate(neut, panel, "rank", 21, 0.5)
    print(f"frozen  (no live signal): net_Sh={mf['net_sharpe']:+.2f}")
    print(f"live    (halflife 21)   : net_Sh={ml['net_sharpe']:+.2f}")
    print("-> a large gap here means the alpha is live stock selection, not a tilt")
