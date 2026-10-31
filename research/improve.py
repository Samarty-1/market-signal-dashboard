"""Pre-registered v2 experiments (see research/PREREGISTRATION_v2.md).

    MSD_UNIVERSE=pit python -m research.improve            # DEV only: H0-H3
    MSD_UNIVERSE=pit python -m research.improve --confirm  # once, adopted stack only

H0  borrow fee on the short leg (accounting fix)
H1  beta-neutral construction
H2  residual-return features (residual reversal / residual momentum)
H3  volatility-target overlay

Every number used to decide anything is computed on DEV (<= 2022-12-31).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.blend_study import HORIZONS, cs_rank
from research.cost_study import fwd_wide
from research.experiment import DEV_END, lgbm_regressor_lowsnr, run
from research.harness import FEATURES, SUFFIX, daily_rank_ic, ic_summary, load_wide, membership_mask
from research.horizon_study import PRED_DIR, get_predictions, target_for
from research.panel import load_panel
from research.portfolio import metrics, signal_to_weights, simulate, smooth

HALFLIFE, BAND, COST_BPS = 42, 0.5, 10.0
BORROW_BPS_YR = 50.0
BETA_WIN, BETA_MIN = 252, 126
VOL_TARGET, VOL_WIN, MAX_LEV = 0.10, 63, 4.0

RESID_FEATURES = ["res_1d", "res_5d", "res_21d", "res_mom_12_1", "res_vol_60d", "beta_252d"]


# --------------------------------------------------------------------------
# market, beta, residual returns -- all causal (rolling windows end at t)
# --------------------------------------------------------------------------
def market_and_beta() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Daily returns, the equal-weight universe return, and each name's trailing beta.

    The market is the equal-weight mean over names that are index members and
    priced on that date -- the same universe the book trades, so beta-neutral
    means neutral to the thing the book's long/short legs are drawn from.
    """
    close, _ = load_wide()
    ret = close.pct_change(fill_method=None)
    member = membership_mask(close.index, close.columns)
    in_univ = ret.notna() if member is None else (ret.notna() & member)
    mkt = ret.where(in_univ).mean(axis=1)

    m = pd.DataFrame(np.repeat(mkt.to_numpy()[:, None], ret.shape[1], axis=1),
                     index=ret.index, columns=ret.columns).where(ret.notna())
    mean_xy = (ret * m).rolling(BETA_WIN, min_periods=BETA_MIN).mean()
    mean_x = ret.rolling(BETA_WIN, min_periods=BETA_MIN).mean()
    mean_y = m.rolling(BETA_WIN, min_periods=BETA_MIN).mean()
    var_y = (m * m).rolling(BETA_WIN, min_periods=BETA_MIN).mean() - mean_y ** 2
    beta = (mean_xy - mean_x * mean_y) / var_y.replace(0, np.nan)
    return ret, mkt, beta


def residual_features(ret: pd.DataFrame, mkt: pd.Series, beta: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Residual return e(t) = r(t) - beta(t-1) * mkt(t): yesterday's beta, so the
    residual on day t never uses day t's own return to estimate the hedge."""
    e = ret - beta.shift(1).mul(mkt, axis=0)
    f = {
        "res_1d": e,
        "res_5d": e.rolling(5).sum(),
        "res_21d": e.rolling(21).sum(),
        "res_vol_60d": e.rolling(60).std(),
        "beta_252d": beta,
    }
    # Blitz/Huij/Martens residual momentum: 12-1 month residual return, scaled
    # by residual volatility so a quiet steady winner outranks a noisy one.
    cum = e.shift(21).rolling(231, min_periods=150).sum()
    f["res_mom_12_1"] = cum / (e.rolling(252, min_periods=150).std() * np.sqrt(231))
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in f.items()}


def augment(panel: pd.DataFrame, feats: dict[str, pd.DataFrame]) -> pd.DataFrame:
    idx = panel.index
    for k, wide in feats.items():
        s = wide.stack()
        s.index.names = ["date", "ticker"]
        panel[f"raw_{k}"] = s.reindex(idx).astype("float32")
    return panel


# --------------------------------------------------------------------------
# signals
# --------------------------------------------------------------------------
def resid_predictions(panel: pd.DataFrame, h: int) -> pd.Series:
    p = PRED_DIR / f"resid_h{h}_full.parquet"
    if p.exists():
        return pd.read_parquet(p)["pred"]
    t0 = time.time()
    pred = run(panel, "raw_", target_for(h), lgbm_regressor_lowsnr, use_ticker=False,
               embargo=h + 2, n_seeds=3, date_end=None, neutralize=True,
               features=FEATURES + RESID_FEATURES)
    print(f"    trained resid h{h} in {time.time()-t0:.0f}s", flush=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pred.to_frame("pred").to_parquet(p)
    return pred


def blend(preds: dict[int, pd.Series]) -> pd.Series:
    idx = preds[HORIZONS[0]].index
    for h in HORIZONS:
        idx = idx.intersection(preds[h].index)
    return sum(cs_rank(preds[h].loc[idx]) for h in HORIZONS) / len(HORIZONS)


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------
def beta_neutral(w: pd.DataFrame, beta: pd.DataFrame) -> pd.DataFrame:
    """Remove each day's beta exposure from the target weights.

    w is already dollar-neutral (sum w = 0). Project it orthogonal to the
    cross-sectionally demeaned beta vector, which keeps sum w = 0 and makes
    sum w*beta = 0, then restore the original gross so leverage is unchanged.
    """
    b = beta.reindex(index=w.index, columns=w.columns)
    live = w.notna()
    b = b.where(live)
    b = b.T.fillna(b.mean(axis=1)).T.where(live)          # unknown beta -> average
    bc = b.sub(b.mean(axis=1), axis=0).fillna(0.0)
    W = w.fillna(0.0)
    k = (W * bc).sum(axis=1) / (bc * bc).sum(axis=1).replace(0, np.nan)
    out = W - bc.mul(k.fillna(0.0), axis=0)
    out = out.sub(out.where(live).mean(axis=1), axis=0)
    gross = W.abs().sum(axis=1)
    out = out.div(out.where(live).abs().sum(axis=1).replace(0, np.nan), axis=0).mul(gross, axis=0)
    return out.where(live)


def book(sig: pd.Series, panel: pd.DataFrame, beta: pd.DataFrame | None,
         borrow: float) -> pd.DataFrame:
    w = smooth(signal_to_weights(sig), HALFLIFE)
    if beta is not None:
        w = beta_neutral(w, beta)
    fwd = fwd_wide(panel, sig.index).reindex(index=w.index, columns=w.columns)
    return simulate(w, fwd, cost_bps=COST_BPS, band=BAND, borrow_bps_yr=borrow)


def vol_target(sim: pd.DataFrame) -> pd.DataFrame:
    """Scale the whole book to VOL_TARGET using realised vol through yesterday.

    Re-sizing the book trades |delta scale| x gross of every position, so that
    is charged at the same one-way cost as any other turnover.
    """
    net = sim["net_return"]
    rv = net.rolling(VOL_WIN, min_periods=VOL_WIN // 2).std().shift(1) * np.sqrt(252)
    scale = (VOL_TARGET / rv).clip(upper=MAX_LEV).fillna(1.0)
    rescale_cost = scale.diff().abs().fillna(0.0) * COST_BPS / 10_000.0
    out = sim.copy()
    out["gross_return"] = sim["gross_return"] * scale
    out["cost"] = sim["cost"] * scale + rescale_cost
    out["net_return"] = out["gross_return"] - out["cost"]
    out["turnover"] = sim["turnover"] * scale + scale.diff().abs().fillna(0.0)
    out["scale"] = scale
    return out


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def score(sim: pd.DataFrame, mkt: pd.Series, lo=None, hi=None) -> dict:
    s = sim
    if lo is not None:
        s = s[s.index > lo]
    if hi is not None:
        s = s[s.index <= hi]
    m = metrics(s)
    # Market beta of the realised book: P&L on day t is earned over t -> t+1.
    mk = mkt.shift(-1).reindex(s.index)
    ok = mk.notna()
    m["mkt_beta"] = round(float(np.polyfit(mk[ok], s["net_return"][ok], 1)[0]), 3)
    yrs = s["net_return"].groupby(s.index.year)
    m["by_year"] = {int(y): round(float(g.mean() / g.std() * np.sqrt(252)), 2) for y, g in yrs if len(g) > 50}
    if "scale" in s:
        m["avg_leverage"] = round(float(s["scale"].mean()), 2)
    return m


def line(name: str, m: dict) -> str:
    yrs = " ".join(f"{y % 100:02d}:{v:+.2f}" for y, v in m["by_year"].items())
    return (f"{name:<34} net_Sh {m['net_sharpe']:+.2f}  ret {m['net_ann_ret']:+.3f}  vol {m['ann_vol']:.3f}  "
            f"dd {m['max_dd']:+.3f}  beta {m['mkt_beta']:+.3f}  turn {m['turnover_daily']:.4f}  | {yrs}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args()

    panel = load_panel()
    ret, mkt, beta = market_and_beta()
    panel = augment(panel, residual_features(ret, mkt, beta))
    print(f"panel {len(panel):,} rows; residual features added\n", flush=True)

    base_preds = {h: get_predictions(panel, "raw_", h, "full") for h in HORIZONS}
    res_preds = {h: resid_predictions(panel, h) for h in HORIZONS}
    base_sig, res_sig = blend(base_preds), blend(res_preds)

    # Signal quality on DEV, before any portfolio is built.
    for name, sig in (("baseline blend", base_sig), ("residual blend", res_sig)):
        d = sig[sig.index.get_level_values("date") <= DEV_END]
        ic = ic_summary(daily_rank_ic(d, panel.loc[d.index, "fwd_ret_5"]))
        print(f"DEV IC vs 5d fwd  {name:<16} IC {ic['mean_ic']:+.4f}  ICIR {ic['icir_ann']:+.2f}  t {ic['t_stat']:+.1f}")
    print()

    configs = {
        "baseline (no borrow)": (base_sig, None, 0.0, False),
        "H0 baseline + borrow": (base_sig, None, BORROW_BPS_YR, False),
        "H1 + beta-neutral": (base_sig, beta, BORROW_BPS_YR, False),
        "H2 + residual features": (res_sig, None, BORROW_BPS_YR, False),
        "H1+H2": (res_sig, beta, BORROW_BPS_YR, False),
        "H3 vol-target on H0": (base_sig, None, BORROW_BPS_YR, True),
    }
    if args.confirm:
        adopted = json.loads(Path("research/adopted_v2.json").read_text())
        configs = {k: v for k, v in configs.items() if k in adopted["compare"]}
        configs["ADOPTED STACK"] = (res_sig if adopted["H2"] else base_sig,
                                    beta if adopted["H1"] else None, BORROW_BPS_YR, adopted["H3"])

    out = {}
    for period, (lo, hi) in ([("DEV", (None, DEV_END))]
                             + ([("CONFIRM", (DEV_END, None))] if args.confirm else [])):
        print(f"================ {period} ================")
        for name, (sig, b, borrow, vt) in configs.items():
            sim = book(sig, panel, b, borrow)
            if vt:
                sim = vol_target(sim)
            m = score(sim, mkt, lo, hi)
            out.setdefault(period, {})[name] = m
            print(line(name, m), flush=True)
        print()

    dest = Path(f"research/results_v2{'_confirm' if args.confirm else '_dev'}{SUFFIX}.json")
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
