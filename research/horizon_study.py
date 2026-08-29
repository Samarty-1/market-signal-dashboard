"""Does training on a longer, more persistent horizon buy a cheaper signal?

The diagnosis this tests: the repo's signal is fine (ICIR ~2.3, zero-cost
Sharpe ~1.7) but it predicts a 1-DAY return, so it must be re-traded every day,
and daily rebalancing costs more than the alpha. A signal trained to predict a
5/10/21-day return should decay more slowly, so it can be held longer for the
same alpha -- turnover falls without the signal having to get stronger.

Trains once per (feature set, horizon) and caches the out-of-sample predictions
so the portfolio sweeps afterwards are free.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from research.experiment import DEV_END, lgbm_regressor_lowsnr, run
from research.harness import daily_rank_ic, ic_summary
from research.panel import load_panel

PRED_DIR = Path("cache/preds")
FEATURE_SETS = ["raw_", "csz_"]
HORIZONS = [1, 5, 10, 21]


def target_for(h: int) -> str:
    return "y_rank" if h == 1 else f"y_rank_{h}"


def fwd_for(h: int) -> str:
    return "fwd_ret" if h == 1 else f"fwd_ret_{h}"


def pred_path(prefix: str, h: int, scope: str) -> Path:
    return PRED_DIR / f"{prefix}h{h}_{scope}.parquet"


def get_predictions(panel: pd.DataFrame, prefix: str, h: int, scope: str = "dev",
                    rebuild: bool = False) -> pd.Series:
    p = pred_path(prefix, h, scope)
    if p.exists() and not rebuild:
        return pd.read_parquet(p)["pred"]

    # Embargo must exceed the label horizon: a training row dated t carries a
    # label realised at t+h, so anything within h days of the test block would
    # peek into it. This is the leak the repo's `train <= T, test > T` split has
    # (at h=1), and it would be h times worse at longer horizons.
    t0 = time.time()
    pred = run(
        panel, prefix, target_for(h), lgbm_regressor_lowsnr,
        use_ticker=False, embargo=h + 2, n_seeds=3,
        date_end=DEV_END if scope == "dev" else None,
        neutralize=True,
    )
    print(f"    trained {prefix}h{h}_{scope} in {time.time()-t0:.0f}s", flush=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pred.to_frame("pred").to_parquet(p)
    return pred


def main() -> None:
    panel = load_panel()
    print(f"panel {len(panel):,} rows | DEV <= {DEV_END.date()}\n", flush=True)

    print(f"{'config':<16} {'IC(1d)':>9} {'ICIR':>7} {'IC(h)':>9} {'ICIR(h)':>8}")
    print("-" * 56)
    for prefix in FEATURE_SETS:
        for h in HORIZONS:
            pred = get_predictions(panel, prefix, h, "dev")
            sub = panel.loc[pred.index]
            ic1 = ic_summary(daily_rank_ic(pred, sub["fwd_ret"]))
            ich = ic_summary(daily_rank_ic(pred, sub[fwd_for(h)]))
            print(f"{prefix}h{h:<12} {ic1['mean_ic']:>+9.5f} {ic1['icir_ann']:>+7.2f} "
                  f"{ich['mean_ic']:>+9.5f} {ich['icir_ann']:>+8.2f}", flush=True)


if __name__ == "__main__":
    main()
