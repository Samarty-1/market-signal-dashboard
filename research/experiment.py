"""Purged walk-forward runner + the ablation ladder.

Protocol discipline (the thing this repo's README is rightly proud of, kept):
  DEV     = dates <= 2022-12-31  -- every design decision is made here
  CONFIRM = dates >  2022-12-31  -- looked at exactly once, at the very end

Nothing below tunes anything against CONFIRM.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.harness import FEATURES, daily_rank_ic, ic_summary, perf, quantile_spread
from research.panel import load_panel

DEV_END = pd.Timestamp("2022-12-31")
TEST_BLOCK = 126          # ~6 months per walk-forward test block
MIN_TRAIN_DAYS = 756      # ~3 years before the first prediction is made
EMBARGO = 2               # trading days dropped between train and test


def make_folds(dates: np.ndarray, block: int = TEST_BLOCK, min_train: int = MIN_TRAIN_DAYS):
    """Expanding-window folds: (train_end_idx, test_start_idx, test_end_idx)."""
    folds = []
    start = min_train
    while start < len(dates):
        end = min(start + block, len(dates))
        if end - start < block // 3:      # don't emit a runt final block
            break
        folds.append((start, start, end))
        start = end
    return folds


# --------------------------------------------------------------------------
# model factories
# --------------------------------------------------------------------------
def xgb_classifier_repo():
    """The estimator the repo currently selects (its own best_model)."""
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        eval_metric="logloss", random_state=42, n_jobs=-1, tree_method="hist",
    )


def lgbm_regressor_plain():
    """Lightly-regularised regressor -- the naive 'just swap to regression' step."""
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=300, num_leaves=31, learning_rate=0.05,
        random_state=42, n_jobs=-1, verbose=-1,
    )


def lgbm_regressor_lowsnr(seed: int = 42):
    """Regularised for a signal with IC ~0.03, i.e. ~99.9% of the variance in
    the target is noise. Shallow trees, large leaves, aggressive row/column
    subsampling and strong L2 all push the model toward the few broad, stable
    relationships instead of memorising noise. This is the standard shape of a
    production cross-sectional GBM, and it is very far from library defaults.
    """
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=500,
        num_leaves=15,            # shallow: few, broad interactions
        max_depth=4,
        learning_rate=0.02,       # slow learning + many trees
        min_child_samples=500,    # a leaf must average over ~1 full day's cross-section
        subsample=0.7, subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=20.0,
        random_state=seed, n_jobs=-1, verbose=-1,
    )


# --------------------------------------------------------------------------
# the runner
# --------------------------------------------------------------------------
def run(
    panel: pd.DataFrame,
    prefix: str,                 # "raw_" or "cs_"
    target: str,                 # "y_binary" or "y_rank"
    model_fn,
    use_ticker: bool = False,
    embargo: int = EMBARGO,
    n_seeds: int = 1,
    date_end: pd.Timestamp | None = None,
    neutralize: bool = False,
    verbose: bool = False,
) -> pd.Series:
    """Returns out-of-sample predictions indexed by (date, ticker)."""
    df = panel if date_end is None else panel[panel.index.get_level_values("date") <= date_end]
    feat_cols = [f"{prefix}{f}" for f in FEATURES]

    dates = np.sort(df.index.get_level_values("date").unique())
    date_arr = df.index.get_level_values("date")

    X_all = df[feat_cols]
    if use_ticker:
        codes = pd.Categorical(df.index.get_level_values("ticker")).codes
        X_all = X_all.assign(_ticker=codes.astype("int16"))
    y_all = df[target]

    preds = []
    for k, (tr_end_i, te_start_i, te_end_i) in enumerate(make_folds(dates)):
        # PURGE: the label on date t is realised on t+1, so training rows dated
        # within `embargo` days of the test block would have labels that peek
        # into it. Drop them. The repo's split (train <= T, test > T) leaks
        # exactly one day this way on every fold.
        train_end_date = dates[tr_end_i - embargo]
        te_lo, te_hi = dates[te_start_i], dates[te_end_i - 1]

        tr = date_arr <= train_end_date
        te = (date_arr >= te_lo) & (date_arr <= te_hi)
        if tr.sum() == 0 or te.sum() == 0:
            continue

        Xtr, ytr, Xte = X_all[tr], y_all[tr], X_all[te]

        fold_pred = np.zeros(len(Xte))
        for s in range(n_seeds):
            m = model_fn(42 + s) if n_seeds > 1 else model_fn()
            m.fit(Xtr, ytr)
            if hasattr(m, "predict_proba"):
                fold_pred += m.predict_proba(Xte)[:, 1]
            else:
                fold_pred += m.predict(Xte)
        fold_pred /= n_seeds

        preds.append(pd.Series(fold_pred, index=Xte.index))
        if verbose:
            print(f"    fold {k}: train<= {train_end_date.date()} "
                  f"test {te_lo.date()}..{te_hi.date()} "
                  f"n_tr={tr.sum():,} n_te={te.sum():,}", flush=True)

    if not preds:
        return pd.Series(dtype=float)
    out = pd.concat(preds)

    if neutralize:
        # Re-rank predictions within each date. A dollar-neutral book only ever
        # acts on the ORDER of today's scores, so making that order the actual
        # signal removes any day-level drift in the raw score scale.
        out = out.groupby(level="date").rank(pct=True) * 2 - 1
    return out


def evaluate(pred: pd.Series, panel: pd.DataFrame, label: str) -> dict:
    fwd = panel.loc[pred.index, "fwd_ret"]
    ic = daily_rank_ic(pred, fwd)
    spread = quantile_spread(pred, fwd, n_q=5)
    res = {"label": label, **ic_summary(ic), "q5_spread": perf(spread)}
    return res


def fmt(res: dict) -> str:
    s = res["q5_spread"]
    return (f"{res['label']:<38} IC={res['mean_ic']:+.5f}  ICIR={res['icir_ann']:+.2f}  "
            f"t={res['t_stat']:+.1f}  hit={res['hit_rate']:.3f}  "
            f"Q5spread Sharpe={s['sharpe']:+.2f} ret={s['ann_return']:+.3f}  days={res['n_days']}")


LADDER = [
    # label,                              prefix, target,     model,                    ticker, embargo, seeds, neutralize
    ("0 baseline (repo: raw+ticker,bin)",  "raw_", "y_binary", xgb_classifier_repo,      True,   0,       1,     False),
    ("1 + purge/embargo",                  "raw_", "y_binary", xgb_classifier_repo,      True,   EMBARGO, 1,     False),
    ("2 + drop ticker identity",           "raw_", "y_binary", xgb_classifier_repo,      False,  EMBARGO, 1,     False),
    ("3 + cross-sectional features",       "cs_",  "y_binary", xgb_classifier_repo,      False,  EMBARGO, 1,     False),
    ("4 + rank target (regression)",       "cs_",  "y_rank",   lgbm_regressor_plain,     False,  EMBARGO, 1,     False),
    ("5 + low-SNR regularisation",         "cs_",  "y_rank",   lgbm_regressor_lowsnr,    False,  EMBARGO, 1,     False),
    ("6 + 5-seed ensemble + neutralise",   "cs_",  "y_rank",   lgbm_regressor_lowsnr,    False,  EMBARGO, 5,     True),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="all", help="comma-separated step indices, or 'all'")
    ap.add_argument("--out", default="research/results_dev.json")
    args = ap.parse_args()

    panel = load_panel()
    print(f"panel: {len(panel):,} rows, DEV = dates <= {DEV_END.date()}\n", flush=True)

    want = None if args.steps == "all" else {int(x) for x in args.steps.split(",")}
    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    for i, (label, prefix, target, mfn, tick, emb, seeds, neut) in enumerate(LADDER):
        if want is not None and i not in want:
            continue
        t0 = time.time()
        pred = run(panel, prefix, target, mfn, use_ticker=tick, embargo=emb,
                   n_seeds=seeds, date_end=DEV_END, neutralize=neut)
        res = evaluate(pred, panel, label)
        res["secs"] = round(time.time() - t0, 1)
        results[str(i)] = res
        print(fmt(res) + f"   [{res['secs']}s]", flush=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
