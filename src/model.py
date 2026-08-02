"""Trains a next-day-direction classifier with walk-forward (time-respecting) validation.

Rows from every ticker are pooled into one model (ticker is a one-hot feature) since
per-ticker history here is too short to train a separate model well. Walk-forward
validation splits on calendar date, never on row index, so no ticker ever leaks
future dates into an earlier fold's training set.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from src.data_ingestion import fetch_prices
from src.features import FEATURE_COLUMNS, build_feature_dataset

LABEL_COLUMN = "label_next_day_up"
N_FOLDS = 5

CANDIDATE_MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42, class_weight="balanced"
    ),
}


def _build_pipeline(estimator) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), FEATURE_COLUMNS),
            ("ticker", OneHotEncoder(handle_unknown="ignore"), ["ticker"]),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def _fold_cutoffs(dates: pd.Series, n_folds: int = N_FOLDS) -> list[pd.Timestamp]:
    unique_dates = np.sort(dates.unique())
    # Skip the first ~40% as a minimum training window before the first fold.
    start = int(len(unique_dates) * 0.4)
    cut_points = np.linspace(start, len(unique_dates) - 1, n_folds + 1).astype(int)
    return [pd.Timestamp(unique_dates[i]) for i in cut_points]


def walk_forward_predictions(df: pd.DataFrame, estimator_name: str) -> pd.DataFrame:
    """Like walk_forward_evaluate, but returns the raw out-of-sample predictions
    (one row per test-fold observation) instead of aggregated metrics. This is what
    the backtest must use — scoring the final full-data-fit model on its own
    training history would leak the future into the "past" and inflate returns."""
    cutoffs = _fold_cutoffs(df["date"])
    predictions = []
    for fold_idx in range(len(cutoffs) - 1):
        train_end, test_end = cutoffs[fold_idx], cutoffs[fold_idx + 1]
        train = df[df["date"] <= train_end]
        test = df[(df["date"] > train_end) & (df["date"] <= test_end)]
        if train.empty or test.empty:
            continue

        pipeline = _build_pipeline(CANDIDATE_MODELS[estimator_name])
        pipeline.fit(train[FEATURE_COLUMNS + ["ticker"]], train[LABEL_COLUMN])
        proba = pipeline.predict_proba(test[FEATURE_COLUMNS + ["ticker"]])[:, 1]

        fold_predictions = test[["date", "ticker", "close", "return_1d", "next_day_return", LABEL_COLUMN]].copy()
        fold_predictions["fold"] = fold_idx
        fold_predictions["proba_up"] = proba
        predictions.append(fold_predictions)

    if not predictions:
        return pd.DataFrame(
            columns=["date", "ticker", "close", "return_1d", "next_day_return", LABEL_COLUMN, "fold", "proba_up"]
        )
    return pd.concat(predictions, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


def walk_forward_evaluate(df: pd.DataFrame, estimator_name: str) -> list[dict]:
    """Expanding-window walk-forward validation. Returns one metrics dict per fold."""
    cutoffs = _fold_cutoffs(df["date"])
    fold_results = []
    for fold_idx in range(len(cutoffs) - 1):
        train_end, test_end = cutoffs[fold_idx], cutoffs[fold_idx + 1]
        train = df[df["date"] <= train_end]
        test = df[(df["date"] > train_end) & (df["date"] <= test_end)]
        if train.empty or test.empty or test[LABEL_COLUMN].nunique() < 2:
            continue

        pipeline = _build_pipeline(CANDIDATE_MODELS[estimator_name])
        pipeline.fit(train[FEATURE_COLUMNS + ["ticker"]], train[LABEL_COLUMN])
        pred = pipeline.predict(test[FEATURE_COLUMNS + ["ticker"]])
        proba = pipeline.predict_proba(test[FEATURE_COLUMNS + ["ticker"]])[:, 1]

        fold_results.append(
            {
                "fold": fold_idx,
                "train_end": str(train_end.date()),
                "test_end": str(test_end.date()),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "accuracy": round(accuracy_score(test[LABEL_COLUMN], pred), 4),
                "roc_auc": round(roc_auc_score(test[LABEL_COLUMN], proba), 4),
                "precision": round(precision_score(test[LABEL_COLUMN], pred, zero_division=0), 4),
                "recall": round(recall_score(test[LABEL_COLUMN], pred, zero_division=0), 4),
            }
        )
    return fold_results


def _mean_metric(fold_results: list[dict], key: str) -> float:
    values = [f[key] for f in fold_results]
    return round(float(np.mean(values)), 4) if values else 0.0


def train_and_select_best(df: pd.DataFrame) -> tuple[str, Pipeline, dict]:
    """Runs walk-forward validation for every candidate, picks the best by mean ROC AUC,
    then refits that model on the full dataset for deployment."""
    comparison = {}
    for name in CANDIDATE_MODELS:
        folds = walk_forward_evaluate(df, name)
        comparison[name] = {
            "folds": folds,
            "mean_accuracy": _mean_metric(folds, "accuracy"),
            "mean_roc_auc": _mean_metric(folds, "roc_auc"),
            "mean_precision": _mean_metric(folds, "precision"),
            "mean_recall": _mean_metric(folds, "recall"),
        }

    best_name = max(comparison, key=lambda name: comparison[name]["mean_roc_auc"])
    final_pipeline = _build_pipeline(CANDIDATE_MODELS[best_name])
    final_pipeline.fit(df[FEATURE_COLUMNS + ["ticker"]], df[LABEL_COLUMN])

    metrics = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(df)),
        "tickers": sorted(df["ticker"].unique().tolist()),
        "best_model": best_name,
        "comparison": comparison,
    }
    return best_name, final_pipeline, metrics


def feature_importance(best_name: str, pipeline: Pipeline) -> dict:
    """Extracts a feature -> importance/coefficient mapping for the numeric features
    (ticker one-hot columns are excluded since they aren't comparable 1:1 with features)."""
    model = pipeline.named_steps["model"]
    if best_name == "random_forest":
        importances = model.feature_importances_[: len(FEATURE_COLUMNS)]
    else:
        importances = np.abs(model.coef_[0][: len(FEATURE_COLUMNS)])
    return dict(sorted(zip(FEATURE_COLUMNS, [round(float(v), 4) for v in importances]), key=lambda kv: -kv[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers (default: data_ingestion defaults)")
    parser.add_argument("--period", default="2y")
    parser.add_argument("--model-out", default="models/model.joblib")
    parser.add_argument("--metrics-out", default="models/metrics.json")
    parser.add_argument("--data-out", default="data/prices.csv")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    prices = fetch_prices(tickers) if tickers else fetch_prices()
    Path(args.data_out).parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(args.data_out, index=False)

    df = build_feature_dataset(prices)
    best_name, pipeline, metrics = train_and_select_best(df)
    metrics["feature_importance"] = feature_importance(best_name, pipeline)

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_out)
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2))

    print(f"Best model: {best_name} (mean ROC AUC {metrics['comparison'][best_name]['mean_roc_auc']})")
    print(f"Saved model to {args.model_out}, metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
