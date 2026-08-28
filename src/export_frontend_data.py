"""Exports a static JSON bundle for the frontend/ static site.

Bridges the offline pipeline (src/model.py, src/backtest.py) to the
Nocturne-styled static frontend in frontend/: reads the out-of-sample
backtest predictions and the latest registry model, and writes
frontend/data/dashboard_data.json with real per-ticker OHLCV+signal rows,
model comparison metrics, and one fresh live inference per ticker.

Deliberately not a live backend — matches the same "no extra infrastructure"
scope already used for the model registry (src/registry.py). The frontend
re-fetches this file on demand (its "Refresh data" button); a new "live"
picture requires re-running this export (done daily by the scheduled retrain,
or on demand: `python -m src.export_frontend_data`).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import registry, sentiment
from src.data_ingestion import fetch_prices
from src.features import FEATURE_COLUMNS, build_feature_dataset

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
BACKTEST_PREDICTIONS_PATH = ROOT / "reports" / "backtest_predictions.csv"
OUT_PATH = ROOT / "frontend" / "data" / "dashboard_data.json"

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
}


def _clean(value):
    """NaN isn't valid JSON — turn it into null so a browser's JSON.parse doesn't choke."""
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def _series_rows(predictions: pd.DataFrame) -> dict[str, list[dict]]:
    series: dict[str, list[dict]] = {}
    for ticker, group in predictions.groupby("ticker"):
        group = group.sort_values("date")
        rows = []
        for _, r in group.iterrows():
            rows.append(
                {
                    "date": r["date"].strftime("%Y-%m-%d"),
                    "open": _clean(round(float(r["open"]), 4)),
                    "high": _clean(round(float(r["high"]), 4)),
                    "low": _clean(round(float(r["low"]), 4)),
                    "close": _clean(round(float(r["close"]), 4)),
                    "volume": _clean(int(r["volume"])) if pd.notna(r["volume"]) else None,
                    "probaUp": _clean(round(float(r["proba_up"]), 6)),
                    "nextDayReturn": _clean(round(float(r["next_day_return"]), 6)) if pd.notna(r["next_day_return"]) else None,
                }
            )
        series[ticker] = rows
    return series


def _live_snapshot(model, tickers: list[str]) -> dict[str, dict]:
    """One fresh live inference per ticker — the same call the old Streamlit
    dashboard made per page-load, just precomputed here instead since a static
    site has no server to run it request-time."""
    live: dict[str, dict] = {}
    for ticker in tickers:
        try:
            prices = fetch_prices([ticker], period="6mo")
            feats = build_feature_dataset(prices)
            latest = feats.iloc[[-1]]
            proba = float(model.predict_proba(latest[FEATURE_COLUMNS + ["ticker"]])[0, 1])
            live[ticker] = {
                "price": round(float(latest["close"].iloc[0]), 4),
                "proba": round(proba, 6),
                "asOf": str(latest["date"].iloc[0].date()),
            }
        except Exception as exc:  # a single ticker's live fetch failing shouldn't kill the whole export
            live[ticker] = {"price": None, "proba": None, "asOf": None, "error": str(exc)}
    return live


def _cross_sectional_snapshot() -> dict | None:
    """Honest comparison: baseline next-day-direction AUC vs the cross-sectional
    "beat the day's median" target (see src/features.add_cross_sectional_label).
    Reads whatever the daily retrain most recently saved -- doesn't retrain here."""
    cs_dir = MODELS_DIR / "cross_sectional"
    try:
        _, cs_metrics = registry.load_latest(cs_dir)
    except FileNotFoundError:
        return None
    _, base_metrics = registry.load_latest(MODELS_DIR)

    return {
        "baselineBestModel": base_metrics["best_model"],
        "baselineMeanRocAuc": base_metrics["comparison"][base_metrics["best_model"]]["mean_roc_auc"],
        "crossSectionalBestModel": cs_metrics["best_model"],
        "crossSectionalMeanRocAuc": cs_metrics["comparison"][cs_metrics["best_model"]]["mean_roc_auc"],
        "trainedAtUtc": cs_metrics["trained_at_utc"],
    }


def _sentiment_snapshot(tickers: list[str]) -> dict:
    """Live headline sentiment per ticker, scored by a classifier trained on the
    Hugging Face `zeroshot/twitter-financial-news-sentiment` dataset. This is
    NOT backtested or folded into modelMetrics/live proba above -- see
    src/sentiment.py for why (no dated historical headlines to align to
    trading days). It's a separate, live-only read shown alongside the model."""
    model = sentiment.load_sentiment_model()
    per_ticker = {}
    for ticker in tickers:
        result = sentiment.score_ticker_sentiment(ticker, model=model)
        per_ticker[ticker] = {
            "score": result["sentiment_score"],
            "label": result["label"],
            "nHeadlines": result["n_headlines"],
            "topHeadlines": [h["text"] for h in result["headlines"][:3]],
        }

    model_metrics = {}
    if sentiment.METRICS_PATH.exists():
        raw = json.loads(sentiment.METRICS_PATH.read_text())
        model_metrics = {
            "dataset": raw["dataset"],
            "trainRows": raw["train_rows"],
            "validationRows": raw["validation_rows"],
            "validationAccuracy": raw["validation_accuracy"],
            "validationF1Macro": raw["validation_f1_macro"],
        }

    return {"perTicker": per_ticker, "classifierMetrics": model_metrics}


def main() -> None:
    if not BACKTEST_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"{BACKTEST_PREDICTIONS_PATH} not found — run `python -m src.backtest` first")

    model, metrics = registry.load_latest(MODELS_DIR)
    predictions = pd.read_csv(BACKTEST_PREDICTIONS_PATH, parse_dates=["date"])
    tickers = sorted(predictions["ticker"].unique().tolist())

    bundle = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "tickers": tickers,
        "series": _series_rows(predictions),
        "live": _live_snapshot(model, tickers),
        "sentiment": _sentiment_snapshot(tickers),
        "crossSectional": _cross_sectional_snapshot(),
        "modelMetrics": {
            "trainedAtUtc": metrics["trained_at_utc"],
            "nRows": metrics["n_rows"],
            "bestModel": metrics["best_model"],
            "registryVersion": registry.latest_version(MODELS_DIR),
            "comparison": {
                name: {
                    "meanAccuracy": vals["mean_accuracy"],
                    "meanRocAuc": vals["mean_roc_auc"],
                    "meanPrecision": vals["mean_precision"],
                    "meanRecall": vals["mean_recall"],
                }
                for name, vals in metrics["comparison"].items()
            },
            "featureImportance": metrics["feature_importance"],
            "displayNames": DISPLAY_NAMES,
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(bundle, indent=2, allow_nan=False))
    total_rows = sum(len(v) for v in bundle["series"].values())
    print(f"Wrote {OUT_PATH} ({len(tickers)} tickers, {total_rows} total rows)")


if __name__ == "__main__":
    main()
