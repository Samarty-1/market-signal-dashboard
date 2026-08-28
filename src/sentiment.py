"""Financial headline sentiment: trained on Hugging Face data, applied to live news.

Model: TF-IDF + Logistic Regression, trained on `zeroshot/twitter-financial-news-sentiment`
(9,543 train / 2,388 validation rows of labeled financial tweets/headlines: 0=Bearish,
1=Bullish, 2=Neutral).

Why this isn't merged into the historical walk-forward direction model in model.py:
that dataset has no publish dates, so there's no way to align a headline's sentiment
to a specific ticker on a specific historical trading day. Backfilling it onto years of
price history would just be fabricating a time series that doesn't exist. Instead this
is applied live, to each ticker's current news (`yfinance`'s `.news`), as a real-time
signal shown alongside the (separately, honestly validated) price-based model -- not
folded into its backtested numbers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import yfinance as yf
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

LABEL_NAMES = {0: "bearish", 1: "bullish", 2: "neutral"}
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "sentiment_classifier.joblib"
METRICS_PATH = Path(__file__).resolve().parent.parent / "models" / "sentiment_metrics.json"


def train_sentiment_model() -> dict:
    """Train on the HF dataset and evaluate honestly on its held-out validation split."""
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    train_texts, train_labels = ds["train"]["text"], ds["train"]["label"]
    val_texts, val_labels = ds["validation"]["text"], ds["validation"]["label"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=5.0)),
    ])
    pipeline.fit(train_texts, train_labels)

    val_preds = pipeline.predict(val_texts)
    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "zeroshot/twitter-financial-news-sentiment",
        "train_rows": len(train_texts),
        "validation_rows": len(val_texts),
        "validation_accuracy": accuracy_score(val_labels, val_preds),
        "validation_f1_macro": f1_score(val_labels, val_preds, average="macro"),
        "classification_report": classification_report(
            val_labels, val_preds, target_names=[LABEL_NAMES[i] for i in range(3)], output_dict=True
        ),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


def load_sentiment_model() -> Pipeline:
    if not MODEL_PATH.exists():
        train_sentiment_model()
    return joblib.load(MODEL_PATH)


def fetch_live_headlines(ticker: str, limit: int = 10) -> list[str]:
    """Pull the most recent news headlines for a ticker from Yahoo Finance."""
    try:
        news = yf.Ticker(ticker).news or []
    except Exception:
        return []
    headlines = []
    for item in news[:limit]:
        content = item.get("content", item)
        title = content.get("title") if isinstance(content, dict) else None
        if title:
            headlines.append(title)
    return headlines


def score_ticker_sentiment(ticker: str, model: Pipeline | None = None) -> dict:
    """Score a ticker's current live headlines. Returns per-headline and aggregate sentiment."""
    model = model or load_sentiment_model()
    headlines = fetch_live_headlines(ticker)
    if not headlines:
        return {"ticker": ticker, "headlines": [], "n_headlines": 0, "sentiment_score": None, "label": "no_data"}

    probs = model.predict_proba(headlines)  # columns ordered [bearish, bullish, neutral]
    preds = model.predict(headlines)

    per_headline = [
        {"text": h, "label": LABEL_NAMES[p], "bearish": float(pr[0]), "bullish": float(pr[1]), "neutral": float(pr[2])}
        for h, p, pr in zip(headlines, preds, probs)
    ]
    # Aggregate score in [-1, 1]: mean(P(bullish) - P(bearish)) across headlines
    agg_score = float((probs[:, 1] - probs[:, 0]).mean())
    if agg_score > 0.15:
        agg_label = "bullish"
    elif agg_score < -0.15:
        agg_label = "bearish"
    else:
        agg_label = "neutral"

    return {
        "ticker": ticker,
        "headlines": per_headline,
        "n_headlines": len(headlines),
        "sentiment_score": agg_score,
        "label": agg_label,
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    metrics = train_sentiment_model()
    print(f"Validation accuracy: {metrics['validation_accuracy']:.3f}")
    print(f"Validation F1 (macro): {metrics['validation_f1_macro']:.3f}")
