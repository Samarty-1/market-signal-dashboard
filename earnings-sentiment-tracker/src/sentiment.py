"""Score headline sentiment with VADER (rule-based, no model download needed)."""
from __future__ import annotations

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def score_text(text: str) -> float:
    """Return VADER's compound sentiment score in [-1, 1] for a single string."""
    if not text:
        return 0.0
    return _analyzer.polarity_scores(text)["compound"]


def score_headlines(df: pd.DataFrame, text_col: str = "title") -> pd.DataFrame:
    """Return a copy of `df` with a `sentiment` column (VADER compound score)."""
    out = df.copy()
    out["sentiment"] = out[text_col].apply(score_text)
    return out
