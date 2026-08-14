import pandas as pd

from src.sentiment import score_headlines, score_text


def test_score_text_positive():
    assert score_text("Great quarter, profits jump and outlook is excellent") > 0.3


def test_score_text_negative():
    assert score_text("Company misses estimates badly, shares plunge") < -0.3


def test_score_text_empty():
    assert score_text("") == 0.0


def test_score_headlines_adds_column():
    df = pd.DataFrame({"title": ["Great quarter, profits jump", "Disappointing results, losses widen"]})
    out = score_headlines(df)
    assert "sentiment" in out.columns
    assert out.loc[0, "sentiment"] > 0
    assert out.loc[1, "sentiment"] < 0
