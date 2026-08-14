import pandas as pd

from src.categorize import categorize, categorize_description


def test_categorize_description_groceries():
    assert categorize_description("WHOLE FOODS MARKET #123") == "Groceries"


def test_categorize_description_dining():
    assert categorize_description("STARBUCKS COFFEE #4521") == "Dining"


def test_categorize_description_rent():
    assert categorize_description("Landlord Rent Payment") == "Rent/Mortgage"


def test_categorize_description_unmatched_is_other():
    assert categorize_description("XYZ Unknown Merchant 99") == "Other"


def test_categorize_description_case_insensitive():
    assert categorize_description("netflix subscription") == "Subscriptions"


def test_categorize_adds_column():
    df = pd.DataFrame({"description": ["Whole Foods Market", "Uber Trip", "Unknown Vendor"]})
    out = categorize(df)
    assert list(out["category"]) == ["Groceries", "Transport", "Other"]
