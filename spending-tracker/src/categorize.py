"""Rule-based transaction categorization by keyword matching on description."""
from __future__ import annotations

import pandas as pd

# Ordered: first matching category wins, so put more specific rules earlier.
CATEGORY_RULES: dict[str, list[str]] = {
    "Groceries": ["grocery", "supermarket", "whole foods", "trader joe", "safeway", "kroger"],
    "Dining": ["restaurant", "cafe", "coffee", "starbucks", "doordash", "ubereats", "grubhub", "pizza"],
    "Transport": ["uber", "lyft", "gas station", "shell", "chevron", "parking", "transit", "metro"],
    "Utilities": ["electric", "water bill", "gas bill", "internet", "comcast", "verizon", "at&t", "utility"],
    "Rent/Mortgage": ["rent", "mortgage", "landlord", "property management"],
    "Subscriptions": ["netflix", "spotify", "hulu", "disney+", "amazon prime", "subscription", "gym membership"],
    "Shopping": ["amazon", "target", "walmart", "best buy", "mall", "clothing", "ebay"],
    "Healthcare": ["pharmacy", "cvs", "walgreens", "doctor", "clinic", "hospital", "dental"],
    "Entertainment": ["movie", "cinema", "concert", "ticketmaster", "steam", "playstation", "xbox"],
    "Travel": ["airline", "airbnb", "hotel", "expedia", "flight"],
    "Income": ["payroll", "salary", "paycheck", "direct deposit", "refund"],
}
FALLBACK_CATEGORY = "Other"


def categorize_description(description: str) -> str:
    """Return the first matching category for a description, else 'Other'."""
    text = description.lower()
    for category, keywords in CATEGORY_RULES.items():
        if any(keyword in text for keyword in keywords):
            return category
    return FALLBACK_CATEGORY


def categorize(df: pd.DataFrame, description_col: str = "description") -> pd.DataFrame:
    """Return a copy of `df` with a `category` column added."""
    out = df.copy()
    out["category"] = out[description_col].apply(categorize_description)
    return out
