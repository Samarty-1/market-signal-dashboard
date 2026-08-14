"""Compare actual spend per category against monthly budgets."""
from __future__ import annotations

import pandas as pd

DEFAULT_BUDGETS: dict[str, float] = {
    "Groceries": 500,
    "Dining": 200,
    "Transport": 150,
    "Utilities": 250,
    "Rent/Mortgage": 1800,
    "Subscriptions": 60,
    "Shopping": 200,
    "Healthcare": 100,
    "Entertainment": 100,
    "Travel": 150,
    "Other": 100,
}


def monthly_spend_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Return spend (positive numbers) per category per calendar month.

    Only negative `amount` rows (money out) count as spend; income rows are
    excluded. Columns: month (Period), category, spend.
    """
    spend = df[df["amount"] < 0].copy()
    spend["month"] = spend["date"].dt.to_period("M")
    grouped = (
        spend.groupby(["month", "category"])["amount"]
        .sum()
        .abs()
        .reset_index()
        .rename(columns={"amount": "spend"})
    )
    return grouped


def budget_status(monthly_spend: pd.DataFrame, budgets: dict[str, float] | None = None) -> pd.DataFrame:
    """For the most recent month in `monthly_spend`, compare spend to budget.

    Returns columns: category, spend, budget, remaining, over_budget (bool).
    Categories with no spend but a budget still appear (spend=0).
    """
    budgets = budgets or DEFAULT_BUDGETS
    if monthly_spend.empty:
        latest_rows = pd.DataFrame(columns=["category", "spend"])
    else:
        latest_month = monthly_spend["month"].max()
        latest_rows = monthly_spend[monthly_spend["month"] == latest_month][["category", "spend"]]

    result = pd.DataFrame({"category": list(budgets.keys())})
    result["budget"] = result["category"].map(budgets)
    result = result.merge(latest_rows, on="category", how="left")
    result["spend"] = result["spend"].fillna(0.0)
    result["remaining"] = result["budget"] - result["spend"]
    result["over_budget"] = result["remaining"] < 0
    return result.sort_values("spend", ascending=False).reset_index(drop=True)
