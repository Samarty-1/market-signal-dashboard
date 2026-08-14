import pandas as pd

from src.budget import budget_status, monthly_spend_by_category


def _sample_transactions():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-06-01", "2026-06-05", "2026-06-10", "2026-07-01", "2026-07-03"]
            ),
            "description": ["Whole Foods", "Whole Foods", "Payroll", "Whole Foods", "Netflix"],
            "amount": [-100.0, -50.0, 2000.0, -300.0, -15.0],
            "category": ["Groceries", "Groceries", "Income", "Groceries", "Subscriptions"],
        }
    )


def test_monthly_spend_by_category_excludes_income():
    result = monthly_spend_by_category(_sample_transactions())
    categories = set(result["category"])
    assert "Income" not in categories
    assert "Groceries" in categories


def test_monthly_spend_by_category_sums_per_month():
    result = monthly_spend_by_category(_sample_transactions())
    june_groceries = result[(result["month"] == pd.Period("2026-06", "M")) & (result["category"] == "Groceries")]
    assert june_groceries["spend"].iloc[0] == 150.0


def test_budget_status_flags_over_budget():
    monthly = monthly_spend_by_category(_sample_transactions())
    status = budget_status(monthly, budgets={"Groceries": 250, "Subscriptions": 60})
    groceries_row = status[status["category"] == "Groceries"].iloc[0]
    assert groceries_row["spend"] == 300.0
    assert groceries_row["over_budget"]
    subs_row = status[status["category"] == "Subscriptions"].iloc[0]
    assert not subs_row["over_budget"]


def test_budget_status_handles_empty_spend():
    status = budget_status(pd.DataFrame(columns=["month", "category", "spend"]), budgets={"Groceries": 100})
    assert status.loc[0, "spend"] == 0.0
    assert not status.loc[0, "over_budget"]
