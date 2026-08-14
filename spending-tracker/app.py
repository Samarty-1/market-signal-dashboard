"""Spending Tracker — Streamlit dashboard.

Upload a transactions CSV (date, description, amount), auto-categorize by
keyword rules, and see spend by category/month against a budget.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.budget import DEFAULT_BUDGETS, budget_status, monthly_spend_by_category
from src.categorize import categorize
from src.parser import load_transactions

st.set_page_config(page_title="Spending Tracker", page_icon="💸", layout="wide")

st.title("💸 Spending Tracker")
st.caption(
    "Upload a transactions export (date, description, amount) to see spend by "
    "category and month, and how it stacks up against a budget. No bank "
    "connection — this reads a CSV you export yourself, so nothing leaves "
    "your machine."
)

SAMPLE_PATH = Path(__file__).parent / "sample_data" / "transactions.csv"

uploaded = st.sidebar.file_uploader("Transactions CSV", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample data", value=uploaded is None)

if uploaded is not None and not use_sample:
    transactions = load_transactions(uploaded)
elif SAMPLE_PATH.exists():
    transactions = load_transactions(SAMPLE_PATH)
    st.sidebar.info("Showing bundled sample data. Upload your own CSV to replace it.")
else:
    st.warning("Upload a transactions CSV to get started.")
    st.stop()

transactions = categorize(transactions)

st.sidebar.subheader("Monthly budgets ($)")
budgets = {}
for category, default in DEFAULT_BUDGETS.items():
    budgets[category] = st.sidebar.number_input(category, min_value=0, value=int(default), step=10)

spend_only = transactions[transactions["amount"] < 0].copy()
spend_only["month"] = spend_only["date"].dt.to_period("M").astype(str)

st.header("Spend by category")
col1, col2 = st.columns(2)
with col1:
    by_category = spend_only.groupby("category")["amount"].sum().abs().sort_values(ascending=False)
    fig_pie = px.pie(
        by_category.reset_index(), names="category", values="amount", title="Total spend by category"
    )
    st.plotly_chart(fig_pie, use_container_width=True)
with col2:
    by_month_cat = spend_only.groupby(["month", "category"])["amount"].sum().abs().reset_index()
    fig_bar = px.bar(
        by_month_cat, x="month", y="amount", color="category", title="Monthly spend by category", barmode="stack"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.header("Budget vs. actual (latest month)")
monthly = monthly_spend_by_category(transactions)
status = budget_status(monthly, budgets)


def _highlight_over(row):
    color = "background-color: #ffcccc" if row["over_budget"] else ""
    return [color] * len(row)


st.dataframe(
    status.style.apply(_highlight_over, axis=1).format({"spend": "${:.2f}", "budget": "${:.2f}", "remaining": "${:.2f}"}),
    use_container_width=True,
)

over_budget = status[status["over_budget"]]
if not over_budget.empty:
    st.warning(f"Over budget in {len(over_budget)} categor{'y' if len(over_budget) == 1 else 'ies'}: " + ", ".join(over_budget["category"]))
else:
    st.success("Within budget in every category this month.")

st.header("All transactions")
st.dataframe(
    transactions.assign(date=transactions["date"].dt.strftime("%Y-%m-%d")).sort_values("date", ascending=False),
    use_container_width=True,
)
