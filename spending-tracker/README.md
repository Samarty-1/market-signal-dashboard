# 💸 Spending Tracker

A local-first personal spending dashboard. Upload a transactions CSV (the
kind you can export from most banks/card issuers), and it auto-categorizes
each transaction by keyword rules, then shows spend by category and month
against a configurable budget.

## What this does

1. **CSV parsing** (`src/parser.py`) — normalizes a transactions export
   (date, description, amount) regardless of header casing.
2. **Rule-based categorization** (`src/categorize.py`) — matches each
   transaction description against keyword rules (e.g. "starbucks" →
   Dining, "netflix" → Subscriptions) to assign a category. Falls back to
   "Other" for anything unmatched.
3. **Budget comparison** (`src/budget.py`) — sums spend per category for the
   latest month in the data and compares it against per-category budgets.
4. **Dashboard** (`app.py`, Streamlit) — category breakdown pie chart,
   monthly stacked bar chart, a budget-vs-actual table (over-budget
   categories highlighted), editable budgets in the sidebar, and the full
   transaction table.

No bank API integration by design — you export a CSV from your bank/card
provider's website and upload it here, so no credentials are ever handled
by this tool and nothing leaves your machine.

## CSV format

```csv
date,description,amount
2026-06-01,Whole Foods Market,-84.90
2026-06-01,Employer Payroll Direct Deposit,2800.00
```

- `amount`: negative = money out (spending), positive = money in
  (income/refunds) — the standard convention for most bank/card exports.
- Column names are matched case-insensitively; extra columns are ignored.

`sample_data/transactions.csv` has ~3 months of synthetic sample data so you
can try the dashboard immediately without your own export.

## Running it

```bash
cd spending-tracker
pip install -r requirements.txt
streamlit run app.py
```

It loads the bundled sample data by default — uncheck "Use sample data" in
the sidebar and upload your own CSV to see your real spending.

## Tests

```bash
pytest
```

Covers CSV parsing, categorization rules, and budget comparison logic —
all pure functions on synthetic data, no external services.
