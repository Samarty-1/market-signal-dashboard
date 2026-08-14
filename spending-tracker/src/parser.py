"""Parse a bank-export-style transactions CSV into a normalized DataFrame."""
from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["date", "description", "amount"]


def load_transactions(source) -> pd.DataFrame:
    """Load transactions from a CSV path or file-like object.

    Expects columns (case-insensitive): date, description, amount.
    Amount convention: negative = money out (spending), positive = money in
    (income/refunds) — matches most bank/card export formats.
    Returns a DataFrame with normalized column names, a parsed `date`
    (datetime64) and `amount` (float), sorted by date ascending.
    """
    df = pd.read_csv(source)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Transactions CSV is missing required column(s): {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = pd.to_numeric(df["amount"])
    df["description"] = df["description"].astype(str).str.strip()
    return df.sort_values("date").reset_index(drop=True)
