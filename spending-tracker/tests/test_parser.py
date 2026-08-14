import io

import pytest

from src.parser import load_transactions


def test_load_transactions_parses_csv():
    csv_text = "date,description,amount\n2026-06-01,Whole Foods,-45.20\n2026-06-02,Payroll,2000.00\n"
    df = load_transactions(io.StringIO(csv_text))
    assert list(df.columns) == ["date", "description", "amount"]
    assert len(df) == 2
    assert df["amount"].iloc[0] == -45.20


def test_load_transactions_is_case_insensitive_on_headers():
    csv_text = "Date,Description,Amount\n2026-06-01,Whole Foods,-45.20\n"
    df = load_transactions(io.StringIO(csv_text))
    assert len(df) == 1


def test_load_transactions_sorts_by_date():
    csv_text = "date,description,amount\n2026-06-05,B,-1\n2026-06-01,A,-2\n"
    df = load_transactions(io.StringIO(csv_text))
    assert list(df["description"]) == ["A", "B"]


def test_load_transactions_missing_column_raises():
    csv_text = "date,description\n2026-06-01,Whole Foods\n"
    with pytest.raises(ValueError):
        load_transactions(io.StringIO(csv_text))
