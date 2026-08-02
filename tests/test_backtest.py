import numpy as np
import pandas as pd
import pytest

from src.backtest import add_strategy_returns, backtest_portfolio, performance_metrics


def test_performance_metrics_constant_positive_return():
    # 252 days of a flat +0.1%/day return -> total return should compound to (1.001)^252 - 1
    returns = pd.Series([0.001] * 252)
    metrics = performance_metrics(returns)
    expected_total = (1.001) ** 252 - 1
    assert metrics["total_return"] == pytest.approx(round(expected_total, 4), abs=1e-3)
    assert metrics["annualized_vol"] == 0.0  # no variance -> no volatility
    assert metrics["sharpe"] == 0.0  # guarded: zero vol means sharpe formula would divide by zero


def test_performance_metrics_empty_series_returns_zeros():
    metrics = performance_metrics(pd.Series([], dtype=float))
    assert metrics == {"total_return": 0.0, "annualized_return": 0.0, "annualized_vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}


def test_performance_metrics_drawdown_detected():
    # Peak after day 1: 1.10. Trough after day 2: 1.10 * 0.80 = 0.88.
    # Drawdown from peak = 0.88 / 1.10 - 1 = -20%.
    returns = pd.Series([0.10, -0.20])
    metrics = performance_metrics(returns)
    assert metrics["max_drawdown"] == pytest.approx(-0.20, abs=1e-6)


def test_add_strategy_returns_only_earns_when_signal_on():
    predictions = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4),
            "ticker": ["AAA"] * 4,
            "proba_up": [0.8, 0.2, 0.6, 0.4],
            "next_day_return": [0.01, 0.02, -0.01, 0.03],
        }
    )
    scored = add_strategy_returns(predictions, threshold=0.5)
    # Signal on for proba 0.8 and 0.6 -> strategy captures those returns; else 0.
    assert scored["strategy_return"].tolist() == [0.01, 0.0, -0.01, 0.0]
    assert scored["buy_hold_return"].tolist() == [0.01, 0.02, -0.01, 0.03]


def test_backtest_portfolio_equal_weights_across_tickers():
    predictions = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "ticker": ["AAA", "BBB"],
            "proba_up": [0.9, 0.9],
            "next_day_return": [0.02, 0.04],
        }
    )
    predictions["date"] = pd.to_datetime(predictions["date"])
    result = backtest_portfolio(predictions, threshold=0.5)
    # Both tickers go long; equal-weighted daily return should be the mean: (0.02+0.04)/2 = 0.03
    assert result["strategy"]["total_return"] == pytest.approx(0.03, abs=1e-6)
    assert result["n_days"] == 1
