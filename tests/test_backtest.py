import numpy as np
import pandas as pd
import pytest

from src.backtest import add_strategy_returns, backtest_portfolio, compute_regime_gate, performance_metrics


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


def test_confidence_sizing_scales_position_with_signal_strength():
    predictions = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4),
            "ticker": ["AAA"] * 4,
            "proba_up": [0.8, 0.2, 0.6, 0.4],
            "next_day_return": [0.01, 0.02, -0.01, 0.03],
        }
    )
    scored = add_strategy_returns(predictions, threshold=0.5, position_sizing="confidence")
    # position = clip((proba - 0.5) / 0.5, 0, 1) -> [0.6, 0.0, 0.2, 0.0]
    assert scored["position"].tolist() == pytest.approx([0.6, 0.0, 0.2, 0.0])
    assert scored["strategy_return"].tolist() == pytest.approx([0.6 * 0.01, 0.0, 0.2 * -0.01, 0.0])


def test_transaction_costs_charged_on_position_turnover():
    predictions = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4),
            "ticker": ["AAA"] * 4,
            "proba_up": [0.8, 0.2, 0.6, 0.4],  # binary positions: 1, 0, 1, 0
            "next_day_return": [0.01, 0.02, -0.01, 0.03],
        }
    )
    scored = add_strategy_returns(predictions, threshold=0.5, cost_bps=100.0)  # 1% per unit of turnover
    # Position flips every day (1,0,1,0 vs prev 0,1,0,1) -> turnover is always 1 -> cost is always 0.01
    assert scored["transaction_cost"].tolist() == pytest.approx([0.01, 0.01, 0.01, 0.01])
    assert scored["strategy_return"].tolist() == pytest.approx([0.01 - 0.01, 0.0 - 0.01, -0.01 - 0.01, 0.0 - 0.01])


def test_zero_cost_bps_matches_original_binary_behavior():
    # Guards against a regression in the default path: cost_bps=0.0 must reproduce
    # the pre-transaction-cost strategy_return exactly.
    predictions = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4),
            "ticker": ["AAA"] * 4,
            "proba_up": [0.8, 0.2, 0.6, 0.4],
            "next_day_return": [0.01, 0.02, -0.01, 0.03],
        }
    )
    scored = add_strategy_returns(predictions, threshold=0.5)
    assert scored["strategy_return"].tolist() == [0.01, 0.0, -0.01, 0.0]


def test_regime_gate_blocks_regime_with_no_prior_positive_edge():
    dates = pd.date_range("2024-01-01", periods=16)
    scored = pd.DataFrame(
        {
            "date": dates,
            "fold": [0] * 12 + [1] * 4,
            "position": [1.0] * 16,
            # fold 0: low-VIX days (12) were profitable when traded; mid (20) and high (30) VIX days were not.
            "next_day_return": [0.02] * 4 + [-0.02] * 4 + [-0.02] * 4 + [0.0] * 4,
        }
    )
    vix = pd.DataFrame(
        {
            "date": dates,
            # fold 0 terciles: 12 -> low, 20 -> mid, 30 -> high. fold 1: two low-VIX days, two high-VIX days.
            "vix_close": [12] * 4 + [20] * 4 + [30] * 4 + [12, 12, 30, 30],
        }
    )
    gate = compute_regime_gate(scored, vix)
    # Low-VIX regime had positive edge in fold 0 -> passes. High-VIX regime didn't -> blocked.
    assert gate.iloc[12:].tolist() == [True, True, False, False]


def test_regime_filter_requires_vix():
    predictions = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2),
            "ticker": ["AAA"] * 2,
            "proba_up": [0.8, 0.2],
            "next_day_return": [0.01, 0.02],
        }
    )
    with pytest.raises(ValueError):
        add_strategy_returns(predictions, regime_filter=True)
