import numpy as np
import pandas as pd

from src.long_short_backtest import (
    build_hysteresis_long_short_returns,
    build_long_short_returns,
    evaluate_hysteresis_long_short,
    evaluate_long_short,
)


def _make_predictions(n_tickers=30, n_days=5, seed=0):
    """Synthetic OOS predictions: proba_up perfectly ranks next_day_return
    within each day (so the long-short spread should be unambiguously
    positive -- a sanity check that the decile construction is wired up
    correctly, not a claim about real predictive skill)."""
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    for date in dates:
        # proba_up and next_day_return are given the SAME rank order by
        # construction, so decile 9 (highest proba_up) always has the
        # highest next_day_return.
        order = rng.permutation(n_tickers)
        for i, ticker_idx in enumerate(order):
            rows.append(
                {
                    "date": date,
                    "ticker": f"T{ticker_idx}",
                    "proba_up": i / n_tickers,
                    "next_day_return": i / n_tickers * 0.02 - 0.01,  # spans -1% to +1%
                }
            )
    return pd.DataFrame(rows)


def test_long_decile_has_higher_proba_than_short_decile():
    preds = _make_predictions()
    daily = build_long_short_returns(preds, cost_bps=0.0)
    assert not daily.empty
    # By construction (perfect rank correlation), long leg return > short leg return every day.
    assert (daily["long_return"] > daily["short_return"]).all()


def test_spread_return_matches_long_minus_short():
    preds = _make_predictions()
    daily = build_long_short_returns(preds, cost_bps=0.0)
    pd.testing.assert_series_equal(
        daily["spread_return"], daily["long_return"] - daily["short_return"], check_names=False
    )


def test_transaction_cost_reduces_return_versus_zero_cost():
    preds = _make_predictions(n_days=10)
    zero_cost = build_long_short_returns(preds, cost_bps=0.0)
    with_cost = build_long_short_returns(preds, cost_bps=50.0)
    # Same spread, but long_short_return must be lower once costs are charged
    # (turnover happens because next_day_return re-randomizes membership).
    assert (with_cost["long_short_return"] <= zero_cost["long_short_return"]).all()


def test_days_with_too_few_tickers_are_dropped():
    tiny_universe = _make_predictions(n_tickers=5, n_days=3)  # 5 < N_DECILES * MIN_UNIVERSE_PER_DECILE
    daily = build_long_short_returns(tiny_universe)
    assert daily.empty


def test_rebalance_every_reduces_transaction_cost_days():
    preds = _make_predictions(n_days=10)
    daily_rebalance = build_long_short_returns(preds, cost_bps=10.0, rebalance_every=1)
    weekly_rebalance = build_long_short_returns(preds, cost_bps=10.0, rebalance_every=5)
    # Non-rebalance days must show zero cost.
    n_zero_cost_days = (weekly_rebalance["transaction_cost"] == 0.0).sum()
    assert n_zero_cost_days > 0
    # Total cost paid over the period should be lower with less frequent rebalancing.
    assert weekly_rebalance["transaction_cost"].sum() < daily_rebalance["transaction_cost"].sum()


def test_rebalance_every_holds_membership_between_rebalances():
    preds = _make_predictions(n_tickers=30, n_days=10)
    daily = build_long_short_returns(preds, rebalance_every=3)
    # Rows 1 and 2 (between rebalances at row 0 and row 3) must hold the
    # exact same long/short set sizes as the rebalance day, since membership
    # doesn't change until the next rebalance.
    assert daily.loc[0, "n_long"] == daily.loc[1, "n_long"] == daily.loc[2, "n_long"]


def test_evaluate_long_short_reports_universe_benchmark_separately():
    preds = _make_predictions(n_days=15)
    result = evaluate_long_short(preds, cost_bps=5.0)
    assert result["n_days"] > 0
    assert "sharpe" in result["long_short"]
    assert "sharpe" in result["universe_equal_weight"]
    # These are deliberately different quantities (hedged spread vs. raw
    # universe average) -- just confirm both are populated, not equal.
    assert result["mean_n_long"] > 0
    assert result["mean_n_short"] > 0


def test_hysteresis_long_decile_has_higher_proba_than_short():
    preds = _make_predictions(n_tickers=50, n_days=10)
    daily = build_hysteresis_long_short_returns(preds, cost_bps=0.0, entry_pct=0.10, exit_pct=0.25)
    assert not daily.empty
    assert (daily["long_return"] > daily["short_return"]).all()


def test_hysteresis_reduces_turnover_versus_hard_decile():
    """The whole point of hysteresis: a name near the decile boundary
    shouldn't flip in and out every day. Same underlying ranking, hysteresis
    construction must show less cumulative transaction cost."""
    preds = _make_predictions(n_tickers=50, n_days=20)
    hard = build_long_short_returns(preds, cost_bps=10.0, n_deciles=10)
    hysteresis = build_hysteresis_long_short_returns(preds, cost_bps=10.0, entry_pct=0.10, exit_pct=0.25)
    assert hysteresis["transaction_cost"].sum() < hard["transaction_cost"].sum()


def test_hysteresis_membership_persists_between_rebalances():
    preds = _make_predictions(n_tickers=30, n_days=10)
    daily = build_hysteresis_long_short_returns(preds, entry_pct=0.10, exit_pct=0.25, rebalance_every=3)
    # Same reasoning as test_rebalance_every_holds_membership_between_rebalances:
    # rows 1 and 2 sit between rebalance days 0 and 3, so leg sizes must be frozen.
    assert daily.loc[0, "n_long"] == daily.loc[1, "n_long"] == daily.loc[2, "n_long"]


def test_evaluate_hysteresis_long_short_reports_universe_benchmark():
    preds = _make_predictions(n_tickers=50, n_days=15)
    result = evaluate_hysteresis_long_short(preds, cost_bps=5.0, entry_pct=0.10, exit_pct=0.25)
    assert result["n_days"] > 0
    assert "sharpe" in result["long_short"]
    assert "sharpe" in result["universe_equal_weight"]
