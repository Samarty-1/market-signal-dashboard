"""Tests for the cost-aware portfolio construction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio_construction import (
    decompose_signal, evaluate_signal, performance, signal_to_weights,
    simulate, smooth_weights,
)


def _score(n_dates: int = 60, n_tickers: int = 20, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    return pd.Series(rng.normal(size=len(idx)), index=idx)


def _returns(weights: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(scale=0.01, size=weights.shape),
        index=weights.index, columns=weights.columns,
    )


def test_weights_are_dollar_neutral_and_unit_gross():
    w = signal_to_weights(_score())
    assert np.allclose(w.sum(axis=1), 0.0, atol=1e-9), "book must be dollar-neutral"
    assert np.allclose(w.abs().sum(axis=1), 1.0, atol=1e-9), "gross exposure must be 1"


def test_weights_are_monotone_in_score():
    """The top-scoring name must carry the largest long weight on its date."""
    s = _score(n_dates=5, n_tickers=10)
    w = signal_to_weights(s)
    d = w.index[0]
    day = s.xs(d, level="date")
    assert w.loc[d].idxmax() == day.idxmax()
    assert w.loc[d].idxmin() == day.idxmin()


def test_smoothing_preserves_neutrality_and_gross():
    w = signal_to_weights(_score())
    sm = smooth_weights(w, halflife=10)
    assert np.allclose(sm.sum(axis=1), 0.0, atol=1e-9)
    # Gross must be restored, otherwise smoothing would silently shrink the
    # book and flatter its volatility.
    assert np.allclose(sm.abs().sum(axis=1), w.abs().sum(axis=1), atol=1e-9)


def test_smoothing_reduces_turnover():
    """The whole point of smoothing: strictly less trading, same gross exposure."""
    w = signal_to_weights(_score())
    fwd = _returns(w)
    raw = simulate(w, fwd, band=0.0)["turnover"].mean()
    sm = simulate(smooth_weights(w, halflife=20), fwd, band=0.0)["turnover"].mean()
    assert sm < raw * 0.6


def test_band_reduces_turnover():
    w = smooth_weights(signal_to_weights(_score()), halflife=10)
    fwd = _returns(w)
    assert (simulate(w, fwd, band=1.0)["turnover"].mean()
            < simulate(w, fwd, band=0.0)["turnover"].mean())


def test_zero_cost_means_net_equals_gross():
    w = signal_to_weights(_score())
    sim = simulate(w, _returns(w), cost_bps=0.0)
    assert np.allclose(sim["net_return"], sim["gross_return"])


def test_cost_scales_with_turnover_and_rate():
    w = signal_to_weights(_score())
    fwd = _returns(w)
    a = simulate(w, fwd, cost_bps=10.0)
    b = simulate(w, fwd, cost_bps=20.0)
    assert np.allclose(b["cost"], 2 * a["cost"])
    assert (a["cost"] >= 0).all()


def test_delisted_name_is_closed_even_inside_the_band():
    """A ticker that leaves the universe must go to zero weight, not linger
    because its drift happened to fall inside the no-trade band."""
    s = _score(n_dates=10, n_tickers=6)
    w = signal_to_weights(s)
    w.iloc[5:, 0] = np.nan                       # first ticker delists midway
    _, pos = simulate(w, _returns(w), band=0.5, return_positions=True)
    assert (pos.iloc[5:, 0] == 0).all(), "a delisted name must be fully closed"
    assert (pos.iloc[:5, 0] != 0).any(), "it must have been held before delisting"


def test_returns_flow_through_to_pnl():
    """A book long exactly one name earns that name's return."""
    dates = pd.bdate_range("2024-01-01", periods=3)
    cols = ["A", "B"]
    w = pd.DataFrame([[1.0, -1.0]] * 3, index=dates, columns=cols)
    fwd = pd.DataFrame([[0.02, 0.01]] * 3, index=dates, columns=cols)
    sim = simulate(w, fwd, cost_bps=0.0, band=0.0)
    assert np.allclose(sim["gross_return"], 0.02 - 0.01)


def test_performance_reports_sane_fields():
    w = signal_to_weights(_score())
    m = performance(simulate(w, _returns(w)))
    assert m["n_days"] > 0
    assert m["ann_vol"] >= 0
    assert m["max_drawdown"] <= 0
    assert m["turnover_daily"] >= 0


def test_decompose_is_causal_and_additive():
    s = _score(n_dates=200, n_tickers=8)
    full, static, timing = decompose_signal(s, min_periods=20)
    assert np.allclose((static + timing).to_numpy(), full.to_numpy(), atol=1e-9)
    # `static` on a date must not depend on that date's own score: recomputing
    # after perturbing only the LAST date must leave every earlier value intact.
    s2 = s.copy()
    last = s2.index.get_level_values("date").max()
    s2.loc[last] = s2.loc[last] + 100.0
    _, static2, _ = decompose_signal(s2, min_periods=20)
    common = static.index.intersection(static2.index)
    common = common[common.get_level_values("date") <= last]
    assert np.allclose(static.loc[common].to_numpy(), static2.loc[common].to_numpy(), atol=1e-9)


def test_evaluate_signal_runs_end_to_end():
    s = _score()
    w = signal_to_weights(s)
    m = evaluate_signal(s, _returns(w))
    assert set(m) >= {"net_sharpe", "gross_sharpe", "turnover_daily", "cost_bps_per_day"}


@pytest.mark.parametrize("halflife", [None, 0, 5, 42])
def test_smoothing_accepts_disabled_and_enabled(halflife):
    w = signal_to_weights(_score())
    sm = smooth_weights(w, halflife=halflife)
    assert sm.shape == w.shape
