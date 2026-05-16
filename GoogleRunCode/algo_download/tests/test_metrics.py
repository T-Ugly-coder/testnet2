"""Unit tests for algo.metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algo.metrics import (
    equity_returns, sharpe_ratio, sortino_ratio, max_drawdown, cagr,
    profit_factor, trade_stats, compute_perf, bars_per_year,
)


def test_equity_returns_simple():
    eq = np.array([100.0, 110.0, 99.0])
    r = equity_returns(eq)
    assert r.shape == (2,)
    assert r[0] == pytest.approx(0.10)
    assert r[1] == pytest.approx(99.0 / 110.0 - 1.0)


def test_equity_returns_short_input():
    assert equity_returns(np.array([])).shape == (0,)
    assert equity_returns(np.array([100.0])).shape == (0,)


def test_max_drawdown_known():
    eq = np.array([100.0, 120.0, 60.0, 90.0, 150.0])
    mdd, dur = max_drawdown(eq)
    assert mdd == pytest.approx(-0.5)  # 60 from peak 120
    assert dur >= 1


def test_max_drawdown_monotone_no_dd():
    eq = np.array([100.0, 110.0, 121.0, 133.0])
    mdd, dur = max_drawdown(eq)
    assert mdd == 0.0
    assert dur == 0


def test_cagr_basic():
    # 1 year of 252 daily bars, doubling
    eq = np.linspace(100.0, 200.0, 252)
    cg = cagr(eq, bars_per_year=252)
    assert cg == pytest.approx(1.0, rel=0.05)


def test_sharpe_zero_for_constant():
    r = np.zeros(100)
    assert sharpe_ratio(r, 252) == 0.0


def test_sharpe_positive_for_positive_drift():
    rng = np.random.default_rng(99)
    r = rng.normal(0.001, 0.01, 1000)
    assert sharpe_ratio(r, 252) > 0


def test_sortino_no_downside():
    r = np.array([0.01, 0.02, 0.005, 0.015])
    assert sortino_ratio(r, 252) == 0.0  # no downside returns -> 0 by convention


def test_profit_factor_basic():
    p = np.array([10.0, -5.0, 20.0, -3.0])
    assert profit_factor(p) == pytest.approx(30.0 / 8.0)


def test_profit_factor_no_losses():
    assert profit_factor(np.array([1.0, 2.0])) == float("inf")


def test_profit_factor_empty():
    assert profit_factor(np.array([])) == 0.0


def test_trade_stats_empty():
    s = trade_stats(pd.DataFrame())
    assert s["n_trades"] == 0
    assert s["win_rate"] == 0.0


def test_compute_perf_shapes():
    rng = np.random.default_rng(7)
    eq = 100_000.0 * np.cumprod(1 + rng.normal(0.0001, 0.005, 1000))
    trades = pd.DataFrame({
        "pnl": rng.normal(50.0, 200.0, 30),
    })
    perf = compute_perf(eq, trades, bars_per_year=252)
    d = perf.to_dict()
    for k in ("sharpe", "sortino", "max_drawdown", "cagr", "profit_factor"):
        assert k in d
    assert perf.n_bars == 1000
    assert perf.n_trades == 30


def test_bars_per_year_known():
    assert bars_per_year("1d") == 365
    with pytest.raises(KeyError):
        bars_per_year("3d")
