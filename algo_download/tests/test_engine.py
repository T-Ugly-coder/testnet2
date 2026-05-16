"""Unit tests for algo.backtest.engine._resolve_exit and run_backtest.

Covers:
  - SL-wins-when-both-touched intrabar (pessimistic ordering).
  - Gap-up TP fill on a long opened on prior bar.
  - Gap-down SL fill on a long opened on prior bar.
  - Same-bar entry: SL/TP measured against intrabar high/low only,
    NOT against the open (we already filled at the open).
  - Direction sanity: invalid SL/TP (sl >= fill for long) is rejected.
  - End-to-end run_backtest equity curve never has NaNs and is finite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algo.backtest.engine import _resolve_exit, run_backtest, _simulate


# --- _resolve_exit unit tests ------------------------------------------------

def test_resolve_exit_long_both_touched_sl_wins():
    # long, SL=99, TP=102, bar [open=100, high=103, low=98]: both touch.
    reason, px = _resolve_exit(1, 99.0, 102.0, 100.0, 103.0, 98.0,
                               0.0, False)
    assert reason == -1
    assert px == pytest.approx(99.0)


def test_resolve_exit_long_gap_up_tp_at_open():
    # long, prior fill < TP; bar gaps above TP -> exit at open (favorable gap).
    reason, px = _resolve_exit(1, 95.0, 102.0, 105.0, 106.0, 104.0,
                               0.0, False)
    assert reason == 1
    assert px == pytest.approx(105.0)


def test_resolve_exit_long_gap_down_sl_at_open():
    # long; bar gaps below SL -> exit at open (adverse gap, no slip).
    reason, px = _resolve_exit(1, 99.0, 102.0, 95.0, 100.0, 94.0,
                               0.0, False)
    assert reason == -1
    assert px == pytest.approx(95.0)


def test_resolve_exit_entry_bar_ignores_open_check():
    # On the entry bar, we already filled at open. Open=100 below SL=101 is
    # NOT an exit signal (we wouldn't have opened); only intrabar matters.
    # Using a long with sl=99, tp=102, open=100, high=101.5, low=99.5: no exit.
    reason, _ = _resolve_exit(1, 99.0, 102.0, 100.0, 101.5, 99.5,
                              0.0, True)
    assert reason == 0


def test_resolve_exit_short_gap_down_tp_at_open():
    # Short, TP below; bar gaps to open well below TP -> favorable fill at open.
    reason, px = _resolve_exit(-1, 105.0, 99.0, 95.0, 96.0, 94.0,
                               0.0, False)
    assert reason == 1
    assert px == pytest.approx(95.0)


def test_resolve_exit_short_gap_up_sl_at_open():
    reason, px = _resolve_exit(-1, 105.0, 99.0, 110.0, 111.0, 109.0,
                               0.0, False)
    assert reason == -1
    assert px == pytest.approx(110.0)


# --- run_backtest integration tests -----------------------------------------

def _make_bars(o, h, l, c):
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c},
                        dtype=np.float64)


def test_run_backtest_long_takes_profit():
    # 5 bars. Signal at bar 1 -> long fills at open of bar 2.
    o = [100, 100, 100, 105, 110]
    h = [101, 101, 106, 110, 115]
    l = [99,  99,  99, 104, 108]
    c = [100, 100, 105, 109, 114]
    bars = _make_bars(o, h, l, c)
    sig = np.zeros(5, dtype=np.int8)
    sl_arr = np.full(5, np.nan)
    tp_arr = np.full(5, np.nan)
    sig[1] = 1
    sl_arr[1] = 95.0
    tp_arr[1] = 105.0  # TP within bar-2 range (high=106)
    out = run_backtest(bars, {"signal": sig, "sl": sl_arr, "tp": tp_arr},
                       fee_bps=0.0, slip_bps=0.0)
    trades = out["trades"]
    assert len(trades) == 1
    assert int(trades.iloc[0]["reason"]) == 1   # TP
    assert int(trades.iloc[0]["direction"]) == 1
    assert out["final_balance"] > 100_000.0


def test_run_backtest_long_stop_loss():
    o = [100, 100, 100,  90,  80]
    h = [101, 101, 101,  95,  85]
    l = [99,  99,  89,  79,  70]
    c = [100, 100,  92,  82,  72]
    bars = _make_bars(o, h, l, c)
    sig = np.zeros(5, dtype=np.int8)
    sl_arr = np.full(5, np.nan)
    tp_arr = np.full(5, np.nan)
    sig[1] = 1
    sl_arr[1] = 95.0
    tp_arr[1] = 110.0
    out = run_backtest(bars, {"signal": sig, "sl": sl_arr, "tp": tp_arr},
                       fee_bps=0.0, slip_bps=0.0)
    trades = out["trades"]
    assert len(trades) == 1
    assert int(trades.iloc[0]["reason"]) == -1
    assert out["final_balance"] < 100_000.0


def test_run_backtest_invalid_direction_rejected():
    # SL above fill on a long should be rejected (direction sanity).
    o = [100, 100, 100, 100, 100]
    h = [101, 101, 101, 101, 101]
    l = [99,   99,  99,  99,  99]
    c = [100, 100, 100, 100, 100]
    bars = _make_bars(o, h, l, c)
    sig = np.zeros(5, dtype=np.int8)
    sl_arr = np.full(5, np.nan)
    tp_arr = np.full(5, np.nan)
    sig[1] = 1
    sl_arr[1] = 110.0  # ABOVE fill — invalid
    tp_arr[1] = 120.0
    out = run_backtest(bars, {"signal": sig, "sl": sl_arr, "tp": tp_arr},
                       fee_bps=0.0, slip_bps=0.0)
    assert len(out["trades"]) == 0


def test_run_backtest_equity_finite():
    rng = np.random.default_rng(42)
    n = 200
    c = 100 + np.cumsum(rng.normal(0, 0.5, n))
    o = c + rng.normal(0, 0.1, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.2, n))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.2, n))
    bars = _make_bars(o, h, l, c)
    sig = np.zeros(n, dtype=np.int8)
    sl_arr = np.full(n, np.nan)
    tp_arr = np.full(n, np.nan)
    # sparse signals
    for i in range(20, n - 1, 30):
        sig[i] = 1
        sl_arr[i] = c[i] - 1.0
        tp_arr[i] = c[i] + 2.0
    out = run_backtest(bars, {"signal": sig, "sl": sl_arr, "tp": tp_arr})
    eq = out["equity"]
    assert np.all(np.isfinite(eq))
    assert eq.shape[0] == n
