from __future__ import annotations

import numpy as np
import pandas as pd

from tick_forward_test import summarize
from tools.auto_algo_finder import _trade_profile_metrics


def test_forward_trade_profile_reports_min_avg_max_hold_time():
    trades = pd.DataFrame({
        "entry_idx": [1, 10, 20],
        "exit_idx": [3, 15, 30],
        "entry": [100.0, 100.0, 100.0],
        "exit": [102.0, 98.0, 104.0],
        "direction": [1, -1, 1],
    })

    metrics = _trade_profile_metrics(trades, "1h")

    assert metrics["min_hold_bars"] == 2.0
    assert metrics["avg_hold_bars"] == (2.0 + 5.0 + 10.0) / 3.0
    assert metrics["max_hold_bars"] == 10.0
    assert metrics["min_hold_hours"] == 2.0
    assert metrics["avg_hold_hours"] == (2.0 + 5.0 + 10.0) / 3.0
    assert metrics["max_hold_hours"] == 10.0


def test_tick_summary_reports_min_avg_max_hold_time():
    equity = np.array([100_000.0, 100_100.0, 100_200.0, 100_300.0])
    trades = pd.DataFrame({
        "entry_idx": [1, 2],
        "exit_idx": [2, 4],
        "entry_ts_ms": [3_600_000, 7_200_000],
        "exit_ts_ms": [7_200_000, 18_000_000],
        "pnl": [100.0, -50.0],
        "risk_amt": [1_000.0, 1_000.0],
    })

    report = summarize(equity, trades, 100_000.0)

    assert report["min_hold_bars"] == 1.0
    assert report["avg_hold_bars"] == 1.5
    assert report["max_hold_bars"] == 2.0
    assert report["min_hold_hours"] == 1.0
    assert report["avg_hold_hours"] == 2.0
    assert report["max_hold_hours"] == 3.0
