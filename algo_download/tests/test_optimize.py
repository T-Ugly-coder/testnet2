"""Unit tests for the auto-optimizer's objective scoring + smoke run."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algo.optimize.objective import (
    ObjectiveConfig, score, trades_per_month, _freq_penalty, _dd_penalty,
)


def _fake_folds(sharpe=2.0, sortino=2.5, pf=1.8, expectancy=0.005,
                win_rate=0.55, max_dd=-0.10, n_trades=12, test_bars=720,
                n_folds=5):
    """720 1h bars/fold = 1 month per fold; 12 trades = 12 trades/month."""
    return pd.DataFrame([{
        "sharpe": sharpe + 0.1 * (i - n_folds / 2),
        "sortino": sortino,
        "profit_factor": pf,
        "expectancy": expectancy,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
        "test_bars": test_bars,
        "fold_id": i,
    } for i in range(n_folds)])


def test_trades_per_month_basic():
    folds = _fake_folds(n_trades=12, test_bars=720, n_folds=3)  # 720 bars = 30d at 1h
    # bars_per_month for 1h = 24*30 = 720
    tpm = trades_per_month(folds, bars_per_month=720.0)
    assert tpm == pytest.approx(12.0)


def test_freq_penalty_zero_inside_band():
    cfg = ObjectiveConfig()
    assert _freq_penalty(12.0, cfg) >= 0.0
    assert _freq_penalty(12.0, cfg) < 0.1     # very small at exact target
    assert _freq_penalty(50.0, cfg) > _freq_penalty(15.0, cfg)
    assert _freq_penalty(0.0, cfg) > _freq_penalty(8.0, cfg)


def test_dd_penalty_monotone():
    cfg = ObjectiveConfig()
    assert _dd_penalty(0.05, cfg) == 0.0
    assert _dd_penalty(0.25, cfg) > 0.0
    assert _dd_penalty(0.40, cfg) >= _dd_penalty(0.30, cfg)


def test_score_prefers_high_sharpe_target_freq_low_dd():
    cfg = ObjectiveConfig()
    good = _fake_folds(sharpe=2.5, max_dd=-0.08, n_trades=12, test_bars=720)
    bad_dd = _fake_folds(sharpe=2.5, max_dd=-0.35, n_trades=12, test_bars=720)
    bad_freq = _fake_folds(sharpe=2.5, max_dd=-0.08, n_trades=2, test_bars=720)
    low_sharpe = _fake_folds(sharpe=0.3, max_dd=-0.08, n_trades=12, test_bars=720)

    s_good = score(good, bars_per_month=720.0, cfg=cfg)["score"]
    s_dd = score(bad_dd, bars_per_month=720.0, cfg=cfg)["score"]
    s_freq = score(bad_freq, bars_per_month=720.0, cfg=cfg)["score"]
    s_lo = score(low_sharpe, bars_per_month=720.0, cfg=cfg)["score"]

    assert s_good > s_dd
    assert s_good > s_freq
    assert s_good > s_lo


def test_score_handles_empty():
    s = score(pd.DataFrame(), bars_per_month=720.0)
    assert s["score"] <= -1e8


# --- Optimizer smoke (skipped if optuna not installed) ----------------------

def _synthetic_bars(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 0.5, n))
    o = c + rng.normal(0, 0.1, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.2, n))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.2, n))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c})


def test_optimizer_smoke_runs():
    optuna = pytest.importorskip("optuna")
    from algo.config import BacktestConfig
    from algo.optimize import OptimizeConfig, optimize

    bars = _synthetic_bars(n=2000)
    bt = BacktestConfig(n_splits=3, purge_days=0, embargo_days=0)
    ocfg = OptimizeConfig(timeframe="1h", holdout_frac=0.3, n_trials=2,
                          n_jobs=1, seed=1, bt=bt)
    result = optimize(bars, ocfg)
    assert "best_params" in result
    assert "holdout_score" in result
    assert result["n_trials"] == 2
