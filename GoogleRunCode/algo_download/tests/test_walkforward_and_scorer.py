"""Smoke tests for walk-forward and the pluggable scorer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algo.config import BacktestConfig
from algo.backtest.walkforward import (
    purged_kfold_indices, walk_forward, aggregate_walk_forward,
)
from algo.strategy.scorer import ConfluenceScorer


def _bars(n: int, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 0.5, n))
    o = c + rng.normal(0, 0.1, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.3, n))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.3, n))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c})


def test_backtest_config_defaults_to_single_position():
    assert BacktestConfig().max_concurrent_trades == 1


def test_purged_kfold_no_overlap_and_purge():
    n = 1000
    folds = list(purged_kfold_indices(n, n_splits=5, purge=10, embargo=0))
    assert len(folds) >= 1
    for f in folds:
        # train ends before test starts, with at least `purge` gap
        assert f.train_end <= f.test_start - 10 + 1e-9 or f.train_end == 0
        # test windows non-empty
        assert f.test_end > f.test_start


def test_purged_kfold_rejects_too_few_bars():
    with pytest.raises(ValueError):
        list(purged_kfold_indices(10, n_splits=5, purge=0, embargo=0))


def test_walk_forward_runs_and_aggregates():
    bars = _bars(800)
    cfg = BacktestConfig(n_splits=4, purge_days=0, embargo_days=0,
                         max_concurrent_trades=3, risk_per_trade=0.01)
    df = walk_forward(bars, cfg=cfg, bars_per_day=1.0,
                      bars_per_year_value=252.0)
    # n_splits=4 -> up to 3 usable folds (k=1..3)
    assert len(df) <= 3
    if len(df) > 0:
        for col in ("sharpe", "max_drawdown", "n_trades", "fold_id"):
            assert col in df.columns
    agg = aggregate_walk_forward(df)
    assert agg["n_folds"] == len(df)


def test_scorer_default_runs():
    bars = _bars(400)
    s = ConfluenceScorer.default()
    out = s.build_signals(bars)
    n = len(bars)
    assert out["signal"].shape == (n,)
    assert out["bull_score"].shape == (n,)
    assert out["bear_score"].shape == (n,)


def test_scorer_direction_sanity():
    bars = _bars(400)
    s = ConfluenceScorer.default(entry_threshold=0.3, min_dominance=0.0)
    out = s.build_signals(bars)
    sig = out["signal"]; sl = out["sl"]; tp = out["tp"]
    c = bars["close"].to_numpy()
    longs = np.where(sig == 1)[0]
    shorts = np.where(sig == -1)[0]
    for i in longs:
        assert sl[i] < c[i] < tp[i]
    for i in shorts:
        assert tp[i] < c[i] < sl[i]


def test_scorer_no_signals_below_threshold():
    bars = _bars(400)
    s = ConfluenceScorer.default(entry_threshold=1e6)  # impossibly high
    out = s.build_signals(bars)
    assert (out["signal"] == 0).all()


def test_scorer_empty_bars():
    out = ConfluenceScorer.default().build_signals(
        pd.DataFrame({"open": [], "high": [], "low": [], "close": []})
    )
    assert out["signal"].shape == (0,)
