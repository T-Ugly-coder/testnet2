"""
Purged & embargoed walk-forward cross-validation.

Why purge + embargo? (López de Prado, "Advances in Financial Machine Learning")
A signal generated at time t can target an outcome that lands inside the test
fold's window. If we naively split, label leakage poisons the test:
  * "purge_days" — drop train rows whose label horizon overlaps the test set.
  * "embargo_days" — drop train rows immediately AFTER the test set, because
    they may have been informationally entangled with the test fold.

This module implements:

  - `purged_kfold_indices` — pure index split generator (no I/O).
  - `walk_forward` — anchored or rolling walk-forward over time-ordered bars.
  - `evaluate_walk_forward` — runs `algo.strategy.runner.backtest_strategy`
    on each fold's test window and aggregates metrics.

All splits operate on integer bar positions; conversion from `purge_days`/
`embargo_days` to bar counts requires `bars_per_day` (caller-supplied).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import numpy as np
import pandas as pd

from ..config import BacktestConfig
from ..metrics import compute_perf, PerfStats
from ..strategy.runner import backtest_strategy


@dataclass
class Fold:
    fold_id: int
    train_start: int
    train_end: int       # exclusive
    test_start: int
    test_end: int        # exclusive


def purged_kfold_indices(n: int, n_splits: int, purge: int, embargo: int
                         ) -> Iterator[Fold]:
    """Generate non-overlapping forward-looking folds with purge+embargo.

    Each fold's TEST window is one of `n_splits` contiguous, non-overlapping
    chunks. The TRAIN window is the past, with the boundary trimmed by
    `max(purge, embargo)` bars to prevent label leakage in BOTH directions:
      - `purge` removes train rows whose forward-looking label horizon would
        overlap the test window.
      - `embargo` removes train rows immediately before the test window which
        may be informationally entangled (auto-correlation).
    Anchored walk-forward never adds future bars to train, so the right-side
    leakage path is already absent.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if n <= n_splits * 2:
        raise ValueError("not enough bars for the requested n_splits")
    if purge < 0 or embargo < 0:
        raise ValueError("purge and embargo must be non-negative")
    fold_size = n // n_splits
    boundary = max(int(purge), int(embargo))
    for k in range(1, n_splits):
        test_start = k * fold_size
        test_end = (k + 1) * fold_size if k < n_splits - 1 else n
        train_end = max(0, test_start - boundary)
        if train_end <= 0:
            continue
        yield Fold(
            fold_id=k,
            train_start=0,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )


def walk_forward(bars: pd.DataFrame, cfg: Optional[BacktestConfig] = None,
                 bars_per_day: Optional[float] = None,
                 strategy_kwargs: Optional[dict] = None,
                 backtest_kwargs: Optional[dict] = None,
                 bars_per_year_value: Optional[float] = None,
                 min_test_bars: int = 50,
                 ) -> pd.DataFrame:
    """Run anchored walk-forward across `bars` and return a per-fold results
    DataFrame with metrics from `algo.metrics.compute_perf`.

    Parameters
    ----------
    bars : pd.DataFrame
        OHLC frame indexed in time order.
    cfg : BacktestConfig
        Provides n_splits, purge_days, embargo_days, risk_per_trade,
        fee_bps, slippage_bps, initial_balance, max_concurrent_trades.
        Defaults to algo.config.CFG.bt if None.
    bars_per_day : float
        Number of bars in one calendar day at this timeframe (e.g. 1440 for
        1m crypto, 1 for 1d). Required to convert purge_days/embargo_days
        to bar counts. If None, assumed = 1 (i.e. purge/embargo are in bars).
    strategy_kwargs : dict | None
        Forwarded to build_signals via runner.
    backtest_kwargs : dict | None
        Engine overrides; values from `cfg` are merged in unless overridden.
    bars_per_year_value : float
        Annualization factor for Sharpe/Sortino. If None, falls back to
        `bars_per_day * 365` (24/7 default) when `bars_per_day` is given,
        else 252.
    """
    if cfg is None:
        from ..config import CFG
        cfg = CFG.bt
    bpd = float(bars_per_day) if bars_per_day else 1.0
    purge = int(round(cfg.purge_days * bpd))
    embargo = int(round(cfg.embargo_days * bpd))

    by = bars_per_year_value
    if by is None:
        by = bpd * 365.0 if bars_per_day else 252.0

    skw = dict(strategy_kwargs or {})
    bkw = {
        "risk": cfg.risk_per_trade,
        "fee_bps": cfg.fee_bps,
        "slip_bps": cfg.slippage_bps,
        "initial_balance": cfg.initial_balance,
        "max_concurrent": cfg.max_concurrent_trades,
    }
    bkw.update(backtest_kwargs or {})

    rows = []
    n = len(bars)
    for fold in purged_kfold_indices(n, cfg.n_splits, purge, embargo):
        # Build test slice. We feed only the test slice to the backtester
        # because build_signals is causal — train has no learnable parameters
        # in the current confluence strategy. When you add an estimator,
        # fit on bars[train_start:train_end] and pass fitted params via skw.
        test_bars = bars.iloc[fold.test_start:fold.test_end].reset_index(drop=True)
        if len(test_bars) < min_test_bars:
            continue
        result = backtest_strategy(test_bars, **skw, **bkw)
        perf = compute_perf(result["equity"], result["trades"],
                            bars_per_year=by)
        d = perf.to_dict()
        d.update({
            "fold_id": fold.fold_id,
            "train_start": fold.train_start,
            "train_end": fold.train_end,
            "test_start": fold.test_start,
            "test_end": fold.test_end,
            "test_bars": len(test_bars),
        })
        rows.append(d)
    return pd.DataFrame(rows)


def aggregate_walk_forward(folds_df: pd.DataFrame) -> dict:
    """Summary across folds: mean/std of key metrics, plus stability stats."""
    if folds_df is None or len(folds_df) == 0:
        return {"n_folds": 0}
    keys = ["sharpe", "sortino", "calmar", "max_drawdown",
            "cagr", "win_rate", "profit_factor", "expectancy"]
    out = {"n_folds": int(len(folds_df))}
    for k in keys:
        if k not in folds_df.columns:
            continue
        v = folds_df[k].to_numpy(np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        out[f"{k}_mean"] = float(v.mean())
        out[f"{k}_std"] = float(v.std(ddof=1)) if v.size > 1 else 0.0
        out[f"{k}_min"] = float(v.min())
        out[f"{k}_max"] = float(v.max())
    return out
