"""
Auto-optimizer over `ConfluenceScorer` parameters.

Pipeline
--------
1. Caller supplies a time-ordered OHLC DataFrame and the bar timeframe.
2. The bars are split into TRAIN and HOLDOUT segments (default 70/30) with
   explicit minimum-size guards on BOTH halves.
3. Optuna's TPE sampler proposes parameters; for each trial we run
   `walk_forward` on TRAIN (NOT on holdout — that data must remain unseen).
4. The composite `objective.score(...)` collapses fold metrics into a scalar.
   Failed parameter combos are reported as `optuna.TrialPruned` so they don't
   pollute the TPE surrogate model.
5. The best trial is re-evaluated on HOLDOUT for an honest, untouched
   out-of-sample number — this is the figure to trust. Train metrics are
   recovered from the trial's user_attrs (no redundant re-evaluation).

The search space is intentionally broad but anchored: thresholds, RR,
SL buffer, swing window (single symmetric param), scorer weights.

Optuna is an *optional* dependency (declared in requirements.txt). If it
isn't installed, this module raises a clear ImportError.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..config import BacktestConfig, CFG
from ..backtest.walkforward import walk_forward, aggregate_walk_forward
from ..metrics import bars_per_year as _bpy_lookup, compute_perf
from ..strategy.runner import backtest_strategy
from .objective import ObjectiveConfig, score as score_run


_TIMEFRAME_BARS_PER_DAY = {
    "1m": 1440.0, "5m": 288.0, "15m": 96.0,
    "1h": 24.0, "4h": 6.0, "1d": 1.0, "1w": 1.0 / 7.0,
}
# Use 30.4375 (= 365.25 / 12) so a "month" is consistent with the 365-day year
# implied by the crypto default in `metrics.bars_per_year`.
_DAYS_PER_MONTH = 365.25 / 12.0
_TIMEFRAME_BARS_PER_MONTH = {
    tf: bpd * _DAYS_PER_MONTH for tf, bpd in _TIMEFRAME_BARS_PER_DAY.items()
}
# Equity/FX market_hours=True: ~6.5 trading hours/day, 252 trading days/year.
# Intraday TFs see far fewer bars/day than 24h crypto; daily/weekly only need
# the trading-day fraction. Build the table explicitly to avoid the dimensional
# bug of scaling 24h-derived totals by 252/365.
_TRADING_HOURS_PER_DAY = 6.5
_TRADING_DAYS_PER_MONTH = 252.0 / 12.0
_TIMEFRAME_BARS_PER_DAY_MH = {
    "1m": 60.0 * _TRADING_HOURS_PER_DAY,
    "5m": 12.0 * _TRADING_HOURS_PER_DAY,
    "15m": 4.0 * _TRADING_HOURS_PER_DAY,
    "1h": _TRADING_HOURS_PER_DAY,
    "4h": _TRADING_HOURS_PER_DAY / 4.0,
    "1d": 1.0,
    "1w": 1.0 / 5.0,  # 5 trading days per week
}
_TIMEFRAME_BARS_PER_MONTH_MH = {
    tf: bpd * _TRADING_DAYS_PER_MONTH
    for tf, bpd in _TIMEFRAME_BARS_PER_DAY_MH.items()
}

# Train/holdout MUST each have at least this many bars or walk-forward is
# meaningless (purged_kfold_indices needs >= n_splits*2 anyway).
_MIN_BARS_FOR_WALKFORWARD = 500


@dataclass
class OptimizeConfig:
    timeframe: str = "1h"
    holdout_frac: float = 0.30
    n_trials: int = 50
    n_jobs: int = 1
    seed: int = 42
    # IMPORTANT: use `replace` to copy from the global CFG.bt so each
    # OptimizeConfig instance gets its own BacktestConfig (not the shared one).
    bt: BacktestConfig = field(default_factory=lambda: replace(CFG.bt))
    objective_cfg: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    market_hours: bool = False


def _bars_per_day(timeframe: str, market_hours: bool) -> float:
    """Bars per *calendar* day for the given timeframe, optionally restricted
    to a 6.5h trading session (market_hours=True). Falls back to 1.0 for
    unknown timeframes (caller likely passed daily-equivalent bars).
    """
    table = _TIMEFRAME_BARS_PER_DAY_MH if market_hours else _TIMEFRAME_BARS_PER_DAY
    return float(table.get(timeframe, 1.0))


def _bars_per_month(timeframe: str, market_hours: bool) -> float:
    table = _TIMEFRAME_BARS_PER_MONTH_MH if market_hours else _TIMEFRAME_BARS_PER_MONTH
    if timeframe in table:
        return float(table[timeframe])
    # Unknown TF: derive consistently from bars/day so units stay coherent.
    bpd = _bars_per_day(timeframe, market_hours)
    days_per_month = _TRADING_DAYS_PER_MONTH if market_hours else _DAYS_PER_MONTH
    return float(bpd * days_per_month)


def _suggest_params(trial) -> dict[str, Any]:
    """Search space. Keep ranges economic: avoid degenerate corners.

    Swing left/right are tied to a single symmetric parameter so we don't
    waste TPE capacity on lookahead-asymmetric pivots.
    """
    swing = trial.suggest_int("swing", 3, 12)
    p: dict[str, Any] = {
        # Thresholds + dominance
        "entry_threshold": trial.suggest_float("entry_threshold", 0.6, 2.5),
        "min_dominance": trial.suggest_float("min_dominance", 0.1, 1.5),
        # Risk shape
        "rr": trial.suggest_float("rr", 1.0, 4.0),
        "sl_buffer_atr": trial.suggest_float("sl_buffer_atr", 0.1, 1.5),
        "wick_min_atr": trial.suggest_float("wick_min_atr", 0.2, 1.2),
        # Structure detection
        "atr_period": trial.suggest_int("atr_period", 7, 28),
        "swing_left": swing,
        "swing_right": swing,
        # Volume / Wyckoff / VWAP knobs
        "vol_ma_period": trial.suggest_int("vol_ma_period", 10, 50),
        "wyckoff_lookback": trial.suggest_int("wyckoff_lookback", 20, 60),
        "wyckoff_compression": trial.suggest_float("wyckoff_compression", 0.4, 0.85),
        "vwap_z_threshold": trial.suggest_float("vwap_z_threshold", 0.5, 2.5),
        # Scorer weights
        "w_trend": trial.suggest_float("w_trend", 0.0, 2.5),
        "w_sfp": trial.suggest_float("w_sfp", 0.0, 2.5),
        "w_candle": trial.suggest_float("w_candle", 0.0, 2.5),
        "w_regime": trial.suggest_float("w_regime", 0.0, 2.5),
        "w_vsa": trial.suggest_float("w_vsa", 0.0, 2.5),
        "w_wyckoff": trial.suggest_float("w_wyckoff", 0.0, 2.5),
        "w_vwap_dev": trial.suggest_float("w_vwap_dev", 0.0, 2.5),
    }
    return p


def _evaluate(bars: pd.DataFrame, params: dict, ocfg: OptimizeConfig
              ) -> tuple[pd.DataFrame, float]:
    """Walk-forward the candidate on `bars`. Returns (folds_df, bpm)."""
    bpd = _bars_per_day(ocfg.timeframe, ocfg.market_hours)
    bpm = _bars_per_month(ocfg.timeframe, ocfg.market_hours)
    bpy = _bpy_lookup(ocfg.timeframe, market_hours=ocfg.market_hours)
    folds_df = walk_forward(
        bars,
        cfg=ocfg.bt,
        bars_per_day=bpd,
        strategy_kwargs={**params, "strategy_kind": "scorer"},
        bars_per_year_value=bpy,
    )
    return folds_df, bpm


def _train_summary_from_trial(trial) -> dict:
    """Reconstruct the train-side score dict from `m_*` user_attrs so we don't
    have to re-run the full walk-forward on TRAIN."""
    out: dict = {}
    for k, v in trial.user_attrs.items():
        if k.startswith("m_"):
            out[k[2:]] = v
    # `m_score` is set by `_obj`; only fill `score` from trial.value if absent.
    if "score" not in out:
        out["score"] = float(trial.value) if trial.value is not None else None
    return out


_PERF_FOLD_DEFAULTS = {
    "sharpe": 0.0, "sortino": 0.0, "profit_factor": 0.0,
    "expectancy": 0.0, "expectancy_R": 0.0,
    "win_rate": 0.0, "max_drawdown": 0.0,
    "n_trades": 0,
}


def _perf_to_fold_row(perf, n_bars: int) -> dict:
    """Coerce a `PerfStats` (or anything with .to_dict()) into a row that
    matches the schema expected by `objective.score`.
    """
    d = perf.to_dict() if hasattr(perf, "to_dict") else dict(perf)
    for k, default in _PERF_FOLD_DEFAULTS.items():
        d.setdefault(k, default)
    d["test_bars"] = int(n_bars)
    return d


def optimize(bars: pd.DataFrame,
             ocfg: Optional[OptimizeConfig] = None,
             study=None) -> dict:
    """Run an Optuna study and return best params + holdout metrics.

    Returns a dict:
        {
          "best_params": {...},
          "train_score": {...},          # composite + components on TRAIN
          "holdout_score": {...},        # composite + components on HOLDOUT
          "holdout_folds": pd.DataFrame, # per-fold holdout metrics
          "holdout_aggregate": {...},
          "study": optuna.Study,
          "n_trials": int,
        }
    """
    try:
        import optuna
    except ImportError as e:
        raise ImportError(
            "optuna is required for algo.optimize.auto.optimize(). "
            "Install with: pip install optuna"
        ) from e

    ocfg = ocfg or OptimizeConfig()
    n = len(bars)

    # Need enough for BOTH halves to host a walk-forward (or a single-shot
    # holdout). Compute the split first and validate each side.
    if not (0.0 < ocfg.holdout_frac < 1.0):
        raise ValueError(f"holdout_frac must be in (0,1), got {ocfg.holdout_frac}")
    split = int(n * (1.0 - ocfg.holdout_frac))
    # Preserve original index — walk-forward and metrics may rely on it for
    # time-aware purging or annualization downstream.
    train_bars = bars.iloc[:split].copy()
    holdout_bars = bars.iloc[split:].copy()

    if len(train_bars) < _MIN_BARS_FOR_WALKFORWARD:
        raise ValueError(
            f"train segment has {len(train_bars)} bars; need "
            f">= {_MIN_BARS_FOR_WALKFORWARD}. Provide more data or lower holdout_frac."
        )
    if len(holdout_bars) < 50:
        raise ValueError(
            f"holdout segment has {len(holdout_bars)} bars; need >= 50."
        )

    # Validate the index BEFORE constructing the sampler — a bad index should
    # short-circuit cheaply. Walk-forward purging assumes time order, so we
    # require a monotonic index (real bug class, unlike a type check that
    # pandas already enforces structurally).
    if not bars.index.is_monotonic_increasing:
        raise ValueError(
            "bars.index must be monotonic increasing for walk-forward purging"
        )
    if not isinstance(bars.index, pd.DatetimeIndex):
        warnings.warn(
            "bars.index is not a DatetimeIndex; time-aware purging in"
            " walk_forward may degrade to positional behavior.",
            RuntimeWarning, stacklevel=2,
        )

    sampler = optuna.samplers.TPESampler(seed=ocfg.seed, multivariate=True)
    if study is None:
        # No pruner: we don't emit intermediate values, so MedianPruner would
        # be dead code. Trials always run to completion or are TrialPruned
        # explicitly on exception.
        study = optuna.create_study(direction="maximize", sampler=sampler)
        if ocfg.n_jobs > 1:
            warnings.warn(
                "n_jobs>1 with the default in-memory study is not safe for"
                " persistence across processes. Pass a `study=` backed by"
                " Optuna RDB/JournalStorage for parallel runs.",
                RuntimeWarning, stacklevel=2,
            )

    # Hoist timeframe-derived constants out of the trial loop.
    bpm_const = _bars_per_month(ocfg.timeframe, ocfg.market_hours)

    def _obj(trial) -> float:
        params = _suggest_params(trial)
        try:
            folds_df, _ = _evaluate(train_bars, params, ocfg)
        except (KeyboardInterrupt, MemoryError, SystemExit):
            # Never swallow these — they signal user/system intent.
            raise
        except Exception as e:
            trial.set_user_attr("error", repr(e))
            # Tell TPE to ignore this point rather than treating -1e9 as a real
            # observation that biases the surrogate.
            raise optuna.TrialPruned() from e
        s = score_run(folds_df, bars_per_month=bpm_const, cfg=ocfg.objective_cfg)
        for k, v in s.items():
            if k != "config":
                trial.set_user_attr(f"m_{k}", v)
        score_val = float(s["score"])
        if not np.isfinite(score_val):
            raise optuna.TrialPruned()
        return score_val

    study.optimize(_obj, n_trials=ocfg.n_trials, n_jobs=ocfg.n_jobs,
                   show_progress_bar=False)

    if not any(t.state.name == "COMPLETE" for t in study.trials):
        raise RuntimeError("no Optuna trials completed successfully; check trial.user_attrs['error']")

    best = study.best_trial
    best_params = dict(best.params)
    # Expand the symmetric "swing" param back into swing_left/swing_right for
    # downstream consumers that expect them explicitly. Always overwrite so
    # changes to the search space (e.g. someone splitting the param) don't
    # silently survive.
    if "swing" in best_params:
        sw = best_params.pop("swing")
        best_params["swing_left"] = sw
        best_params["swing_right"] = sw

    train_summary = _train_summary_from_trial(best)
    train_summary.setdefault("config", {})

    # Holdout: untouched data. Walk-forward inside it for stability when large
    # enough; otherwise a single-shot backtest with the same metric schema.
    if len(holdout_bars) >= _MIN_BARS_FOR_WALKFORWARD:
        holdout_folds, _ = _evaluate(holdout_bars, best_params, ocfg)
        holdout_summary = score_run(holdout_folds, bars_per_month=bpm_const,
                                    cfg=ocfg.objective_cfg)
        holdout_agg = aggregate_walk_forward(holdout_folds)
    else:
        bpy = _bpy_lookup(ocfg.timeframe, market_hours=ocfg.market_hours)
        result = backtest_strategy(holdout_bars,
                                   strategy_kind="scorer",
                                   risk=ocfg.bt.risk_per_trade,
                                   fee_bps=ocfg.bt.fee_bps,
                                   slip_bps=ocfg.bt.slippage_bps,
                                   initial_balance=ocfg.bt.initial_balance,
                                   max_concurrent=ocfg.bt.max_concurrent_trades,
                                   **best_params)
        perf = compute_perf(result["equity"], result["trades"], bars_per_year=bpy)
        # Build a schema-complete fold row so `score_run` doesn't trip on a
        # missing key from a future PerfStats refactor.
        holdout_folds = pd.DataFrame([_perf_to_fold_row(perf, len(holdout_bars))])
        holdout_summary = score_run(holdout_folds, bars_per_month=bpm_const,
                                    cfg=ocfg.objective_cfg)
        holdout_agg = aggregate_walk_forward(holdout_folds)

    return {
        "best_params": best_params,
        "train_score": train_summary,
        "holdout_score": holdout_summary,
        "holdout_folds": holdout_folds,
        "holdout_aggregate": holdout_agg,
        "study": study,
        "n_trials": ocfg.n_trials,
    }


def rank_top_strategies(study, top_k: int = 10,
                        dedupe_round: int = 2) -> pd.DataFrame:
    """Return the top-k trials sorted by score, with their per-trial metrics.

    TPE often re-samples near the same mode; deduplicate by rounding param
    values to `dedupe_round` decimals so the report surfaces diverse
    candidates rather than ten clones of one peak.
    """
    rows = []
    for t in study.trials:
        if t.state.name != "COMPLETE":
            continue
        if t.value is None or not np.isfinite(t.value):
            continue
        row = {"trial": t.number, "score": float(t.value), **t.params}
        for k, v in t.user_attrs.items():
            if k.startswith("m_"):
                row[k[2:]] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    if dedupe_round is not None and dedupe_round >= 0:
        param_cols = [c for c in df.columns
                      if c not in ("trial", "score") and not c.startswith("m_")]
        numeric = df[param_cols].select_dtypes(include=[np.number]).columns
        sig = df[param_cols].copy()
        if len(numeric):
            sig[numeric] = sig[numeric].round(dedupe_round)
        keep = ~sig.duplicated(keep="first")
        df = df.loc[keep].reset_index(drop=True)
    return df.head(top_k).reset_index(drop=True)
