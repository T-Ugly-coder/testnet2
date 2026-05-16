from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import optuna

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algo_download.config import BacktestConfig
from algo_download.metrics import compute_perf
from algo_download.optimize import ObjectiveConfig, OptimizeConfig, optimize
from algo_download.optimize.auto import score_run
from algo_download.strategy.runner import backtest_strategy
from algo_download.backtest import walkforward as walkforward_mod
from tools.enhanced_strategy import enhanced_backtest_strategy


BARS_PER_YEAR = {
    "1m": 365 * 24 * 60,
    "5m": 365 * 24 * 12,
    "15m": 365 * 24 * 4,
    "1h": 365 * 24,
    "4h": 365 * 6,
    "1d": 365,
    "1w": 52,
}

SEED_PARAM_FILES = (
    Path("data_store/multi_seed_auto_150_v2/BTCUSDT_1h_backtest_5y_tf-1h_target-22_seed-42.json"),
    Path("data_store/managed_exit_selective/BTCUSDT_4h_backtest_5y_tf-4h_target-8_seed-11.json"),
    Path("data_store/breakout_confirm_alt_seeds/best_passed.json"),
    Path("data_store/alt_champion.json"),
)

PARAM_BOUNDS: dict[str, tuple[float, float, bool]] = {
    "entry_threshold": (0.5, 2.7, False),
    "min_dominance": (0.1, 1.5, False),
    "rr": (0.8, 4.0, False),
    "sl_buffer_atr": (0.1, 1.5, False),
    "wick_min_atr": (0.2, 1.2, False),
    "atr_period": (7, 28, True),
    "vol_ma_period": (10, 50, True),
    "wyckoff_lookback": (20, 60, True),
    "wyckoff_compression": (0.4, 0.85, False),
    "vwap_z_threshold": (0.5, 2.5, False),
    "breakout_lookback": (12, 96, True),
    "breakout_atr_buffer": (0.0, 0.6, False),
    "retest_window": (3, 18, True),
    "retest_atr_buffer": (0.05, 0.8, False),
    "failed_breakout_lookback": (12, 96, True),
    "failed_breakout_atr_buffer": (0.0, 0.6, False),
    "failed_breakout_reclaim_atr": (0.0, 0.4, False),
    "volume_breakout_lookback": (12, 96, True),
    "volume_breakout_atr_buffer": (0.0, 0.6, False),
    "volume_breakout_volume_period": (10, 80, True),
    "volume_breakout_volume_mult": (1.0, 3.0, False),
    "volume_breakout_body_atr_min": (0.05, 1.2, False),
    "directional_adx_di_gap": (2.0, 14.0, False),
    "pullback_fast_ema": (13, 34, True),
    "pullback_slow_ema": (55, 180, True),
    "pullback_rsi_period": (7, 21, True),
    "pullback_rsi_floor": (38.0, 52.0, False),
    "pullback_rsi_ceiling": (48.0, 62.0, False),
    "pullback_atr_touch": (0.15, 0.90, False),
    "range_bb_period": (14, 40, True),
    "range_bb_mult": (1.6, 2.8, False),
    "range_rsi_period": (7, 21, True),
    "range_rsi_long_max": (25.0, 42.0, False),
    "range_rsi_short_min": (58.0, 75.0, False),
    "range_max_adx": (14.0, 30.0, False),
    "rsi_period": (7, 28, True),
    "rsi_band": (2.0, 15.0, False),
    "bb_period": (12, 40, True),
    "bb_mult": (1.5, 3.0, False),
    "squeeze_window": (40, 160, True),
    "squeeze_quantile": (0.15, 0.55, False),
    "adx_period": (7, 28, True),
    "adx_threshold": (12.0, 45.0, False),
    "w_trend": (0.0, 2.5, False),
    "w_sfp": (0.0, 2.5, False),
    "w_candle": (0.0, 2.5, False),
    "w_regime": (0.0, 2.5, False),
    "w_vsa": (0.0, 2.5, False),
    "w_wyckoff": (0.0, 2.5, False),
    "w_vwap_dev": (0.0, 2.5, False),
    "w_breakout": (0.0, 2.5, False),
    "w_breakout_retest": (0.0, 2.5, False),
    "w_failed_breakout": (0.0, 2.5, False),
    "w_volume_breakout": (0.0, 2.5, False),
    "w_momentum": (0.0, 2.5, False),
    "w_volatility": (0.0, 2.5, False),
    "w_adx": (0.0, 2.5, False),
    "w_directional_adx": (0.0, 2.5, False),
    "w_trend_pullback": (0.0, 2.5, False),
    "w_range_reversion": (0.0, 2.5, False),
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def _sort_score(row: dict[str, Any]) -> float:
    return _safe_float(row.get("rank_score"), -math.inf)


def _infer_tf(path: Path, fallback: str) -> str:
    parts = path.stem.split("_")
    for part in parts:
        if part in BARS_PER_YEAR:
            return part
    return fallback


def _forward_pair(backtest_path: Path) -> Path | None:
    candidates = [
        Path(str(backtest_path).replace("backtest_5y", "forward_2y")),
        Path(str(backtest_path).replace("backtest", "forward")),
    ]
    for candidate in candidates:
        if candidate != backtest_path and candidate.exists():
            return candidate
    return None


def _load_bars(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported bars file: {path}")


def _load_seed_param_sets(seed_files: list[str] | tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    seeds: list[dict[str, Any]] = []
    paths = [Path(p) for p in seed_files] if seed_files else list(SEED_PARAM_FILES)
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        params = data.get("best_params", data)
        if isinstance(params, dict) and params:
            seeds.append(dict(params))
    return seeds


def _jitter_seed_params(
    params: dict[str, Any],
    *,
    rng: np.random.Generator,
    jitter: float,
    rr_min: float,
    rr_max: float,
) -> dict[str, Any]:
    variant = dict(params)
    scale = max(0.0, float(jitter))
    for name, value in list(params.items()):
        if name not in PARAM_BOUNDS or not isinstance(value, int | float):
            continue
        low, high, is_int = PARAM_BOUNDS[name]
        if name == "rr":
            low, high = max(low, rr_min), min(high, rr_max)
        span = high - low
        if span <= 0:
            continue
        moved = float(value) + rng.normal(0.0, span * scale)
        moved = min(max(moved, low), high)
        variant[name] = int(round(moved)) if is_int else moved
    if "swing_left" in variant or "swing_right" in variant:
        swing = int(round(float(variant.get("swing_left", variant.get("swing_right", 5)))))
        swing = min(max(swing, 3), 12)
        variant["swing_left"] = swing
        variant["swing_right"] = swing
    return variant


def _quality_params(trial: optuna.Trial) -> dict[str, Any]:
    args = trial.study.user_attrs.get("args")
    min_target_move_pct = _safe_float(getattr(args, "min_target_move_pct", 0.0), 0.0) if args else 0.0
    min_tp_floor = max(0.0, min_target_move_pct / 100.0)
    mode = trial.suggest_categorical("quality_mode", ["off", "light", "balanced"])
    if mode == "off":
        return {
            "quality_mode": mode,
            "min_tp_pct": 0.0,
            "max_sl_pct": 1.0,
            "min_signal_score": 0.0,
            "min_long_signal_score": 0.0,
            "min_short_signal_score": 0.0,
            "min_score_edge": 0.0,
            "min_long_dominance": 0.0,
            "min_short_dominance": 0.0,
            "min_score_rank": 0.0,
            "min_long_score_rank": 0.0,
            "min_short_score_rank": 0.0,
            "score_rank_lookback": 120,
            "min_lookout_score": 0.0,
            "min_adx_entry": 0.0,
            "trend_filter": "none",
            "trend_ema_period": 200,
            "cooldown_bars": trial.suggest_int("cooldown_bars_off", 0, 24),
        }
    if mode == "light":
        min_tp_hi = max(0.035, min_tp_floor + 0.02)
        return {
            "quality_mode": mode,
            "min_tp_pct": trial.suggest_float("min_tp_pct_light", min_tp_floor, min_tp_hi),
            "max_sl_pct": trial.suggest_float("max_sl_pct_light", 0.03, 0.20),
            "min_signal_score": trial.suggest_float("min_signal_score_light", 0.0, 1.5),
            "min_long_signal_score": trial.suggest_float("min_long_signal_score_light", 0.0, 1.8),
            "min_short_signal_score": trial.suggest_float("min_short_signal_score_light", 0.0, 1.8),
            "min_score_edge": trial.suggest_float("min_score_edge_light", 0.0, 0.9),
            "min_long_dominance": trial.suggest_float("min_long_dominance_light", 0.0, 1.0),
            "min_short_dominance": trial.suggest_float("min_short_dominance_light", 0.0, 1.0),
            "min_score_rank": trial.suggest_float("min_score_rank_light", 0.0, 0.75),
            "min_long_score_rank": trial.suggest_float("min_long_score_rank_light", 0.0, 0.80),
            "min_short_score_rank": trial.suggest_float("min_short_score_rank_light", 0.0, 0.80),
            "score_rank_lookback": trial.suggest_int("score_rank_lookback_light", 60, 220),
            "min_lookout_score": trial.suggest_float("min_lookout_score_light", 0.60, 0.82),
            "min_adx_entry": trial.suggest_float("min_adx_entry_light", 0.0, 25.0),
            "trend_filter": trial.suggest_categorical("trend_filter_light", ["none", "ema"]),
            "trend_ema_period": trial.suggest_int("trend_ema_period_light", 100, 260),
            "cooldown_bars": trial.suggest_int("cooldown_bars_light", 0, 48),
        }
    min_tp_hi = max(0.06, min_tp_floor + 0.03)
    return {
        "quality_mode": mode,
        "min_tp_pct": trial.suggest_float("min_tp_pct_balanced", min_tp_floor, min_tp_hi),
        "max_sl_pct": trial.suggest_float("max_sl_pct_balanced", 0.025, 0.16),
        "min_signal_score": trial.suggest_float("min_signal_score_balanced", 0.0, 2.5),
        "min_long_signal_score": trial.suggest_float("min_long_signal_score_balanced", 0.0, 3.0),
        "min_short_signal_score": trial.suggest_float("min_short_signal_score_balanced", 0.0, 3.0),
        "min_score_edge": trial.suggest_float("min_score_edge_balanced", 0.0, 1.5),
        "min_long_dominance": trial.suggest_float("min_long_dominance_balanced", 0.0, 1.8),
        "min_short_dominance": trial.suggest_float("min_short_dominance_balanced", 0.0, 1.8),
        "min_score_rank": trial.suggest_float("min_score_rank_balanced", 0.0, 0.9),
        "min_long_score_rank": trial.suggest_float("min_long_score_rank_balanced", 0.0, 0.95),
        "min_short_score_rank": trial.suggest_float("min_short_score_rank_balanced", 0.0, 0.95),
        "score_rank_lookback": trial.suggest_int("score_rank_lookback_balanced", 60, 260),
        "min_lookout_score": trial.suggest_float("min_lookout_score_balanced", 0.72, 0.90),
        "min_adx_entry": trial.suggest_float("min_adx_entry_balanced", 0.0, 35.0),
        "trend_filter": trial.suggest_categorical("trend_filter_balanced", ["none", "ema", "ema_slope"]),
        "trend_ema_period": trial.suggest_int("trend_ema_period_balanced", 100, 300),
        "cooldown_bars": trial.suggest_int("cooldown_bars_balanced", 0, 72),
    }


def _exit_params(trial: optuna.Trial, args: argparse.Namespace) -> dict[str, Any]:
    max_hold_hours = (
        trial.suggest_float("max_hold_hours", args.min_exit_hold_hours, args.max_exit_hold_hours)
        if args.optimize_hold_time
        else 0.0
    )
    if not args.optimize_exits:
        return {
            "exit_mode": "none",
            "breakeven_after_r": 0.0,
            "partial_tp_r": 0.0,
            "partial_close_frac": 0.5,
            "trail_after_r": 0.0,
            "trail_atr_mult": 0.0,
            "trail_atr_period": 14,
            "max_hold_hours": max_hold_hours,
        }
    if args.exit_profile == "swing":
        modes = [
            "none",
            "breakeven",
            "partial",
            "partial_breakeven",
            "trail",
            "partial_trail",
            "partial_breakeven_trail",
        ]
        partial_min, partial_max = 0.8, 2.0
        close_min, close_max = 0.15, 0.45
        breakeven_min, breakeven_max = 0.8, 2.2
    else:
        modes = ["none", "breakeven", "partial", "partial_breakeven"]
        partial_min, partial_max = 0.4, 1.1
        close_min, close_max = 0.25, 0.75
        breakeven_min, breakeven_max = 0.5, 1.4

    mode = trial.suggest_categorical("exit_mode", modes)
    uses_partial = mode in {"partial", "partial_breakeven", "partial_trail", "partial_breakeven_trail"}
    uses_breakeven = mode in {"breakeven", "partial_breakeven", "partial_breakeven_trail"}
    uses_trail = mode in {"trail", "partial_trail", "partial_breakeven_trail"}
    return {
        "exit_mode": mode,
        "breakeven_after_r": trial.suggest_float("breakeven_after_r", breakeven_min, breakeven_max)
        if uses_breakeven else 0.0,
        "partial_tp_r": trial.suggest_float("partial_tp_r", partial_min, partial_max)
        if uses_partial else 0.0,
        "partial_close_frac": trial.suggest_float("partial_close_frac", close_min, close_max)
        if uses_partial else 0.5,
        "trail_after_r": trial.suggest_float("trail_after_r", 0.8, 2.5)
        if uses_trail else 0.0,
        "trail_atr_mult": trial.suggest_float("trail_atr_mult", 1.5, 5.0)
        if uses_trail else 0.0,
        "trail_atr_period": trial.suggest_int("trail_atr_period", 7, 28)
        if uses_trail else 14,
        "max_hold_hours": max_hold_hours,
    }


def _objective_for_target(target: float, args: argparse.Namespace) -> ObjectiveConfig:
    return ObjectiveConfig(
        target_trades_per_month=target,
        min_trades_per_month=max(1.0, target * args.min_tpm_ratio),
        max_trades_per_month=max(2.0, target * args.max_tpm_ratio),
        soft_dd_cap=args.soft_dd_cap,
        hard_dd_cap=args.hard_dd_cap,
        min_win_rate=args.min_win_rate,
        stability_lambda=args.stability_lambda,
        dd_penalty_weight=args.dd_penalty_weight,
        freq_penalty_weight=args.freq_penalty_weight,
    )


def _tf_hours(tf: str) -> float:
    if tf.endswith("m"):
        return float(tf[:-1]) / 60.0
    if tf.endswith("h"):
        return float(tf[:-1])
    if tf.endswith("d"):
        return float(tf[:-1]) * 24.0
    if tf.endswith("w"):
        return float(tf[:-1]) * 24.0 * 7.0
    return 1.0


def _trade_profile_metrics(trades: pd.DataFrame, tf: str) -> dict[str, Any]:
    if trades.empty:
        return {
            "avg_abs_price_move_pct": 0.0,
            "median_abs_price_move_pct": 0.0,
            "avg_aligned_price_move_pct": 0.0,
            "avg_hold_bars": 0.0,
            "median_hold_bars": 0.0,
            "avg_hold_hours": 0.0,
            "median_hold_hours": 0.0,
            "funding_exposure_days": 0.0,
        }
    entry = trades["entry"].to_numpy(float)
    exit_ = trades["exit"].to_numpy(float)
    direction = trades["direction"].to_numpy(float)
    move_pct = ((exit_ - entry) / np.maximum(np.abs(entry), 1e-12)) * direction * 100.0
    hold_bars = (trades["exit_idx"].to_numpy(float) - trades["entry_idx"].to_numpy(float)).clip(min=0.0)
    hold_hours = hold_bars * _tf_hours(tf)
    return {
        "avg_abs_price_move_pct": float(np.mean(np.abs(move_pct))),
        "median_abs_price_move_pct": float(np.median(np.abs(move_pct))),
        "avg_aligned_price_move_pct": float(np.mean(move_pct)),
        "avg_hold_bars": float(np.mean(hold_bars)),
        "median_hold_bars": float(np.median(hold_bars)),
        "avg_hold_hours": float(np.mean(hold_hours)),
        "median_hold_hours": float(np.median(hold_hours)),
        "funding_exposure_days": float(np.sum(hold_hours) / 24.0),
    }


def _monte_carlo_forward(
    trades: pd.DataFrame,
    *,
    runs: int,
    seed: int,
    risk_per_trade: float,
    initial_balance: float,
) -> dict[str, Any]:
    if runs <= 0 or trades.empty or "risk_amt" not in trades:
        return {}
    risk_amt = trades["risk_amt"].to_numpy(float)
    pnl = trades["pnl"].to_numpy(float)
    ok = np.isfinite(risk_amt) & (risk_amt > 0.0) & np.isfinite(pnl)
    pnl_r = pnl[ok] / risk_amt[ok]
    if pnl_r.size == 0:
        return {}

    rng = np.random.default_rng(seed)
    n = int(pnl_r.size)
    returns = np.empty(runs, dtype=float)
    max_dds = np.empty(runs, dtype=float)
    for run_idx in range(runs):
        sampled = rng.choice(pnl_r, size=n, replace=True)
        trade_returns = np.maximum(1.0 + sampled * risk_per_trade, 0.01)
        equity = initial_balance * np.cumprod(trade_returns)
        peaks = np.maximum.accumulate(equity)
        dd = equity / peaks - 1.0
        returns[run_idx] = equity[-1] / initial_balance - 1.0
        max_dds[run_idx] = abs(float(dd.min()))

    return {
        "mc_runs": int(runs),
        "mc_profit_prob": float(np.mean(returns > 0.0)),
        "mc_median_return": float(np.median(returns)),
        "mc_p05_return": float(np.quantile(returns, 0.05)),
        "mc_p95_drawdown": float(np.quantile(max_dds, 0.95)),
        "mc_median_drawdown": float(np.median(max_dds)),
    }


def _strategy_metrics(
    bars: pd.DataFrame,
    tf: str,
    params: dict[str, Any],
    strategy: str,
    args: argparse.Namespace,
    *,
    fee_bps: float,
    slip_bps: float,
) -> dict[str, Any]:
    runtime_params = dict(params)
    runtime_params.pop("fee_bps", None)
    runtime_params.pop("slip_bps", None)
    runtime_params.pop("slippage_bps", None)
    if strategy == "enhanced":
        result = enhanced_backtest_strategy(
            bars,
            risk=args.risk_per_trade,
            fee_bps=fee_bps,
            slip_bps=slip_bps,
            **runtime_params,
        )
    else:
        result = backtest_strategy(
            bars,
            risk=args.risk_per_trade,
            fee_bps=fee_bps,
            slippage_bps=slip_bps,
            **runtime_params,
        )
    perf = compute_perf(result["equity"], result["trades"], BARS_PER_YEAR.get(tf, 365 * 24))
    metrics = asdict(perf)
    years = max(float(metrics["n_bars"]) / BARS_PER_YEAR.get(tf, 365 * 24), 1e-9)
    metrics["trades_per_month"] = float(metrics["n_trades"]) / (years * 12.0)
    metrics["max_drawdown_abs"] = abs(float(metrics["max_drawdown"]))
    metrics.update(_trade_profile_metrics(result["trades"], tf))
    metrics.update(_monte_carlo_forward(
        result["trades"],
        runs=args.monte_carlo_runs,
        seed=args.monte_carlo_seed,
        risk_per_trade=args.risk_per_trade,
        initial_balance=100000.0,
    ))
    metrics["fee_bps"] = float(fee_bps)
    metrics["slip_bps"] = float(slip_bps)
    return metrics


def _forward_metrics(
    path: Path,
    tf: str,
    params: dict[str, Any],
    strategy: str,
    args: argparse.Namespace,
    *,
    fee_bps: float,
    slip_bps: float,
) -> dict[str, Any]:
    return _strategy_metrics(
        _load_bars(path),
        tf,
        params,
        strategy,
        args,
        fee_bps=fee_bps,
        slip_bps=slip_bps,
    )


def _sample_enhanced_params(trial: optuna.Trial) -> dict[str, Any]:
    swing = trial.suggest_int("swing", 3, 12)
    params = {
        "entry_threshold": trial.suggest_float("entry_threshold", 0.5, 2.7),
        "min_dominance": trial.suggest_float("min_dominance", 0.1, 1.5),
        "rr": trial.suggest_float("rr", trial.study.user_attrs.get("rr_min", 0.8), trial.study.user_attrs.get("rr_max", 4.0)),
        "sl_buffer_atr": trial.suggest_float("sl_buffer_atr", 0.1, 1.5),
        "wick_min_atr": trial.suggest_float("wick_min_atr", 0.2, 1.2),
        "atr_period": trial.suggest_int("atr_period", 7, 28),
        "vol_ma_period": trial.suggest_int("vol_ma_period", 10, 50),
        "wyckoff_lookback": trial.suggest_int("wyckoff_lookback", 20, 60),
        "wyckoff_compression": trial.suggest_float("wyckoff_compression", 0.4, 0.85),
        "vwap_z_threshold": trial.suggest_float("vwap_z_threshold", 0.5, 2.5),
        "breakout_lookback": trial.suggest_int("breakout_lookback", 12, 96),
        "breakout_atr_buffer": trial.suggest_float("breakout_atr_buffer", 0.0, 0.6),
        "retest_window": trial.suggest_int("retest_window", 3, 18),
        "retest_atr_buffer": trial.suggest_float("retest_atr_buffer", 0.05, 0.8),
        "failed_breakout_lookback": trial.suggest_int("failed_breakout_lookback", 12, 96),
        "failed_breakout_atr_buffer": trial.suggest_float("failed_breakout_atr_buffer", 0.0, 0.6),
        "failed_breakout_reclaim_atr": trial.suggest_float("failed_breakout_reclaim_atr", 0.0, 0.4),
        "volume_breakout_lookback": trial.suggest_int("volume_breakout_lookback", 12, 96),
        "volume_breakout_atr_buffer": trial.suggest_float("volume_breakout_atr_buffer", 0.0, 0.6),
        "volume_breakout_volume_period": trial.suggest_int("volume_breakout_volume_period", 10, 80),
        "volume_breakout_volume_mult": trial.suggest_float("volume_breakout_volume_mult", 1.0, 3.0),
        "volume_breakout_body_atr_min": trial.suggest_float("volume_breakout_body_atr_min", 0.05, 1.2),
        "directional_adx_di_gap": trial.suggest_float("directional_adx_di_gap", 2.0, 14.0),
        "pullback_fast_ema": trial.suggest_int("pullback_fast_ema", 13, 34),
        "pullback_slow_ema": trial.suggest_int("pullback_slow_ema", 55, 180),
        "pullback_rsi_period": trial.suggest_int("pullback_rsi_period", 7, 21),
        "pullback_rsi_floor": trial.suggest_float("pullback_rsi_floor", 38.0, 52.0),
        "pullback_rsi_ceiling": trial.suggest_float("pullback_rsi_ceiling", 48.0, 62.0),
        "pullback_atr_touch": trial.suggest_float("pullback_atr_touch", 0.15, 0.90),
        "range_bb_period": trial.suggest_int("range_bb_period", 14, 40),
        "range_bb_mult": trial.suggest_float("range_bb_mult", 1.6, 2.8),
        "range_rsi_period": trial.suggest_int("range_rsi_period", 7, 21),
        "range_rsi_long_max": trial.suggest_float("range_rsi_long_max", 25.0, 42.0),
        "range_rsi_short_min": trial.suggest_float("range_rsi_short_min", 58.0, 75.0),
        "range_max_adx": trial.suggest_float("range_max_adx", 14.0, 30.0),
        "rsi_period": trial.suggest_int("rsi_period", 7, 28),
        "rsi_band": trial.suggest_float("rsi_band", 2.0, 15.0),
        "bb_period": trial.suggest_int("bb_period", 12, 40),
        "bb_mult": trial.suggest_float("bb_mult", 1.5, 3.0),
        "squeeze_window": trial.suggest_int("squeeze_window", 40, 160),
        "squeeze_quantile": trial.suggest_float("squeeze_quantile", 0.15, 0.55),
        "adx_period": trial.suggest_int("adx_period", 7, 28),
        "adx_threshold": trial.suggest_float("adx_threshold", 12.0, 45.0),
        "w_trend": trial.suggest_float("w_trend", 0.0, 2.5),
        "w_sfp": trial.suggest_float("w_sfp", 0.0, 2.5),
        "w_candle": trial.suggest_float("w_candle", 0.0, 2.5),
        "w_regime": trial.suggest_float("w_regime", 0.0, 2.5),
        "w_vsa": trial.suggest_float("w_vsa", 0.0, 2.5),
        "w_wyckoff": trial.suggest_float("w_wyckoff", 0.0, 2.5),
        "w_vwap_dev": trial.suggest_float("w_vwap_dev", 0.0, 2.5),
        "w_breakout": trial.suggest_float("w_breakout", 0.0, 2.5),
        "w_breakout_retest": trial.suggest_float("w_breakout_retest", 0.0, 2.5),
        "w_failed_breakout": trial.suggest_float("w_failed_breakout", 0.0, 2.5),
        "w_volume_breakout": trial.suggest_float("w_volume_breakout", 0.0, 2.5),
        "w_momentum": trial.suggest_float("w_momentum", 0.0, 2.5),
        "w_volatility": trial.suggest_float("w_volatility", 0.0, 2.5),
        "w_adx": trial.suggest_float("w_adx", 0.0, 2.5),
        "w_directional_adx": trial.suggest_float("w_directional_adx", 0.0, 2.5),
        "w_trend_pullback": trial.suggest_float("w_trend_pullback", 0.0, 2.5),
        "w_range_reversion": trial.suggest_float("w_range_reversion", 0.0, 2.5),
        "swing_left": swing,
        "swing_right": swing,
    }
    params.update(_quality_params(trial))
    params.update(_exit_params(trial, trial.study.user_attrs["args"]))
    return params


def _seed_to_enqueued_trial(
    params: dict[str, Any],
    target: float,
    rr_min: float,
    rr_max: float,
    min_target_move_pct: float = 0.0,
    exit_profile: str = "high_win",
) -> dict[str, Any]:
    seeded = {k: v for k, v in params.items() if isinstance(v, int | float | str)}
    if "rr" in seeded:
        seeded["rr"] = min(max(float(seeded["rr"]), rr_min), rr_max)
    min_tp_floor = max(0.0, min_target_move_pct / 100.0)
    seeded.setdefault("quality_mode", "light")
    seed_tp = 0.0 if target <= 8 else 0.015
    seeded["min_tp_pct_light"] = max(min_tp_floor, min(max(0.035, min_tp_floor + 0.02), seed_tp))
    seeded["max_sl_pct_light"] = 0.20
    seeded["min_signal_score_light"] = 0.0
    seeded["min_long_signal_score_light"] = float(seeded.get("min_long_signal_score", 0.0))
    seeded["min_short_signal_score_light"] = float(seeded.get("min_short_signal_score", 0.0))
    seeded["min_score_edge_light"] = 0.0
    seeded["min_long_dominance_light"] = float(seeded.get("min_long_dominance", 0.0))
    seeded["min_short_dominance_light"] = float(seeded.get("min_short_dominance", 0.0))
    seeded["min_score_rank_light"] = 0.0
    seeded["min_long_score_rank_light"] = float(seeded.get("min_long_score_rank", 0.0))
    seeded["min_short_score_rank_light"] = float(seeded.get("min_short_score_rank", 0.0))
    seeded["score_rank_lookback_light"] = 120
    seeded["min_lookout_score_light"] = max(0.60, min(0.82, float(seeded.get("min_lookout_score", 0.75))))
    seeded["min_adx_entry_light"] = 0.0
    seeded["trend_filter_light"] = "none"
    seeded["trend_ema_period_light"] = int(seeded.get("trend_ema_period", 200))
    seeded["cooldown_bars_light"] = 0
    seeded.setdefault("exit_mode", "none")
    if exit_profile == "swing":
        modes = {
            "none",
            "breakeven",
            "partial",
            "partial_breakeven",
            "trail",
            "partial_trail",
            "partial_breakeven_trail",
        }
        partial_min, partial_max = 0.8, 2.0
        close_min, close_max = 0.15, 0.45
        breakeven_min, breakeven_max = 0.8, 2.2
    else:
        modes = {"none", "breakeven", "partial", "partial_breakeven"}
    if seeded["exit_mode"] not in modes:
        seeded["exit_mode"] = "none"
    seeded["swing"] = int(seeded.get("swing_left", seeded.get("swing", 5)))
    return seeded


def _bars_per_day(tf: str) -> float:
    return BARS_PER_YEAR.get(tf, 365 * 24) / 365.0


def _score_with_user_targets(summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    adjusted = dict(summary)
    win_rate = float(adjusted.get("win_rate_mean", 0.0))
    tpm = float(adjusted.get("trades_per_month", 0.0))
    if tpm <= 0.0:
        adjusted["score"] = -1_000_000.0
        adjusted["reason"] = "zero_trades"
        return adjusted
    if args.min_objective_win_rate > 0 and win_rate < args.min_objective_win_rate:
        adjusted["score"] = -1_000_000.0 + win_rate
        adjusted["reason"] = "objective_win_rate_below_floor"
        return adjusted
    if args.min_objective_tpm > 0 and tpm < args.min_objective_tpm:
        adjusted["score"] = -1_000_000.0 + tpm / max(args.min_objective_tpm, 1e-9)
        adjusted["reason"] = "objective_tpm_below_floor"
        return adjusted
    if args.hard_max_objective_tpm and args.max_objective_tpm > 0 and tpm > args.max_objective_tpm:
        adjusted["score"] = -1_000_000.0 - tpm / args.max_objective_tpm
        adjusted["reason"] = "objective_tpm_above_ceiling"
        return adjusted

    win_gap = win_rate - args.win_rate_target
    adjusted["score"] = float(adjusted.get("score", 0.0)) + args.win_rate_weight * win_gap
    adjusted["score"] += args.pf_over_weight * max(0.0, float(adjusted.get("profit_factor_mean", 0.0)) - 1.0)
    adjusted["score"] += args.expectancy_r_weight * float(adjusted.get("expectancy_R_mean", 0.0))
    if args.max_objective_tpm > 0 and tpm > args.max_objective_tpm:
        adjusted["score"] -= args.tpm_over_weight * ((tpm / args.max_objective_tpm) - 1.0)
    return adjusted


def _optimize_enhanced(bars: pd.DataFrame, ocfg: OptimizeConfig, args: argparse.Namespace) -> dict[str, Any]:
    split = int(len(bars) * (1.0 - ocfg.holdout_frac))
    train_bars = bars.iloc[:split].copy()
    holdout_bars = bars.iloc[split:].copy()
    bars_per_year = BARS_PER_YEAR.get(ocfg.timeframe, 365 * 24)
    bars_per_month = bars_per_year / 12.0

    old_backtest = walkforward_mod.backtest_strategy
    walkforward_mod.backtest_strategy = enhanced_backtest_strategy
    try:
        seed_param_sets = _load_seed_param_sets(args.seed_param_files)

        def objective(trial: optuna.Trial) -> float:
            params = _sample_enhanced_params(trial)
            folds = walkforward_mod.walk_forward(
                train_bars,
                cfg=ocfg.bt,
                bars_per_day=_bars_per_day(ocfg.timeframe),
                strategy_kwargs=params,
                bars_per_year_value=bars_per_year,
            )
            summary = score_run(folds, bars_per_month, ocfg.objective_cfg)
            summary = _score_with_user_targets(summary, args)
            trial.set_user_attr("params", params)
            trial.set_user_attr("summary", summary)
            return float(summary["score"])

        sampler = optuna.samplers.TPESampler(seed=ocfg.seed, multivariate=False)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.set_user_attr("rr_min", args.rr_min)
        study.set_user_attr("rr_max", args.rr_max)
        study.set_user_attr("args", args)
        rng = np.random.default_rng(ocfg.seed + 10_000)
        for seed_params in seed_param_sets:
            study.enqueue_trial(
                _seed_to_enqueued_trial(
                    seed_params,
                    ocfg.objective_cfg.target_trades_per_month,
                    args.rr_min,
                    args.rr_max,
                    args.min_target_move_pct,
                    args.exit_profile,
                )
            )
            if args.enqueue_low_rr:
                low_rr_seed = dict(seed_params)
                low_rr_seed["rr"] = args.rr_min
                low_rr_seed["sl_buffer_atr"] = max(float(low_rr_seed.get("sl_buffer_atr", 0.5)), 1.0)
                study.enqueue_trial(
                    _seed_to_enqueued_trial(
                        low_rr_seed,
                        ocfg.objective_cfg.target_trades_per_month,
                        args.rr_min,
                        args.rr_max,
                        args.min_target_move_pct,
                        args.exit_profile,
                    )
                )
            for _ in range(max(0, int(args.seed_variant_count))):
                variant = _jitter_seed_params(
                    seed_params,
                    rng=rng,
                    jitter=args.seed_jitter,
                    rr_min=args.rr_min,
                    rr_max=args.rr_max,
                )
                study.enqueue_trial(
                    _seed_to_enqueued_trial(
                        variant,
                        ocfg.objective_cfg.target_trades_per_month,
                        args.rr_min,
                        args.rr_max,
                        args.min_target_move_pct,
                        args.exit_profile,
                    )
                )
        study.optimize(objective, n_trials=ocfg.n_trials, n_jobs=ocfg.n_jobs, show_progress_bar=False)

        best_params = dict(study.best_trial.user_attrs["params"])
        train_score = dict(study.best_trial.user_attrs["summary"])
        holdout_folds = walkforward_mod.walk_forward(
            holdout_bars,
            cfg=ocfg.bt,
            bars_per_day=_bars_per_day(ocfg.timeframe),
            strategy_kwargs=best_params,
            bars_per_year_value=bars_per_year,
        )
        holdout_score = score_run(holdout_folds, bars_per_month, ocfg.objective_cfg)
        completed_trials = [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
            and trial.value is not None
            and math.isfinite(float(trial.value))
            and "params" in trial.user_attrs
            and "summary" in trial.user_attrs
        ]
        sorted_trials = sorted(completed_trials, key=lambda trial: float(trial.value), reverse=True)
        raw_limit = max(0, int(args.save_top_trials))
        total_limit = max(raw_limit, int(args.max_candidate_trials))
        candidate_trials = sorted_trials[:raw_limit]
        if args.candidate_max_train_score > 0:
            sane_trials = [
                trial
                for trial in sorted_trials
                if args.candidate_min_train_score <= float(trial.value) <= args.candidate_max_train_score
            ][:total_limit]
            seen_numbers = {trial.number for trial in candidate_trials}
            candidate_trials.extend(trial for trial in sane_trials if trial.number not in seen_numbers)
        candidate_trials = candidate_trials[:total_limit]

        trial_candidates: list[dict[str, Any]] = []
        for candidate_rank, trial in enumerate(candidate_trials, start=1):
            candidate_params = dict(trial.user_attrs["params"])
            candidate_holdout_folds = walkforward_mod.walk_forward(
                holdout_bars,
                cfg=ocfg.bt,
                bars_per_day=_bars_per_day(ocfg.timeframe),
                strategy_kwargs=candidate_params,
                bars_per_year_value=bars_per_year,
            )
            candidate_holdout_score = score_run(candidate_holdout_folds, bars_per_month, ocfg.objective_cfg)
            trial_candidates.append({
                "candidate_rank": candidate_rank,
                "trial_number": int(trial.number),
                "trial_value": float(trial.value),
                "best_params": candidate_params,
                "train_score": dict(trial.user_attrs["summary"]),
                "holdout_score": candidate_holdout_score,
            })
        return {
            "best_params": best_params,
            "train_score": train_score,
            "holdout_score": holdout_score,
            "holdout_folds": holdout_folds,
            "trial_candidates": trial_candidates,
            "study": study,
            "n_trials": ocfg.n_trials,
        }
    finally:
        walkforward_mod.backtest_strategy = old_backtest


def _guardrail_failures(row: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    train = row["train_score"]
    holdout = row["holdout_score"]
    train_score = _safe_float(train.get("score"), 0.0)
    holdout_score = _safe_float(holdout.get("score"), 0.0)
    if train_score - holdout_score > args.max_train_holdout_score_gap:
        failures.append("train_holdout_score_gap")
    train_pf = _safe_float(train.get("profit_factor_mean"), 0.0)
    holdout_pf = _safe_float(holdout.get("profit_factor_mean"), 0.0)
    if holdout_pf > 0 and train_pf / holdout_pf > args.max_train_holdout_pf_ratio:
        failures.append("train_holdout_pf_ratio")
    if holdout_pf < args.min_holdout_pf:
        failures.append("holdout_pf")
    if _safe_float(holdout.get("win_rate_mean"), 0.0) < args.min_holdout_win_rate:
        failures.append("holdout_win_rate")
    if _safe_float(holdout.get("expectancy_R_mean"), 0.0) <= args.min_holdout_expectancy_r:
        failures.append("holdout_expectancy_r")
    if _safe_float(holdout.get("max_drawdown_abs"), 1.0) > args.max_holdout_dd:
        failures.append("holdout_dd")
    if args.min_holdout_tpm > 0 and _safe_float(holdout.get("trades_per_month"), 0.0) < args.min_holdout_tpm:
        failures.append("holdout_tpm_low")
    if args.max_holdout_tpm > 0 and _safe_float(holdout.get("trades_per_month"), 0.0) > args.max_holdout_tpm:
        failures.append("holdout_tpm_high")
    forward = row.get("forward_score")
    if forward:
        if _safe_float(forward.get("profit_factor"), 0.0) < args.min_forward_pf:
            failures.append("forward_pf")
        if _safe_float(forward.get("win_rate"), 0.0) < args.min_forward_win_rate:
            failures.append("forward_win_rate")
        if _safe_float(forward.get("expectancy_R"), 0.0) <= args.min_forward_expectancy_r:
            failures.append("forward_expectancy_r")
        if _safe_float(forward.get("max_drawdown_abs"), 1.0) > args.max_forward_dd:
            failures.append("forward_dd")
        if _safe_float(forward.get("total_return"), 0.0) < args.min_forward_return:
            failures.append("forward_return")
        if _safe_float(forward.get("cagr"), 0.0) < args.min_forward_cagr:
            failures.append("forward_cagr")
        if args.min_forward_tpm > 0 and _safe_float(forward.get("trades_per_month"), 0.0) < args.min_forward_tpm:
            failures.append("forward_tpm_low")
        if args.max_forward_tpm > 0 and _safe_float(forward.get("trades_per_month"), 0.0) > args.max_forward_tpm:
            failures.append("forward_tpm_high")
        if _safe_float(forward.get("avg_abs_price_move_pct"), 0.0) < args.min_avg_trade_move_pct:
            failures.append("forward_trade_move")
        if args.max_avg_hold_hours > 0 and _safe_float(forward.get("avg_hold_hours"), 1e9) > args.max_avg_hold_hours:
            failures.append("forward_avg_hold")
        if args.max_median_hold_hours > 0 and _safe_float(forward.get("median_hold_hours"), 1e9) > args.max_median_hold_hours:
            failures.append("forward_median_hold")
        if args.min_mc_profit_prob > 0 and _safe_float(forward.get("mc_profit_prob"), 0.0) < args.min_mc_profit_prob:
            failures.append("monte_carlo_profit_prob")
        if args.min_mc_p05_return > -1.0 and _safe_float(forward.get("mc_p05_return"), -1.0) < args.min_mc_p05_return:
            failures.append("monte_carlo_p05_return")
        if args.max_mc_p95_drawdown > 0 and _safe_float(forward.get("mc_p95_drawdown"), 1.0) > args.max_mc_p95_drawdown:
            failures.append("monte_carlo_p95_dd")
    stress = row.get("forward_stress_score")
    if stress:
        if _safe_float(stress.get("profit_factor"), 0.0) < args.min_stress_forward_pf:
            failures.append("stress_forward_pf")
        if _safe_float(stress.get("expectancy_R"), 0.0) <= args.min_stress_forward_expectancy_r:
            failures.append("stress_forward_expectancy_r")
        if _safe_float(stress.get("max_drawdown_abs"), 1.0) > args.max_stress_forward_dd:
            failures.append("stress_forward_dd")
        if _safe_float(stress.get("total_return"), 0.0) < args.min_stress_forward_return:
            failures.append("stress_forward_return")
    return failures


def _passes_guardrails(row: dict[str, Any], args: argparse.Namespace) -> bool:
    return not _guardrail_failures(row, args)


def _rank_score(row: dict[str, Any], args: argparse.Namespace) -> float:
    holdout = row["holdout_score"]
    score = _safe_float(holdout.get("score"), -1e9)
    score += 3.0 * max(0.0, _safe_float(holdout.get("profit_factor_mean"), 0.0) - 1.0)
    score += 3.0 * _safe_float(holdout.get("expectancy_R_mean"), 0.0)
    score += args.win_rate_weight * (_safe_float(holdout.get("win_rate_mean"), 0.0) - args.win_rate_target)
    score -= 3.0 * _safe_float(holdout.get("max_drawdown_abs"), 0.0)
    if args.max_holdout_tpm > 0:
        score -= args.tpm_over_weight * max(
            0.0,
            (_safe_float(holdout.get("trades_per_month"), 0.0) / args.max_holdout_tpm) - 1.0,
        )
    forward = row.get("forward_score")
    if forward:
        score += 0.5 * _safe_float(forward.get("sharpe"), 0.0)
        score += 3.0 * max(0.0, _safe_float(forward.get("profit_factor"), 0.0) - 1.0)
        score += 3.0 * _safe_float(forward.get("expectancy_R"), 0.0)
        score += args.win_rate_weight * (_safe_float(forward.get("win_rate"), 0.0) - args.win_rate_target)
        score += args.forward_return_weight * _safe_float(forward.get("total_return"), 0.0)
        score += args.forward_cagr_weight * _safe_float(forward.get("cagr"), 0.0)
        score += args.trade_move_weight * (
            _safe_float(forward.get("avg_abs_price_move_pct"), 0.0) / max(args.min_avg_trade_move_pct, 1.0)
        )
        score += args.monte_carlo_weight * _safe_float(forward.get("mc_p05_return"), 0.0)
        if args.max_avg_hold_hours > 0:
            score -= args.hold_time_weight * max(
                0.0,
                (_safe_float(forward.get("avg_hold_hours"), 0.0) / args.max_avg_hold_hours) - 1.0,
            )
        score -= 3.0 * _safe_float(forward.get("max_drawdown_abs"), 0.0)
        score -= args.monte_carlo_dd_weight * _safe_float(forward.get("mc_p95_drawdown"), 0.0)
        if args.min_forward_tpm > 0:
            score -= args.tpm_under_weight * max(
                0.0,
                1.0 - (_safe_float(forward.get("trades_per_month"), 0.0) / args.min_forward_tpm),
            )
        if args.max_forward_tpm > 0:
            score -= args.tpm_over_weight * max(
                0.0,
                (_safe_float(forward.get("trades_per_month"), 0.0) / args.max_forward_tpm) - 1.0,
            )
    stress = row.get("forward_stress_score")
    if stress:
        score += args.stress_forward_weight * (
            2.0 * max(0.0, _safe_float(stress.get("profit_factor"), 0.0) - 1.0)
            + 3.0 * _safe_float(stress.get("expectancy_R"), 0.0)
            + 0.25 * _safe_float(stress.get("sharpe"), 0.0)
            + 0.5 * _safe_float(stress.get("total_return"), 0.0)
            - 2.0 * _safe_float(stress.get("max_drawdown_abs"), 0.0)
        )
    return _safe_float(score, -1e9)


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    if args.rebuild_existing:
        skip_names = {
            "leaderboard.json",
            "best_passed.json",
            "best_passed_params.json",
        }
        for result_path in sorted(out_dir.glob("*.json")):
            if result_path.name in skip_names or result_path.name.endswith("_candidates.json"):
                continue
            with result_path.open("r", encoding="utf-8") as f:
                row = json.load(f)
            if not {"train_score", "holdout_score", "best_params"}.issubset(row):
                continue
            row["guardrail_failures"] = _guardrail_failures(row, args)
            row["passed_guardrails"] = not row["guardrail_failures"]
            row["rank_score"] = _rank_score(row, args)
            rows.append(_json_safe(row))
        print(f"rebuilt from {len(rows)} saved run files in {out_dir}")
    else:
        for pattern in args.patterns:
            for path in sorted(Path().glob(pattern)):
                if "forward" in path.name:
                    continue
                tf = args.tf or _infer_tf(path, args.default_tf)
                bars = _load_bars(path)
                forward_path = _forward_pair(path) if args.forward else None

                for target in args.targets:
                    for seed in args.seeds:
                        run_id = f"{path.stem}_tf-{tf}_target-{target:g}_seed-{seed}"
                        result_path = out_dir / f"{run_id}.json"
                        if args.skip_existing and result_path.exists():
                            with result_path.open("r", encoding="utf-8") as f:
                                row = json.load(f)
                            if {"train_score", "holdout_score", "best_params"}.issubset(row):
                                row["guardrail_failures"] = _guardrail_failures(row, args)
                                row["passed_guardrails"] = not row["guardrail_failures"]
                                row["rank_score"] = _rank_score(row, args)
                                rows.append(_json_safe(row))
                                print(f"\n=== {run_id} ===\nskipped existing", flush=True)
                                continue
                        print(f"\n=== {run_id} ===", flush=True)
                        ocfg = OptimizeConfig(
                            timeframe=tf,
                            holdout_frac=args.holdout,
                            n_trials=args.trials,
                            n_jobs=args.jobs,
                            seed=seed,
                            bt=BacktestConfig(
                                risk_per_trade=args.risk_per_trade,
                                fee_bps=args.fee_bps,
                                slippage_bps=args.slip_bps,
                            ),
                            objective_cfg=_objective_for_target(target, args),
                        )
                        if args.strategy == "enhanced":
                            result = _optimize_enhanced(bars, ocfg, args)
                        else:
                            result = optimize(bars, ocfg)
                        base_row = {
                            "run_id": run_id,
                            "bars": str(path),
                            "strategy": args.strategy,
                            "timeframe": tf,
                            "target_tpm": target,
                            "seed": seed,
                            "trials": args.trials,
                        }
                        trial_candidates = result.get("trial_candidates") or [{
                            "candidate_rank": 1,
                            "trial_number": None,
                            "trial_value": _safe_float(result["train_score"].get("score"), -math.inf),
                            "best_params": result["best_params"],
                            "train_score": result["train_score"],
                            "holdout_score": result["holdout_score"],
                        }]
                        candidate_rows: list[dict[str, Any]] = []
                        for candidate in trial_candidates:
                            candidate_row = {
                                **base_row,
                                "candidate_rank": candidate.get("candidate_rank"),
                                "trial_number": candidate.get("trial_number"),
                                "trial_value": candidate.get("trial_value"),
                                "best_params": candidate["best_params"],
                                "train_score": candidate["train_score"],
                                "holdout_score": candidate["holdout_score"],
                            }
                            if forward_path:
                                candidate_row["forward_bars"] = str(forward_path)
                                candidate_row["forward_score"] = _forward_metrics(
                                    forward_path,
                                    tf,
                                    candidate["best_params"],
                                    args.strategy,
                                    args,
                                    fee_bps=args.fee_bps,
                                    slip_bps=args.slip_bps,
                                )
                                if args.stress_slip_bps > 0:
                                    candidate_row["forward_stress_score"] = _forward_metrics(
                                        forward_path,
                                        tf,
                                        candidate["best_params"],
                                        args.strategy,
                                        args,
                                        fee_bps=args.stress_fee_bps,
                                        slip_bps=args.stress_slip_bps,
                                    )
                            candidate_row["guardrail_failures"] = _guardrail_failures(candidate_row, args)
                            candidate_row["passed_guardrails"] = not candidate_row["guardrail_failures"]
                            candidate_row["rank_score"] = _rank_score(candidate_row, args)
                            candidate_rows.append(_json_safe(candidate_row))

                        candidate_rows.sort(
                            key=lambda candidate: (
                                bool(candidate.get("passed_guardrails")),
                                _sort_score(candidate),
                            ),
                            reverse=True,
                        )
                        safe_row = dict(candidate_rows[0])
                        safe_row["selected_from_top_trials"] = bool(result.get("trial_candidates"))
                        safe_row["n_saved_trial_candidates"] = len(candidate_rows)
                        rows.append(safe_row)

                        with result_path.open("w", encoding="utf-8") as f:
                            json.dump(safe_row, f, indent=2)
                        if result.get("trial_candidates"):
                            with (out_dir / f"{run_id}_candidates.json").open("w", encoding="utf-8") as f:
                                json.dump(candidate_rows, f, indent=2)

    rows.sort(key=_sort_score, reverse=True)
    with (out_dir / "leaderboard.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    passed_rows = [row for row in rows if row["passed_guardrails"]]
    if passed_rows:
        best = passed_rows[0]
        champion = {
            "source": "auto_algo_finder",
            "run_id": best["run_id"],
            "bars": best["bars"],
            "forward_bars": best.get("forward_bars"),
            "strategy": best["strategy"],
            "timeframe": best["timeframe"],
            "target_tpm": best["target_tpm"],
            "seed": best["seed"],
            "trials": best["trials"],
            "candidate_rank": best.get("candidate_rank"),
            "trial_number": best.get("trial_number"),
            "trial_value": best.get("trial_value"),
            "selected_from_top_trials": best.get("selected_from_top_trials"),
            "rank_score": best["rank_score"],
            "best_params": best["best_params"],
            "train_score": best["train_score"],
            "holdout_score": best["holdout_score"],
            "forward_score": best.get("forward_score"),
            "forward_stress_score": best.get("forward_stress_score"),
        }
        with (out_dir / "best_passed.json").open("w", encoding="utf-8") as f:
            json.dump(champion, f, indent=2)
        with (out_dir / "best_passed_params.json").open("w", encoding="utf-8") as f:
            json.dump(best["best_params"], f, indent=2)

    flat_rows = []
    for row in rows:
        holdout = row["holdout_score"]
        forward = row.get("forward_score", {})
        stress = row.get("forward_stress_score", {})
        flat_rows.append(
            {
                "rank_score": row["rank_score"],
                "passed": row["passed_guardrails"],
                "guardrail_failures": ",".join(row.get("guardrail_failures", [])),
                "run_id": row["run_id"],
                "bars": row["bars"],
                "target_tpm": row["target_tpm"],
                "seed": row["seed"],
                "candidate_rank": row.get("candidate_rank"),
                "trial_number": row.get("trial_number"),
                "trial_value": row.get("trial_value"),
                "selected_from_top_trials": row.get("selected_from_top_trials"),
                "n_saved_trial_candidates": row.get("n_saved_trial_candidates"),
                "holdout_score": holdout.get("score"),
                "holdout_pf": holdout.get("profit_factor_mean"),
                "holdout_win_rate": holdout.get("win_rate_mean"),
                "holdout_exp_r": holdout.get("expectancy_R_mean"),
                "holdout_dd": holdout.get("max_drawdown_abs"),
                "holdout_tpm": holdout.get("trades_per_month"),
                "forward_pf": forward.get("profit_factor"),
                "forward_win_rate": forward.get("win_rate"),
                "forward_exp_r": forward.get("expectancy_R"),
                "forward_dd": forward.get("max_drawdown_abs"),
                "forward_return": forward.get("total_return"),
                "forward_cagr": forward.get("cagr"),
                "forward_tpm": forward.get("trades_per_month"),
                "avg_abs_move_pct": forward.get("avg_abs_price_move_pct"),
                "median_abs_move_pct": forward.get("median_abs_price_move_pct"),
                "avg_hold_hours": forward.get("avg_hold_hours"),
                "median_hold_hours": forward.get("median_hold_hours"),
                "mc_profit_prob": forward.get("mc_profit_prob"),
                "mc_p05_return": forward.get("mc_p05_return"),
                "mc_p95_drawdown": forward.get("mc_p95_drawdown"),
                "stress_slip_bps": stress.get("slip_bps"),
                "stress_forward_pf": stress.get("profit_factor"),
                "stress_forward_win_rate": stress.get("win_rate"),
                "stress_forward_exp_r": stress.get("expectancy_R"),
                "stress_forward_dd": stress.get("max_drawdown_abs"),
                "stress_forward_return": stress.get("total_return"),
                "stress_forward_cagr": stress.get("cagr"),
                "stress_forward_tpm": stress.get("trades_per_month"),
            }
        )
    pd.DataFrame(flat_rows).to_csv(out_dir / "leaderboard.csv", index=False)
    flat_df = pd.DataFrame(flat_rows)
    if not flat_df.empty:
        seed_summary = (
            flat_df.groupby("seed", dropna=False)
            .agg(
                runs=("run_id", "count"),
                passed=("passed", "sum"),
                best_rank=("rank_score", "max"),
                best_forward_pf=("forward_pf", "max"),
                best_forward_exp_r=("forward_exp_r", "max"),
                min_forward_dd=("forward_dd", "min"),
                best_forward_cagr=("forward_cagr", "max"),
                best_mc_p05_return=("mc_p05_return", "max"),
                best_stress_forward_pf=("stress_forward_pf", "max"),
                best_stress_forward_exp_r=("stress_forward_exp_r", "max"),
                min_stress_forward_dd=("stress_forward_dd", "min"),
                median_forward_pf=("forward_pf", "median"),
            )
            .reset_index()
            .sort_values(["passed", "best_rank"], ascending=[False, False])
        )
        seed_summary.to_csv(out_dir / "seed_summary.csv", index=False)

        target_summary = (
            flat_df.groupby(["bars", "target_tpm"], dropna=False)
            .agg(
                runs=("run_id", "count"),
                passed=("passed", "sum"),
                best_rank=("rank_score", "max"),
                best_forward_pf=("forward_pf", "max"),
                best_forward_exp_r=("forward_exp_r", "max"),
                min_forward_dd=("forward_dd", "min"),
                best_forward_cagr=("forward_cagr", "max"),
                best_mc_p05_return=("mc_p05_return", "max"),
                best_stress_forward_pf=("stress_forward_pf", "max"),
                best_stress_forward_exp_r=("stress_forward_exp_r", "max"),
                min_stress_forward_dd=("stress_forward_dd", "min"),
            )
            .reset_index()
            .sort_values(["passed", "best_rank"], ascending=[False, False])
        )
        target_summary.to_csv(out_dir / "target_summary.csv", index=False)

    print("\n=== leaderboard ===")
    for row in flat_rows[: args.top]:
        print(
            f"{row['passed']} rank={row['rank_score']:.3f} "
            f"holdout_pf={row['holdout_pf']:.3f} holdout_expR={row['holdout_exp_r']:.3f} "
            f"forward_pf={row['forward_pf']} stress_pf={row['stress_forward_pf']} run={row['run_id']}"
        )
    print(f"\nwrote {out_dir / 'leaderboard.json'}")
    print(f"wrote {out_dir / 'leaderboard.csv'}")
    if not flat_rows:
        print("no rows produced")
    else:
        print(f"wrote {out_dir / 'seed_summary.csv'}")
        print(f"wrote {out_dir / 'target_summary.csv'}")
    if passed_rows:
        print(f"wrote {out_dir / 'best_passed.json'}")
        print(f"wrote {out_dir / 'best_passed_params.json'}")
    else:
        print("no candidate passed guardrails")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run robust multi-seed optimizer sweeps and rank only by out-of-sample evidence."
    )
    parser.add_argument("--patterns", nargs="+", default=["data_store/*_backtest_5y.parquet"])
    parser.add_argument("--out-dir", default="data_store/auto_finder")
    parser.add_argument("--tf", default=None)
    parser.add_argument("--default-tf", default="1h")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--strategy", choices=["enhanced", "base"], default="enhanced")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--holdout", type=float, default=0.30)
    parser.add_argument("--targets", type=float, nargs="+", default=[8.0, 12.0, 16.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 42, 101])
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--forward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--fee-bps", type=float, default=4.0)
    parser.add_argument("--slip-bps", type=float, default=1.5)
    parser.add_argument(
        "--stress-fee-bps",
        type=float,
        default=4.0,
        help="Fee bps used for the separate stressed forward re-test.",
    )
    parser.add_argument(
        "--stress-slip-bps",
        type=float,
        default=0.0,
        help="If >0, run an additional forward test at this slippage and rank/guardrail it.",
    )
    parser.add_argument("--min-tpm-ratio", type=float, default=0.60)
    parser.add_argument("--max-tpm-ratio", type=float, default=1.60)
    parser.add_argument("--soft-dd-cap", type=float, default=0.18)
    parser.add_argument("--hard-dd-cap", type=float, default=0.35)
    parser.add_argument("--min-win-rate", type=float, default=0.35)
    parser.add_argument("--stability-lambda", type=float, default=0.80)
    parser.add_argument("--dd-penalty-weight", type=float, default=3.0)
    parser.add_argument("--freq-penalty-weight", type=float, default=1.0)
    parser.add_argument("--min-holdout-pf", type=float, default=1.05)
    parser.add_argument("--min-holdout-win-rate", type=float, default=0.35)
    parser.add_argument("--min-holdout-expectancy-r", type=float, default=0.02)
    parser.add_argument("--max-holdout-dd", type=float, default=0.30)
    parser.add_argument("--min-holdout-tpm", type=float, default=0.0)
    parser.add_argument("--max-holdout-tpm", type=float, default=0.0)
    parser.add_argument("--max-train-holdout-score-gap", type=float, default=4.0)
    parser.add_argument("--max-train-holdout-pf-ratio", type=float, default=1.75)
    parser.add_argument("--min-forward-pf", type=float, default=1.03)
    parser.add_argument("--min-forward-win-rate", type=float, default=0.35)
    parser.add_argument("--min-forward-expectancy-r", type=float, default=0.01)
    parser.add_argument("--max-forward-dd", type=float, default=0.35)
    parser.add_argument("--min-forward-tpm", type=float, default=0.0)
    parser.add_argument("--max-forward-tpm", type=float, default=0.0)
    parser.add_argument("--min-forward-return", type=float, default=0.0)
    parser.add_argument("--min-forward-cagr", type=float, default=0.0)
    parser.add_argument("--min-stress-forward-pf", type=float, default=1.0)
    parser.add_argument("--min-stress-forward-expectancy-r", type=float, default=0.0)
    parser.add_argument("--max-stress-forward-dd", type=float, default=0.45)
    parser.add_argument("--min-stress-forward-return", type=float, default=-0.25)
    parser.add_argument(
        "--min-target-move-pct",
        type=float,
        default=0.0,
        help="Minimum planned target distance from entry, in percent, used while sampling quality filters.",
    )
    parser.add_argument("--min-avg-trade-move-pct", type=float, default=0.0)
    parser.add_argument("--max-avg-hold-hours", type=float, default=0.0)
    parser.add_argument("--max-median-hold-hours", type=float, default=0.0)
    parser.add_argument(
        "--min-objective-win-rate",
        type=float,
        default=0.0,
        help="Hard win-rate floor during optimization; trials below this score as invalid.",
    )
    parser.add_argument("--win-rate-target", type=float, default=0.45)
    parser.add_argument("--win-rate-weight", type=float, default=4.0)
    parser.add_argument(
        "--pf-over-weight",
        type=float,
        default=0.0,
        help="Extra objective reward for profit factor above 1.0.",
    )
    parser.add_argument(
        "--expectancy-r-weight",
        type=float,
        default=0.0,
        help="Extra objective reward for expectancy measured in R.",
    )
    parser.add_argument(
        "--min-objective-tpm",
        type=float,
        default=0.0,
        help="Hard trades-per-month floor during optimization; trials below this score as invalid.",
    )
    parser.add_argument("--max-objective-tpm", type=float, default=0.0)
    parser.add_argument(
        "--hard-max-objective-tpm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Treat --max-objective-tpm as a hard optimization ceiling instead of only a penalty.",
    )
    parser.add_argument("--tpm-over-weight", type=float, default=2.0)
    parser.add_argument("--tpm-under-weight", type=float, default=2.0)
    parser.add_argument("--forward-return-weight", type=float, default=0.0)
    parser.add_argument("--forward-cagr-weight", type=float, default=0.0)
    parser.add_argument("--trade-move-weight", type=float, default=0.0)
    parser.add_argument("--hold-time-weight", type=float, default=0.0)
    parser.add_argument("--monte-carlo-weight", type=float, default=0.0)
    parser.add_argument("--monte-carlo-dd-weight", type=float, default=0.0)
    parser.add_argument("--stress-forward-weight", type=float, default=1.0)
    parser.add_argument("--rr-min", type=float, default=0.8)
    parser.add_argument("--rr-max", type=float, default=4.0)
    parser.add_argument("--enqueue-low-rr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--seed-variant-count",
        type=int,
        default=0,
        help="Enqueue this many jittered variants around each seed-param file before random trials.",
    )
    parser.add_argument(
        "--seed-jitter",
        type=float,
        default=0.08,
        help="Neighborhood size for --seed-variant-count, as a fraction of each parameter range.",
    )
    parser.add_argument("--optimize-exits", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--optimize-hold-time", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-exit-hold-hours", type=float, default=24.0)
    parser.add_argument("--max-exit-hold-hours", type=float, default=120.0)
    parser.add_argument("--monte-carlo-runs", type=int, default=0)
    parser.add_argument("--monte-carlo-seed", type=int, default=20240512)
    parser.add_argument("--min-mc-profit-prob", type=float, default=0.0)
    parser.add_argument("--min-mc-p05-return", type=float, default=-1.0)
    parser.add_argument("--max-mc-p95-drawdown", type=float, default=0.0)
    parser.add_argument(
        "--save-top-trials",
        type=int,
        default=20,
        help="Always evaluate and save this many top raw training trials per run, even if scores look suspicious.",
    )
    parser.add_argument(
        "--max-candidate-trials",
        type=int,
        default=120,
        help="Maximum total candidates to holdout/forward/stress evaluate per run after adding the sane score band.",
    )
    parser.add_argument(
        "--candidate-min-train-score",
        type=float,
        default=4.0,
        help="Preserve trials at or above this train score when they are also below --candidate-max-train-score.",
    )
    parser.add_argument(
        "--candidate-max-train-score",
        type=float,
        default=25.0,
        help="Preserve trials at or below this train score, so suspicious huge scores do not hide sane candidates. Use <=0 to disable the sane band.",
    )
    parser.add_argument(
        "--exit-profile",
        choices=["high_win", "swing"],
        default="high_win",
        help="Exit optimization range. swing searches larger partial targets and ATR trailing exits.",
    )
    parser.add_argument(
        "--seed-param-files",
        nargs="+",
        default=[str(path) for path in SEED_PARAM_FILES],
        help="Saved JSON files whose best_params should be enqueued as optimizer seed trials.",
    )
    parser.add_argument(
        "--rebuild-existing",
        action="store_true",
        help="Rebuild leaderboard files from saved per-run JSONs in --out-dir without optimizing again.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="When optimizing, load existing per-run JSONs from --out-dir instead of recomputing them.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
