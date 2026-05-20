"""Tick-level validation for optimized candle strategies.

Workflow:
1. Load forward-test candles and optimizer JSON.
2. Build the same candle signals used by the optimizer.
3. Run the normal candle backtest to identify candidate trades.
4. Fetch/cache Binance aggTrades only for those trade windows.
5. Re-resolve entry/exit and SL/TP ordering with tick sequence.

This keeps tick usage focused where it matters: execution validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from algo.config import CFG
from algo.metrics import profit_factor
from algo.strategy.runner import _build_scorer_signals
from algo.backtest.engine import run_backtest

from tick_cache import (
    TICK_DB_PATH,
    ensure_ticks,
    read_first_price_cross,
    read_first_tick,
    read_last_tick,
    read_ticks,
)
from tools.enhanced_strategy import build_enhanced_signals, enhanced_backtest_strategy


INTERVAL_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
    "1w": 7 * 24 * 60 * 60_000,
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, float):
        if np.isfinite(value):
            return value
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
        return None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def load_bars(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        bars = pd.read_parquet(p)
    elif p.suffix.lower() == ".csv":
        bars = pd.read_csv(p)
    else:
        raise ValueError(f"unsupported bars file type: {p.suffix}")

    missing = {"open", "high", "low", "close"} - set(bars.columns)
    if missing:
        raise ValueError(f"bars file is missing columns: {sorted(missing)}")

    if "ts_ms" in bars.columns:
        bars = bars.sort_values("ts_ms").copy()
        bars.index = pd.to_datetime(bars["ts_ms"], unit="ms", utc=True)
    elif isinstance(bars.index, pd.DatetimeIndex):
        bars = bars.sort_index().copy()
        bars["ts_ms"] = (bars.index.astype("int64") // 1_000_000).astype(np.int64)
    else:
        raise ValueError("tick validation needs ts_ms or a DatetimeIndex")
    return bars


def load_best_params(path: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    params = data.get("best_params")
    if not isinstance(params, dict) or not params:
        raise ValueError(f"{path} does not contain a non-empty best_params object")
    return params


def _signal_value(values: Any, idx: int, default: float = 0.0) -> float:
    if np.isscalar(values):
        return float(values)
    arr = np.asarray(values)
    if idx < 0 or idx >= arr.shape[0]:
        return default
    return float(arr[idx])


def resolve_trade_with_ticks(
    *,
    bars: pd.DataFrame,
    ticks: pd.DataFrame,
    trade: pd.Series,
    signals: dict,
    tf_ms: int,
    balance_at_entry: float,
    risk: float,
    fee_bps: float,
    slip_bps: float,
    partial_tp_r: float = 0.0,
    partial_close_frac: float = 0.5,
) -> dict[str, Any]:
    entry_idx = int(trade["entry_idx"])
    signal_idx = entry_idx - 1
    direction = int(trade["direction"])
    candle_exit_idx = int(trade["exit_idx"])

    entry_ts = int(bars["ts_ms"].iloc[entry_idx])
    candle_exit_end = int(bars["ts_ms"].iloc[candle_exit_idx]) + tf_ms
    window = ticks[(ticks["ts_ms"] >= entry_ts) & (ticks["ts_ms"] < candle_exit_end)]

    candle_entry = float(trade["entry"])
    if window.empty:
        entry_px = candle_entry
        exit_px_raw = float(trade["exit"])
        exit_ts = candle_exit_end - 1
        reason = 0
    else:
        entry_px = float(window["price"].iloc[0])
        exit_px_raw = float(window["price"].iloc[-1])
        exit_ts = int(window["ts_ms"].iloc[-1])
        reason = 0

    slip = slip_bps / 10_000.0
    fee = fee_bps / 10_000.0
    entry_fill = entry_px * (1.0 + slip) if direction == 1 else entry_px * (1.0 - slip)
    sl = _signal_value(signals["sl"], signal_idx, float("nan"))
    tp = _signal_value(signals["tp"], signal_idx, float("nan"))
    rr = _signal_value(signals.get("rr", 0.0), signal_idx, 0.0)

    if not np.isfinite(sl):
        raise ValueError(f"missing SL for trade at entry_idx={entry_idx}")

    stop_dist = abs(entry_fill - sl)
    if stop_dist <= 0:
        return {
            "entry_idx": entry_idx,
            "exit_idx": candle_exit_idx,
            "direction": direction,
            "entry": entry_fill,
            "exit": entry_fill,
            "sl": sl,
            "tp": tp,
            "size": 0.0,
            "risk_amt": 0.0,
            "pnl": 0.0,
            "reason": "invalid_stop",
            "entry_ts_ms": entry_ts,
            "exit_ts_ms": entry_ts,
        }

    if rr > 0:
        tp = entry_fill + rr * stop_dist if direction == 1 else entry_fill - rr * stop_dist

    if direction == 1 and not (sl < entry_fill < tp):
        reason = "invalid_levels"
        exit_fill = entry_fill
        exit_ts = entry_ts
    elif direction == -1 and not (tp < entry_fill < sl):
        reason = "invalid_levels"
        exit_fill = entry_fill
        exit_ts = entry_ts
    else:
        reason = "eow"
        exit_fill = exit_px_raw * (1.0 - slip) if direction == 1 else exit_px_raw * (1.0 + slip)

    risk_amt = balance_at_entry * risk
    size = risk_amt / stop_dist
    remaining_size = size
    partial_pnl = 0.0
    partial_done = False

    if window.empty or reason == "invalid_levels":
        pnl = (exit_fill - entry_fill) * size * direction
        pnl -= (entry_fill + exit_fill) * size * fee
    else:
        reason = "eow"
        exit_fill = exit_px_raw * (1.0 - slip) if direction == 1 else exit_px_raw * (1.0 + slip)
        if partial_tp_r > 0.0:
            partial_level = entry_fill + direction * partial_tp_r * stop_dist
            partial_frac = min(max(float(partial_close_frac), 0.0), 1.0)
        else:
            partial_level = np.nan
            partial_frac = 0.0

        for row in window.itertuples(index=False):
            px = float(row.price)
            if partial_frac > 0.0 and not partial_done:
                hit_partial = px >= partial_level if direction == 1 else px <= partial_level
                if hit_partial:
                    partial_fill = partial_level * (1.0 - slip) if direction == 1 else partial_level * (1.0 + slip)
                    partial_size = remaining_size * partial_frac
                    partial_pnl += (partial_fill - entry_fill) * partial_size * direction
                    partial_pnl -= (entry_fill + partial_fill) * partial_size * fee
                    remaining_size -= partial_size
                    partial_done = True

            if direction == 1:
                if px <= sl:
                    reason = "sl"
                    exit_px_raw = px
                    exit_ts = int(row.ts_ms)
                    exit_fill = px * (1.0 - slip)
                    break
                if px >= tp:
                    reason = "tp"
                    exit_px_raw = px
                    exit_ts = int(row.ts_ms)
                    exit_fill = px * (1.0 - slip)
                    break
            else:
                if px >= sl:
                    reason = "sl"
                    exit_px_raw = px
                    exit_ts = int(row.ts_ms)
                    exit_fill = px * (1.0 + slip)
                    break
                if px <= tp:
                    reason = "tp"
                    exit_px_raw = px
                    exit_ts = int(row.ts_ms)
                    exit_fill = px * (1.0 + slip)
                    break

        pnl = partial_pnl
        if remaining_size > 0.0:
            pnl += (exit_fill - entry_fill) * remaining_size * direction
            pnl -= (entry_fill + exit_fill) * remaining_size * fee
    bar_ts = bars["ts_ms"].to_numpy(np.int64)
    exit_idx = int(np.searchsorted(bar_ts, exit_ts, side="right") - 1)
    exit_idx = min(max(exit_idx, entry_idx), len(bars) - 1)

    return {
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
        "direction": direction,
        "entry": entry_fill,
        "exit": exit_fill,
        "sl": sl,
        "tp": tp,
        "size": size,
        "remaining_size": remaining_size,
        "risk_amt": risk_amt,
        "pnl": pnl,
        "partial_done": partial_done,
        "partial_pnl": partial_pnl,
        "reason": reason,
        "entry_ts_ms": entry_ts,
        "exit_ts_ms": exit_ts,
        "tick_window_end_ms": candle_exit_end,
    }


def _tick_sort_key(tick: dict | None) -> tuple[int, int]:
    if tick is None:
        return (2**63 - 1, 2**63 - 1)
    return (int(tick["ts_ms"]), int(tick["agg_id"]))


def resolve_trade_from_cache(
    *,
    symbol: str,
    bars: pd.DataFrame,
    trade: pd.Series,
    signals: dict,
    tf_ms: int,
    balance_at_entry: float,
    risk: float,
    fee_bps: float,
    slip_bps: float,
    partial_tp_r: float = 0.0,
    partial_close_frac: float = 0.5,
) -> dict[str, Any]:
    entry_idx = int(trade["entry_idx"])
    signal_idx = entry_idx - 1
    direction = int(trade["direction"])
    candle_exit_idx = int(trade["exit_idx"])

    entry_ts = int(bars["ts_ms"].iloc[entry_idx])
    candle_exit_end = int(bars["ts_ms"].iloc[candle_exit_idx]) + tf_ms
    first_tick = read_first_tick(symbol, entry_ts, candle_exit_end)
    last_tick = read_last_tick(symbol, entry_ts, candle_exit_end)

    candle_entry = float(trade["entry"])
    entry_px = float(first_tick["price"]) if first_tick else candle_entry
    exit_px_raw = float(last_tick["price"]) if last_tick else float(trade["exit"])
    exit_ts = int(last_tick["ts_ms"]) if last_tick else candle_exit_end - 1

    slip = slip_bps / 10_000.0
    fee = fee_bps / 10_000.0
    entry_fill = entry_px * (1.0 + slip) if direction == 1 else entry_px * (1.0 - slip)
    sl = _signal_value(signals["sl"], signal_idx, float("nan"))
    tp = _signal_value(signals["tp"], signal_idx, float("nan"))
    rr = _signal_value(signals.get("rr", 0.0), signal_idx, 0.0)

    if not np.isfinite(sl):
        raise ValueError(f"missing SL for trade at entry_idx={entry_idx}")

    stop_dist = abs(entry_fill - sl)
    if stop_dist <= 0:
        return {
            "entry_idx": entry_idx,
            "exit_idx": candle_exit_idx,
            "direction": direction,
            "entry": entry_fill,
            "exit": entry_fill,
            "sl": sl,
            "tp": tp,
            "size": 0.0,
            "remaining_size": 0.0,
            "risk_amt": 0.0,
            "pnl": 0.0,
            "partial_done": False,
            "partial_pnl": 0.0,
            "reason": "invalid_stop",
            "entry_ts_ms": entry_ts,
            "exit_ts_ms": entry_ts,
        }

    if rr > 0:
        tp = entry_fill + rr * stop_dist if direction == 1 else entry_fill - rr * stop_dist

    if direction == 1 and not (sl < entry_fill < tp):
        reason = "invalid_levels"
        exit_fill = entry_fill
        exit_ts = entry_ts
        terminal_tick = None
    elif direction == -1 and not (tp < entry_fill < sl):
        reason = "invalid_levels"
        exit_fill = entry_fill
        exit_ts = entry_ts
        terminal_tick = None
    else:
        if direction == 1:
            sl_tick = read_first_price_cross(symbol, entry_ts, candle_exit_end, op="<=", level=sl)
            tp_tick = read_first_price_cross(symbol, entry_ts, candle_exit_end, op=">=", level=tp)
        else:
            sl_tick = read_first_price_cross(symbol, entry_ts, candle_exit_end, op=">=", level=sl)
            tp_tick = read_first_price_cross(symbol, entry_ts, candle_exit_end, op="<=", level=tp)

        if _tick_sort_key(sl_tick) <= _tick_sort_key(tp_tick):
            terminal_tick = sl_tick
            reason = "sl" if sl_tick is not None else "eow"
        else:
            terminal_tick = tp_tick
            reason = "tp" if tp_tick is not None else "eow"

        if terminal_tick is not None:
            exit_px_raw = float(terminal_tick["price"])
            exit_ts = int(terminal_tick["ts_ms"])
        exit_fill = exit_px_raw * (1.0 - slip) if direction == 1 else exit_px_raw * (1.0 + slip)

    risk_amt = balance_at_entry * risk
    size = risk_amt / stop_dist
    remaining_size = size
    partial_pnl = 0.0
    partial_done = False

    if reason != "invalid_levels" and first_tick is not None and partial_tp_r > 0.0:
        partial_level = entry_fill + direction * partial_tp_r * stop_dist
        partial_frac = min(max(float(partial_close_frac), 0.0), 1.0)
        if direction == 1:
            partial_tick = read_first_price_cross(
                symbol, entry_ts, candle_exit_end, op=">=", level=partial_level
            )
        else:
            partial_tick = read_first_price_cross(
                symbol, entry_ts, candle_exit_end, op="<=", level=partial_level
            )
        if partial_frac > 0.0 and partial_tick is not None:
            terminal_key = _tick_sort_key(terminal_tick)
            partial_key = _tick_sort_key(partial_tick)
            if partial_key <= terminal_key:
                partial_fill = partial_level * (1.0 - slip) if direction == 1 else partial_level * (1.0 + slip)
                partial_size = remaining_size * partial_frac
                partial_pnl += (partial_fill - entry_fill) * partial_size * direction
                partial_pnl -= (entry_fill + partial_fill) * partial_size * fee
                remaining_size -= partial_size
                partial_done = True

    pnl = partial_pnl
    if remaining_size > 0.0:
        pnl += (exit_fill - entry_fill) * remaining_size * direction
        pnl -= (entry_fill + exit_fill) * remaining_size * fee

    bar_ts = bars["ts_ms"].to_numpy(np.int64)
    exit_idx = int(np.searchsorted(bar_ts, exit_ts, side="right") - 1)
    exit_idx = min(max(exit_idx, entry_idx), len(bars) - 1)

    return {
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
        "direction": direction,
        "entry": entry_fill,
        "exit": exit_fill,
        "sl": sl,
        "tp": tp,
        "size": size,
        "remaining_size": remaining_size,
        "risk_amt": risk_amt,
        "pnl": pnl,
        "partial_done": partial_done,
        "partial_pnl": partial_pnl,
        "reason": reason,
        "entry_ts_ms": entry_ts,
        "exit_ts_ms": exit_ts,
        "tick_window_end_ms": candle_exit_end,
    }


def equity_from_trades(
    bars: pd.DataFrame,
    trades: pd.DataFrame,
    initial_balance: float,
) -> np.ndarray:
    close = bars["close"].to_numpy(np.float64)
    equity = np.empty(len(bars), dtype=np.float64)
    realized = initial_balance
    if trades.empty:
        equity.fill(initial_balance)
        return equity

    exits_by_bar: dict[int, list[float]] = {}
    for row in trades.itertuples(index=False):
        exits_by_bar.setdefault(int(row.exit_idx), []).append(float(row.pnl))

    for i in range(len(bars)):
        for pnl in exits_by_bar.get(i, []):
            realized += pnl
        mtm = 0.0
        active = trades[(trades["entry_idx"] <= i) & (trades["exit_idx"] > i)]
        for row in active.itertuples(index=False):
            mtm += (close[i] - float(row.entry)) * float(row.size) * int(row.direction)
        equity[i] = realized + mtm
    return equity


def summarize(equity: np.ndarray, trades: pd.DataFrame, initial_balance: float) -> dict[str, Any]:
    pnls = trades["pnl"].to_numpy(np.float64) if not trades.empty else np.zeros(0)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    peaks = np.maximum.accumulate(equity) if equity.size else np.array([initial_balance])
    drawdown = equity / peaks - 1.0 if equity.size else np.array([0.0])
    risk = trades["risk_amt"].to_numpy(np.float64) if not trades.empty else np.zeros(0)
    ok_r = np.isfinite(risk) & (risk > 0)
    if not trades.empty:
        hold_bars = (
            trades["exit_idx"].to_numpy(np.float64)
            - trades["entry_idx"].to_numpy(np.float64)
        ).clip(min=0.0)
        if {"entry_ts_ms", "exit_ts_ms"}.issubset(trades.columns):
            hold_hours = (
                trades["exit_ts_ms"].to_numpy(np.float64)
                - trades["entry_ts_ms"].to_numpy(np.float64)
            ).clip(min=0.0) / 3_600_000.0
        else:
            hold_hours = np.zeros_like(hold_bars)
    else:
        hold_bars = np.zeros(0, dtype=np.float64)
        hold_hours = np.zeros(0, dtype=np.float64)
    return {
        "n_bars": int(equity.shape[0]),
        "n_trades": int(pnls.shape[0]),
        "final_balance": float(equity[-1]) if equity.size else initial_balance,
        "total_return": float(equity[-1] / initial_balance - 1.0) if equity.size else 0.0,
        "max_drawdown": float(drawdown.min()) if drawdown.size else 0.0,
        "win_rate": float(wins.shape[0] / pnls.shape[0]) if pnls.shape[0] else 0.0,
        "profit_factor": profit_factor(pnls),
        "expectancy": float(pnls.mean()) if pnls.shape[0] else 0.0,
        "expectancy_R": float(np.mean(pnls[ok_r] / risk[ok_r])) if ok_r.any() else 0.0,
        "avg_win": float(wins.mean()) if wins.size else 0.0,
        "avg_loss": float(losses.mean()) if losses.size else 0.0,
        "largest_win": float(wins.max()) if wins.size else 0.0,
        "largest_loss": float(losses.min()) if losses.size else 0.0,
        "avg_hold_bars": float(hold_bars.mean()) if hold_bars.size else 0.0,
        "min_hold_bars": float(hold_bars.min()) if hold_bars.size else 0.0,
        "max_hold_bars": float(hold_bars.max()) if hold_bars.size else 0.0,
        "avg_hold_hours": float(hold_hours.mean()) if hold_hours.size else 0.0,
        "min_hold_hours": float(hold_hours.min()) if hold_hours.size else 0.0,
        "max_hold_hours": float(hold_hours.max()) if hold_hours.size else 0.0,
    }


def run_tick_validation(
    *,
    bars_path: str,
    optimized_path: str,
    symbol: str,
    tf: str,
    strategy: str = "enhanced",
    initial_balance: float = CFG.bt.initial_balance,
    risk: float = CFG.bt.risk_per_trade,
    fee_bps: float = CFG.bt.fee_bps,
    slip_bps: float = CFG.bt.slippage_bps,
    max_concurrent: int = CFG.bt.max_concurrent_trades,
    max_trades: int = 0,
    tick_max_requests: int = 25_000,
    tick_chunk_hours: float = 6.0,
) -> tuple[dict[str, Any], pd.DataFrame]:
    bars = load_bars(bars_path)
    params = load_best_params(optimized_path)
    if strategy == "enhanced":
        signals = build_enhanced_signals(bars, **params)
        candle_result = enhanced_backtest_strategy(
            bars,
            initial_balance=initial_balance,
            risk=risk,
            fee_bps=fee_bps,
            slip_bps=slip_bps,
            max_concurrent=max_concurrent,
            **params,
        )
    else:
        signals = _build_scorer_signals(bars, **params)
        candle_result = run_backtest(
            bars,
            signals,
            initial_balance=initial_balance,
            risk=risk,
            fee_bps=fee_bps,
            slip_bps=slip_bps,
            max_concurrent=max_concurrent,
        )

    candle_trades = candle_result["trades"].sort_values("entry_idx").reset_index(drop=True)
    if max_trades > 0:
        candle_trades = candle_trades.head(max_trades)

    tf_ms = INTERVAL_MS[tf]
    partial_tp_r = float(params.get("partial_tp_r", 0.0) or 0.0)
    partial_close_frac = float(params.get("partial_close_frac", 0.5) or 0.5)
    tick_trades = []
    for i, trade in candle_trades.iterrows():
        entry_idx = int(trade["entry_idx"])
        exit_idx = int(trade["exit_idx"])
        start_ms = int(bars["ts_ms"].iloc[entry_idx])
        end_ms = int(bars["ts_ms"].iloc[exit_idx]) + tf_ms
        print(f"trade {i + 1}/{len(candle_trades)}: caching ticks {start_ms} -> {end_ms}")
        ensure_ticks(
            symbol,
            start_ms,
            end_ms,
            max_requests=tick_max_requests,
            chunk_ms=int(max(0.0, tick_chunk_hours) * 3_600_000),
        )
        balance_at_entry = initial_balance + sum(
            t["pnl"] for t in tick_trades if int(t["exit_idx"]) <= entry_idx
        )
        tick_trades.append(resolve_trade_from_cache(
            symbol=symbol,
            bars=bars,
            trade=trade,
            signals=signals,
            tf_ms=tf_ms,
            balance_at_entry=balance_at_entry,
            risk=risk,
            fee_bps=fee_bps,
            slip_bps=slip_bps,
            partial_tp_r=partial_tp_r,
            partial_close_frac=partial_close_frac,
        ))

    trades_df = pd.DataFrame(tick_trades)
    equity = equity_from_trades(bars, trades_df, initial_balance)
    report = summarize(equity, trades_df, initial_balance)
    report.update({
        "symbol": symbol.upper(),
        "timeframe": tf,
        "strategy": strategy,
        "bars_file": str(Path(bars_path)),
        "optimized_file": str(Path(optimized_path)),
        "tick_cache": str(TICK_DB_PATH),
        "candle_candidate_trades": int(len(candle_result["trades"])),
        "tick_validated_trades": int(len(trades_df)),
        "fee_bps": float(fee_bps),
        "slip_bps": float(slip_bps),
    })
    return report, trades_df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate optimized forward trades with on-demand Binance aggTrades"
    )
    parser.add_argument("bars", help="Forward OHLC .parquet/.csv file")
    parser.add_argument("optimized", help="Optimizer JSON containing best_params")
    parser.add_argument("--symbol", required=True, help="Binance symbol, e.g. BTCUSDT")
    parser.add_argument("--tf", required=True, choices=sorted(INTERVAL_MS))
    parser.add_argument("--strategy", choices=["base", "enhanced"], default="enhanced")
    parser.add_argument("--initial-balance", type=float, default=CFG.bt.initial_balance)
    parser.add_argument("--risk", type=float, default=CFG.bt.risk_per_trade)
    parser.add_argument("--fee-bps", type=float, default=CFG.bt.fee_bps)
    parser.add_argument("--slip-bps", type=float, default=CFG.bt.slippage_bps)
    parser.add_argument("--max-concurrent", type=int, default=CFG.bt.max_concurrent_trades)
    parser.add_argument(
        "--max-trades",
        type=int,
        default=0,
        help="Limit tick validation to first N candle trades; 0 means all",
    )
    parser.add_argument(
        "--tick-max-requests",
        type=int,
        default=25_000,
        help="Max Binance aggTrade requests per cache chunk",
    )
    parser.add_argument(
        "--tick-chunk-hours",
        type=float,
        default=6.0,
        help="Split long trade windows into this many hours per cache chunk; 0 disables chunking",
    )
    parser.add_argument("--out", help="Optional JSON report path")
    parser.add_argument("--trades-out", help="Optional tick-resolved trades CSV path")
    args = parser.parse_args()

    report, trades_df = run_tick_validation(
        bars_path=args.bars,
        optimized_path=args.optimized,
        symbol=args.symbol,
        tf=args.tf,
        strategy=args.strategy,
        initial_balance=args.initial_balance,
        risk=args.risk,
        fee_bps=args.fee_bps,
        slip_bps=args.slip_bps,
        max_concurrent=args.max_concurrent,
        max_trades=args.max_trades,
        tick_max_requests=args.tick_max_requests,
        tick_chunk_hours=args.tick_chunk_hours,
    )

    print("=== TICK FORWARD TEST ===")
    for key, value in report.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6g}")
        else:
            print(f"{key}: {value}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_jsonable(report), indent=2))
        print(f"wrote {out}")

    if args.trades_out:
        trades_out = Path(args.trades_out)
        trades_out.parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(trades_out, index=False)
        print(f"wrote {trades_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
