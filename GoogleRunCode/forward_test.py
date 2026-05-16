"""Run untouched forward-test data with optimizer-selected parameters.

This script is intentionally outside ``algo_download``.  It reads the JSON
written by ``python -m algo.optimize``, applies ``best_params`` to a separate
bars file, and reports out-of-sample performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from algo.config import CFG
from algo.metrics import bars_per_year, compute_perf
from algo.strategy.runner import backtest_strategy
from tools.enhanced_strategy import enhanced_backtest_strategy


REPORT_KEYS = (
    "n_bars",
    "n_trades",
    "final_balance",
    "total_return",
    "cagr",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "win_rate",
    "profit_factor",
    "expectancy",
    "expectancy_R",
    "avg_win",
    "avg_loss",
    "largest_win",
    "largest_loss",
)


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
    suffix = p.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        bars = pd.read_parquet(p)
    elif suffix == ".csv":
        bars = pd.read_csv(p)
    else:
        raise ValueError(f"unsupported bars file type: {p.suffix}")

    required = {"open", "high", "low", "close"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars file is missing columns: {sorted(missing)}")

    if "ts_ms" in bars.columns:
        bars = bars.sort_values("ts_ms").copy()
        bars.index = pd.to_datetime(bars["ts_ms"], unit="ms", utc=True)
        bars.index.name = "ts"
    else:
        bars = bars.sort_index()

    return bars


def load_best_params(path: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    params = data.get("best_params")
    if not isinstance(params, dict) or not params:
        raise ValueError(f"{path} does not contain a non-empty best_params object")
    return params


def print_report(stats: dict[str, Any]) -> None:
    print("=== FORWARD TEST ===")
    for key in REPORT_KEYS:
        value = stats.get(key)
        if isinstance(value, float):
            print(f"{key}: {value:.6g}")
        else:
            print(f"{key}: {value}")


def enrich_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    out = trades.copy()
    direction = out["direction"].astype(float)
    out["price_move_pct"] = ((out["exit"] - out["entry"]) / out["entry"]) * direction
    out["price_move_pct"] *= 100.0
    risk = out["risk_amt"].replace(0, np.nan)
    out["pnl_R"] = out["pnl"] / risk
    out["trade_quality_score"] = out["pnl_R"]
    return out.sort_values("trade_quality_score", ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run optimized params on untouched forward-test candles"
    )
    parser.add_argument("bars", help="Forward OHLC .parquet/.csv file")
    parser.add_argument("optimized", help="Optimizer JSON containing best_params")
    parser.add_argument("--tf", required=True, help="Timeframe, e.g. 1h or 4h")
    parser.add_argument("--strategy", choices=["base", "enhanced"], default="enhanced")
    parser.add_argument("--initial-balance", type=float, default=CFG.bt.initial_balance)
    parser.add_argument("--risk", type=float, default=CFG.bt.risk_per_trade)
    parser.add_argument("--fee-bps", type=float, default=CFG.bt.fee_bps)
    parser.add_argument("--slip-bps", type=float, default=CFG.bt.slippage_bps)
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=CFG.bt.max_concurrent_trades,
    )
    parser.add_argument(
        "--market-hours",
        action="store_true",
        help="Use equity/FX annualization instead of 24/7 crypto",
    )
    parser.add_argument("--out", help="Optional JSON report path")
    parser.add_argument("--trades-out", help="Optional trades CSV path")
    args = parser.parse_args()

    bars = load_bars(args.bars)
    params = load_best_params(args.optimized)
    if args.strategy == "enhanced":
        result = enhanced_backtest_strategy(
            bars,
            initial_balance=args.initial_balance,
            risk=args.risk,
            fee_bps=args.fee_bps,
            slip_bps=args.slip_bps,
            max_concurrent=args.max_concurrent,
            **params,
        )
    else:
        result = backtest_strategy(
            bars,
            strategy_kind="scorer",
            initial_balance=args.initial_balance,
            risk=args.risk,
            fee_bps=args.fee_bps,
            slip_bps=args.slip_bps,
            max_concurrent=args.max_concurrent,
            **params,
        )
    perf = compute_perf(
        result["equity"],
        result["trades"],
        bars_per_year=bars_per_year(args.tf, market_hours=args.market_hours),
    )
    stats = perf.to_dict()
    stats["bars_file"] = str(Path(args.bars))
    stats["optimized_file"] = str(Path(args.optimized))
    stats["timeframe"] = args.tf
    stats["strategy"] = args.strategy
    stats["best_params"] = params

    print_report(stats)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_jsonable(stats), indent=2))
        print(f"wrote {out}")

    if args.trades_out:
        trades_out = Path(args.trades_out)
        trades_out.parent.mkdir(parents=True, exist_ok=True)
        enrich_trades(result["trades"]).to_csv(trades_out, index=False)
        print(f"wrote {trades_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
