"""Batch tick-level validation for saved optimizer runs.

This reads a leaderboard CSV, replays each saved strategy JSON on its untouched
forward bars, and writes one tick report per strategy plus a combined summary.
It does not run Optuna or search new parameters.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algo.config import CFG
from tick_forward_test import (
    _jsonable,
    load_bars,
    load_best_params,
    run_tick_validation,
)
from tools.enhanced_strategy import enhanced_backtest_strategy


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _tick_verdict(report: dict[str, Any], source_row: dict[str, Any]) -> str:
    pf = _safe_float(report.get("profit_factor"))
    exp_r = _safe_float(report.get("expectancy_R"))
    ret = _safe_float(report.get("total_return"))
    dd = abs(_safe_float(report.get("max_drawdown")))
    passed = _safe_bool(source_row.get("passed"))
    failures = str(source_row.get("guardrail_failures", ""))

    if not passed and "train_holdout" in failures:
        prefix = "profitable tick result, but original run is overfit-risk"
    elif not passed:
        prefix = "failed original guardrail"
    else:
        prefix = "passed original guardrails"

    if pf >= 1.5 and exp_r > 0.10 and ret > 0.0 and dd <= 0.25:
        return f"{prefix}; tick result looks good"
    if pf >= 1.1 and exp_r > 0.0 and ret > 0.0:
        return f"{prefix}; tick result is acceptable but weaker"
    return f"{prefix}; tick result is weak"


def _select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = rows
    if args.passed_only:
        selected = [r for r in selected if _safe_bool(r.get("passed"))]
    if args.failed_only:
        selected = [r for r in selected if not _safe_bool(r.get("passed"))]
    if args.timeframe:
        selected = [r for r in selected if f"_tf-{args.timeframe}_" in str(r.get("run_id", ""))]
    selected = sorted(selected, key=lambda r: _safe_float(r.get("rank_score")), reverse=True)
    if args.max_strategies > 0:
        selected = selected[: args.max_strategies]
    return selected


def _estimate_run(
    *,
    run_json: Path,
    initial_balance: float,
    risk: float,
    fee_bps: float,
    slip_bps: float,
    max_concurrent: int,
) -> dict[str, Any]:
    data = _load_json(run_json)
    bars_path = str(data["forward_bars"])
    params = load_best_params(str(run_json))
    bars = load_bars(bars_path)
    result = enhanced_backtest_strategy(
        bars,
        initial_balance=initial_balance,
        risk=risk,
        fee_bps=fee_bps,
        slip_bps=slip_bps,
        max_concurrent=max_concurrent,
        **params,
    )
    trades = result["trades"]
    if trades.empty:
        return {
            "candle_candidate_trades": 0,
            "estimated_trade_window_hours": 0.0,
            "estimated_merged_window_hours": 0.0,
        }
    ts = bars["ts_ms"].to_numpy()
    hours = 0.0
    windows: list[tuple[int, int]] = []
    for row in trades.itertuples(index=False):
        start = int(ts[int(row.entry_idx)])
        end = int(ts[int(row.exit_idx)])
        windows.append((start, end))
        hours += max(0.0, (end - start) / 3_600_000.0)
    merged_hours = 0.0
    merged: list[tuple[int, int]] = []
    for start, end in sorted(windows):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    for start, end in merged:
        merged_hours += max(0.0, (end - start) / 3_600_000.0)
    return {
        "candle_candidate_trades": int(len(trades)),
        "estimated_trade_window_hours": float(hours),
        "estimated_merged_window_hours": float(merged_hours),
    }


def run(args: argparse.Namespace) -> int:
    leaderboard = Path(args.leaderboard)
    base_dir = leaderboard.parent
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with leaderboard.open(newline="") as f:
        rows = list(csv.DictReader(f))
    selected = _select_rows(rows, args)
    if not selected:
        raise SystemExit("no leaderboard rows matched the filters")

    summary: list[dict[str, Any]] = []
    for idx, row in enumerate(selected, start=1):
        run_id = str(row["run_id"])
        run_json = base_dir / f"{run_id}.json"
        if not run_json.exists():
            print(f"[{idx}/{len(selected)}] missing {run_json}; skipping")
            continue

        data = _load_json(run_json)
        bars_path = str(data["forward_bars"])
        tf = str(data["timeframe"])
        report_path = out_dir / f"{run_id}_tick.json"
        trades_path = out_dir / f"{run_id}_tick_trades.csv"

        common = {
            "run_id": run_id,
            "source_passed": row.get("passed", ""),
            "source_rank_score": row.get("rank_score", ""),
            "source_guardrail_failures": row.get("guardrail_failures", ""),
            "source_forward_pf": row.get("forward_pf", ""),
            "source_forward_return": row.get("forward_return", ""),
            "source_stress_forward_pf": row.get("stress_forward_pf", ""),
            "timeframe": tf,
            "optimized_file": str(run_json),
            "bars_file": bars_path,
        }

        if args.skip_existing and report_path.exists():
            report = _load_json(report_path)
            print(f"[{idx}/{len(selected)}] existing {run_id}")
            summary.append({
                **common,
                **{f"tick_{k}": v for k, v in report.items() if not isinstance(v, (dict, list))},
                "tick_verdict": _tick_verdict(report, row),
            })
            continue

        print(f"[{idx}/{len(selected)}] {run_id}")
        if args.dry_run:
            estimate = _estimate_run(
                run_json=run_json,
                initial_balance=args.initial_balance,
                risk=args.risk,
                fee_bps=args.fee_bps,
                slip_bps=args.slip_bps,
                max_concurrent=args.max_concurrent,
            )
            summary.append({**common, **estimate, "tick_verdict": "dry run only"})
            continue

        report, trades = run_tick_validation(
            bars_path=bars_path,
            optimized_path=str(run_json),
            symbol=args.symbol,
            tf=tf,
            strategy="enhanced",
            initial_balance=args.initial_balance,
            risk=args.risk,
            fee_bps=args.fee_bps,
            slip_bps=args.slip_bps,
            max_concurrent=args.max_concurrent,
            max_trades=args.max_trades,
            tick_max_requests=args.tick_max_requests,
            tick_chunk_hours=args.tick_chunk_hours,
        )
        report_path.write_text(json.dumps(_jsonable(report), indent=2))
        trades.to_csv(trades_path, index=False)
        summary.append({
            **common,
            **{f"tick_{k}": v for k, v in report.items() if not isinstance(v, (dict, list))},
            "tick_report": str(report_path),
            "tick_trades": str(trades_path),
            "tick_verdict": _tick_verdict(report, row),
        })

    summary_df = pd.DataFrame(summary)
    summary_csv = out_dir / "tick_summary.csv"
    summary_json = out_dir / "tick_summary.json"
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(_jsonable(summary), indent=2))
    print(f"wrote {summary_csv}")
    print(f"wrote {summary_json}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch tick-test saved strategy JSON files")
    parser.add_argument("--leaderboard", required=True, help="leaderboard CSV from auto_algo_finder")
    parser.add_argument("--out-dir", required=True, help="folder for tick reports")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--passed-only", action="store_true", help="only test rows with passed=True")
    parser.add_argument("--failed-only", action="store_true", help="only test rows with passed=False")
    parser.add_argument("--timeframe", choices=["1h", "4h"], help="optional timeframe filter")
    parser.add_argument("--max-strategies", type=int, default=0, help="0 means all selected rows")
    parser.add_argument("--max-trades", type=int, default=0, help="0 means all trades per strategy")
    parser.add_argument("--tick-max-requests", type=int, default=25_000)
    parser.add_argument("--tick-chunk-hours", type=float, default=6.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="count candle trades without fetching ticks")
    parser.add_argument("--initial-balance", type=float, default=CFG.bt.initial_balance)
    parser.add_argument("--risk", type=float, default=CFG.bt.risk_per_trade)
    parser.add_argument("--fee-bps", type=float, default=4.0)
    parser.add_argument("--slip-bps", type=float, default=3.0)
    parser.add_argument("--max-concurrent", type=int, default=CFG.bt.max_concurrent_trades)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.passed_only and args.failed_only:
        raise SystemExit("use only one of --passed-only or --failed-only")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
