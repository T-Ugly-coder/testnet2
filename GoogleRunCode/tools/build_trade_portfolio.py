"""Build a combined trade stream from tick-validated strategies.

This script does not optimize strategy parameters. It takes completed tick
validation folders, selects strategies by tick quality, combines their trades,
and simulates one shared account with portfolio-level risk/concurrency caps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _jsonable(value: Any) -> Any:
    if isinstance(value, float):
        if np.isfinite(value):
            return value
        return None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_summaries(tick_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for tick_dir in tick_dirs:
        summary_path = tick_dir / "tick_summary_evaluated.csv"
        if not summary_path.exists():
            summary_path = tick_dir / "tick_summary.csv"
        if not summary_path.exists():
            continue
        df = pd.read_csv(summary_path)
        df["tick_dir"] = str(tick_dir)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("no tick_summary.csv/tick_summary_evaluated.csv found")
    out = pd.concat(frames, ignore_index=True)
    for col in [
        "tick_profit_factor",
        "tick_total_return",
        "tick_expectancy_R",
        "tick_max_drawdown",
        "tick_win_rate",
        "source_rank_score",
    ]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _select_strategies(summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = summary.copy()
    df = df[df["tick_profit_factor"] >= args.min_tick_pf]
    df = df[df["tick_expectancy_R"] >= args.min_tick_exp_r]
    df = df[df["tick_total_return"] >= args.min_tick_return]
    df = df[df["tick_max_drawdown"].abs() <= args.max_tick_dd]
    df = df.sort_values(
        ["tick_profit_factor", "tick_expectancy_R", "tick_total_return"],
        ascending=False,
    )
    if args.top_strategies > 0:
        df = df.head(args.top_strategies)
    if df.empty:
        raise SystemExit("no strategies matched the portfolio filters")
    return df.reset_index(drop=True)


def _load_trades(selected: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for priority, row in selected.iterrows():
        run_id = str(row["run_id"])
        trades_path = Path(row["tick_dir"]) / f"{run_id}_tick_trades.csv"
        if not trades_path.exists():
            continue
        trades = pd.read_csv(trades_path)
        if trades.empty:
            continue
        trades["run_id"] = run_id
        trades["strategy_priority"] = int(priority)
        trades["strategy_tick_pf"] = float(row["tick_profit_factor"])
        trades["strategy_tick_exp_r"] = float(row["tick_expectancy_R"])
        frames.append(trades)
    if not frames:
        raise FileNotFoundError("selected strategies did not have trade CSV files")
    out = pd.concat(frames, ignore_index=True)
    out["entry_ts_ms"] = pd.to_numeric(out["entry_ts_ms"], errors="coerce").astype("int64")
    out["exit_ts_ms"] = pd.to_numeric(out["exit_ts_ms"], errors="coerce").astype("int64")
    out["direction"] = pd.to_numeric(out["direction"], errors="coerce").astype("int64")
    out["risk_amt"] = pd.to_numeric(out["risk_amt"], errors="coerce")
    out["pnl"] = pd.to_numeric(out["pnl"], errors="coerce")
    out["pnl_R"] = out["pnl"] / out["risk_amt"].replace(0.0, np.nan)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["pnl_R"])
    return out.sort_values(["entry_ts_ms", "strategy_priority"]).reset_index(drop=True)


def _dedupe_trades(trades: pd.DataFrame, window_hours: float) -> pd.DataFrame:
    if window_hours <= 0 or trades.empty:
        return trades
    window_ms = int(window_hours * 3_600_000)
    kept = []
    last_by_side: dict[int, int] = {}
    for row in trades.itertuples(index=False):
        direction = int(row.direction)
        entry_ts = int(row.entry_ts_ms)
        last_ts = last_by_side.get(direction)
        if last_ts is not None and entry_ts - last_ts <= window_ms:
            continue
        kept.append(row._asdict())
        last_by_side[direction] = entry_ts
    return pd.DataFrame(kept)


def _add_agreement_counts(trades: pd.DataFrame, window_hours: float) -> pd.DataFrame:
    if trades.empty:
        return trades
    window_ms = int(max(0.0, window_hours) * 3_600_000)
    frames = []
    for _, side in trades.groupby("direction", sort=False):
        side = side.sort_values("entry_ts_ms").reset_index(drop=True)
        entries = side["entry_ts_ms"].to_numpy(np.int64)
        run_ids = side["run_id"].astype(str).to_numpy()
        counts = []
        names = []
        for entry_ts in entries:
            lo = np.searchsorted(entries, entry_ts - window_ms, side="left")
            hi = np.searchsorted(entries, entry_ts + window_ms, side="right")
            agreeing = sorted(set(run_ids[lo:hi]))
            counts.append(len(agreeing))
            names.append("|".join(agreeing))
        side["agreement_count"] = counts
        side["agreement_run_ids"] = names
        frames.append(side)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["entry_ts_ms", "strategy_priority"]).reset_index(drop=True)


def _simulate_portfolio(
    trades: pd.DataFrame,
    *,
    initial_balance: float,
    risk_per_trade: float,
    max_open_trades: int,
    max_strategy_open_trades: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    balance = float(initial_balance)
    peak = balance
    max_dd = 0.0
    accepted = []
    open_positions: list[dict[str, Any]] = []
    pnls = []

    for row in trades.itertuples(index=False):
        entry_ts = int(row.entry_ts_ms)
        still_open = []
        for pos in open_positions:
            if int(pos["exit_ts_ms"]) <= entry_ts:
                balance += float(pos["portfolio_pnl"])
                pnls.append(float(pos["portfolio_pnl"]))
                peak = max(peak, balance)
                max_dd = min(max_dd, balance / peak - 1.0)
            else:
                still_open.append(pos)
        open_positions = still_open

        if len(open_positions) >= max_open_trades:
            continue
        strategy_open = sum(1 for pos in open_positions if pos["run_id"] == row.run_id)
        if strategy_open >= max_strategy_open_trades:
            continue

        risk_amt = balance * risk_per_trade
        pnl = risk_amt * float(row.pnl_R)
        rec = row._asdict()
        rec["portfolio_risk_amt"] = risk_amt
        rec["portfolio_pnl"] = pnl
        accepted.append(rec)
        open_positions.append(rec)

    for pos in sorted(open_positions, key=lambda x: int(x["exit_ts_ms"])):
        balance += float(pos["portfolio_pnl"])
        pnls.append(float(pos["portfolio_pnl"]))
        peak = max(peak, balance)
        max_dd = min(max_dd, balance / peak - 1.0)

    pnls_arr = np.asarray(pnls, dtype=float)
    wins = pnls_arr[pnls_arr > 0]
    losses = pnls_arr[pnls_arr < 0]
    gross_profit = float(wins.sum()) if wins.size else 0.0
    gross_loss = float(-losses.sum()) if losses.size else 0.0
    if gross_loss > 0:
        pf: float | str = gross_profit / gross_loss
    else:
        pf = "inf" if gross_profit > 0 else 0.0

    accepted_df = pd.DataFrame(accepted)
    report = {
        "initial_balance": initial_balance,
        "final_balance": balance,
        "total_return": balance / initial_balance - 1.0,
        "n_trades": int(len(accepted_df)),
        "win_rate": float((pnls_arr > 0).mean()) if pnls_arr.size else 0.0,
        "profit_factor": pf,
        "expectancy": float(pnls_arr.mean()) if pnls_arr.size else 0.0,
        "expectancy_R": float(np.nanmean(accepted_df["pnl_R"])) if not accepted_df.empty else 0.0,
        "max_drawdown": max_dd,
        "max_open_trades": max_open_trades,
        "max_strategy_open_trades": max_strategy_open_trades,
        "risk_per_trade": risk_per_trade,
        "unique_strategies": int(accepted_df["run_id"].nunique()) if not accepted_df.empty else 0,
    }
    return report, accepted_df


def run(args: argparse.Namespace) -> int:
    tick_dirs = [Path(p) for p in args.tick_dirs]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_summaries(tick_dirs)
    selected = _select_strategies(summary, args)
    trades = _load_trades(selected)
    trades = _add_agreement_counts(trades, args.agree_window_hours)
    if args.min_agree > 1:
        trades = trades[trades["agreement_count"] >= args.min_agree].copy()
        if trades.empty:
            raise SystemExit("no trades matched the agreement filter")
    trades = _dedupe_trades(trades, args.dedupe_window_hours)
    report, accepted = _simulate_portfolio(
        trades,
        initial_balance=args.initial_balance,
        risk_per_trade=args.risk_per_trade,
        max_open_trades=args.max_open_trades,
        max_strategy_open_trades=args.max_strategy_open_trades,
    )
    report["min_agree"] = args.min_agree
    report["agree_window_hours"] = args.agree_window_hours

    selected_path = out_dir / "selected_strategies.csv"
    trades_path = out_dir / "portfolio_trades.csv"
    report_path = out_dir / "portfolio_report.json"
    selected.to_csv(selected_path, index=False)
    accepted.to_csv(trades_path, index=False)
    report_path.write_text(json.dumps(_jsonable(report), indent=2))

    print("=== PORTFOLIO REPORT ===")
    for key, value in report.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6g}")
        else:
            print(f"{key}: {value}")
    print(f"wrote {selected_path}")
    print(f"wrote {trades_path}")
    print(f"wrote {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a combined tick-validated trade portfolio")
    parser.add_argument("--tick-dirs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-strategies", type=int, default=4)
    parser.add_argument("--min-tick-pf", type=float, default=1.5)
    parser.add_argument("--min-tick-exp-r", type=float, default=0.20)
    parser.add_argument("--min-tick-return", type=float, default=0.0)
    parser.add_argument("--max-tick-dd", type=float, default=0.22)
    parser.add_argument("--risk-per-trade", type=float, default=0.003)
    parser.add_argument("--initial-balance", type=float, default=100000.0)
    parser.add_argument("--max-open-trades", type=int, default=6)
    parser.add_argument("--max-strategy-open-trades", type=int, default=2)
    parser.add_argument("--min-agree", type=int, default=1)
    parser.add_argument("--agree-window-hours", type=float, default=4.0)
    parser.add_argument("--dedupe-window-hours", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
