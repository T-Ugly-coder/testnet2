"""CLI entry point: `python -m algo.optimize <bars.parquet> --tf 1h --trials 200`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import OptimizeConfig, ObjectiveConfig, optimize, rank_top_strategies


def _load_bars(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix in (".parquet", ".pq"):
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"unsupported file type: {p.suffix}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="algo.optimize")
    ap.add_argument("bars", help="OHLC parquet/csv with columns open,high,low,close")
    ap.add_argument("--tf", default="1h", help="timeframe key (1m/5m/15m/1h/4h/1d/1w)")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout", type=float, default=0.30)
    ap.add_argument("--target-tpm", type=float, default=12.0,
                    help="target trades per month (default 12, mid of 10-15)")
    ap.add_argument("--top", type=int, default=10, help="report top-K trials")
    ap.add_argument("--out", default=None,
                    help="optional path to write best_params + summary as JSON")
    args = ap.parse_args(argv)

    bars = _load_bars(args.bars)
    ocfg = OptimizeConfig(
        timeframe=args.tf,
        holdout_frac=args.holdout,
        n_trials=args.trials,
        n_jobs=args.jobs,
        seed=args.seed,
        objective_cfg=ObjectiveConfig(
            target_trades_per_month=args.target_tpm,
            min_trades_per_month=max(1.0, args.target_tpm * 0.7),
            max_trades_per_month=args.target_tpm * 1.7,
        ),
    )

    result = optimize(bars, ocfg)
    top = rank_top_strategies(result["study"], top_k=args.top)

    print("=== best params ===")
    for k, v in result["best_params"].items():
        print(f"  {k}: {v}")
    print("\n=== TRAIN summary ===")
    for k, v in result["train_score"].items():
        if k != "config":
            print(f"  {k}: {v}")
    print("\n=== HOLDOUT summary (untouched) ===")
    for k, v in result["holdout_score"].items():
        if k != "config":
            print(f"  {k}: {v}")
    print(f"\n=== top-{len(top)} trials ===")
    if not top.empty:
        with pd.option_context("display.max_columns", None, "display.width", 220):
            print(top.to_string(index=False))

    if args.out:
        import json
        out = {
            "best_params": result["best_params"],
            "train_score": {k: v for k, v in result["train_score"].items() if k != "config"},
            "holdout_score": {k: v for k, v in result["holdout_score"].items() if k != "config"},
        }
        Path(args.out).write_text(json.dumps(out, indent=2, default=float))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
