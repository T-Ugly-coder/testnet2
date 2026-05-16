"""CLI for the reusable strategy research framework."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import download_many_sync
from .optimizer import backtest_patterns_for_results, run_finder


def _add_optimizer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 42, 101, 137])
    parser.add_argument("--targets", type=float, nargs="+", default=[8.0, 12.0])
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--strategy", choices=["enhanced", "base"], default="enhanced")
    parser.add_argument("--holdout", type=float, default=0.30)
    parser.add_argument("--fee-bps", type=float, default=4.0)
    parser.add_argument("--slip-bps", type=float, default=3.0)
    parser.add_argument("--stress-fee-bps", type=float, default=4.0)
    parser.add_argument("--stress-slip-bps", type=float, default=10.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--save-top-trials", type=int, default=25)
    parser.add_argument("--max-candidate-trials", type=int, default=150)
    parser.add_argument("--candidate-min-train-score", type=float, default=4.0)
    parser.add_argument("--candidate-max-train-score", type=float, default=25.0)
    parser.add_argument("--seed-variant-count", type=int, default=60)
    parser.add_argument("--seed-jitter", type=float, default=0.04)
    parser.add_argument("--seed-param-files", nargs="+", default=[])
    parser.add_argument("--component-module", action="append", default=[])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--optimize-exits", action="store_true")
    parser.add_argument("--optimize-hold-time", action="store_true")


def _finder_kwargs(args: argparse.Namespace, patterns: list[str], out_dir: Path) -> dict:
    return {
        "patterns": patterns,
        "out_dir": str(out_dir),
        "trials": args.trials,
        "jobs": args.jobs,
        "seeds": args.seeds,
        "targets": args.targets,
        "top": args.top,
        "strategy": args.strategy,
        "holdout": args.holdout,
        "fee_bps": args.fee_bps,
        "slip_bps": args.slip_bps,
        "stress_fee_bps": args.stress_fee_bps,
        "stress_slip_bps": args.stress_slip_bps,
        "risk_per_trade": args.risk_per_trade,
        "save_top_trials": args.save_top_trials,
        "max_candidate_trials": args.max_candidate_trials,
        "candidate_min_train_score": args.candidate_min_train_score,
        "candidate_max_train_score": args.candidate_max_train_score,
        "seed_variant_count": args.seed_variant_count,
        "seed_jitter": args.seed_jitter,
        "seed_param_files": args.seed_param_files,
        "component_modules": args.component_module,
        "skip_existing": args.skip_existing,
        "optimize_exits": args.optimize_exits,
        "optimize_hold_time": args.optimize_hold_time,
    }


def _cmd_download(args: argparse.Namespace) -> int:
    results = download_many_sync(
        symbol=args.symbol,
        timeframes=args.tf,
        out_dir=Path(args.out_dir),
        years=args.years,
        start_value=args.start,
        end_value=args.end,
        label=args.label,
        forward_frac=args.forward_frac,
        refresh=args.refresh,
    )
    for result in results:
        print(
            f"{result.symbol} {result.timeframe}: {result.rows} rows -> "
            f"{result.backtest_path}"
        )
        if result.forward_path:
            print(f"{result.symbol} {result.timeframe}: forward -> {result.forward_path}")
    return 0


def _cmd_find(args: argparse.Namespace) -> int:
    patterns = [str(Path(p)) for p in args.bars]
    return run_finder(**_finder_kwargs(args, patterns, Path(args.out_dir)))


def _cmd_run(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    results = download_many_sync(
        symbol=args.symbol,
        timeframes=args.tf,
        out_dir=data_dir,
        years=args.years,
        start_value=args.start,
        end_value=args.end,
        label=args.label,
        forward_frac=args.forward_frac,
        refresh=args.refresh,
    )
    patterns = backtest_patterns_for_results(results)
    out_dir = Path(args.out_dir or data_dir / "strategy_runs")
    return run_finder(**_finder_kwargs(args, patterns, out_dir))


def _synth_bars(n: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC")
    drift = 0.00015 + 0.0006 * np.sin(np.linspace(0.0, 24.0, n))
    ret = drift + rng.normal(0.0, 0.006, size=n)
    close = 100.0 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.maximum(close * rng.uniform(0.001, 0.012, size=n), 0.01)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.lognormal(mean=7.0, sigma=0.4, size=n)
    return pd.DataFrame(
        {
            "ts_ms": (ts.view("int64") // 1_000_000).astype("int64"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=ts,
    )


def _cmd_smoke(args: argparse.Namespace) -> int:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    backtest = _synth_bars(args.backtest_bars, seed=11)
    forward = _synth_bars(args.forward_bars, seed=22)
    backtest_path = out / "TESTUSDT_1h_backtest_smoke.parquet"
    forward_path = out / "TESTUSDT_1h_forward_smoke.parquet"
    backtest.to_parquet(backtest_path)
    forward.to_parquet(forward_path)
    print(json.dumps({"backtest": str(backtest_path), "forward": str(forward_path)}, indent=2))
    args.bars = [str(backtest_path)]
    args.out_dir = str(out / "runs")
    args.targets = [args.target]
    args.seeds = [args.seed]
    args.strategy = "enhanced"
    args.jobs = 1
    args.seed_param_files = []
    args.skip_existing = False
    args.optimize_exits = False
    args.optimize_hold_time = False
    args.stress_slip_bps = 0.0
    args.save_top_trials = 1
    args.max_candidate_trials = 1
    args.candidate_min_train_score = -1_000_000.0
    args.candidate_max_train_score = 0.0
    return _cmd_find(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="strategy_framework",
        description="Download market parquet files and run multi-seed strategy discovery.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="download Binance candle parquet files")
    download.add_argument("--symbol", required=True)
    download.add_argument("--tf", nargs="+", default=["1h"])
    download.add_argument("--out-dir", required=True)
    download.add_argument("--years", type=int, default=4)
    download.add_argument("--start")
    download.add_argument("--end")
    download.add_argument("--label")
    download.add_argument("--forward-frac", type=float, default=0.30)
    download.add_argument("--refresh", action="store_true")
    download.set_defaults(func=_cmd_download)

    find = sub.add_parser("find", help="run strategy discovery on existing parquet files")
    find.add_argument("--bars", nargs="+", required=True)
    find.add_argument("--out-dir", required=True)
    _add_optimizer_args(find)
    find.set_defaults(func=_cmd_find)

    run = sub.add_parser("run", help="download parquet files, then run discovery")
    run.add_argument("--symbol", required=True)
    run.add_argument("--tf", nargs="+", default=["1h"])
    run.add_argument("--data-dir", required=True)
    run.add_argument("--out-dir")
    run.add_argument("--years", type=int, default=4)
    run.add_argument("--start")
    run.add_argument("--end")
    run.add_argument("--label")
    run.add_argument("--forward-frac", type=float, default=0.30)
    run.add_argument("--refresh", action="store_true")
    _add_optimizer_args(run)
    run.set_defaults(func=_cmd_run)

    smoke = sub.add_parser("smoke", help="run a local synthetic end-to-end smoke test")
    smoke.add_argument("--out-dir", default="data_store/framework_smoke")
    smoke.add_argument("--seed", type=int, default=11)
    smoke.add_argument("--target", type=float, default=4.0)
    smoke.add_argument("--backtest-bars", type=int, default=1800)
    smoke.add_argument("--forward-bars", type=int, default=600)
    _add_optimizer_args(smoke)
    smoke.set_defaults(
        func=_cmd_smoke,
        trials=1,
        jobs=1,
        seed_variant_count=0,
        targets=[4.0],
        seeds=[11],
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
