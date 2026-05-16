"""Programmatic wrapper around the existing multi-seed auto optimizer."""
from __future__ import annotations

from pathlib import Path

from tools import auto_algo_finder


def build_finder_argv(
    *,
    patterns: list[str],
    out_dir: str,
    trials: int,
    jobs: int,
    seeds: list[int],
    targets: list[float],
    top: int,
    strategy: str,
    holdout: float,
    fee_bps: float,
    slip_bps: float,
    stress_fee_bps: float,
    stress_slip_bps: float,
    risk_per_trade: float,
    save_top_trials: int,
    max_candidate_trials: int,
    candidate_min_train_score: float,
    candidate_max_train_score: float,
    seed_variant_count: int,
    seed_jitter: float,
    seed_param_files: list[str],
    component_modules: list[str],
    skip_existing: bool,
    optimize_exits: bool,
    optimize_hold_time: bool,
) -> list[str]:
    argv: list[str] = [
        "--patterns",
        *patterns,
        "--out-dir",
        out_dir,
        "--trials",
        str(trials),
        "--strategy",
        strategy,
        "--jobs",
        str(jobs),
        "--holdout",
        str(holdout),
        "--top",
        str(top),
        "--targets",
        *[str(v) for v in targets],
        "--seeds",
        *[str(v) for v in seeds],
        "--fee-bps",
        str(fee_bps),
        "--slip-bps",
        str(slip_bps),
        "--stress-fee-bps",
        str(stress_fee_bps),
        "--stress-slip-bps",
        str(stress_slip_bps),
        "--risk-per-trade",
        str(risk_per_trade),
        "--save-top-trials",
        str(save_top_trials),
        "--max-candidate-trials",
        str(max_candidate_trials),
        "--candidate-min-train-score",
        str(candidate_min_train_score),
        "--candidate-max-train-score",
        str(candidate_max_train_score),
        "--seed-variant-count",
        str(seed_variant_count),
        "--seed-jitter",
        str(seed_jitter),
    ]
    if seed_param_files:
        argv += ["--seed-param-files", *seed_param_files]
    for module in component_modules:
        argv += ["--component-module", module]
    if skip_existing:
        argv.append("--skip-existing")
    if optimize_exits:
        argv.append("--optimize-exits")
    if optimize_hold_time:
        argv.append("--optimize-hold-time")
    return argv


def run_finder(**kwargs) -> int:
    argv = build_finder_argv(**kwargs)
    args = auto_algo_finder.parse_args(argv)
    return auto_algo_finder.run(args)


def backtest_patterns_for_results(results) -> list[str]:
    return [str(Path(result.backtest_path)) for result in results]
