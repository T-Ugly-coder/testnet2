# OpenClaude Operating Rules

This project is on Windows PowerShell. Prefer PowerShell commands and the local venv:

```powershell
.\.venv\Scripts\python.exe -m algo_download.optimize --help
.\.venv\Scripts\python.exe -m algo.optimize --help
.\run_auto_algo_finder.ps1 -Trials 100
.\run_auto_algo_finder.ps1 -Trials 100 -Strategy enhanced
.\run_auto_algo_finder.ps1 -Trials 100 -Strategy base
.\run_auto_algo_finder.ps1 -Trials 150 -Strategy enhanced -Seeds 7,29,73,101,137 -Targets 10,14,18,22 -OutDir data_store\multi_seed_run
```

Do not use Unix shell setup commands such as `export PYTHONPATH=...` or `$(pwd)`. Do not run `.venv\Scripts\python.exe` from a Bash-only shell; that is what caused exit code 127. If a tool only exposes Bash, call PowerShell explicitly:

```bash
powershell -NoProfile -ExecutionPolicy Bypass -Command ".\\.venv\\Scripts\\python.exe -m algo_download.optimize --help"
```

The correct optimizer module is `algo_download.optimize` or the compatibility alias `algo.optimize`. Avoid `algo_download.optimize.auto` as a CLI target; it is an internal module.

Optimization workflow:

1. Run tests before changing behavior:
   `.\.venv\Scripts\python.exe -m pytest algo_download\tests -q`
2. Run the robust sweep wrapper:
   `.\run_auto_algo_finder.ps1 -Trials 100 -Strategy enhanced`
3. Read `data_store\auto_finder\leaderboard.csv` and `leaderboard.json`.
4. Use `data_store\auto_finder\best_passed.json` as the current champion only when it exists. It contains the best candidate that passed holdout and forward guardrails.
5. Treat train score as weak evidence. Prefer candidates that pass guardrails on holdout and forward data.
6. Do not loosen guardrails just to make a result look profitable. BTC strategies with holdout profit factor below 1.05, negative expectancy_R, or drawdown above the configured cap should be rejected.
7. Make one small strategy or objective change at a time, then rerun tests and a short optimizer sweep before trying a longer run.
8. Compare `-Strategy enhanced` against `-Strategy base`. Keep enhanced changes only if forward profit factor and expectancy_R improve without a drawdown increase.

Current diagnosis: `data_store\BTCUSDT_1h_optimized.json` overfits. Train score is positive, but holdout score is negative with profit factor below 1 and negative expectancy, so more blind trials are unlikely to solve it alone.

Enhanced strategy notes: `tools\enhanced_strategy.py` adds breakout, breakout retest, failed breakout, volume-confirmed breakout, RSI/MACD momentum, ADX, and volatility squeeze scorers. The optimizer samples weights for these signals plus the existing trend, SFP, candle, regime, VSA, Wyckoff, and VWAP deviation scorers. More indicators are not automatically better; every new signal must earn its place on holdout and forward data. Newly added signals should default to weight `0.0` unless their weight is explicitly optimized, so old champions stay reproducible.

Anti-overfit guardrails are part of `tools\auto_algo_finder.py`: candidates fail if holdout/forward metrics miss minimums, if drawdown is too high, or if train performance is much stronger than holdout performance. Do not remove these checks to make a leaderboard look better.

Multi-seed automation: `run_auto_algo_finder.ps1` accepts `-Seeds` and `-Targets`. One run evaluates every seed/target/timeframe combination, ranks all rows together, writes `best_passed.json`, and also writes `seed_summary.csv` plus `target_summary.csv`.
