# Strategy Research Memory

Last updated: 2026-05-14

## Goal

Find BTCUSDT strategies on 7-year data that balance:

- Good profit and CAGR
- Enough trades per month
- Robustness through crash periods
- Survival under stressed slippage
- Low chance of selecting fake overfit trials

This is research tooling, not live-trading approval.

## Important Data

Use 7-year files for serious testing:

- `data_store/BTCUSDT_1h_backtest_7y.parquet`
- `data_store/BTCUSDT_1h_forward_7y.parquet`
- `data_store/BTCUSDT_4h_backtest_7y.parquet`
- `data_store/BTCUSDT_4h_forward_7y.parquet`

Avoid mixing 5y and 7y outputs in one folder when judging results.

## Main Code Paths

- Strategy construction: `tools/enhanced_strategy.py`
  - `build_enhanced_signals()`
  - `_apply_entry_quality_filters()`
  - `enhanced_backtest_strategy()`

- Optimizer and ranking: `tools/auto_algo_finder.py`
  - `_optimize_enhanced()`
  - `_score_with_user_targets()`
  - `_guardrail_failures()`
  - `_rank_score()`

## Key Code Changes Made

1. Added normal and stress slippage support:
   - `--fee-bps`
   - `--slip-bps`
   - `--stress-fee-bps`
   - `--stress-slip-bps`

2. Added seed-file support and focused neighborhood search:
   - `--seed-param-files`
   - `--seed-variant-count`
   - `--seed-jitter`

3. Added resumable runs:
   - `--skip-existing`

4. Added candidate preservation so good normal trials are not lost when suspicious huge train scores appear:
   - `--save-top-trials`
   - `--candidate-min-train-score`
   - `--candidate-max-train-score`
   - `--max-candidate-trials`

5. Each run now can write:
   - `<run_id>.json` selected candidate for that run
   - `<run_id>_candidates.json` preserved candidate list
   - `leaderboard.csv`
   - `leaderboard.json`
   - `best_passed.json`
   - `best_passed_params.json`

6. Added batch tick validation:
   - `tick_forward_test.py` now supports `--strategy enhanced`
   - tick replay now uses the enhanced confluence signal builder for enhanced optimizer JSONs
   - partial exits are handled in tick replay
   - `tools/batch_tick_forward_test.py` can tick-test saved leaderboard rows and write combined summaries
   - long trade windows are split into cache chunks with `--tick-chunk-hours` so a single long trade does not hit the Binance request cap and reruns can reuse completed chunks
   - `tick_cache.py` now prefers official Binance monthly `aggTrades` archive files from `data.binance.vision` before REST fallback
   - `tools/download_tick_archives.py` can preload archive data into `data_store/tick_cache.duckdb`
   - tick replay now queries DuckDB for first entry/exit/partial-hit ticks instead of loading huge multi-week tick windows into pandas

Official tick archive preload completed:

- Range: `2024-01-01` to `2025-05-01`
- Symbol: `BTCUSDT`
- Archives imported: `16`
- Rows imported: `746,102,244`
- DB file: `data_store/tick_cache.duckdb`
- DB size after preload: about `29.5 GB`
- Archive zips folder: `data_store/binance_archive/spot/monthly/aggTrades/BTCUSDT`
- Smoke test through trade `20/354` now passes from local archive cache.

Completed passed-strategy tick validation:

- Folder: `data_store/focused_robust_7y_candidates_v3/tick_validation_passed`
- Rows tested: `12`
- All 12 remained profitable after tick replay.
- Tick PF range: `1.398` to `1.922`
- Median tick PF: `1.615`
- Tick return range: `100.7%` to `275.5%`
- Median tick return: `172.3%`
- Tick expR range: `0.236` to `0.465`
- 9 of 12 had tick PF above `1.5`
- Best tick PF:
  - `BTCUSDT_1h_backtest_7y_tf-1h_target-22_seed-11`
  - Tick PF: `1.922`
  - Tick expR: `0.465`
  - Tick return: `275.5%`
  - Tick max drawdown: `16.8%`
- Best original candle-rank candidate after tick:
  - `BTCUSDT_1h_backtest_7y_tf-1h_target-26_seed-42`
  - Tick PF: `1.796`
  - Tick expR: `0.340`
  - Tick return: `214.6%`
  - Tick max drawdown: `14.6%`
- Evaluation CSV:
  - `data_store/focused_robust_7y_candidates_v3/tick_validation_passed/tick_summary_evaluated.csv`

Current short list for next-stage review:

1. `BTCUSDT_1h_backtest_7y_tf-1h_target-22_seed-11`
2. `BTCUSDT_1h_backtest_7y_tf-1h_target-26_seed-42`
3. `BTCUSDT_1h_backtest_7y_tf-1h_target-22_seed-101`
4. `BTCUSDT_1h_backtest_7y_tf-1h_target-26_seed-202`

Before any live or paper trading, inspect trade timing, monthly returns, crash-period behavior, and exchange constraints/funding assumptions.

Portfolio combination tooling:

- Added `tools/build_trade_portfolio.py`
- It combines tick-validated trade CSVs into one shared-account simulation.
- It uses portfolio-level risk/concurrency caps instead of blindly adding all strategy PnL.
- Initial passed-only top-4 portfolio:
  - Folder: `data_store/focused_robust_7y_candidates_v3/portfolio_top4_passed`
  - Risk/trade: `0.3%`
  - Max open trades: `6`
  - Strategies: `4`
  - Trades: `497`
  - Return: `55.1%`
  - PF: `1.667`
  - expR: `0.305`
  - Max drawdown: `7.0%`
- Broader top-9 portfolio:
  - Folder: `data_store/focused_robust_7y_candidates_v3/portfolio_top9_passed`
  - Return: `63.1%`
  - PF: `1.493`
  - Max drawdown: `9.1%`
- Interpretation: more strategies increased return but reduced quality. Prefer a curated top-4/top-5 source before blindly using every strategy.

All-strategy tick validation completed:

- Folder: `data_store/focused_robust_7y_candidates_v3/tick_validation_all`
- Rows tested: `24`
- All 24 were profitable after tick replay.
- Strategies with tick PF above `1.5`: `16`
- 4h rows tested: `12`
- 4h rows with tick PF above `1.5`: `7`
- Important: all 4h rows still have original overfit warnings such as `train_holdout_score_gap`, so use them only as secondary confirmation until more checks pass.

Confluence wrapper / agreement filter:

- `tools/build_trade_portfolio.py` now supports:
  - `--min-agree`
  - `--agree-window-hours`
- Meaning: only take a trade when at least N selected strategies give the same-direction trade near the same time.
- Best current wrapper:
  - Folder: `data_store/focused_robust_7y_candidates_v3/portfolio_confluence_top8_all_agree2`
  - Selected strategies: top 8 tick strategies
  - Rule: at least 2 strategies agree within 4 hours
  - Risk/trade: `0.3%`
  - Max open trades: `6`
  - Trades: `407`
  - Return: `65.9%`
  - PF: `2.069`
  - expR: `0.424`
  - Max drawdown: `4.4%`
- This is currently better than both:
  - top-4 passed-only no-agreement portfolio: `55.1%` return, PF `1.667`, DD `7.0%`
  - top-10 all-tick no-agreement portfolio: `52.1%` return, PF `1.468`, DD `10.5%`

Risk sweep for best wrapper:

- File: `data_store/focused_robust_7y_candidates_v3/portfolio_confluence_risk_sweep.csv`
- Same wrapper: top 8 tick strategies, 2 agree within 4 hours, max 6 open trades.
- Results:
  - `0.3%` risk/trade: return `65.9%`, PF `2.069`, DD `4.4%`
  - `0.5%` risk/trade: return `129.6%`, PF `1.981`, DD `7.2%`
  - `0.75%` risk/trade: return `240.0%`, PF `1.881`, DD `10.6%`
  - `1.0%` risk/trade: return `396.7%`, PF `1.793`, DD `13.9%`
  - `1.5%` risk/trade: return `919.3%`, PF `1.648`, DD `20.1%`
- For a small account like `$200`, the `0.75%` backtest maps to about `$680` final balance before real-world frictions/min-order constraints.
- Warning: this does not model liquidation, futures funding, exchange minimum order sizes, or live execution failures. Do not equate this with safe 10x leverage.

## Candidate Selection Logic

The optimizer no longer trusts only Optuna's single highest training value.

It keeps:

- Top raw train-score trials, even if suspicious
- Sane strong trials inside a score band

Recommended settings:

```powershell
"--save-top-trials", "25",
"--candidate-min-train-score", "4",
"--candidate-max-train-score", "25",
"--max-candidate-trials", "150",
```

Plain meaning:

- Save top 25 raw training scores.
- Also save strong normal trials with train score 4 to 25.
- Evaluate up to 150 candidates per run on holdout, forward, and stress forward.

If a run has 50 trials with score `100+` and 100 trials with score `6+`, this should preserve the `6+` trials too, up to the max candidate cap.

## Current Notable Results

Folder checked:

- `data_store/focused_robust_7y`

Summary:

- Rows: 21
- Passed: 5
- Best passed:
  - `BTCUSDT_1h_backtest_7y_tf-1h_target-26_seed-202`
  - Rank: `11.267`
  - Forward PF: `1.462`
  - Stress PF: `1.187`

Interesting high-rank rows from cancelled/partial output:

- `BTCUSDT_4h_backtest_7y_tf-4h_target-8_seed-42`
  - Rank: `15.162`
  - Holdout PF: `1.678`
  - Forward PF: `1.551`
  - Stress PF: `1.682`
  - `passed=False`, inspect guardrail failures before trusting.

- `BTCUSDT_1h_backtest_7y_tf-1h_target-22_seed-101`
  - Rank: `13.668`
  - Holdout PF: `1.384`
  - Forward PF: `1.560`
  - Stress PF: `1.474`
  - `passed=False`, inspect guardrail failures before trusting.

- `BTCUSDT_1h_backtest_7y_tf-1h_target-26_seed-202`
  - Rank: `11.267`
  - Forward PF: `1.462`
  - Stress PF: `1.187`
  - `passed=True`

## Current Active Run

The latest completed clean run is:

- Output folder: `data_store/focused_robust_7y_candidates_v3`
- Data: 7-year BTCUSDT only
- Candidate preservation enabled:
  - `--save-top-trials 25`
  - `--candidate-min-train-score 4`
  - `--candidate-max-train-score 25`
  - `--max-candidate-trials 150`
- Focused seed neighborhood search enabled:
  - `--seed-variant-count 60`
  - `--seed-jitter 0.04`
- Normal execution assumption:
  - `--fee-bps 4`
  - `--slip-bps 3`
- Stress execution assumption:
  - `--stress-fee-bps 4`
  - `--stress-slip-bps 10`

Completed result:

- Rows: `24`
- Passed: `12`
- Failed: `12`
- All passed rows were `1h`.
- All `4h` rows failed because of train/holdout overfit guardrails, mostly `train_holdout_score_gap`.

Best passed strategy:

- `BTCUSDT_1h_backtest_7y_tf-1h_target-26_seed-42`
- File: `data_store/focused_robust_7y_candidates_v3/BTCUSDT_1h_backtest_7y_tf-1h_target-26_seed-42.json`
- Best params: `data_store/focused_robust_7y_candidates_v3/best_passed_params.json`
- Rank score: `23.278`
- Train value: `5.427`
- Holdout PF: `1.582`
- Holdout expR: `0.190`
- Forward PF: `2.075`
- Forward expR: `0.333`
- Forward return: `212.3%`
- Forward CAGR: `135.2%`
- Forward trades/month: `22.16`
- Forward max drawdown: `8.27%`
- Stress forward PF at 10 bps slippage: `1.929`
- Stress forward expR: `0.313`
- Stress forward return: `178.8%`
- Stress forward CAGR: `116.0%`
- Stress forward max drawdown: `8.45%`
- Monte Carlo profit probability: `1.0`
- Monte Carlo 5th percentile return: `131.7%`

This is the strongest current candle-level candidate. It is not live-trading approval yet. The next filter is tick-level validation and trade inspection.

## Next Recommended Steps

1. Run tick-level validation on the best passed candidate:
   - `data_store/focused_robust_7y_candidates_v3/best_passed.json`
   - `data_store/focused_robust_7y_candidates_v3/best_passed_params.json`
2. Inspect the top 3 passed candidates, not just the single winner:
   - `BTCUSDT_1h_backtest_7y_tf-1h_target-26_seed-42`
   - `BTCUSDT_1h_backtest_7y_tf-1h_target-22_seed-101`
   - `BTCUSDT_1h_backtest_7y_tf-1h_target-26_seed-202`
3. For future candidates, require:
   - `passed=True`
   - forward PF above `1.10`
   - stress PF above `1.02`
   - forward expectancy R above `0.04`
   - stress expectancy R above `0`
   - acceptable drawdown
   - enough trades per month
4. Inspect high-rank `passed=False` candidates only after reading their `guardrail_failures`.
5. Only after stable candle and tick candidates exist, build ML as a trade filter, not as a direct BTC price predictor.

Tick validation command shape:

```powershell
.\.venv\Scripts\python.exe tools\batch_tick_forward_test.py --leaderboard data_store\focused_robust_7y_candidates_v3\leaderboard_with_verdicts.csv --out-dir data_store\focused_robust_7y_candidates_v3\tick_validation --symbol BTCUSDT --passed-only --fee-bps 4 --slip-bps 3 --risk 0.01 --skip-existing
```

Use `--max-strategies 3` or `--max-trades 50` for a smaller first run. Full tick validation may take a long time because the best strategies have hundreds of trades and are exposed across most of the forward period.

If a long trade hits Binance request limits, resume with chunking:

```powershell
.\.venv\Scripts\python.exe tools\batch_tick_forward_test.py --leaderboard data_store\focused_robust_7y_candidates_v3\leaderboard_with_verdicts.csv --out-dir data_store\focused_robust_7y_candidates_v3\tick_validation_passed --symbol BTCUSDT --passed-only --fee-bps 4 --slip-bps 3 --risk 0.01 --tick-chunk-hours 6 --tick-max-requests 25000 --skip-existing
```

After the archive preload, the same command should mostly read local DuckDB instead of downloading from REST.

## ML Direction

Do not start with deep learning.

Best ML use:

- Base strategy creates a trade.
- ML model decides take/skip.

First model should likely be tabular:

- LightGBM / XGBoost / RandomForest / Logistic Regression

GPU can help later, but the first priority is clean labels and robust walk-forward validation.
