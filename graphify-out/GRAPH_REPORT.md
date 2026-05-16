# Graph Report - tryingnew  (2026-05-16)

## Corpus Check
- 112 files · ~61,974 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 654 nodes · 1138 edges · 46 communities (41 shown, 5 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 109 edges (avg confidence: 0.79)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]

## God Nodes (most connected - your core abstractions)
1. `run()` - 16 edges
2. `run_tick_validation()` - 15 edges
3. `compute_perf()` - 15 edges
4. `build_enhanced_signals()` - 15 edges
5. `optimize()` - 13 edges
6. `GoogleRunCode` - 13 edges
7. `connect()` - 11 edges
8. `ensure_ticks()` - 11 edges
9. `connect()` - 11 edges
10. `run_backtest()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `enhanced_backtest_strategy()`  [INFERRED]
  forward_test.py → tools/enhanced_strategy.py
- `summarize()` --calls--> `profit_factor()`  [INFERRED]
  tick_forward_test.py → GoogleRunCode/algo_download/metrics.py
- `run_tick_validation()` --calls--> `_build_scorer_signals()`  [INFERRED]
  tick_forward_test.py → GoogleRunCode/algo_download/strategy/runner.py
- `run_tick_validation()` --calls--> `run_backtest()`  [INFERRED]
  tick_forward_test.py → GoogleRunCode/algo_download/backtest/engine.py
- `run_tick_validation()` --calls--> `build_enhanced_signals()`  [INFERRED]
  tick_forward_test.py → tools/enhanced_strategy.py

## Communities (46 total, 5 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (47): BacktestConfig, Config, DataConfig, HardwareConfig, LLMConfig, Central config — no cron, all timing handled by event loop., aggregate_walk_forward(), Fold (+39 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (47): bars_per_year(), cagr(), compute_perf(), equity_returns(), max_drawdown(), PerfStats, profit_factor(), Performance analytics for an equity curve and a trade ledger.  All functions are (+39 more)

### Community 2 - "Community 2"
Cohesion: 0.11
Nodes (47): adx(), anchored_vwap(), atr_wilder(), bollinger(), cvd_from_ticks(), cvd_proxy(), ema(), macd() (+39 more)

### Community 3 - "Community 3"
Cohesion: 0.09
Nodes (46): _bars_range_ms(), main(), _parse_date_ms(), Download/import official Binance monthly aggTrades archives into tick cache., _archive_path(), _archive_url(), connect(), _covered_by_ranges() (+38 more)

### Community 4 - "Community 4"
Cohesion: 0.15
Nodes (28): _archive_path(), _archive_url(), connect(), _covered_by_ranges(), download_monthly_archive(), ensure_archive_ticks(), ensure_ticks(), get_ticks() (+20 more)

### Community 5 - "Community 5"
Cohesion: 0.11
Nodes (25): _check_required_columns(), _clean_series(), _dd_penalty(), _freq_penalty(), ObjectiveConfig, Composite objective for the auto-optimizer.  The objective collapses a multi-met, Weighted mean and (unbiased) weighted std for a 1-D array.      Falls back grace, Average trades per month across folds (weighted by fold bar-count). (+17 more)

### Community 6 - "Community 6"
Cohesion: 0.21
Nodes (26): _bars_per_day(), _exit_params(), _forward_metrics(), _forward_pair(), _guardrail_failures(), _infer_tf(), _jitter_seed_params(), _json_safe() (+18 more)

### Community 7 - "Community 7"
Cohesion: 0.16
Nodes (26): Alert, append_ledger(), Candidate, close_paper_trade(), env_float(), env_int(), fetch_klines(), format_alert() (+18 more)

### Community 8 - "Community 8"
Cohesion: 0.12
Nodes (23): atr_percentile_rank(), autocorr_lag1_rolling(), classify_regime(), hurst_rolling(), parkinson_vol(), Regime classification — what professionals run BEFORE any pattern fires.  Implem, Average R/S statistic over non-overlapping windows of size `lag`., Rolling Hurst exponent via R/S analysis.      Uses sub-series sizes [window/16, (+15 more)

### Community 9 - "Community 9"
Cohesion: 0.1
Nodes (22): code:bash (sudo apt update), code:bash (source .venv/bin/activate), code:bash (source .venv/bin/activate), code:bash (source .venv/bin/activate), code:bash (source .venv/bin/activate), code:env (PAPER_TRADING_ENABLED=1), code:bash (source .venv/bin/activate), code:bash (source .venv/bin/activate) (+14 more)

### Community 10 - "Community 10"
Cohesion: 0.17
Nodes (19): Vectorized event-driven backtester.  Design: - Strategy emits SignalEvents on ba, bars: DataFrame with open/high/low/close.     signals: dict with 'signal' (+1/-1, Determine SL/TP resolution for one bar.      Returns (reason, exit_px):, _resolve_exit(), run_backtest(), _simulate(), TradeResult, _make_bars() (+11 more)

### Community 11 - "Community 11"
Cohesion: 0.13
Nodes (16): Volume Spread Analysis (VSA) — Tom Williams style.  Detects:   - No Demand bar:, Returns (idx, code).     code: 1=no_demand, -1=no_supply, 2=stopping_volume_bull, vsa_signals(), ConfluenceScorer, _ctx_or_compute(), default(), Pluggable confluence scorer.  Idea ---- Each *scorer* is a stateless function th, Boost both sides equally when regime is trending; suppress when chop. (+8 more)

### Community 12 - "Community 12"
Cohesion: 0.18
Nodes (17): connect(), parquet_path(), DuckDB + Parquet storage for tick and bar data — replaces SQLite., Append ticks to partitioned parquet (by date).      df columns: ts_ms, price, qt, read_bars(), upsert_bars(), write_ticks(), download_one() (+9 more)

### Community 13 - "Community 13"
Cohesion: 0.15
Nodes (9): cvd_divergence(), liquidation_cascades(), oi_regime(), _pivots(), Microstructure & orderflow patterns — the layer pros use to confirm setups.  Imp, Return code per bar:         1 longs adding, 2 short covering,        -1 shorts, Returns (idx, side). side=1 long-liq cascade (down move), -1 short-liq., Return (idx, type) for fractal pivots: type=1 high, -1 low.     A pivot at i req (+1 more)

### Community 14 - "Community 14"
Cohesion: 0.14
Nodes (13): detect_fvg(), detect_inducement(), detect_order_blocks(), detect_sweeps(), equal_levels(), fvg_filled_mask(), SMC primitives — Order Blocks, FVGs, Sweeps, Inducements. Vectorized with numba;, Return (bar_idx, side, level). side: 1=buy-side sweep (bearish),     -1=sell-sid (+5 more)

### Community 15 - "Community 15"
Cohesion: 0.36
Nodes (12): _add_agreement_counts(), _dedupe_trades(), _jsonable(), _load_summaries(), _load_trades(), main(), parse_args(), Build a combined trade stream from tick-validated strategies.  This script does (+4 more)

### Community 16 - "Community 16"
Cohesion: 0.14
Nodes (13): Candidate Selection Logic, code:powershell ("--save-top-trials", "25",), code:powershell (.\.venv\Scripts\python.exe tools\batch_tick_forward_test.py ), code:powershell (.\.venv\Scripts\python.exe tools\batch_tick_forward_test.py ), Current Active Run, Current Notable Results, Goal, Important Data (+5 more)

### Community 17 - "Community 17"
Cohesion: 0.14
Nodes (13): Candidate Selection Logic, code:powershell ("--save-top-trials", "25",), code:powershell (.\.venv\Scripts\python.exe tools\batch_tick_forward_test.py ), code:powershell (.\.venv\Scripts\python.exe tools\batch_tick_forward_test.py ), Current Active Run, Current Notable Results, Goal, Important Data (+5 more)

### Community 18 - "Community 18"
Cohesion: 0.28
Nodes (12): equity_from_trades(), _jsonable(), load_bars(), load_best_params(), main(), Tick-level validation for optimized candle strategies.  Workflow: 1. Load forwar, resolve_trade_from_cache(), resolve_trade_with_ticks() (+4 more)

### Community 19 - "Community 19"
Cohesion: 0.19
Nodes (11): detect_sfp(), Swing Failure Pattern (SFP) — Tom Dante / TopStepTrader style.  An SFP forms whe, Returns (idx, side, level, wick_size).     side: 1 = bearish SFP (wick above res, _bool_mask(), build_signals(), Confluence strategy — bias + sweep + trigger.  Causality contract:     signal[i], Forward-fill structural trend (+1 bull, -1 bear, 0 unknown).      ev_type from b, Return length-n boolean mask: True at indices where side == want_side. (+3 more)

### Community 20 - "Community 20"
Cohesion: 0.47
Nodes (10): _estimate_run(), _load_json(), main(), parse_args(), Batch tick-level validation for saved optimizer runs.  This reads a leaderboard, run(), _safe_bool(), _safe_float() (+2 more)

### Community 21 - "Community 21"
Cohesion: 0.18
Nodes (9): bos_choch(), detect_swings(), displacement(), Market structure: vectorized swing detection, BOS, CHoCH.  Tick-precision: when, Detect displacement candles (body > mult * ATR)., Fractal swing detection.      Returns (swing_high_idx, swing_high_px, swing_low_, Compute BOS and CHoCH events without look-ahead.      A swing at index j is only, +1 to bull/bear when BOS/CHoCH gives a bullish/bearish bias. (+1 more)

### Community 22 - "Community 22"
Cohesion: 0.24
Nodes (9): detect_springs_upthrusts(), detect_trading_range(), Wyckoff phases: accumulation, distribution, spring, upthrust.  Approximation — a, Mark indices where price is in a contraction TR.      Returns boolean mask. A ba, Spring (bull): break TR low intrabar, close back inside.     Upthrust (bear): br, SOS: large bull bar (close - open) on rising volume after spring.     SOW: large, sign_of_strength_weakness(), Wyckoff phase confluence: spring/upthrust + SOS/SOW.      Springs (bull) and upt (+1 more)

### Community 23 - "Community 23"
Cohesion: 0.2
Nodes (9): initial_balance(), Auction Market Theory: POC, VAH/VAL, Initial Balance, balance/imbalance.  Tick-p, Aggregate volume into price bins from OHLCV.      Approximation: distribute cand, True volume profile from tick data., Compute POC, VAH, VAL from a binned profile., Compute IB (first ``ib_minutes`` of session) high/low PER session day.      ts:, value_area(), volume_profile_bars() (+1 more)

### Community 24 - "Community 24"
Cohesion: 0.31
Nodes (8): _equal_with_nans(), Causality tests for algo.strategy.confluence.build_signals.  The contract: signa, Truncating bars at K must not change outputs at indices < K (modulo     the righ, _synth_bars(), test_build_signals_causality_prefix_invariance(), test_build_signals_direction_sanity(), test_build_signals_nan_where_no_signal(), test_build_signals_shape()

### Community 25 - "Community 25"
Cohesion: 0.39
Nodes (7): compute_dealing_range(), DealingRange, in_ote(), is_discount(), is_premium(), Dealing range, premium/discount, OTE — corrected math.  Fixes vs. spec: - Remove, validate_entry_zone()

### Community 26 - "Community 26"
Cohesion: 0.29
Nodes (6): detect_candles(), filter_at_poi(), Candlestick patterns — corrected from spec.  Fixes: - Three White Soldiers / Thr, Keep only patterns whose bar overlaps an active POI zone.      pois: list of (st, Returns flat arrays: (idx, code, signal, strength).     code:        1=bull_engu, candle_scorer()

### Community 27 - "Community 27"
Cohesion: 0.38
Nodes (6): _change_mask(), htf_anchored_vwaps(), Anchored VWAP, session VWAP and dev bands as a pattern layer.  Calendar boundari, Return indices marking the start of each session day.      A "session day" rolls, Anchor at the most recent ISO-week start and calendar-month start.      Uses ``n, session_anchors()

### Community 28 - "Community 28"
Cohesion: 0.33
Nodes (5): _agg(), make_bars(), Tick-to-bar conversion: time, volume, dollar, range, and renko bars.  Tick preci, Generic bar aggregator.      mode:         0 = volume bars (threshold = base qty, ticks columns: ts_ms, price, qty, is_buyer_maker.

### Community 29 - "Community 29"
Cohesion: 0.33
Nodes (4): Strategy sub-package — orchestrates indicators + structure + patterns into causa, _build_scorer_signals(), End-to-end runner: build causal signals -> run backtest engine.  Two strategy "k, Adapter: instantiate a ConfluenceScorer from flat kwargs.

### Community 30 - "Community 30"
Cohesion: 0.5
Nodes (3): get_top_5(), extract_best_strategies(), main()

### Community 31 - "Community 31"
Cohesion: 0.6
Nodes (4): calculate_rank_score(), extract_best_strategies(), main(), Replicates the scoring logic from auto_algo_finder.py

### Community 32 - "Community 32"
Cohesion: 0.67
Nodes (3): download_workspace_dir(), main(), Download all files under the `algo/` directory from a Snowflake Workspace to yo

### Community 33 - "Community 33"
Cohesion: 0.5
Nodes (3): code:powershell (.\.venv\Scripts\python.exe -m algo_download.optimize --help), code:bash (powershell -NoProfile -ExecutionPolicy Bypass -Command ".\\.), OpenClaude Operating Rules

## Knowledge Gaps
- **165 isolated node(s):** `Download Binance OHLC candles into the local DuckDB cache.  This script is inten`, `Parse YYYY-MM-DD or ISO datetime as UTC.`, `Run untouched forward-test data with optimizer-selected parameters.  This script`, `Download all files under the `algo/` directory from a Snowflake Workspace to yo`, `Replicates the scoring logic from auto_algo_finder.py` (+160 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_tick_validation()` connect `Community 3` to `Community 10`, `Community 2`, `Community 20`, `Community 29`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Why does `_build_scorer_signals()` connect `Community 29` to `Community 11`, `Community 18`, `Community 3`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **Why does `backtest_strategy()` connect `Community 1` to `Community 0`, `Community 10`, `Community 29`, `Community 6`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `run()` (e.g. with `OptimizeConfig` and `BacktestConfig`) actually correct?**
  _`run()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `run_tick_validation()` (e.g. with `_build_scorer_signals()` and `run_backtest()`) actually correct?**
  _`run_tick_validation()` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `compute_perf()` (e.g. with `main()` and `main()`) actually correct?**
  _`compute_perf()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `optimize()` (e.g. with `main()` and `test_optimizer_smoke_runs()`) actually correct?**
  _`optimize()` has 3 INFERRED edges - model-reasoned connections that need verification._