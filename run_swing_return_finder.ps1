param(
    [int]$Trials = 1000,
    [string]$OutDir = "data_store\swing_return_v1_prod"
)

.\.venv\Scripts\python.exe .\tools\auto_algo_finder.py `
  --patterns "data_store\BTCUSDT_1h_backtest_7y.parquet" "data_store\BTCUSDT_4h_backtest_7y.parquet" `
  --out-dir $OutDir `
  --strategy enhanced --trials $Trials --targets 10 12 16 `
  --seeds 11 42 101 137 202 --holdout 0.30 --top 20 `
  --min-win-rate 0.30 --min-holdout-win-rate 0.30 --min-forward-win-rate 0.30 `
  --win-rate-target 0.45 --win-rate-weight 0.5 `
  --pf-over-weight 6.0 --expectancy-r-weight 35.0 `
  --min-holdout-pf 1.15 --min-forward-pf 1.10 `
  --min-holdout-expectancy-r 0.10 --min-forward-expectancy-r 0.08 `
  --min-objective-tpm 10 --min-holdout-tpm 10 --min-forward-tpm 10 `
  --max-objective-tpm 45 --max-forward-tpm 45 --max-holdout-tpm 45 `
  --min-forward-return 0.60 --min-forward-cagr 0.50 `
  --min-target-move-pct 4.0 --min-avg-trade-move-pct 3.0 --max-avg-hold-hours 96 --max-median-hold-hours 72 `
  --rr-min 2.0 --rr-max 4.5 --optimize-exits --exit-profile swing --optimize-hold-time `
  --min-exit-hold-hours 24 --max-exit-hold-hours 96 `
  --risk-per-trade 0.006 `
  --monte-carlo-runs 1000 --min-mc-profit-prob 0.80 --min-mc-p05-return 0.0 --max-mc-p95-drawdown 0.30 `
  --forward-return-weight 0.75 --forward-cagr-weight 0.75 --trade-move-weight 1.5 `
  --hold-time-weight 1.0 --monte-carlo-weight 1.0 --monte-carlo-dd-weight 1.5 --tpm-under-weight 4.0 `
  --soft-dd-cap 0.16 --max-holdout-dd 0.22 --max-forward-dd 0.22 `
  --max-train-holdout-score-gap 999.0 --max-train-holdout-pf-ratio 2.5
