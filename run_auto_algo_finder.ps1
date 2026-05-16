param(
    [int]$Trials = 100,
    [string]$OutDir = "data_store\auto_finder",
    [ValidateSet("enhanced", "base")]
    [string]$Strategy = "enhanced",
    [string[]]$Seeds = @("11", "42", "101", "137"),
    [string[]]$Targets = @("10", "12"),
    [int]$Top = 20,
    [int]$Jobs = 8,
    [double]$MinForwardWinRate = 0.60,
    [double]$MinHoldoutWinRate = 0.55,
    [double]$MinForwardPf = 1.10,
    [double]$MinHoldoutPf = 1.10,
    [double]$MinForwardExpectancyR = 0.08,
    [double]$MinHoldoutExpectancyR = 0.08,
    [double]$MaxForwardTpm = 12.0,
    [double]$MaxHoldoutTpm = 12.0,
    [double]$MinForwardTpm = 10.0,
    [double]$MinHoldoutTpm = 8.0,
    [double]$MinForwardReturn = 0.60,
    [double]$MinForwardCagr = 0.50,
    [double]$MinTargetMovePct = 2.0,
    [double]$MinAvgTradeMovePct = 2.0,
    [double]$MaxAvgHoldHours = 120.0,
    [double]$MaxMedianHoldHours = 96.0,
    [double]$RiskPerTrade = 0.006,
    [double]$WinRateTarget = 0.60,
    [double]$WinRateWeight = 10.0,
    [double]$MinOptimizerWinRate = 0.55,
    [double]$MaxObjectiveTpm = 12.0,
    [double]$MinObjectiveTpm = 8.0,
    [string]$HardMaxObjectiveTpm = "true",
    [double]$TpmOverWeight = 6.0,
    [double]$RrMin = 0.7,
    [double]$RrMax = 2.5,
    [switch]$OptimizeExits,
    [switch]$OptimizeHoldTime = $true
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$seedValues = @($Seeds | ForEach-Object { "$_".Split(",") } | Where-Object { $_ -ne "" } | ForEach-Object { "$_".Trim() })
$targetValues = @($Targets | ForEach-Object { "$_".Split(",") } | Where-Object { $_ -ne "" } | ForEach-Object { "$_".Trim() })

$finderArgs = @(
    ".\tools\auto_algo_finder.py",
    "--patterns",
    "data_store\BTCUSDT_1h_backtest_5y.parquet",
    "data_store\BTCUSDT_4h_backtest_5y.parquet",
    "--trials",
    "$Trials",
    "--strategy",
    "$Strategy",
    "--jobs",
    "$Jobs",
    "--targets"
)
$finderArgs += $targetValues
$finderArgs += "--seeds"
$finderArgs += $seedValues
$finderArgs += @(
    "--holdout",
    "0.30",
    "--out-dir",
    "$OutDir",
    "--top",
    "$Top",
    "--min-forward-win-rate",
    "$MinForwardWinRate",
    "--min-holdout-win-rate",
    "$MinHoldoutWinRate",
    "--min-forward-pf",
    "$MinForwardPf",
    "--min-holdout-pf",
    "$MinHoldoutPf",
    "--min-forward-expectancy-r",
    "$MinForwardExpectancyR",
    "--min-holdout-expectancy-r",
    "$MinHoldoutExpectancyR",
    "--max-forward-tpm",
    "$MaxForwardTpm",
    "--max-holdout-tpm",
    "$MaxHoldoutTpm",
    "--min-forward-tpm",
    "$MinForwardTpm",
    "--min-holdout-tpm",
    "$MinHoldoutTpm",
    "--min-forward-return",
    "$MinForwardReturn",
    "--min-forward-cagr",
    "$MinForwardCagr",
    "--min-target-move-pct",
    "$MinTargetMovePct",
    "--min-avg-trade-move-pct",
    "$MinAvgTradeMovePct",
    "--max-avg-hold-hours",
    "$MaxAvgHoldHours",
    "--max-median-hold-hours",
    "$MaxMedianHoldHours",
    "--monte-carlo-runs",
    "1000",
    "--min-mc-profit-prob",
    "0.80",
    "--min-mc-p05-return",
    "0.0",
    "--max-mc-p95-drawdown",
    "0.30",
    "--risk-per-trade",
    "$RiskPerTrade",
    "--win-rate-target",
    "$WinRateTarget",
    "--win-rate-weight",
    "$WinRateWeight",
    "--min-win-rate",
    "$MinOptimizerWinRate",
    "--min-objective-win-rate",
    "$MinOptimizerWinRate",
    "--min-objective-tpm",
    "$MinObjectiveTpm",
    "--max-objective-tpm",
    "$MaxObjectiveTpm",
    "--tpm-over-weight",
    "$TpmOverWeight",
    "--rr-min",
    "$RrMin",
    "--rr-max",
    "$RrMax"
)

if ($OptimizeExits) {
    $finderArgs += "--optimize-exits"
}
$hardMaxObjectiveTpmEnabled = "$HardMaxObjectiveTpm".Trim().ToLowerInvariant() -notin @("0", "false", "`$false", "no", "off")
if ($hardMaxObjectiveTpmEnabled) {
    $finderArgs += "--hard-max-objective-tpm"
}
if ($OptimizeHoldTime) {
    $finderArgs += @("--optimize-hold-time", "--min-exit-hold-hours", "24", "--max-exit-hold-hours", "96")
}

& ".\.venv\Scripts\python.exe" @finderArgs
