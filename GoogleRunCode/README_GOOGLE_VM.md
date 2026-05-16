# GoogleRunCode

This folder is the smaller VM package for the current BTCUSDT strategy research.

## Did we get a good strategy?

Yes, for research/paper trading. The strongest result is not one Pine script. It is a confluence portfolio:

- Source: `data_store/focused_robust_7y_candidates_v3/portfolio_confluence_top8_all_agree2`
- Rule: take a trade only when at least 2 selected strategies agree within about 4 hours
- Backtest/tick-tested result: `65.9%` return, `2.069` PF, `4.4%` max drawdown at `0.3%` risk/trade

This is still not live-trading approval. Treat it as a paper-trading candidate first.

## What this folder contains

- `algo/` and `algo_download/`: local strategy/backtest package
- `tools/`: optimizer, tick validation, portfolio builder
- `tick_forward_test.py`, `tick_cache.py`, `forward_test.py`: validation runners
- `data_store/*_7y.parquet`: 7-year BTCUSDT candle data
- `data_store/focused_robust_7y_candidates_v3`: selected strategy JSONs and validation results
- `scripts/live_trade_email_alert.py`: checks live Binance candles and emails alerts with entry, stop, take profit, and reason

The huge `data_store/tick_cache.duckdb` file is intentionally not copied. It is about 31 GB. Rebuild it on the VM only if you need full tick validation again.

## Google VM recommendation

Use Ubuntu, not Windows, unless you strongly prefer Windows.

Recommended VM:

- Machine: 8 vCPU, 32 GB RAM
- Disk: 150-250 GB balanced persistent disk
- OS: Ubuntu 22.04 or 24.04 LTS
- Network: no public web server needed

## Secure VM setup

1. Use SSH keys only. Do not enable password login.
2. Do not put Binance exchange API keys on this VM for now. The email alert script only reads public Binance candles.
3. Allow only SSH port `22` from your own IP in Google Cloud firewall.
4. Keep mail password in `.env`, not inside code.
5. Use a Gmail app password, not your normal Gmail password.
6. After creating `.env`, run: `chmod 600 .env`

## Setup on Ubuntu

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip unzip
cd ~/GoogleRunCode
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-google-vm.txt
cp .env.example .env
nano .env
chmod 600 .env
```

## Test imports

```bash
source .venv/bin/activate
python -c "from tools.enhanced_strategy import build_enhanced_signals; print('ok')"
```

## Send a one-time live scan email

```bash
source .venv/bin/activate
python scripts/live_trade_email_alert.py --once
```

## Test without sending email

```bash
source .venv/bin/activate
python scripts/live_trade_email_alert.py --once --dry-run
```

## Run continuously

```bash
source .venv/bin/activate
python scripts/live_trade_email_alert.py
```

It checks every `ALERT_INTERVAL_SECONDS` seconds. Default is 5 minutes.

## Paper-trade journal

The alert runner now records paper trades automatically when a confluence alert appears.

Files written:

- `data_store/paper_trade_ledger.csv` - append-only open/close event log
- `data_store/paper_open_trades.csv` - current open paper trades
- `data_store/paper_closed_trades.csv` - closed paper trades with PnL
- `data_store/paper_trades_state.json` - internal state used to resume after restart, including opposite-signal warnings

Paper settings in `.env`:

```env
PAPER_TRADING_ENABLED=1
PAPER_INITIAL_BALANCE=200
PAPER_RISK_PER_TRADE=0.003
PAPER_MONITOR_KLINE_LIMIT=1000
PAPER_SKIP_IF_SAME_SIDE_OPEN=1
PAPER_OPPOSITE_SIGNAL_ACTION=reverse
PAPER_REVERSAL_MIN_AGREE=2
PAPER_REVERSAL_MIN_TICK_PF=1.2
PAPER_REVERSAL_MIN_EXP_R=0.0
PAPER_REVERSAL_MIN_SCORE_RATIO=0.9
```

Meaning:

- `PAPER_INITIAL_BALANCE=200`: pretend account size is 200 USDT.
- `PAPER_RISK_PER_TRADE=0.003`: risk 0.3% of that pretend account per trade.
- The script calculates position size from entry to stop loss.
- It checks recent 1-minute candles to close the paper trade if SL or TP is touched.
- Opposite signals are treated as conflict/reversal events, not hedges. With `PAPER_OPPOSITE_SIGNAL_ACTION=reverse`, the script closes an open same-symbol opposite paper trade and opens the new one only when the new setup passes the reversal gates above. If it does not pass, it records a warning and does not open the opposite trade.
- Use `PAPER_OPPOSITE_SIGNAL_ACTION=warn` to only journal and email opposite-signal warnings without reversing.

This is better than only sending alerts because you get a real paper PnL log. It is still not real exchange execution.

If email is not working yet, run paper tracking without email:

```bash
source .venv/bin/activate
python scripts/live_trade_email_alert.py --no-email
```

To test only email:

```bash
source .venv/bin/activate
python scripts/live_trade_email_alert.py --test-email
```

## Rebuild a portfolio report with the current code

```bash
source .venv/bin/activate
python tools/build_trade_portfolio.py \
  --tick-dirs data_store/focused_robust_7y_candidates_v3/tick_validation_all \
  --out-dir data_store/focused_robust_7y_candidates_v3/portfolio_confluence_top8_all_agree2_rerun \
  --top-strategies 8 \
  --risk-per-trade 0.003 \
  --max-open-trades 6 \
  --min-agree 2 \
  --agree-window-hours 4
```

The archived best report is already included at:

`data_store/focused_robust_7y_candidates_v3/portfolio_confluence_top8_all_agree2/portfolio_report.json`

Use that archived report as the current reference result unless we intentionally re-run and replace the portfolio selection logic.

## Plain English

Python is the engine. Pine is only the display/alert layer later. Run the heavy work on the Google VM, and let this script email you when the selected strategies agree on a trade.
