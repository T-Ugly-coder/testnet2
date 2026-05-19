from __future__ import annotations

import argparse
import csv
import json
import os
import smtplib
import ssl
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "data_store" / "live_alert_state.json"
PAPER_STATE_PATH = ROOT / "data_store" / "paper_trades_state.json"
PAPER_LEDGER_PATH = ROOT / "data_store" / "paper_trade_ledger.csv"
PAPER_OPEN_PATH = ROOT / "data_store" / "paper_open_trades.csv"
PAPER_CLOSED_PATH = ROOT / "data_store" / "paper_closed_trades.csv"
DEFAULT_SELECTED = ROOT / "data_store" / "focused_robust_7y_candidates_v3" / "portfolio_confluence_top8_all_agree2" / "selected_strategies.csv"
DEFAULT_CANDIDATE_DIR = ROOT / "data_store" / "focused_robust_7y_candidates_v3"

import sys

sys.path.insert(0, str(ROOT))

from tools.enhanced_strategy import build_enhanced_signals  # noqa: E402


@dataclass
class Candidate:
    run_id: str
    interval: str
    direction: int
    candle_time_ms: int
    signal_close: float
    entry: float
    entry_source: str
    stop: float
    take_profit: float
    rr: float
    bull_score: float
    bear_score: float
    source_pf: float
    tick_pf: float
    tick_exp_r: float


@dataclass
class Alert:
    key: str
    subject: str
    body: str
    symbol: str
    direction: int
    primary: Candidate
    candidates: list[Candidate]


@dataclass
class PaperTradeDecision:
    opened: dict[str, Any] | None = None
    closed: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


def infer_interval(run_id: str) -> str:
    if "_tf-4h_" in run_id:
        return "4h"
    return "1h"


def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    params = urllib.parse.urlencode({"symbol": symbol, "interval": interval, "limit": limit})
    url = f"https://api.binance.com/api/v3/klines?{params}"
    with urllib.request.urlopen(url, timeout=20) as response:
        rows = json.loads(response.read().decode("utf-8"))

    records = []
    for row in rows:
        records.append(
            {
                "ts_ms": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "close_time_ms": int(row[6]),
            }
        )
    bars = pd.DataFrame(records)
    bars.index = pd.to_datetime(bars["ts_ms"], unit="ms", utc=True)
    bars.index.name = None
    return bars


def fetch_latest_price(symbol: str) -> float:
    params = urllib.parse.urlencode({"symbol": symbol})
    url = f"https://api.binance.com/api/v3/ticker/price?{params}"
    with urllib.request.urlopen(url, timeout=20) as response:
        data = json.loads(response.read().decode("utf-8"))
    return float(data["price"])


def load_selected(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_params(run_id: str, candidate_dir: Path) -> dict[str, Any]:
    data = json.loads((candidate_dir / f"{run_id}.json").read_text())
    params = data.get("best_params")
    if not isinstance(params, dict):
        raise ValueError(f"{run_id}.json has no best_params")
    return params


def latest_closed_index(bars: pd.DataFrame) -> int:
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    closed = bars.index[bars["close_time_ms"] < now_ms]
    if len(closed) == 0:
        raise ValueError("no closed candles returned")
    return int(bars.index.get_loc(closed[-1]))


def signal_candidate(row: dict[str, str], bars: pd.DataFrame, candidate_dir: Path) -> Candidate | None:
    run_id = row["run_id"]
    params = load_params(run_id, candidate_dir)
    signals = build_enhanced_signals(bars, **params)
    i = latest_closed_index(bars)

    signal = int(np.asarray(signals["signal"])[i])
    if signal == 0:
        return None

    signal_close = float(bars["close"].iloc[i])
    stop = float(np.asarray(signals["sl"], dtype=float)[i])
    tp_arr = np.asarray(signals.get("tp", np.full(len(bars), np.nan)), dtype=float)
    rr_arr = np.asarray(signals.get("rr", np.full(len(bars), np.nan)), dtype=float)
    rr = float(rr_arr[i]) if np.isfinite(rr_arr[i]) else float(params.get("rr", 0.0))
    next_i = i + 1
    if next_i < len(bars):
        entry = float(bars["open"].iloc[next_i])
        entry_source = "next_candle_open"
    else:
        entry = signal_close
        entry_source = "signal_close_fallback"

    if not np.isfinite(stop):
        return None
    if rr <= 0:
        take_profit = float(tp_arr[i]) if np.isfinite(tp_arr[i]) else float("nan")
    else:
        risk = abs(entry - stop)
        take_profit = entry + signal * rr * risk
    if not np.isfinite(take_profit):
        return None

    if signal == 1 and not stop < entry < take_profit:
        return None
    if signal == -1 and not take_profit < entry < stop:
        return None

    bull = np.asarray(signals.get("bull_score", np.zeros(len(bars))), dtype=float)
    bear = np.asarray(signals.get("bear_score", np.zeros(len(bars))), dtype=float)

    return Candidate(
        run_id=run_id,
        interval=infer_interval(run_id),
        direction=signal,
        candle_time_ms=int(bars["ts_ms"].iloc[i]),
        signal_close=signal_close,
        entry=entry,
        entry_source=entry_source,
        stop=stop,
        take_profit=take_profit,
        rr=rr,
        bull_score=float(bull[i]),
        bear_score=float(bear[i]),
        source_pf=float(row.get("source_forward_pf") or 0.0),
        tick_pf=float(row.get("tick_profit_factor") or 0.0),
        tick_exp_r=float(row.get("tick_expectancy_R") or 0.0),
    )


def load_state() -> set[str]:
    if not STATE_PATH.exists():
        return set()
    data = json.loads(STATE_PATH.read_text())
    return set(data.get("sent_keys", []))


def save_state(sent: set[str]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps({"sent_keys": sorted(sent)[-500:]}, indent=2))


def load_paper_state() -> dict[str, Any]:
    if not PAPER_STATE_PATH.exists():
        return {"open_trades": [], "closed_trades": []}
    return json.loads(PAPER_STATE_PATH.read_text())


def save_paper_state(state: dict[str, Any]) -> None:
    PAPER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAPER_STATE_PATH.write_text(json.dumps(state, indent=2))
    write_trade_snapshots(state)


def append_ledger(row: dict[str, Any]) -> None:
    PAPER_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "event",
        "trade_id",
        "symbol",
        "side",
        "status",
        "time_utc",
        "entry",
        "entry_source",
        "reference_entry",
        "entry_drift_bps",
        "stop",
        "take_profit",
        "exit_price",
        "exit_reason",
        "qty",
        "risk_amount",
        "pnl",
        "pnl_R",
        "agreement_count",
        "primary_run_id",
        "run_ids",
    ]
    exists = PAPER_LEDGER_PATH.exists()
    with PAPER_LEDGER_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_trade_snapshots(state: dict[str, Any]) -> None:
    PAPER_OPEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_csv(PAPER_OPEN_PATH, state.get("open_trades", []))
    write_csv(PAPER_CLOSED_PATH, state.get("closed_trades", []))


def utc_now_ms() -> int:
    return int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)


def ms_to_iso(ms: int) -> str:
    return pd.to_datetime(ms, unit="ms", utc=True).isoformat()


def send_mail(subject: str, body: str) -> None:
    host = os.environ["SMTP_HOST"]
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.environ["SMTP_USER"].strip()
    password = "".join(os.environ["SMTP_PASS"].split())
    mail_from = os.getenv("MAIL_FROM", user)
    mail_to = os.environ["MAIL_TO"]

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = mail_from
    msg["To"] = mail_to
    msg.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.starttls(context=context)
        smtp.login(user, password)
        smtp.send_message(msg)


def format_alert(symbol: str, direction: int, candidates: list[Candidate]) -> tuple[str, str, Candidate]:
    side = "LONG" if direction == 1 else "SHORT"
    primary = sorted(candidates, key=lambda c: (c.tick_pf, c.tick_exp_r), reverse=True)[0]
    names = ", ".join(c.run_id for c in candidates)
    candle_time = pd.to_datetime(primary.candle_time_ms, unit="ms", utc=True).isoformat()
    live_price_text = "unavailable"
    entry_drift_text = "unavailable"
    paper_action_text = "Paper entry uses live ticker when the trade is recorded."
    try:
        live_price = fetch_latest_price(symbol)
        entry_drift_bps = abs(live_price - primary.entry) / max(abs(primary.entry), 1e-12) * 10_000.0
        max_entry_drift_bps = env_float("PAPER_MAX_ENTRY_DRIFT_BPS", 15.0)
        live_price_text = f"{live_price:.2f}"
        entry_drift_text = f"{entry_drift_bps:.2f} bps"
        if entry_drift_bps > max_entry_drift_bps:
            paper_action_text = (
                f"Paper trade will be skipped if drift is still above {max_entry_drift_bps:.2f} bps "
                "when recording."
            )
        else:
            paper_action_text = "Paper entry will use live ticker and TP will be recalculated from that fill."
    except Exception as exc:
        paper_action_text = f"Paper entry uses live ticker when recording, but live ticker check failed for email: {exc}"

    subject = f"{symbol} {side} alert: {len(candidates)} strategies agree"
    body = [
        f"Symbol: {symbol}",
        f"Side: {side}",
        f"Agreement count: {len(candidates)}",
        f"Candle time UTC: {candle_time}",
        "",
        "Primary trade levels:",
        f"Signal close: {primary.signal_close:.2f}",
        f"Entry reference: {primary.entry:.2f} ({primary.entry_source})",
        f"Live ticker now: {live_price_text}",
        f"Live/reference drift: {entry_drift_text}",
        f"Stop loss: {primary.stop:.2f}",
        f"Take profit: {primary.take_profit:.2f}",
        f"RR: {primary.rr:.3f}",
        f"Paper handling: {paper_action_text}",
        "",
        "Reason:",
        f"{len(candidates)} selected strategies agree in the same direction. The primary strategy has tick PF {primary.tick_pf:.3f} and tick expectancy R {primary.tick_exp_r:.3f}.",
        f"Bull score: {primary.bull_score:.3f}",
        f"Bear score: {primary.bear_score:.3f}",
        "",
        "Agreeing strategies:",
        names,
        "",
        "Important: this is a paper-trading/research alert, not automatic live-trading approval.",
    ]
    return subject, "\n".join(body), primary


def paper_signal_score(agreement_count: int, tick_pf: float, tick_exp_r: float) -> float:
    return agreement_count * 2.0 + tick_pf + max(tick_exp_r, 0.0)


def adverse_risk_per_unit(direction: int, entry: float, stop: float) -> float:
    if direction == 1:
        return max(0.0, entry - stop)
    return max(0.0, stop - entry)


def maybe_tighten_same_side_stop(
    alert: Alert,
    existing_trade: dict[str, Any],
    market_price: float,
) -> tuple[bool, str]:
    if os.getenv("PAPER_SAME_SIDE_UPGRADE_ENABLED", "1") == "0":
        return False, "same-side stop upgrade disabled"

    direction = int(existing_trade.get("direction") or 0)
    if direction != alert.direction:
        return False, "same-side stop upgrade skipped because direction does not match"

    old_agreement = int(existing_trade.get("agreement_count") or 0)
    old_pf = float(existing_trade.get("primary_tick_pf") or 0.0)
    old_exp_r = float(existing_trade.get("primary_tick_exp_r") or 0.0)
    new_agreement = len(alert.candidates)
    new_pf = float(alert.primary.tick_pf)
    new_exp_r = float(alert.primary.tick_exp_r)
    old_score = paper_signal_score(old_agreement, old_pf, old_exp_r)
    new_score = paper_signal_score(new_agreement, new_pf, new_exp_r)
    min_score_ratio = env_float("PAPER_SAME_SIDE_UPGRADE_MIN_SCORE_RATIO", 1.0)

    if new_score < old_score * min_score_ratio:
        return False, f"same-side signal score {new_score:.3f} is below required {min_score_ratio:.2f}x open score {old_score:.3f}"

    entry = float(existing_trade["entry"])
    old_stop = float(existing_trade["stop"])
    new_stop = float(alert.primary.stop)
    if not np.isfinite(new_stop):
        return False, "same-side signal has invalid stop"

    if direction == 1 and not (old_stop < new_stop < market_price):
        return False, f"same-side long stop {new_stop:.2f} does not tighten safely from {old_stop:.2f} with market {market_price:.2f}"
    if direction == -1 and not (market_price < new_stop < old_stop):
        return False, f"same-side short stop {new_stop:.2f} does not tighten safely from {old_stop:.2f} with market {market_price:.2f}"

    old_risk = adverse_risk_per_unit(direction, entry, old_stop)
    new_risk = adverse_risk_per_unit(direction, entry, new_stop)
    if new_risk >= old_risk:
        return False, f"same-side stop does not reduce open trade risk: old {old_risk:.2f}, new {new_risk:.2f}"

    now_ms = utc_now_ms()
    existing_trade["stop"] = new_stop
    existing_trade["last_stop_update_ms"] = now_ms
    existing_trade["last_stop_update_utc"] = ms_to_iso(now_ms)
    existing_trade["last_stop_update_reason"] = "same_side_stronger_signal"
    existing_trade["last_stop_update_from"] = old_stop
    existing_trade["stop_update_count"] = int(existing_trade.get("stop_update_count") or 0) + 1
    existing_trade["agreement_count"] = max(old_agreement, new_agreement)
    existing_trade["primary_tick_pf"] = max(old_pf, new_pf)
    existing_trade["primary_tick_exp_r"] = max(old_exp_r, new_exp_r)
    existing_trade["management_primary_run_id"] = alert.primary.run_id
    existing_trade["management_signal_key"] = alert.key
    append_ledger(
        {
            "event": "stop_update",
            "trade_id": existing_trade["trade_id"],
            "symbol": existing_trade["symbol"],
            "side": existing_trade["side"],
            "status": "open",
            "time_utc": existing_trade["last_stop_update_utc"],
            "entry": existing_trade["entry"],
            "stop": existing_trade["stop"],
            "take_profit": existing_trade["take_profit"],
            "qty": existing_trade["qty"],
            "risk_amount": existing_trade["risk_amount"],
            "agreement_count": existing_trade["agreement_count"],
            "primary_run_id": existing_trade.get("primary_run_id"),
            "run_ids": existing_trade.get("run_ids"),
            "exit_reason": f"tightened_stop_from_{old_stop:.2f}",
        }
    )
    return True, (
        f"Same-side stronger signal tightened stop from {old_stop:.2f} to {new_stop:.2f}; "
        f"qty and take-profit stayed unchanged."
    )


def reversal_quality_decision(alert: Alert, existing_trade: dict[str, Any]) -> tuple[bool, str]:
    new_agreement = len(alert.candidates)
    new_pf = float(alert.primary.tick_pf)
    new_exp_r = float(alert.primary.tick_exp_r)
    old_agreement = int(existing_trade.get("agreement_count") or 0)
    old_pf = float(existing_trade.get("primary_tick_pf") or 0.0)
    old_exp_r = float(existing_trade.get("primary_tick_exp_r") or 0.0)

    min_agree = env_int("PAPER_REVERSAL_MIN_AGREE", env_int("ALERT_MIN_AGREE", 2))
    min_pf = env_float("PAPER_REVERSAL_MIN_TICK_PF", 1.2)
    min_exp_r = env_float("PAPER_REVERSAL_MIN_EXP_R", 0.0)
    min_score_ratio = env_float("PAPER_REVERSAL_MIN_SCORE_RATIO", 0.9)

    new_score = paper_signal_score(new_agreement, new_pf, new_exp_r)
    old_score = paper_signal_score(old_agreement, old_pf, old_exp_r)

    if new_agreement < min_agree:
        return False, f"opposite signal agreement {new_agreement} is below reversal minimum {min_agree}"
    if new_pf < min_pf:
        return False, f"opposite signal tick PF {new_pf:.3f} is below reversal minimum {min_pf:.3f}"
    if new_exp_r < min_exp_r:
        return False, f"opposite signal expectancy R {new_exp_r:.3f} is below reversal minimum {min_exp_r:.3f}"
    if old_score > 0 and new_score < old_score * min_score_ratio:
        return (
            False,
            f"opposite signal score {new_score:.3f} is below {min_score_ratio:.2f}x open trade score {old_score:.3f}",
        )
    return (
        True,
        f"opposite signal passed reversal gates: agreement {new_agreement}, tick PF {new_pf:.3f}, expectancy R {new_exp_r:.3f}",
    )


def record_opposite_signal_warning(
    alert: Alert,
    existing_trade: dict[str, Any],
    state: dict[str, Any],
    action: str,
    reason: str,
) -> None:
    now_ms = utc_now_ms()
    warnings = state.setdefault("opposite_signal_warnings", [])
    warnings.append(
        {
            "time_ms": now_ms,
            "time_utc": ms_to_iso(now_ms),
            "symbol": alert.symbol,
            "new_side": "LONG" if alert.direction == 1 else "SHORT",
            "open_trade_id": existing_trade.get("trade_id"),
            "open_side": existing_trade.get("side"),
            "action": action,
            "reason": reason,
            "new_agreement_count": len(alert.candidates),
            "new_primary_run_id": alert.primary.run_id,
            "new_primary_tick_pf": alert.primary.tick_pf,
            "new_primary_tick_exp_r": alert.primary.tick_exp_r,
        }
    )
    state["opposite_signal_warnings"] = warnings[-200:]


def latest_reversal_exit_price(symbol: str, fallback: float) -> float:
    try:
        return fetch_latest_price(symbol)
    except Exception as exc:
        print(f"using alert entry as reversal exit price for {symbol}: {exc}")
        return fallback


def open_paper_trade(alert: Alert, state: dict[str, Any]) -> PaperTradeDecision:
    decision = PaperTradeDecision()
    open_trades = state.setdefault("open_trades", [])
    duplicate = any(
        t.get("symbol") == alert.symbol
        and t.get("direction") == alert.direction
        and t.get("opened_candle_time_ms") == alert.primary.candle_time_ms
        for t in open_trades
    )
    if duplicate:
        decision.notes.append("Duplicate paper setup skipped: same symbol, direction, and candle time are already open.")
        return decision

    same_side_trades = [
        t
        for t in open_trades
        if t.get("symbol") == alert.symbol and int(t.get("direction") or 0) == alert.direction
    ]
    skip_same_side = os.getenv("PAPER_SKIP_IF_SAME_SIDE_OPEN", "1") != "0"
    if skip_same_side and same_side_trades:
        market_price = latest_reversal_exit_price(alert.symbol, alert.primary.entry)
        upgraded = 0
        for trade in same_side_trades:
            did_upgrade, reason = maybe_tighten_same_side_stop(alert, trade, market_price)
            if did_upgrade:
                upgraded += 1
            decision.notes.append(reason)
        if upgraded == 0:
            decision.notes.append("No same-side paper trade was changed; entry, qty, and take-profit stayed unchanged.")
        return decision

    opposite_trades = [
        t
        for t in open_trades
        if t.get("symbol") == alert.symbol and int(t.get("direction") or 0) == -alert.direction
    ]
    if opposite_trades:
        action = os.getenv("PAPER_OPPOSITE_SIGNAL_ACTION", "reverse").strip().lower()
        if action not in {"reverse", "warn"}:
            action = "warn"
        should_reverse = action == "reverse"
        reversal_checks: list[tuple[dict[str, Any], bool, str]] = []
        for trade in opposite_trades:
            passed, reason = reversal_quality_decision(alert, trade)
            reversal_checks.append((trade, passed, reason))
            if not passed:
                should_reverse = False

        reverse_reasons = [reason for _, _, reason in reversal_checks]
        if action != "reverse":
            reverse_reasons.insert(0, "PAPER_OPPOSITE_SIGNAL_ACTION is set to warn")
        recorded_action = "reverse" if should_reverse else "warn"
        for trade, _, reason in reversal_checks:
            record_opposite_signal_warning(
                alert,
                trade,
                state,
                recorded_action,
                reason,
            )

        if not should_reverse:
            decision.notes.append(
                "Opposite signal detected while a same-symbol trade is open; no hedge was opened. "
                + " | ".join(reverse_reasons)
            )
            return decision

        exit_price = latest_reversal_exit_price(alert.symbol, alert.primary.entry)
        exit_ms = utc_now_ms()
        closed_state = state.setdefault("closed_trades", [])
        for trade in opposite_trades:
            closed_trade = close_paper_trade(trade, exit_price, "opposite_signal_reversal", exit_ms)
            closed_trade["reversal_signal_key"] = alert.key
            closed_trade["reversal_primary_run_id"] = alert.primary.run_id
            decision.closed.append(closed_trade)
            closed_state.append(closed_trade)
        opposite_ids = {t.get("trade_id") for t in opposite_trades}
        open_trades[:] = [t for t in open_trades if t.get("trade_id") not in opposite_ids]
        decision.notes.append(
            f"Opposite signal passed reversal gates; closed {len(opposite_trades)} open opposite trade(s) "
            f"at {exit_price:.2f} before opening the new setup."
        )

    initial_balance = env_float("PAPER_INITIAL_BALANCE", 10000.0)
    risk_per_trade = env_float("PAPER_RISK_PER_TRADE", 0.003)
    risk_amount = initial_balance * risk_per_trade
    reference_entry = float(alert.primary.entry)
    actual_entry = latest_reversal_exit_price(alert.symbol, reference_entry)
    entry_drift_bps = abs(actual_entry - reference_entry) / max(abs(reference_entry), 1e-12) * 10_000.0
    max_entry_drift_bps = env_float("PAPER_MAX_ENTRY_DRIFT_BPS", 15.0)
    if entry_drift_bps > max_entry_drift_bps:
        decision.notes.append(
            f"Paper setup skipped because live entry drift {entry_drift_bps:.2f} bps exceeds "
            f"PAPER_MAX_ENTRY_DRIFT_BPS={max_entry_drift_bps:.2f}. "
            f"Reference {reference_entry:.2f}, live {actual_entry:.2f}."
        )
        return decision

    stop = float(alert.primary.stop)
    if not np.isfinite(stop):
        decision.notes.append("Paper setup skipped because stop is invalid.")
        return decision
    if alert.direction == 1 and not stop < actual_entry:
        decision.notes.append(
            f"Paper setup skipped because live long entry {actual_entry:.2f} is not above stop {stop:.2f}."
        )
        return decision
    if alert.direction == -1 and not actual_entry < stop:
        decision.notes.append(
            f"Paper setup skipped because live short entry {actual_entry:.2f} is not below stop {stop:.2f}."
        )
        return decision

    risk_per_unit = abs(actual_entry - stop)
    if risk_per_unit <= 0:
        decision.notes.append("Paper setup skipped because entry and stop create zero risk per unit.")
        return decision
    take_profit = actual_entry + alert.direction * alert.primary.rr * risk_per_unit

    qty = risk_amount / risk_per_unit
    opened_ms = utc_now_ms()
    trade_id = f"{alert.symbol}-{alert.primary.candle_time_ms}-{alert.direction}"
    run_ids = [c.run_id for c in alert.candidates]
    trade = {
        "trade_id": trade_id,
        "symbol": alert.symbol,
        "side": "LONG" if alert.direction == 1 else "SHORT",
        "direction": alert.direction,
        "status": "open",
        "opened_at_ms": opened_ms,
        "opened_at_utc": ms_to_iso(opened_ms),
        "opened_candle_time_ms": alert.primary.candle_time_ms,
        "opened_candle_time_utc": ms_to_iso(alert.primary.candle_time_ms),
        "signal_close": alert.primary.signal_close,
        "reference_entry": reference_entry,
        "entry": actual_entry,
        "entry_source": "live_ticker",
        "entry_drift_bps": entry_drift_bps,
        "stop": stop,
        "take_profit": take_profit,
        "rr": alert.primary.rr,
        "qty": qty,
        "risk_amount": risk_amount,
        "risk_per_trade": risk_per_trade,
        "agreement_count": len(alert.candidates),
        "primary_run_id": alert.primary.run_id,
        "primary_tick_pf": alert.primary.tick_pf,
        "primary_tick_exp_r": alert.primary.tick_exp_r,
        "run_ids": "|".join(run_ids),
        "last_checked_ms": opened_ms,
        "last_price": alert.primary.entry,
        "unrealized_pnl": 0.0,
        "unrealized_pnl_R": 0.0,
    }
    decision.notes.append(
        f"Paper trade opened at live ticker price {actual_entry:.2f}, not the candle reference {reference_entry:.2f}. "
        f"Entry drift was {entry_drift_bps:.2f} bps; TP was recalculated from the live entry."
    )
    open_trades.append(trade)
    append_ledger(
        {
            "event": "open",
            "trade_id": trade_id,
            "symbol": alert.symbol,
            "side": trade["side"],
            "status": "open",
            "time_utc": trade["opened_at_utc"],
            "entry": trade["entry"],
            "entry_source": trade["entry_source"],
            "reference_entry": trade["reference_entry"],
            "entry_drift_bps": trade["entry_drift_bps"],
            "stop": trade["stop"],
            "take_profit": trade["take_profit"],
            "qty": trade["qty"],
            "risk_amount": trade["risk_amount"],
            "agreement_count": trade["agreement_count"],
            "primary_run_id": trade["primary_run_id"],
            "run_ids": trade["run_ids"],
        }
    )
    decision.opened = trade
    return decision


def close_paper_trade(
    trade: dict[str, Any],
    exit_price: float,
    exit_reason: str,
    exit_ms: int,
    *,
    write_ledger: bool = True,
) -> dict[str, Any]:
    pnl = (exit_price - float(trade["entry"])) * float(trade["qty"]) * int(trade["direction"])
    risk_amount = float(trade["risk_amount"])
    closed = dict(trade)
    closed.update(
        {
            "status": "closed",
            "closed_at_ms": exit_ms,
            "closed_at_utc": ms_to_iso(exit_ms),
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "pnl": pnl,
            "pnl_R": pnl / risk_amount if risk_amount else 0.0,
        }
    )
    if write_ledger:
        append_ledger(
            {
                "event": "close",
                "trade_id": closed["trade_id"],
                "symbol": closed["symbol"],
                "side": closed["side"],
                "status": "closed",
                "time_utc": closed["closed_at_utc"],
                "entry": closed["entry"],
                "stop": closed["stop"],
                "take_profit": closed["take_profit"],
                "exit_price": closed["exit_price"],
                "exit_reason": closed["exit_reason"],
                "qty": closed["qty"],
                "risk_amount": closed["risk_amount"],
                "pnl": closed["pnl"],
                "pnl_R": closed["pnl_R"],
                "agreement_count": closed["agreement_count"],
                "primary_run_id": closed["primary_run_id"],
                "run_ids": closed["run_ids"],
            }
        )
    return closed


def monitor_paper_trades(state: dict[str, Any], dry_run: bool) -> list[tuple[str, str]]:
    open_trades = state.get("open_trades", [])
    if not open_trades:
        return []

    closed_messages: list[tuple[str, str]] = []
    remaining: list[dict[str, Any]] = []
    closed = state.setdefault("closed_trades", [])
    minute_bars_by_symbol: dict[str, pd.DataFrame] = {}

    for trade in open_trades:
        symbol = str(trade["symbol"])
        if symbol not in minute_bars_by_symbol:
            minute_bars_by_symbol[symbol] = fetch_klines(symbol, "1m", env_int("PAPER_MONITOR_KLINE_LIMIT", 1000))
        minute_bars = minute_bars_by_symbol[symbol]
        direction = int(trade["direction"])
        stop = float(trade["stop"])
        take_profit = float(trade["take_profit"])
        last_checked_ms = int(trade.get("last_checked_ms") or trade.get("opened_at_ms") or 0)
        check_from_ms = max(0, last_checked_ms - 60_000)

        exit_price = None
        exit_reason = ""
        exit_ms = utc_now_ms()
        recent = minute_bars[minute_bars["close_time_ms"] >= check_from_ms].sort_values("ts_ms")
        for row in recent.itertuples(index=False):
            high = float(row.high)
            low = float(row.low)
            if direction == 1:
                if low <= stop:
                    exit_price = stop
                    exit_reason = "sl"
                elif high >= take_profit:
                    exit_price = take_profit
                    exit_reason = "tp"
            else:
                if high >= stop:
                    exit_price = stop
                    exit_reason = "sl"
                elif low <= take_profit:
                    exit_price = take_profit
                    exit_reason = "tp"
            if exit_price is not None:
                exit_ms = int(row.close_time_ms)
                break

        latest_price = float(minute_bars["close"].iloc[-1])

        if exit_price is None:
            pnl = (latest_price - float(trade["entry"])) * float(trade["qty"]) * direction
            risk_amount = float(trade["risk_amount"])
            trade["last_checked_ms"] = utc_now_ms()
            trade["last_price"] = latest_price
            trade["unrealized_pnl"] = pnl
            trade["unrealized_pnl_R"] = pnl / risk_amount if risk_amount else 0.0
            remaining.append(trade)
            continue

        closed_trade = close_paper_trade(trade, exit_price, exit_reason, exit_ms, write_ledger=not dry_run)
        closed.append(closed_trade)
        subject = f"{symbol} paper trade closed: {closed_trade['side']} {exit_reason.upper()}"
        body = "\n".join(
            [
                f"Symbol: {symbol}",
                f"Side: {closed_trade['side']}",
                f"Exit reason: {exit_reason}",
                f"Entry: {float(closed_trade['entry']):.2f}",
                f"Exit: {float(closed_trade['exit_price']):.2f}",
                f"Stop loss: {float(closed_trade['stop']):.2f}",
                f"Take profit: {float(closed_trade['take_profit']):.2f}",
                f"PnL: {float(closed_trade['pnl']):.2f}",
                f"PnL R: {float(closed_trade['pnl_R']):.3f}",
                f"Primary strategy: {closed_trade['primary_run_id']}",
            ]
        )
        closed_messages.append((subject, body))

    if not dry_run:
        state["open_trades"] = remaining
        save_paper_state(state)
    return closed_messages


def scan_once() -> list[Alert]:
    symbol = os.getenv("ALERT_SYMBOL", "BTCUSDT")
    min_agree = env_int("ALERT_MIN_AGREE", 2)
    limit = env_int("ALERT_KLINE_LIMIT", 800)
    selected_path = Path(os.getenv("ALERT_SELECTED_CSV", str(DEFAULT_SELECTED)))
    candidate_dir = Path(os.getenv("ALERT_CANDIDATE_DIR", str(DEFAULT_CANDIDATE_DIR)))

    rows = load_selected(selected_path)
    bars_by_interval = {
        interval: fetch_klines(symbol, interval, limit)
        for interval in sorted({infer_interval(row["run_id"]) for row in rows})
    }

    candidates: list[Candidate] = []
    for row in rows:
        interval = infer_interval(row["run_id"])
        try:
            candidate = signal_candidate(row, bars_by_interval[interval], candidate_dir)
        except Exception as exc:
            print(f"skip {row.get('run_id')}: {exc}")
            continue
        if candidate:
            candidates.append(candidate)

    alerts = []
    for direction in (1, -1):
        same_side = [c for c in candidates if c.direction == direction]
        if len(same_side) < min_agree:
            continue
        key_parts = [str(direction), str(max(c.candle_time_ms for c in same_side))]
        key_parts.extend(sorted(c.run_id for c in same_side))
        key = "|".join(key_parts)
        subject, body, primary = format_alert(symbol, direction, same_side)
        alerts.append(Alert(key, subject, body, symbol, direction, primary, same_side))
    return alerts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one scan then exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print alerts instead of sending email.")
    parser.add_argument("--no-email", action="store_true", help="Record paper trades but do not send emails.")
    parser.add_argument("--test-email", action="store_true", help="Send a simple test email and exit.")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    if args.test_email:
        symbol = os.getenv("ALERT_SYMBOL", "BTCUSDT")
        send_mail(
            f"{symbol} alert system test",
            "This is a test email from GoogleRunCode. If you received this, SMTP mail is working.",
        )
        print("test email sent")
        return

    interval_seconds = env_int("ALERT_INTERVAL_SECONDS", 300)
    sent = load_state()
    paper_state = load_paper_state()
    paper_enabled = os.getenv("PAPER_TRADING_ENABLED", "1") != "0"

    while True:
        if paper_enabled:
            for subject, body in monitor_paper_trades(paper_state, args.dry_run):
                if args.dry_run:
                    print(f"DRY RUN CLOSE: {subject}")
                    print(body)
                elif args.no_email:
                    print(f"paper close recorded: {subject}")
                else:
                    send_mail(subject, body)
                    print(f"sent: {subject}")

        alerts = scan_once()
        for alert in alerts:
            if alert.key in sent:
                continue
            decision = PaperTradeDecision()
            if paper_enabled and not args.dry_run:
                decision = open_paper_trade(alert, paper_state)
                save_paper_state(paper_state)
            if args.dry_run:
                print(f"DRY RUN: {alert.subject}")
                print(alert.body)
                continue
            body = alert.body
            if decision.notes:
                body += "\n\nPaper-trade decision:\n" + "\n".join(f"- {note}" for note in decision.notes)
            if decision.closed:
                closed_lines = [
                    f"- Closed {trade['side']} {trade['trade_id']} at {float(trade['exit_price']):.2f} "
                    f"({float(trade['pnl_R']):.3f} R)"
                    for trade in decision.closed
                ]
                body += "\n\nReversal close(s):\n" + "\n".join(closed_lines)
            if decision.opened:
                body += "\n\nPaper trade was recorded in data_store/paper_trade_ledger.csv."
            if args.no_email:
                print(f"paper alert recorded: {alert.subject}")
            else:
                send_mail(alert.subject, body)
            sent.add(alert.key)
            save_state(sent)
            if not args.no_email:
                print(f"sent: {alert.subject}")
        if args.once:
            if not alerts:
                print("no confluence trade found")
            break
        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()
