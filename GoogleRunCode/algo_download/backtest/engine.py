"""
Vectorized event-driven backtester.

Design:
- Strategy emits SignalEvents on bar close.
- Engine simulates next-bar fills with slippage + fees.
- No look-ahead: signal at bar i can only be filled at bar i+1 open.
- Tracks per-trade PnL, drawdown, equity curve.
- Numba-compatible inner loop (single symbol).
- Multi-symbol uses parallel.pool.

Tick precision: feed tick-aggregated bars (volume/dollar bars) to reduce
heteroskedasticity and reflect true microstructure timing of fills.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from numba import njit


@dataclass
class TradeResult:
    entry_idx: int
    exit_idx: int
    entry: float
    exit: float
    direction: int
    sl: float
    tp: float
    size: float
    pnl: float
    reason: str


@njit(cache=True, fastmath=True)
def _resolve_exit(d, sl, tp, open_px, high_px, low_px, slip, is_entry_bar):
    """Determine SL/TP resolution for one bar.

    Returns (reason, exit_px):
        reason:  1 = TP hit, -1 = SL hit, 0 = no exit.
    Pessimistic ordering: when both SL and TP are touched intrabar, SL wins.
    For non-entry bars we honor gap fills at the open price (favorable or
    adverse). For entry bars we filled at the open already, so we only
    check the intrabar high/low.
    """
    if d == 1:
        # long
        if not is_entry_bar:
            if open_px >= tp:
                return 1, open_px * (1.0 - slip)
            if open_px <= sl:
                return -1, open_px * (1.0 - slip)
        if low_px <= sl:
            return -1, sl * (1.0 - slip)
        if high_px >= tp:
            return 1, tp * (1.0 - slip)
        return 0, 0.0
    else:
        # short
        if not is_entry_bar:
            if open_px <= tp:
                return 1, open_px * (1.0 + slip)
            if open_px >= sl:
                return -1, open_px * (1.0 + slip)
        if high_px >= sl:
            return -1, sl * (1.0 + slip)
        if low_px <= tp:
            return 1, tp * (1.0 + slip)
        return 0, 0.0


@njit(cache=True, fastmath=True)
def _simulate(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
              close: np.ndarray,
              entry_signal: np.ndarray,  # +1 long, -1 short, 0 nothing
              sl_arr: np.ndarray, tp_arr: np.ndarray,
              rr_arr: np.ndarray,        # per-bar RR; <=0 => use precomputed tp_arr
              risk: float, fee_bps: float, slip_bps: float,
              initial_balance: float, max_concurrent: int):
    n = close.shape[0]
    eq = np.empty(n, dtype=np.float64)
    bal = initial_balance

    # active trades
    open_idx = np.full(max_concurrent, -1, dtype=np.int64)
    open_dir = np.zeros(max_concurrent, dtype=np.int8)
    open_entry = np.zeros(max_concurrent, dtype=np.float64)
    open_sl = np.zeros(max_concurrent, dtype=np.float64)
    open_tp = np.zeros(max_concurrent, dtype=np.float64)
    open_size = np.zeros(max_concurrent, dtype=np.float64)
    open_risk = np.zeros(max_concurrent, dtype=np.float64)  # $-risk at entry

    # results — bounded by n: at most one new trade per bar enters; closes are
    # also at most one per slot per bar. Worst case is n (single-slot churn);
    # we keep n*max_concurrent for safety but allocations are now lazy via the
    # `growing buffer` not adopted in numba — so we keep the conservative cap.
    cap = n * max_concurrent + 1
    res_entry = np.empty(cap, dtype=np.int64)
    res_exit = np.empty(cap, dtype=np.int64)
    res_dir = np.empty(cap, dtype=np.int8)
    res_entry_px = np.empty(cap, dtype=np.float64)
    res_exit_px = np.empty(cap, dtype=np.float64)
    res_pnl = np.empty(cap, dtype=np.float64)
    res_reason = np.empty(cap, dtype=np.int8)  # 1=tp,-1=sl,0=eod
    res_risk = np.empty(cap, dtype=np.float64)
    rcnt = 0

    fee = fee_bps / 10_000.0
    slip = slip_bps / 10_000.0

    for i in range(n):
        occupied_at_open = 0
        for k in range(max_concurrent):
            if open_idx[k] >= 0:
                occupied_at_open += 1

        # 1) Check exits on already-open trades against bar i (with gap fills).
        for k in range(max_concurrent):
            if open_idx[k] < 0:
                continue
            d = open_dir[k]
            ep = open_entry[k]
            sl = open_sl[k]
            tp = open_tp[k]
            sz = open_size[k]
            reason, exit_px = _resolve_exit(d, sl, tp, open_[i], high[i],
                                            low[i], slip, False)
            if reason != 0:
                pnl = (exit_px - ep) * sz * d
                pnl -= (ep + exit_px) * sz * fee
                bal += pnl
                res_entry[rcnt] = open_idx[k]
                res_exit[rcnt] = i
                res_dir[rcnt] = d
                res_entry_px[rcnt] = ep
                res_exit_px[rcnt] = exit_px
                res_pnl[rcnt] = pnl
                res_reason[rcnt] = reason
                res_risk[rcnt] = open_risk[k]
                rcnt += 1
                open_idx[k] = -1

        # 2) Entry from prior-bar signal, filled at this bar's open.
        # Capacity is based on positions already open at the bar open, before
        # any intrabar SL/TP resolution. Otherwise a later-in-bar exit can
        # incorrectly free a slot for an entry at the same candle open.
        new_slot = -1
        if i > 0 and entry_signal[i - 1] != 0 and occupied_at_open < max_concurrent:
            slot = -1
            for k in range(max_concurrent):
                if open_idx[k] < 0:
                    slot = k; break
            if slot >= 0:
                d = entry_signal[i - 1]
                fill = open_[i] * (1.0 + slip) if d == 1 else open_[i] * (1.0 - slip)
                sl = sl_arr[i - 1]
                tp = tp_arr[i - 1]
                # direction sanity: SL/TP must straddle fill on the right sides
                valid = False
                if d == 1 and sl < fill and fill < tp:
                    valid = True
                elif d == -1 and tp < fill and fill < sl:
                    valid = True
                stop_dist = abs(fill - sl)
                if valid and stop_dist > 0:
                    risk_amt = bal * risk
                    sz = risk_amt / stop_dist
                    # If a per-bar RR is supplied (>0), re-derive TP from the
                    # ACTUAL fill so configured reward:risk is honored even on
                    # gap fills where the precomputed `tp` was anchored to
                    # close[i-1]. Otherwise keep the precomputed `tp`.
                    rr_i = rr_arr[i - 1]
                    if rr_i > 0.0:
                        if d == 1:
                            tp = fill + rr_i * stop_dist
                        else:
                            tp = fill - rr_i * stop_dist
                    open_idx[slot] = i
                    open_dir[slot] = d
                    open_entry[slot] = fill
                    open_sl[slot] = sl
                    open_tp[slot] = tp
                    open_size[slot] = sz
                    open_risk[slot] = risk_amt
                    new_slot = slot

        # 3) Same-bar SL/TP check on the just-opened trade (intrabar only).
        if new_slot >= 0:
            d = open_dir[new_slot]
            ep = open_entry[new_slot]
            sl = open_sl[new_slot]
            tp = open_tp[new_slot]
            sz = open_size[new_slot]
            reason, exit_px = _resolve_exit(d, sl, tp, open_[i], high[i],
                                            low[i], slip, True)
            if reason != 0:
                pnl = (exit_px - ep) * sz * d
                pnl -= (ep + exit_px) * sz * fee
                bal += pnl
                res_entry[rcnt] = i
                res_exit[rcnt] = i
                res_dir[rcnt] = d
                res_entry_px[rcnt] = ep
                res_exit_px[rcnt] = exit_px
                res_pnl[rcnt] = pnl
                res_reason[rcnt] = reason
                res_risk[rcnt] = open_risk[new_slot]
                rcnt += 1
                open_idx[new_slot] = -1

        # 4) Mark-to-market equity using close[i].
        m = bal
        for k in range(max_concurrent):
            if open_idx[k] >= 0:
                m += (close[i] - open_entry[k]) * open_size[k] * open_dir[k]
        eq[i] = m

    return (eq,
            res_entry[:rcnt], res_exit[:rcnt], res_dir[:rcnt],
            res_entry_px[:rcnt], res_exit_px[:rcnt],
            res_pnl[:rcnt], res_reason[:rcnt], res_risk[:rcnt])


def run_backtest(bars: pd.DataFrame, signals: dict,
                 risk: float = 0.01, fee_bps: float = 4.0,
                 slip_bps: float = 1.5, initial_balance: float = 100_000.0,
                 max_concurrent: int = 1):
    """
    bars: DataFrame with open/high/low/close.
    signals: dict with 'signal' (+1/-1/0), 'sl' (float per bar), 'tp' (float per bar).
    """
    max_concurrent = int(max_concurrent)
    if max_concurrent < 1:
        raise ValueError("max_concurrent must be >= 1")

    o = bars["open"].to_numpy(np.float64)
    h = bars["high"].to_numpy(np.float64)
    l = bars["low"].to_numpy(np.float64)
    c = bars["close"].to_numpy(np.float64)
    _s = signals["signal"]
    sig = _s.to_numpy(np.int8) if hasattr(_s, "to_numpy") else np.asarray(_s, np.int8)
    sl = np.asarray(signals["sl"], np.float64)
    tp = np.asarray(signals["tp"], np.float64)
    n = sig.shape[0]
    # Optional per-bar RR. Either a scalar (broadcast) or a length-n array.
    if "rr" in signals:
        _rr = signals["rr"]
        if np.isscalar(_rr):
            rr_arr = np.full(n, float(_rr), dtype=np.float64)
        else:
            rr_arr = np.asarray(_rr, np.float64)
            if rr_arr.shape[0] != n:
                raise ValueError("signals['rr'] must be scalar or length-n")
    else:
        rr_arr = np.zeros(n, dtype=np.float64)  # 0 => keep precomputed tp
    out = _simulate(o, h, l, c, sig, sl, tp, rr_arr,
                    risk, fee_bps, slip_bps, initial_balance, max_concurrent)
    eq, e_idx, x_idx, d, e_px, x_px, pnl, reason, risk_amt = out
    trades = pd.DataFrame({
        "entry_idx": e_idx, "exit_idx": x_idx,
        "direction": d, "entry": e_px, "exit": x_px,
        "pnl": pnl, "reason": reason,
        "risk_amt": risk_amt,
    })
    return {"equity": eq, "trades": trades, "final_balance": float(eq[-1] if len(eq) else initial_balance)}
