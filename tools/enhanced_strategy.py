from __future__ import annotations

import numpy as np
import pandas as pd

from algo_download.backtest.engine import run_backtest
from algo_download.core.indicators import bollinger, ema, macd, rsi, adx, cvd_from_ticks, cvd_proxy
from algo_download.strategy.scorer import ConfluenceScorer
from strategy_framework.registry import apply_registered_components


def _wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    period = max(1, int(period))
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0] if close.shape[0] else np.nan
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    out = np.full_like(close, np.nan, dtype=float)
    if close.shape[0] < period:
        return out
    out[period - 1] = np.nanmean(tr[:period])
    for i in range(period, close.shape[0]):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def _run_managed_backtest(
    bars: pd.DataFrame,
    signals: dict,
    *,
    risk: float = 0.01,
    fee_bps: float = 4.0,
    slip_bps: float = 1.5,
    initial_balance: float = 100000.0,
    max_concurrent: int = 1,
    breakeven_after_r: float = 0.0,
    partial_tp_r: float = 0.0,
    partial_close_frac: float = 0.5,
    trail_after_r: float = 0.0,
    trail_atr_mult: float = 0.0,
    trail_atr_period: int = 14,
    max_hold_hours: float = 0.0,
) -> dict:
    max_concurrent = int(max_concurrent)
    if max_concurrent < 1:
        raise ValueError("max_concurrent must be >= 1")

    open_ = bars["open"].to_numpy(float)
    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    close = bars["close"].to_numpy(float)
    signal = np.asarray(signals["signal"])
    sl_arr = np.asarray(signals["sl"], dtype=float)
    rr_arr = np.asarray(signals.get("rr", np.zeros_like(signal)), dtype=float)
    n = len(bars)
    fee = fee_bps / 10_000.0
    slip = slip_bps / 10_000.0
    trail_atr = _wilder_atr(high, low, close, trail_atr_period) if trail_atr_mult > 0 else None
    if max_hold_hours > 0:
        bar_hours = _infer_bar_hours(bars)
        max_hold_bars = max(1, int(np.ceil(max_hold_hours / max(bar_hours, 1e-9))))
    else:
        max_hold_bars = 0

    balance = float(initial_balance)
    equity = np.empty(n, dtype=float)
    open_trades: list[dict] = []
    closed: list[dict] = []

    for i in range(n):
        open_count_at_bar_open = len(open_trades)
        still_open: list[dict] = []
        for trade in open_trades:
            direction = trade["direction"]
            stop = trade["stop"]
            target = trade["target"]
            entry = trade["entry"]
            risk_per_unit = trade["risk_per_unit"]
            size = trade["size"]
            exit_price = None
            reason = 0

            if partial_tp_r > 0 and not trade["partial_done"]:
                partial_level = entry + direction * partial_tp_r * risk_per_unit
                hit_partial = high[i] >= partial_level if direction == 1 else low[i] <= partial_level
                if hit_partial:
                    fill = partial_level * (1.0 - slip) if direction == 1 else partial_level * (1.0 + slip)
                    partial_size = size * partial_close_frac
                    pnl = (fill - entry) * partial_size * direction
                    pnl -= (entry + fill) * partial_size * fee
                    balance += pnl
                    trade["size"] -= partial_size
                    trade["partial_done"] = True
                    trade["partial_pnl"] += pnl
                    if breakeven_after_r > 0:
                        trade["stop"] = max(trade["stop"], entry) if direction == 1 else min(trade["stop"], entry)

            if breakeven_after_r > 0:
                be_level = entry + direction * breakeven_after_r * risk_per_unit
                hit_be = high[i] >= be_level if direction == 1 else low[i] <= be_level
                if hit_be:
                    trade["stop"] = max(trade["stop"], entry) if direction == 1 else min(trade["stop"], entry)
                    stop = trade["stop"]

            if trail_atr is not None and np.isfinite(trail_atr[i]):
                trail_level = entry + direction * trail_after_r * risk_per_unit
                trail_active = high[i] >= trail_level if direction == 1 else low[i] <= trail_level
                if trail_active:
                    if direction == 1:
                        trade["stop"] = max(trade["stop"], high[i] - trail_atr_mult * trail_atr[i])
                    else:
                        trade["stop"] = min(trade["stop"], low[i] + trail_atr_mult * trail_atr[i])
                    stop = trade["stop"]

            if direction == 1:
                if low[i] <= stop:
                    exit_price = stop * (1.0 - slip)
                    reason = -1
                elif high[i] >= target:
                    exit_price = target * (1.0 - slip)
                    reason = 1
            else:
                if high[i] >= stop:
                    exit_price = stop * (1.0 + slip)
                    reason = -1
                elif low[i] <= target:
                    exit_price = target * (1.0 + slip)
                    reason = 1
            if exit_price is None and max_hold_bars > 0 and i - trade["entry_idx"] >= max_hold_bars:
                exit_price = close[i] * (1.0 - slip) if direction == 1 else close[i] * (1.0 + slip)
                reason = 2

            if exit_price is None:
                still_open.append(trade)
                continue

            runner_pnl = (exit_price - entry) * trade["size"] * direction
            runner_pnl -= (entry + exit_price) * trade["size"] * fee
            total_pnl = runner_pnl + trade["partial_pnl"]
            balance += runner_pnl
            closed.append({
                "entry_idx": trade["entry_idx"],
                "exit_idx": i,
                "direction": direction,
                "entry": entry,
                "exit": exit_price,
                "pnl": total_pnl,
                "reason": reason,
                "risk_amt": trade["risk_amt"],
            })

        open_trades = still_open

        if i > 0 and open_count_at_bar_open < max_concurrent and signal[i - 1] != 0:
            direction = int(signal[i - 1])
            raw_entry = open_[i]
            entry = raw_entry * (1.0 + slip) if direction == 1 else raw_entry * (1.0 - slip)
            stop = float(sl_arr[i - 1])
            if np.isfinite(stop):
                risk_per_unit = abs(entry - stop)
                if risk_per_unit > 0:
                    rr = float(rr_arr[i - 1]) if np.ndim(rr_arr) else float(rr_arr)
                    target = entry + direction * rr * risk_per_unit
                    risk_amt = balance * risk
                    size = risk_amt / risk_per_unit
                    open_trades.append({
                        "entry_idx": i,
                        "direction": direction,
                        "entry": entry,
                        "stop": stop,
                        "target": target,
                        "risk_per_unit": risk_per_unit,
                        "size": size,
                        "risk_amt": risk_amt,
                        "partial_done": False,
                        "partial_pnl": 0.0,
                    })

        mtm = 0.0
        for trade in open_trades:
            mtm += (close[i] - trade["entry"]) * trade["size"] * trade["direction"]
        equity[i] = balance + mtm

    last_idx = n - 1
    for trade in open_trades:
        direction = trade["direction"]
        exit_price = close[last_idx] * (1.0 - slip) if direction == 1 else close[last_idx] * (1.0 + slip)
        runner_pnl = (exit_price - trade["entry"]) * trade["size"] * direction
        runner_pnl -= (trade["entry"] + exit_price) * trade["size"] * fee
        total_pnl = runner_pnl + trade["partial_pnl"]
        balance += runner_pnl
        closed.append({
            "entry_idx": trade["entry_idx"],
            "exit_idx": last_idx,
            "direction": direction,
            "entry": trade["entry"],
            "exit": exit_price,
            "pnl": total_pnl,
            "reason": 0,
            "risk_amt": trade["risk_amt"],
        })
    if n:
        equity[-1] = balance
    return {"equity": equity, "trades": pd.DataFrame(closed), "final_balance": balance}


def _infer_bar_hours(bars: pd.DataFrame) -> float:
    if "ts_ms" in bars.columns and len(bars) > 1:
        ts = bars["ts_ms"].to_numpy(np.int64)
        diffs = np.diff(ts)
        diffs = diffs[diffs > 0]
        if diffs.size:
            return float(np.median(diffs) / 3_600_000.0)
    if isinstance(bars.index, pd.DatetimeIndex) and len(bars) > 1:
        diffs = bars.index.to_series().diff().dropna().dt.total_seconds().to_numpy()
        diffs = diffs[diffs > 0]
        if diffs.size:
            return float(np.median(diffs) / 3600.0)
    return 1.0


def _rolling_max_shifted(x: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(x).rolling(window, min_periods=window).max().shift(1).to_numpy()


def _rolling_min_shifted(x: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(x).rolling(window, min_periods=window).min().shift(1).to_numpy()


def adx_scorer(bars: pd.DataFrame, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    close = bars["close"].to_numpy(float)
    period = int(ctx.get("adx_period", 14))
    threshold = float(ctx.get("adx_threshold", 25.0))

    adx_vals = adx(high, low, close, period)

    # ADX is non-directional. If ADX > threshold, we allow both long and short
    # as it indicates a strong trend exists.
    signal = (adx_vals > threshold).astype(float)
    return signal, signal

def make_adx_scorer(adx_period: int, adx_threshold: float):
    def scorer(bars: pd.DataFrame, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
        ctx = dict(ctx)
        ctx["adx_period"] = adx_period
        ctx["adx_threshold"] = adx_threshold
        return adx_scorer(bars, ctx)
    return scorer


def _wilder_rma(x: np.ndarray, period: int) -> np.ndarray:
    period = max(1, int(period))
    out = np.full_like(x, np.nan, dtype=float)
    if x.shape[0] < period:
        return out
    out[period - 1] = np.nanmean(x[:period])
    for i in range(period, x.shape[0]):
        out[i] = (out[i - 1] * (period - 1) + x[i]) / period
    return out


def _directional_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0] if close.shape[0] else np.nan
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0.0
    down_move[0] = 0.0
    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)
    tr_rma = _wilder_rma(tr, period)
    plus_di = 100.0 * _wilder_rma(plus_dm, period) / np.maximum(tr_rma, 1e-12)
    minus_di = 100.0 * _wilder_rma(minus_dm, period) / np.maximum(tr_rma, 1e-12)
    dx = 100.0 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-12)
    return _wilder_rma(np.nan_to_num(dx, nan=0.0), period), plus_di, minus_di


def directional_adx_scorer(bars: pd.DataFrame, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    close = bars["close"].to_numpy(float)
    period = int(ctx.get("directional_adx_period", ctx.get("adx_period", 14)))
    threshold = float(ctx.get("directional_adx_threshold", ctx.get("adx_threshold", 25.0)))
    di_gap = float(ctx.get("directional_adx_di_gap", 4.0))

    adx_vals, plus_di, minus_di = _directional_adx(high, low, close, period)
    strong = adx_vals >= threshold
    long = (strong & (plus_di >= minus_di + di_gap)).astype(float)
    short = (strong & (minus_di >= plus_di + di_gap)).astype(float)
    long[~np.isfinite(adx_vals)] = 0.0
    short[~np.isfinite(adx_vals)] = 0.0
    return long, short


def make_directional_adx_scorer(period: int, threshold: float, di_gap: float):
    def scorer(bars: pd.DataFrame, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
        ctx = dict(ctx)
        ctx["directional_adx_period"] = period
        ctx["directional_adx_threshold"] = threshold
        ctx["directional_adx_di_gap"] = di_gap
        return directional_adx_scorer(bars, ctx)
    return scorer

def breakout_scorer(bars: pd.DataFrame, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    close = bars["close"].to_numpy(float)
    atr = ctx["atr"]
    lookback = int(ctx.get("breakout_lookback", 40))
    atr_buffer = float(ctx.get("breakout_atr_buffer", 0.10))

    prev_high = _rolling_max_shifted(high, lookback)
    prev_low = _rolling_min_shifted(low, lookback)
    long = (close > (prev_high + atr_buffer * atr)).astype(float)
    short = (close < (prev_low - atr_buffer * atr)).astype(float)
    long[~np.isfinite(prev_high)] = 0.0
    short[~np.isfinite(prev_low)] = 0.0
    return long, short


def make_breakout_scorer(lookback: int, atr_buffer: float):
    def scorer(bars: pd.DataFrame, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
        ctx = dict(ctx)
        ctx["breakout_lookback"] = lookback
        ctx["breakout_atr_buffer"] = atr_buffer
        return breakout_scorer(bars, ctx)

    return scorer


def breakout_retest_scorer(bars: pd.DataFrame, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    close = bars["close"].to_numpy(float)
    open_ = bars["open"].to_numpy(float)
    atr = ctx["atr"]
    lookback = int(ctx.get("breakout_lookback", 40))
    atr_buffer = float(ctx.get("breakout_atr_buffer", 0.10))
    retest_window = int(ctx.get("retest_window", 8))
    retest_atr_buffer = float(ctx.get("retest_atr_buffer", 0.30))

    prev_high = _rolling_max_shifted(high, lookback)
    prev_low = _rolling_min_shifted(low, lookback)
    broke_long = close > (prev_high + atr_buffer * atr)
    broke_short = close < (prev_low - atr_buffer * atr)

    long_level = pd.Series(np.where(broke_long, prev_high, np.nan)).ffill(limit=retest_window).shift(1).to_numpy()
    short_level = pd.Series(np.where(broke_short, prev_low, np.nan)).ffill(limit=retest_window).shift(1).to_numpy()

    long = (
        np.isfinite(long_level)
        & (low <= long_level + retest_atr_buffer * atr)
        & (close > long_level)
        & (close > open_)
    ).astype(float)
    short = (
        np.isfinite(short_level)
        & (high >= short_level - retest_atr_buffer * atr)
        & (close < short_level)
        & (close < open_)
    ).astype(float)
    return long, short


def make_breakout_retest_scorer(
    lookback: int,
    atr_buffer: float,
    retest_window: int,
    retest_atr_buffer: float,
):
    def scorer(bars: pd.DataFrame, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
        ctx = dict(ctx)
        ctx["breakout_lookback"] = lookback
        ctx["breakout_atr_buffer"] = atr_buffer
        ctx["retest_window"] = retest_window
        ctx["retest_atr_buffer"] = retest_atr_buffer
        return breakout_retest_scorer(bars, ctx)

    return scorer


def _rolling_profile_levels(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    lookback: int,
    bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast causal volume-profile proxy.

    The first version rebuilt a binned histogram for every bar. That is too
    slow inside thousands of optimizer trials. This uses prior-window
    volume-weighted mean price as POC and one weighted standard deviation as
    VAH/VAL. It keeps the same location intent without making optimization
    impractical.
    """
    lookback = max(10, int(lookback))
    typical = (high + low + close) / 3.0
    vol = pd.Series(np.where(np.isfinite(volume), volume, 0.0))
    pv = pd.Series(typical * volume)
    pv2 = pd.Series(typical * typical * volume)
    vol_sum = vol.rolling(lookback, min_periods=lookback).sum().shift(1).to_numpy()
    pv_sum = pv.rolling(lookback, min_periods=lookback).sum().shift(1).to_numpy()
    pv2_sum = pv2.rolling(lookback, min_periods=lookback).sum().shift(1).to_numpy()
    poc = pv_sum / np.maximum(vol_sum, 1e-12)
    var = pv2_sum / np.maximum(vol_sum, 1e-12) - poc * poc
    sd = np.sqrt(np.maximum(var, 0.0))
    poc[vol_sum <= 0.0] = np.nan
    vah = poc + sd
    val = poc - sd
    return poc, vah, val


def _local_sweep_levels(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    sh_idx: np.ndarray,
    sh_px: np.ndarray,
    sl_idx: np.ndarray,
    sl_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = close.shape[0]
    bull = np.full(n, np.nan, dtype=float)
    bear = np.full(n, np.nan, dtype=float)
    swept_h = np.zeros(sh_px.shape[0], dtype=bool)
    swept_l = np.zeros(sl_px.shape[0], dtype=bool)
    for i in range(n):
        for kh in range(sh_px.shape[0] - 1, -1, -1):
            if sh_idx[kh] >= i or swept_h[kh]:
                continue
            level = sh_px[kh]
            if high[i] > level:
                if close[i] < level:
                    bear[i] = level
                swept_h[kh] = True
            break
        for kl in range(sl_px.shape[0] - 1, -1, -1):
            if sl_idx[kl] >= i or swept_l[kl]:
                continue
            level = sl_px[kl]
            if low[i] < level:
                if close[i] > level:
                    bull[i] = level
                swept_l[kl] = True
            break
    return bull, bear


def _local_absorption(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    atr: np.ndarray,
    vol_ma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = close.shape[0]
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)
    for i in range(1, n):
        a = atr[i]
        vm = vol_ma[i]
        if not np.isfinite(a) or not np.isfinite(vm) or a <= 0.0 or vm <= 0.0:
            continue
        spread = high[i] - low[i]
        if spread < 0.4 * a and volume[i] > 2.0 * vm:
            if close[i] > open_[i]:
                bull[i] = True
            else:
                bear[i] = True
        elif close[i] < open_[i] and spread > 1.5 * a and volume[i] > 2.0 * vm and (close[i] - low[i]) > 0.6 * spread:
            bull[i] = True
        elif close[i] > open_[i] and spread > 1.5 * a and volume[i] > 2.0 * vm and (high[i] - close[i]) > 0.6 * spread:
            bear[i] = True
    return bull, bear


def _local_structure_shift(
    close: np.ndarray,
    sh_idx: np.ndarray,
    sh_px: np.ndarray,
    sl_idx: np.ndarray,
    sl_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = close.shape[0]
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)
    h_pos = 0
    l_pos = 0
    last_high = np.nan
    last_low = np.nan
    for i in range(n):
        while h_pos < sh_idx.shape[0] and sh_idx[h_pos] < i:
            last_high = sh_px[h_pos]
            h_pos += 1
        while l_pos < sl_idx.shape[0] and sl_idx[l_pos] < i:
            last_low = sl_px[l_pos]
            l_pos += 1
        if np.isfinite(last_high) and close[i] > last_high:
            bull[i] = True
        if np.isfinite(last_low) and close[i] < last_low:
            bear[i] = True
    return bull, bear


def liquidity_confluence_scorer(bars: pd.DataFrame, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    close = bars["close"].to_numpy(float)
    open_ = bars["open"].to_numpy(float)
    volume = bars["volume"].to_numpy(float) if "volume" in bars.columns else None
    n = close.shape[0]
    long = np.zeros(n, dtype=float)
    short = np.zeros(n, dtype=float)
    if volume is None:
        return long, short

    atr = ctx["atr"]
    sh_idx, sh_px, sl_idx, sl_px = ctx["swings"]
    swing_right = int(ctx.get("swing_right", 5))
    htf_window = int(ctx.get("liq_htf_window", 200))
    profile_lookback = int(ctx.get("liq_profile_lookback", 120))
    profile_bins = int(ctx.get("liq_profile_bins", 64))
    level_atr = float(ctx.get("liq_level_atr", 0.6))
    setup_window = int(ctx.get("liq_setup_window", 5))
    retest_window = int(ctx.get("liq_retest_window", 8))
    retest_atr = float(ctx.get("liq_retest_atr", 0.35))
    min_score = float(ctx.get("liq_min_score", 5.0))

    shift = max(2, htf_window // 6)
    htf_ma = pd.Series(close).rolling(htf_window, min_periods=htf_window).mean().shift(1).to_numpy()
    htf_prev = pd.Series(close).rolling(htf_window, min_periods=htf_window).mean().shift(shift).to_numpy()
    htf_bull = (close > htf_ma) & (htf_ma > htf_prev)
    htf_bear = (close < htf_ma) & (htf_ma < htf_prev)

    bull_sweep_level, bear_sweep_level = _local_sweep_levels(
        high, low, close, sh_idx + swing_right, sh_px, sl_idx + swing_right, sl_px
    )

    poc, vah, val = _rolling_profile_levels(high, low, close, volume, profile_lookback, profile_bins)
    near_bull_level = (
        np.isfinite(val) & np.isfinite(atr)
        & (np.minimum(np.abs(low - val), np.abs(close - val)) <= level_atr * atr)
    )
    near_bear_level = (
        np.isfinite(vah) & np.isfinite(atr)
        & (np.minimum(np.abs(high - vah), np.abs(close - vah)) <= level_atr * atr)
    )

    vol_ma = pd.Series(volume).rolling(int(ctx.get("vol_ma_period", 20)), min_periods=1).mean().to_numpy()
    absorb_bull, absorb_bear = _local_absorption(open_, high, low, close, volume, atr, vol_ma)

    if "buy_volume" in bars.columns and "sell_volume" in bars.columns:
        cvd = cvd_from_ticks(
            bars["buy_volume"].to_numpy(float),
            bars["sell_volume"].to_numpy(float),
        )
    else:
        cvd = cvd_proxy(open_, high, low, close, volume)
    cvd_delta = np.diff(cvd, prepend=cvd[0] if n else 0.0)
    cvd_ma = pd.Series(cvd_delta).rolling(int(ctx.get("liq_cvd_window", 5)), min_periods=1).mean().to_numpy()
    cvd_bull = cvd_ma > 0
    cvd_bear = cvd_ma < 0

    bull_shift, bear_shift = _local_structure_shift(
        close, sh_idx + swing_right, sh_px, sl_idx + swing_right, sl_px
    )

    bull_setup_level = np.full(n, np.nan, dtype=float)
    bear_setup_level = np.full(n, np.nan, dtype=float)
    for i in range(n):
        s = max(0, i - setup_window + 1)
        if (
            htf_bull[i]
            and np.isfinite(bull_sweep_level[s:i + 1]).any()
            and near_bull_level[s:i + 1].any()
            and absorb_bull[s:i + 1].any()
            and cvd_bull[i]
            and bull_shift[s:i + 1].any()
        ):
            levels = bull_sweep_level[s:i + 1]
            bull_setup_level[i] = levels[np.isfinite(levels)][-1]
        if (
            htf_bear[i]
            and np.isfinite(bear_sweep_level[s:i + 1]).any()
            and near_bear_level[s:i + 1].any()
            and absorb_bear[s:i + 1].any()
            and cvd_bear[i]
            and bear_shift[s:i + 1].any()
        ):
            levels = bear_sweep_level[s:i + 1]
            bear_setup_level[i] = levels[np.isfinite(levels)][-1]

    bull_retest = pd.Series(bull_setup_level).ffill(limit=retest_window).shift(1).to_numpy()
    bear_retest = pd.Series(bear_setup_level).ffill(limit=retest_window).shift(1).to_numpy()
    long_mask = (
        np.isfinite(bull_retest) & np.isfinite(atr)
        & (low <= bull_retest + retest_atr * atr)
        & (close > bull_retest)
        & (close > open_)
        & cvd_bull
    )
    short_mask = (
        np.isfinite(bear_retest) & np.isfinite(atr)
        & (high >= bear_retest - retest_atr * atr)
        & (close < bear_retest)
        & (close < open_)
        & cvd_bear
    )
    long[long_mask] = min_score
    short[short_mask] = min_score
    return long, short


def make_liquidity_confluence_scorer(
    htf_window: int,
    profile_lookback: int,
    profile_bins: int,
    level_atr: float,
    setup_window: int,
    retest_window: int,
    retest_atr: float,
    cvd_window: int,
    min_score: float,
):
    def scorer(bars: pd.DataFrame, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
        ctx = dict(ctx)
        ctx["liq_htf_window"] = htf_window
        ctx["liq_profile_lookback"] = profile_lookback
        ctx["liq_profile_bins"] = profile_bins
        ctx["liq_level_atr"] = level_atr
        ctx["liq_setup_window"] = setup_window
        ctx["liq_retest_window"] = retest_window
        ctx["liq_retest_atr"] = retest_atr
        ctx["liq_cvd_window"] = cvd_window
        ctx["liq_min_score"] = min_score
        return liquidity_confluence_scorer(bars, ctx)

    return scorer


def failed_breakout_scorer(bars: pd.DataFrame, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    close = bars["close"].to_numpy(float)
    open_ = bars["open"].to_numpy(float)
    atr = ctx["atr"]
    lookback = int(ctx.get("failed_breakout_lookback", ctx.get("breakout_lookback", 40)))
    atr_buffer = float(ctx.get("failed_breakout_atr_buffer", 0.10))
    reclaim_atr = float(ctx.get("failed_breakout_reclaim_atr", 0.05))

    prev_high = _rolling_max_shifted(high, lookback)
    prev_low = _rolling_min_shifted(low, lookback)
    short = (
        (high > prev_high + atr_buffer * atr)
        & (close < prev_high - reclaim_atr * atr)
        & (close < open_)
    ).astype(float)
    long = (
        (low < prev_low - atr_buffer * atr)
        & (close > prev_low + reclaim_atr * atr)
        & (close > open_)
    ).astype(float)
    long[~np.isfinite(prev_low)] = 0.0
    short[~np.isfinite(prev_high)] = 0.0
    return long, short


def make_failed_breakout_scorer(
    lookback: int,
    atr_buffer: float,
    reclaim_atr: float,
):
    def scorer(bars: pd.DataFrame, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
        ctx = dict(ctx)
        ctx["failed_breakout_lookback"] = lookback
        ctx["failed_breakout_atr_buffer"] = atr_buffer
        ctx["failed_breakout_reclaim_atr"] = reclaim_atr
        return failed_breakout_scorer(bars, ctx)

    return scorer


def volume_breakout_scorer(bars: pd.DataFrame, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    open_ = bars["open"].to_numpy(float)
    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    close = bars["close"].to_numpy(float)
    volume = bars["volume"].to_numpy(float)
    atr = ctx["atr"]
    lookback = int(ctx.get("volume_breakout_lookback", ctx.get("breakout_lookback", 40)))
    atr_buffer = float(ctx.get("volume_breakout_atr_buffer", ctx.get("breakout_atr_buffer", 0.10)))
    volume_period = int(ctx.get("volume_breakout_volume_period", 30))
    volume_mult = float(ctx.get("volume_breakout_volume_mult", 1.25))
    body_atr_min = float(ctx.get("volume_breakout_body_atr_min", 0.35))

    prev_high = _rolling_max_shifted(high, lookback)
    prev_low = _rolling_min_shifted(low, lookback)
    vol_ma = pd.Series(volume).rolling(volume_period, min_periods=volume_period).mean().shift(1).to_numpy()
    body = np.abs(close - open_)
    confirmed = (volume > volume_mult * vol_ma) & (body > body_atr_min * atr)
    long = ((close > prev_high + atr_buffer * atr) & confirmed & (close > open_)).astype(float)
    short = ((close < prev_low - atr_buffer * atr) & confirmed & (close < open_)).astype(float)
    long[~np.isfinite(prev_high) | ~np.isfinite(vol_ma)] = 0.0
    short[~np.isfinite(prev_low) | ~np.isfinite(vol_ma)] = 0.0
    return long, short


def trend_pullback_scorer(bars: pd.DataFrame, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    close = bars["close"].to_numpy(float)
    atr = ctx["atr"]
    fast_period = int(ctx.get("pullback_fast_ema", 21))
    slow_period = int(ctx.get("pullback_slow_ema", 100))
    rsi_period = int(ctx.get("pullback_rsi_period", 14))
    rsi_floor = float(ctx.get("pullback_rsi_floor", 42.0))
    rsi_ceiling = float(ctx.get("pullback_rsi_ceiling", 58.0))
    atr_touch = float(ctx.get("pullback_atr_touch", 0.45))

    fast = ema(close, fast_period)
    slow = ema(close, slow_period)
    r = rsi(close, rsi_period)
    fast_slope = fast - pd.Series(fast).shift(5).to_numpy()
    slow_slope = slow - pd.Series(slow).shift(5).to_numpy()
    long = (
        (close > slow)
        & (fast > slow)
        & (fast_slope > 0)
        & (slow_slope > 0)
        & (low <= fast + atr_touch * atr)
        & (close > fast)
        & (r >= rsi_floor)
    ).astype(float)
    short = (
        (close < slow)
        & (fast < slow)
        & (fast_slope < 0)
        & (slow_slope < 0)
        & (high >= fast - atr_touch * atr)
        & (close < fast)
        & (r <= rsi_ceiling)
    ).astype(float)
    long[~np.isfinite(fast) | ~np.isfinite(slow) | ~np.isfinite(r)] = 0.0
    short[~np.isfinite(fast) | ~np.isfinite(slow) | ~np.isfinite(r)] = 0.0
    return long, short


def make_trend_pullback_scorer(
    fast_ema: int,
    slow_ema: int,
    rsi_period: int,
    rsi_floor: float,
    rsi_ceiling: float,
    atr_touch: float,
):
    def scorer(bars: pd.DataFrame, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
        ctx = dict(ctx)
        ctx["pullback_fast_ema"] = fast_ema
        ctx["pullback_slow_ema"] = slow_ema
        ctx["pullback_rsi_period"] = rsi_period
        ctx["pullback_rsi_floor"] = rsi_floor
        ctx["pullback_rsi_ceiling"] = rsi_ceiling
        ctx["pullback_atr_touch"] = atr_touch
        return trend_pullback_scorer(bars, ctx)
    return scorer


def range_reversion_scorer(bars: pd.DataFrame, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    open_ = bars["open"].to_numpy(float)
    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    close = bars["close"].to_numpy(float)
    period = int(ctx.get("range_bb_period", 20))
    mult = float(ctx.get("range_bb_mult", 2.0))
    rsi_period = int(ctx.get("range_rsi_period", 14))
    rsi_long_max = float(ctx.get("range_rsi_long_max", 35.0))
    rsi_short_min = float(ctx.get("range_rsi_short_min", 65.0))
    max_adx = float(ctx.get("range_max_adx", 22.0))
    adx_period = int(ctx.get("range_adx_period", ctx.get("adx_period", 14)))

    mid, upper, lower = bollinger(close, period, mult)
    r = rsi(close, rsi_period)
    adx_vals = adx(high, low, close, adx_period)
    long = (
        (low < lower)
        & (close > lower)
        & (close > open_)
        & (r <= rsi_long_max)
        & (adx_vals <= max_adx)
    ).astype(float)
    short = (
        (high > upper)
        & (close < upper)
        & (close < open_)
        & (r >= rsi_short_min)
        & (adx_vals <= max_adx)
    ).astype(float)
    long[~np.isfinite(mid) | ~np.isfinite(r) | ~np.isfinite(adx_vals)] = 0.0
    short[~np.isfinite(mid) | ~np.isfinite(r) | ~np.isfinite(adx_vals)] = 0.0
    return long, short


def make_range_reversion_scorer(
    bb_period: int,
    bb_mult: float,
    rsi_period: int,
    rsi_long_max: float,
    rsi_short_min: float,
    max_adx: float,
):
    def scorer(bars: pd.DataFrame, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
        ctx = dict(ctx)
        ctx["range_bb_period"] = bb_period
        ctx["range_bb_mult"] = bb_mult
        ctx["range_rsi_period"] = rsi_period
        ctx["range_rsi_long_max"] = rsi_long_max
        ctx["range_rsi_short_min"] = rsi_short_min
        ctx["range_max_adx"] = max_adx
        return range_reversion_scorer(bars, ctx)
    return scorer


def make_volume_breakout_scorer(
    lookback: int,
    atr_buffer: float,
    volume_period: int,
    volume_mult: float,
    body_atr_min: float,
):
    def scorer(bars: pd.DataFrame, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
        ctx = dict(ctx)
        ctx["volume_breakout_lookback"] = lookback
        ctx["volume_breakout_atr_buffer"] = atr_buffer
        ctx["volume_breakout_volume_period"] = volume_period
        ctx["volume_breakout_volume_mult"] = volume_mult
        ctx["volume_breakout_body_atr_min"] = body_atr_min
        return volume_breakout_scorer(bars, ctx)

    return scorer


def momentum_scorer(bars: pd.DataFrame, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    close = bars["close"].to_numpy(float)
    rsi_period = int(ctx.get("rsi_period", 14))
    rsi_mid = float(ctx.get("rsi_mid", 50.0))
    rsi_band = float(ctx.get("rsi_band", 5.0))
    macd_fast = int(ctx.get("macd_fast", 12))
    macd_slow = int(ctx.get("macd_slow", 26))
    macd_signal = int(ctx.get("macd_signal", 9))

    r = rsi(close, rsi_period)
    macd_line, signal_line, hist = macd(close, macd_fast, macd_slow, macd_signal)
    ema_fast = ema(close, macd_fast)
    ema_slow = ema(close, macd_slow)

    long = ((r > rsi_mid + rsi_band) & (hist > 0) & (ema_fast > ema_slow)).astype(float)
    short = ((r < rsi_mid - rsi_band) & (hist < 0) & (ema_fast < ema_slow)).astype(float)
    long[~np.isfinite(r) | ~np.isfinite(hist)] = 0.0
    short[~np.isfinite(r) | ~np.isfinite(hist)] = 0.0
    return long, short


def make_momentum_scorer(rsi_period: int, rsi_band: float):
    def scorer(bars: pd.DataFrame, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
        ctx = dict(ctx)
        ctx["rsi_period"] = rsi_period
        ctx["rsi_band"] = rsi_band
        return momentum_scorer(bars, ctx)

    return scorer


def volatility_scorer(bars: pd.DataFrame, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    close = bars["close"].to_numpy(float)
    period = int(ctx.get("bb_period", 20))
    mult = float(ctx.get("bb_mult", 2.0))
    squeeze_window = int(ctx.get("squeeze_window", 80))
    squeeze_quantile = float(ctx.get("squeeze_quantile", 0.35))

    mid, upper, lower = bollinger(close, period, mult)
    width = (upper - lower) / np.maximum(np.abs(mid), 1e-12)
    width_rank = pd.Series(width).rolling(squeeze_window, min_periods=squeeze_window).rank(pct=True).to_numpy()
    was_squeezed = pd.Series(width_rank < squeeze_quantile).rolling(5, min_periods=1).max().shift(1).fillna(0).to_numpy(bool)

    long = ((close > upper) & was_squeezed).astype(float)
    short = ((close < lower) & was_squeezed).astype(float)
    long[~np.isfinite(width_rank)] = 0.0
    short[~np.isfinite(width_rank)] = 0.0
    return long, short


def _apply_entry_quality_filters(
    bars: pd.DataFrame,
    signals: dict,
    *,
    min_tp_pct: float = 0.0,
    max_sl_pct: float = 1.0,
    min_signal_score: float = 0.0,
    min_long_signal_score: float = 0.0,
    min_short_signal_score: float = 0.0,
    min_score_edge: float = 0.0,
    min_long_dominance: float = 0.0,
    min_short_dominance: float = 0.0,
    min_score_rank: float = 0.0,
    min_long_score_rank: float = 0.0,
    min_short_score_rank: float = 0.0,
    score_rank_lookback: int = 120,
    min_lookout_score: float = 0.0,
    min_adx_entry: float = 0.0,
    adx_period: int = 14,
    trend_filter: str = "none",
    trend_ema_period: int = 200,
    cooldown_bars: int = 0,
) -> dict:
    close = bars["close"].to_numpy(float)
    signal = np.asarray(signals["signal"]).copy()
    sl = np.asarray(signals["sl"], dtype=float)
    tp = np.asarray(signals["tp"], dtype=float)
    bull = np.asarray(signals.get("bull_score", np.zeros_like(signal)), dtype=float)
    bear = np.asarray(signals.get("bear_score", np.zeros_like(signal)), dtype=float)
    side_score = np.where(signal > 0, bull, bear)
    opposite_score = np.where(signal > 0, bear, bull)
    score_total = np.maximum(bull + bear, 1e-12)
    lookout_score = np.where(signal != 0, side_score / score_total, 0.0)

    valid = signal != 0
    if min_tp_pct > 0.0:
        tp_pct = np.abs(tp - close) / np.maximum(np.abs(close), 1e-12)
        valid &= tp_pct >= min_tp_pct
    if max_sl_pct < 1.0:
        sl_pct = np.abs(close - sl) / np.maximum(np.abs(close), 1e-12)
        valid &= sl_pct <= max_sl_pct
    if min_signal_score > 0.0:
        valid &= side_score >= min_signal_score
    if min_long_signal_score > 0.0:
        valid &= (signal <= 0) | (bull >= min_long_signal_score)
    if min_short_signal_score > 0.0:
        valid &= (signal >= 0) | (bear >= min_short_signal_score)
    if min_score_edge > 0.0:
        valid &= np.abs(bull - bear) >= min_score_edge
    if min_long_dominance > 0.0:
        valid &= (signal <= 0) | ((bull - bear) >= min_long_dominance)
    if min_short_dominance > 0.0:
        valid &= (signal >= 0) | ((bear - bull) >= min_short_dominance)
    if min_score_rank > 0.0:
        lookback = max(20, int(score_rank_lookback))
        bull_rank = pd.Series(bull).rolling(lookback, min_periods=lookback).rank(pct=True).to_numpy()
        bear_rank = pd.Series(bear).rolling(lookback, min_periods=lookback).rank(pct=True).to_numpy()
        side_rank = np.where(signal > 0, bull_rank, bear_rank)
        valid &= side_rank >= min_score_rank
    if min_long_score_rank > 0.0 or min_short_score_rank > 0.0:
        lookback = max(20, int(score_rank_lookback))
        bull_rank = pd.Series(bull).rolling(lookback, min_periods=lookback).rank(pct=True).to_numpy()
        bear_rank = pd.Series(bear).rolling(lookback, min_periods=lookback).rank(pct=True).to_numpy()
        if min_long_score_rank > 0.0:
            valid &= (signal <= 0) | (bull_rank >= min_long_score_rank)
        if min_short_score_rank > 0.0:
            valid &= (signal >= 0) | (bear_rank >= min_short_score_rank)
    if min_lookout_score > 0.0:
        valid &= lookout_score >= min_lookout_score
    if min_adx_entry > 0.0:
        high = bars["high"].to_numpy(float)
        low = bars["low"].to_numpy(float)
        adx_vals = adx(high, low, close, adx_period)
        valid &= adx_vals >= min_adx_entry
    if trend_filter != "none":
        ema_vals = ema(close, max(2, int(trend_ema_period)))
        if trend_filter == "ema":
            valid &= ((signal > 0) & (close > ema_vals)) | ((signal < 0) & (close < ema_vals))
        elif trend_filter == "ema_slope":
            ema_slope = ema_vals - pd.Series(ema_vals).shift(5).to_numpy()
            valid &= (
                ((signal > 0) & (close > ema_vals) & (ema_slope > 0))
                | ((signal < 0) & (close < ema_vals) & (ema_slope < 0))
            )

    signal[~valid] = 0
    cooldown = max(0, int(cooldown_bars))
    if cooldown > 0:
        last_entry = -cooldown - 1
        for i in range(signal.shape[0]):
            if signal[i] == 0:
                continue
            if i - last_entry <= cooldown:
                signal[i] = 0
            else:
                last_entry = i

    out = dict(signals)
    out["signal"] = signal
    out["side_score"] = np.where(signal != 0, side_score, 0.0)
    out["lookout_score"] = np.where(signal != 0, lookout_score, 0.0)
    return out


def make_volatility_scorer(
    bb_period: int,
    bb_mult: float,
    squeeze_window: int,
    squeeze_quantile: float,
):
    def scorer(bars: pd.DataFrame, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
        ctx = dict(ctx)
        ctx["bb_period"] = bb_period
        ctx["bb_mult"] = bb_mult
        ctx["squeeze_window"] = squeeze_window
        ctx["squeeze_quantile"] = squeeze_quantile
        return volatility_scorer(bars, ctx)

    return scorer


def build_enhanced_signals(
    bars: pd.DataFrame,
    *,
    entry_threshold: float = 1.0,
    min_dominance: float = 0.5,
    rr: float = 2.0,
    sl_buffer_atr: float = 0.5,
    atr_period: int = 14,
    swing_left: int = 5,
    swing_right: int = 5,
    wick_min_atr: float = 0.5,
    breakout_lookback: int = 40,
    breakout_atr_buffer: float = 0.10,
    rsi_period: int = 14,
    rsi_band: float = 5.0,
    bb_period: int = 20,
    bb_mult: float = 2.0,
    squeeze_window: int = 80,
    squeeze_quantile: float = 0.35,
    adx_period: int = 14,
    adx_threshold: float = 25.0,
    retest_window: int = 8,
    retest_atr_buffer: float = 0.30,
    failed_breakout_lookback: int = 40,
    failed_breakout_atr_buffer: float = 0.10,
    failed_breakout_reclaim_atr: float = 0.05,
    volume_breakout_lookback: int = 40,
    volume_breakout_atr_buffer: float = 0.10,
    volume_breakout_volume_period: int = 30,
    volume_breakout_volume_mult: float = 1.25,
    volume_breakout_body_atr_min: float = 0.35,
    directional_adx_di_gap: float = 4.0,
    pullback_fast_ema: int = 21,
    pullback_slow_ema: int = 100,
    pullback_rsi_period: int = 14,
    pullback_rsi_floor: float = 42.0,
    pullback_rsi_ceiling: float = 58.0,
    pullback_atr_touch: float = 0.45,
    range_bb_period: int = 20,
    range_bb_mult: float = 2.0,
    range_rsi_period: int = 14,
    range_rsi_long_max: float = 35.0,
    range_rsi_short_min: float = 65.0,
    range_max_adx: float = 22.0,
    liq_htf_window: int = 200,
    liq_profile_lookback: int = 120,
    liq_profile_bins: int = 64,
    liq_level_atr: float = 0.6,
    liq_setup_window: int = 5,
    liq_retest_window: int = 8,
    liq_retest_atr: float = 0.35,
    liq_cvd_window: int = 5,
    liq_min_score: float = 5.0,
    w_trend: float = 1.0,
    w_sfp: float = 1.0,
    w_candle: float = 1.0,
    w_regime: float = 1.0,
    w_vsa: float = 1.0,
    w_wyckoff: float = 1.0,
    w_vwap_dev: float = 1.0,
    w_breakout: float = 1.0,
    w_breakout_retest: float = 0.0,
    w_failed_breakout: float = 0.0,
    w_volume_breakout: float = 0.0,
    w_momentum: float = 1.0,
    w_volatility: float = 1.0,
    w_adx: float = 0.0,
    w_directional_adx: float = 0.0,
    w_trend_pullback: float = 0.0,
    w_range_reversion: float = 0.0,
    w_liquidity_confluence: float = 0.0,
    min_tp_pct: float = 0.0,
    max_sl_pct: float = 1.0,
    min_signal_score: float = 0.0,
    min_long_signal_score: float = 0.0,
    min_short_signal_score: float = 0.0,
    min_score_edge: float = 0.0,
    min_long_dominance: float = 0.0,
    min_short_dominance: float = 0.0,
    min_score_rank: float = 0.0,
    min_long_score_rank: float = 0.0,
    min_short_score_rank: float = 0.0,
    score_rank_lookback: int = 120,
    min_lookout_score: float = 0.0,
    min_adx_entry: float = 0.0,
    trend_filter: str = "none",
    trend_ema_period: int = 200,
    cooldown_bars: int = 0,
    **extra_params: object,
) -> dict:
    scorer = ConfluenceScorer.default(
        entry_threshold=entry_threshold,
        min_dominance=min_dominance,
        rr=rr,
        sl_buffer_atr=sl_buffer_atr,
        atr_period=atr_period,
        swing_left=swing_left,
        swing_right=swing_right,
        wick_min_atr=wick_min_atr,
    )

    for spec in scorer.scorers:
        if spec.name == "trend":
            spec.weight = w_trend
        elif spec.name == "sfp":
            spec.weight = w_sfp
        elif spec.name == "candle":
            spec.weight = w_candle
        elif spec.name == "regime":
            spec.weight = w_regime
        elif spec.name == "vsa":
            spec.weight = w_vsa
        elif spec.name == "wyckoff":
            spec.weight = w_wyckoff
        elif spec.name == "vwap_dev":
            spec.weight = w_vwap_dev

    scorer.register("breakout", make_breakout_scorer(breakout_lookback, breakout_atr_buffer), w_breakout)
    scorer.register(
        "breakout_retest",
        make_breakout_retest_scorer(
            breakout_lookback,
            breakout_atr_buffer,
            retest_window,
            retest_atr_buffer,
        ),
        w_breakout_retest,
    )
    scorer.register(
        "failed_breakout",
        make_failed_breakout_scorer(
            failed_breakout_lookback,
            failed_breakout_atr_buffer,
            failed_breakout_reclaim_atr,
        ),
        w_failed_breakout,
    )
    scorer.register(
        "volume_breakout",
        make_volume_breakout_scorer(
            volume_breakout_lookback,
            volume_breakout_atr_buffer,
            volume_breakout_volume_period,
            volume_breakout_volume_mult,
            volume_breakout_body_atr_min,
        ),
        w_volume_breakout,
    )
    scorer.register("momentum", make_momentum_scorer(rsi_period, rsi_band), w_momentum)
    scorer.register(
        "volatility",
        make_volatility_scorer(bb_period, bb_mult, squeeze_window, squeeze_quantile),
        w_volatility,
    )
    scorer.register("adx", make_adx_scorer(adx_period, adx_threshold), w_adx)
    scorer.register(
        "directional_adx",
        make_directional_adx_scorer(adx_period, adx_threshold, directional_adx_di_gap),
        w_directional_adx,
    )
    scorer.register(
        "trend_pullback",
        make_trend_pullback_scorer(
            pullback_fast_ema,
            pullback_slow_ema,
            pullback_rsi_period,
            pullback_rsi_floor,
            pullback_rsi_ceiling,
            pullback_atr_touch,
        ),
        w_trend_pullback,
    )
    scorer.register(
        "range_reversion",
        make_range_reversion_scorer(
            range_bb_period,
            range_bb_mult,
            range_rsi_period,
            range_rsi_long_max,
            range_rsi_short_min,
            range_max_adx,
        ),
        w_range_reversion,
    )
    if w_liquidity_confluence > 0.0:
        scorer.register(
            "liquidity_confluence",
            make_liquidity_confluence_scorer(
                liq_htf_window,
                liq_profile_lookback,
                liq_profile_bins,
                liq_level_atr,
                liq_setup_window,
                liq_retest_window,
                liq_retest_atr,
                liq_cvd_window,
                liq_min_score,
            ),
            w_liquidity_confluence,
        )
    apply_registered_components(scorer, extra_params)
    signals = scorer.build_signals(bars)
    return _apply_entry_quality_filters(
        bars,
        signals,
        min_tp_pct=min_tp_pct,
        max_sl_pct=max_sl_pct,
        min_signal_score=min_signal_score,
        min_long_signal_score=min_long_signal_score,
        min_short_signal_score=min_short_signal_score,
        min_score_edge=min_score_edge,
        min_long_dominance=min_long_dominance,
        min_short_dominance=min_short_dominance,
        min_score_rank=min_score_rank,
        min_long_score_rank=min_long_score_rank,
        min_short_score_rank=min_short_score_rank,
        score_rank_lookback=score_rank_lookback,
        min_lookout_score=min_lookout_score,
        min_adx_entry=min_adx_entry,
        adx_period=adx_period,
        trend_filter=trend_filter,
        trend_ema_period=trend_ema_period,
        cooldown_bars=cooldown_bars,
    )


def enhanced_backtest_strategy(
    bars: pd.DataFrame,
    strategy_kind: str = "enhanced",
    risk: float = 0.01,
    fee_bps: float = 4.0,
    slip_bps: float = 1.5,
    slippage_bps: float | None = None,
    initial_balance: float = 100000.0,
    max_concurrent: int = 1,
    breakeven_after_r: float = 0.0,
    partial_tp_r: float = 0.0,
    partial_close_frac: float = 0.5,
    trail_after_r: float = 0.0,
    trail_atr_mult: float = 0.0,
    trail_atr_period: int = 14,
    max_hold_hours: float = 0.0,
    **kwargs,
) -> dict:
    if slippage_bps is not None:
        slip_bps = float(slippage_bps)
    signals = build_enhanced_signals(bars, **kwargs)
    if breakeven_after_r > 0.0 or partial_tp_r > 0.0 or trail_atr_mult > 0.0 or max_hold_hours > 0.0:
        return _run_managed_backtest(
            bars,
            signals,
            risk=risk,
            fee_bps=fee_bps,
            slip_bps=slip_bps,
            initial_balance=initial_balance,
            max_concurrent=max_concurrent,
            breakeven_after_r=breakeven_after_r,
            partial_tp_r=partial_tp_r,
            partial_close_frac=partial_close_frac,
            trail_after_r=trail_after_r,
            trail_atr_mult=trail_atr_mult,
            trail_atr_period=trail_atr_period,
            max_hold_hours=max_hold_hours,
        )
    return run_backtest(
        bars,
        signals,
        risk=risk,
        fee_bps=fee_bps,
        slip_bps=slip_bps,
        initial_balance=initial_balance,
        max_concurrent=max_concurrent,
    )
