"""
Vectorized indicators with corrected math.

Key fixes vs. spec:
- ATR uses Wilder's RMA (not SMA)
- EMA seeded with SMA of first n
- All returns are numpy arrays the same length as input (NaN-padded)
- Numba-jitted hot paths; CuPy fallback if available

All functions accept 1-D numpy arrays. For tick precision, pass tick-derived bars.
"""
from __future__ import annotations
import numpy as np
from numba import njit, prange

try:
    import cupy as cp  # type: ignore
    _HAS_CUPY = True
except Exception:  # pragma: no cover
    cp = None
    _HAS_CUPY = False


# ---------------------------------------------------------------------------
# True Range / ATR (Wilder)
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    n = high.shape[0]
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        a = high[i] - low[i]
        b = abs(high[i] - close[i - 1])
        c = abs(low[i] - close[i - 1])
        tr[i] = a if a >= b and a >= c else (b if b >= c else c)
    return tr


@njit(cache=True, fastmath=True)
def atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = 14) -> np.ndarray:
    """Wilder's RMA-based ATR. Length == input."""
    n = high.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    tr = true_range(high, low, close)
    if n < period:
        return out
    s = 0.0
    for i in range(period):
        s += tr[i]
    out[period - 1] = s / period
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


# ---------------------------------------------------------------------------
# EMA / SMA / RMA
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def sma(x: np.ndarray, period: int) -> np.ndarray:
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out
    s = 0.0
    for i in range(period):
        s += x[i]
    out[period - 1] = s / period
    for i in range(period, n):
        s += x[i] - x[i - period]
        out[i] = s / period
    return out


@njit(cache=True, fastmath=True)
def ema(x: np.ndarray, period: int) -> np.ndarray:
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out
    s = 0.0
    for i in range(period):
        s += x[i]
    out[period - 1] = s / period
    k = 2.0 / (period + 1.0)
    for i in range(period, n):
        out[i] = x[i] * k + out[i - 1] * (1.0 - k)
    return out


@njit(cache=True, fastmath=True)
def rma(x: np.ndarray, period: int) -> np.ndarray:
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out
    s = 0.0
    for i in range(period):
        s += x[i]
    out[period - 1] = s / period
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + x[i]) / period
    return out


# ---------------------------------------------------------------------------
# RSI (Wilder)
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    n = close.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= period:
        return out
    gain = 0.0
    loss = 0.0
    for i in range(1, period + 1):
        d = close[i] - close[i - 1]
        if d >= 0:
            gain += d
        else:
            loss -= d
    avg_g = gain / period
    avg_l = loss / period
    out[period] = 100.0 - 100.0 / (1.0 + (avg_g / avg_l if avg_l > 0 else 1e9))
    for i in range(period + 1, n):
        d = close[i] - close[i - 1]
        g = d if d > 0 else 0.0
        l = -d if d < 0 else 0.0
        avg_g = (avg_g * (period - 1) + g) / period
        avg_l = (avg_l * (period - 1) + l) / period
        out[i] = 100.0 - 100.0 / (1.0 + (avg_g / avg_l if avg_l > 0 else 1e9))
    return out


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------
def macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    f = ema(close, fast)
    s = ema(close, slow)
    line = f - s
    sig = ema(line[~np.isnan(line)], signal)
    sig_full = np.full_like(line, np.nan)
    sig_full[-len(sig):] = sig
    hist = line - sig_full
    return line, sig_full, hist


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------
def bollinger(close: np.ndarray, period: int = 20, mult: float = 2.0):
    m = sma(close, period)
    # rolling std via cumulative trick
    n = len(close)
    out = np.full(n, np.nan)
    if n < period:
        return m, m, m
    c2 = close * close
    cs = np.concatenate(([0.0], np.cumsum(close)))
    cs2 = np.concatenate(([0.0], np.cumsum(c2)))
    for i in range(period - 1, n):
        s1 = cs[i + 1] - cs[i + 1 - period]
        s2 = cs2[i + 1] - cs2[i + 1 - period]
        var = s2 / period - (s1 / period) ** 2
        out[i] = np.sqrt(max(var, 0.0))
    return m, m + mult * out, m - mult * out


# ---------------------------------------------------------------------------
# CVD — tick-aware. If aggressive_buy_vol/aggressive_sell_vol arrays provided,
# uses real aggressor flow (correct tick-precise method). Otherwise falls back
# to OHLCV proxy (Bookmap-style close-position weighting).
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def cvd_from_ticks(buy_vol: np.ndarray, sell_vol: np.ndarray) -> np.ndarray:
    n = buy_vol.shape[0]
    out = np.empty(n, dtype=np.float64)
    s = 0.0
    for i in range(n):
        s += buy_vol[i] - sell_vol[i]
        out[i] = s
    return out


@njit(cache=True, fastmath=True)
def cvd_proxy(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
              close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    n = close.shape[0]
    out = np.empty(n, dtype=np.float64)
    s = 0.0
    for i in range(n):
        rng = high[i] - low[i]
        if rng <= 0:
            delta = 0.0
        else:
            buy = volume[i] * (close[i] - low[i]) / rng
            sell = volume[i] * (high[i] - close[i]) / rng
            delta = buy - sell
        s += delta
        out[i] = s
    return out


# ---------------------------------------------------------------------------
# VWAP & anchored VWAP with std-dev bands
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def anchored_vwap(price: np.ndarray, volume: np.ndarray,
                  anchor_idx: int) -> np.ndarray:
    n = price.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    pv = 0.0
    v = 0.0
    for i in range(anchor_idx, n):
        pv += price[i] * volume[i]
        v += volume[i]
        out[i] = pv / v if v > 0 else np.nan
    return out


@njit(cache=True, fastmath=True)
def vwap_bands(price: np.ndarray, volume: np.ndarray, anchor_idx: int,
               mult: float = 1.0):
    n = price.shape[0]
    vw = anchored_vwap(price, volume, anchor_idx)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    pv2 = 0.0
    v = 0.0
    pv = 0.0
    for i in range(anchor_idx, n):
        pv += price[i] * volume[i]
        pv2 += price[i] * price[i] * volume[i]
        v += volume[i]
        if v > 0:
            mean = pv / v
            var = pv2 / v - mean * mean
            sd = np.sqrt(max(var, 0.0))
            upper[i] = mean + mult * sd
            lower[i] = mean - mult * sd
    return vw, upper, lower


# ---------------------------------------------------------------------------
# ADX (Average Directional Index)
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average Directional Index (ADX) using Wilder's smoothing."""
    n = high.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period * 2:
        return out

    tr = true_range(high, low, close)

    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        elif down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    def _rma_inline(x, p):
        res = np.full(len(x), np.nan)
        if len(x) < p: return res
        s = 0.0
        for i in range(p):
            s += x[i]
        res[p - 1] = s / p
        for i in range(p, len(x)):
            res[i] = (res[i - 1] * (p - 1) + x[i]) / p
        return res

    tr_rma = _rma_inline(tr, period)
    plus_dm_rma = _rma_inline(plus_dm, period)
    minus_dm_rma = _rma_inline(minus_dm, period)

    plus_di = 100.0 * plus_dm_rma / np.maximum(tr_rma, 1e-12)
    minus_di = 100.0 * minus_dm_rma / np.maximum(tr_rma, 1e-12)

    dx = 100.0 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-12)
    # Fill NaNs in DX for RMA to work
    dx_filled = np.nan_to_num(dx, nan=0.0)
    adx_vals = _rma_inline(dx_filled, period)

    # Restore NaNs where DX was NaN
    adx_vals[np.isnan(dx)] = np.nan
    return adx_vals

# ---------------------------------------------------------------------------
# GPU-accelerated rolling z-score (uses CuPy if available)
# ---------------------------------------------------------------------------

def rolling_zscore(x: np.ndarray, period: int = 100, use_gpu: bool = True) -> np.ndarray:
    if use_gpu and _HAS_CUPY:
        gx = cp.asarray(x)
        n = gx.shape[0]
        out = cp.full(n, cp.nan)
        if n < period:
            return cp.asnumpy(out)
        # cumulative trick on GPU
        cs = cp.concatenate((cp.array([0.0]), cp.cumsum(gx)))
        cs2 = cp.concatenate((cp.array([0.0]), cp.cumsum(gx * gx)))
        idx = cp.arange(period - 1, n)
        s1 = cs[idx + 1] - cs[idx + 1 - period]
        s2 = cs2[idx + 1] - cs2[idx + 1 - period]
        mean = s1 / period
        var = s2 / period - mean * mean
        sd = cp.sqrt(cp.maximum(var, 0.0))
        z = (gx[idx] - mean) / cp.where(sd == 0, 1, sd)
        out[idx] = z
        return cp.asnumpy(out)
    # CPU path
    n = x.shape[0]
    out = np.full(n, np.nan)
    if n < period:
        return out
    cs = np.concatenate(([0.0], np.cumsum(x)))
    cs2 = np.concatenate(([0.0], np.cumsum(x * x)))
    for i in range(period - 1, n):
        s1 = cs[i + 1] - cs[i + 1 - period]
        s2 = cs2[i + 1] - cs2[i + 1 - period]
        m = s1 / period
        var = s2 / period - m * m
        sd = np.sqrt(max(var, 0.0))
        out[i] = (x[i] - m) / sd if sd > 0 else 0.0
    return out
