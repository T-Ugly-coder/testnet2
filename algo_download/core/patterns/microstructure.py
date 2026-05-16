"""
Microstructure & orderflow patterns — the layer pros use to confirm setups.

Implements:
  - CVD divergence vs price (regular & hidden)
  - Open Interest rate-of-change & divergence
  - Funding-rate z-score (crypto-perp specific)
  - Liquidation cascade detector
  - Kyle's lambda (rolling price-impact estimator)
  - Aggressive / passive flow ratio
  - Basis / premium z-score (perp vs spot)

All formulas referenced inline. NaN-padded outputs, length == input.
"""
from __future__ import annotations
import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Pivot detection helper — fractal swings of order k
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _pivots(x: np.ndarray, k: int = 3):
    """Return (idx, type) for fractal pivots: type=1 high, -1 low.
    A pivot at i requires x[i] strict extremum vs k bars on each side.
    """
    n = x.shape[0]
    out_i = np.empty(n, dtype=np.int64)
    out_t = np.empty(n, dtype=np.int8)
    cnt = 0
    for i in range(k, n - k):
        v = x[i]
        is_h = True
        is_l = True
        for j in range(1, k + 1):
            if x[i - j] >= v or x[i + j] >= v:
                is_h = False
            if x[i - j] <= v or x[i + j] <= v:
                is_l = False
            if not is_h and not is_l:
                break
        if is_h:
            out_i[cnt] = i; out_t[cnt] = 1; cnt += 1
        elif is_l:
            out_i[cnt] = i; out_t[cnt] = -1; cnt += 1
    return out_i[:cnt], out_t[:cnt]


# ---------------------------------------------------------------------------
# CVD divergence
# ---------------------------------------------------------------------------
# Regular bearish: price makes higher high, CVD makes lower high  -> short
# Regular bullish: price makes lower  low,  CVD makes higher low  -> long
# Hidden bullish:  price makes higher low,  CVD makes lower  low  -> trend cont.
# Hidden bearish:  price makes lower  high, CVD makes higher high -> trend cont.
@njit(cache=True, fastmath=True)
def cvd_divergence(price: np.ndarray, cvd: np.ndarray, k: int = 3,
                   min_bars: int = 5, max_bars: int = 100):
    """Returns (idx, code).
    code:  1 regular bull, -1 regular bear,
           2 hidden bull,  -2 hidden bear
    Detected at the *second* pivot, comparing the two most recent same-type
    pivots within [min_bars, max_bars] apart.
    """
    pi, pt = _pivots(price, k)
    npiv = pi.shape[0]
    out_i = np.empty(npiv, dtype=np.int64)
    out_c = np.empty(npiv, dtype=np.int8)
    cnt = 0
    # Walk pivots; for each, find prior pivot of the same type within range.
    for b in range(1, npiv):
        for a in range(b - 1, -1, -1):
            if pt[a] != pt[b]:
                continue
            gap = pi[b] - pi[a]
            if gap < min_bars:
                continue
            if gap > max_bars:
                break
            p_a = price[pi[a]]; p_b = price[pi[b]]
            c_a = cvd[pi[a]];   c_b = cvd[pi[b]]
            if pt[b] == 1:  # highs
                if p_b > p_a and c_b < c_a:
                    out_i[cnt] = pi[b]; out_c[cnt] = -1; cnt += 1
                elif p_b < p_a and c_b > c_a:
                    out_i[cnt] = pi[b]; out_c[cnt] = -2; cnt += 1
            else:  # lows
                if p_b < p_a and c_b > c_a:
                    out_i[cnt] = pi[b]; out_c[cnt] = 1; cnt += 1
                elif p_b > p_a and c_b < c_a:
                    out_i[cnt] = pi[b]; out_c[cnt] = 2; cnt += 1
            break
    return out_i[:cnt], out_c[:cnt]


# ---------------------------------------------------------------------------
# Open Interest rate-of-change & divergence
# ---------------------------------------------------------------------------
# OI semantics for perps:
#   price up + OI up   -> longs adding (bullish, but watch for crowding)
#   price up + OI down -> short covering (rally on weak hands closing)
#   price down + OI up -> shorts adding (bearish)
#   price down + OI down -> longs flushed
@njit(cache=True, fastmath=True)
def oi_regime(price: np.ndarray, oi: np.ndarray,
              window: int = 24) -> np.ndarray:
    """Return code per bar:
        1 longs adding, 2 short covering,
       -1 shorts adding, -2 longs flushed,
        0 neutral / NaN.
    Uses % change over `window` bars."""
    n = price.shape[0]
    out = np.zeros(n, dtype=np.int8)
    if n <= window:
        return out
    for i in range(window, n):
        if price[i - window] <= 0 or oi[i - window] <= 0:
            continue
        dp = (price[i] - price[i - window]) / price[i - window]
        do = (oi[i] - oi[i - window]) / oi[i - window]
        # Threshold: |dp| > 0.5%, |do| > 1% — tunable.
        if abs(dp) < 0.005 or abs(do) < 0.01:
            continue
        if dp > 0 and do > 0:
            out[i] = 1
        elif dp > 0 and do < 0:
            out[i] = 2
        elif dp < 0 and do > 0:
            out[i] = -1
        else:
            out[i] = -2
    return out


# ---------------------------------------------------------------------------
# Funding rate z-score (crypto perpetual swaps)
# ---------------------------------------------------------------------------
# Extreme positive z (crowded longs) -> contrarian short bias.
# Extreme negative z (crowded shorts) -> contrarian long bias.
@njit(cache=True, fastmath=True)
def funding_zscore(funding: np.ndarray, window: int = 168) -> np.ndarray:
    n = funding.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    for i in range(window - 1, n):
        s = i - window + 1
        m = 0.0
        for j in range(s, i + 1):
            m += funding[j]
        m /= window
        v = 0.0
        for j in range(s, i + 1):
            d = funding[j] - m
            v += d * d
        v /= window
        if v <= 0:
            continue
        out[i] = (funding[i] - m) / np.sqrt(v)
    return out


# ---------------------------------------------------------------------------
# Liquidation cascade detector
# ---------------------------------------------------------------------------
# A cascade is identified when within a short window:
#   - volume spikes >= vol_z standard deviations above rolling mean
#   - price moves >= price_atr * ATR
#   - move and aggressor flow are aligned in same direction
# Caller passes `signed_volume` = aggressive_buy - aggressive_sell (or proxy).
@njit(cache=True, fastmath=True)
def liquidation_cascades(close: np.ndarray, volume: np.ndarray,
                         signed_vol: np.ndarray, atr: np.ndarray,
                         window: int = 50, vol_z: float = 3.0,
                         price_atr: float = 2.0):
    """Returns (idx, side). side=1 long-liq cascade (down move), -1 short-liq."""
    n = close.shape[0]
    out_i = np.empty(n, dtype=np.int64)
    out_s = np.empty(n, dtype=np.int8)
    cnt = 0
    for i in range(window, n):
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue
        # Rolling mean / std of volume on window ending at i-1 (no leakage)
        m = 0.0
        for j in range(i - window, i):
            m += volume[j]
        m /= window
        v = 0.0
        for j in range(i - window, i):
            d = volume[j] - m
            v += d * d
        v /= window
        if v <= 0:
            continue
        sd = np.sqrt(v)
        z = (volume[i] - m) / sd
        if z < vol_z:
            continue
        dp = close[i] - close[i - 1]
        if abs(dp) < price_atr * a:
            continue
        # Direction must align with aggressor flow
        if dp < 0 and signed_vol[i] < 0:
            out_i[cnt] = i; out_s[cnt] = 1; cnt += 1   # long liquidations
        elif dp > 0 and signed_vol[i] > 0:
            out_i[cnt] = i; out_s[cnt] = -1; cnt += 1  # short liquidations
    return out_i[:cnt], out_s[:cnt]


# ---------------------------------------------------------------------------
# Kyle's lambda — rolling price-impact estimator
# ---------------------------------------------------------------------------
# Δp_t = λ * S_t + ε         where S_t = signed_volume_t
# OLS slope:  λ = Cov(Δp, S) / Var(S)
# High λ  -> illiquid, large impact per unit flow
# Low  λ  -> deep book, flow absorbs without moving price
@njit(cache=True, fastmath=True)
def kyle_lambda_rolling(close: np.ndarray, signed_vol: np.ndarray,
                        window: int = 50) -> np.ndarray:
    n = close.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window + 1:
        return out
    for i in range(window, n):
        s = i - window + 1
        # Means
        mp = 0.0; ms = 0.0
        for j in range(s, i + 1):
            mp += close[j] - close[j - 1]
            ms += signed_vol[j]
        mp /= window; ms /= window
        cov = 0.0; var = 0.0
        for j in range(s, i + 1):
            dp = (close[j] - close[j - 1]) - mp
            ds = signed_vol[j] - ms
            cov += dp * ds
            var += ds * ds
        if var <= 0:
            continue
        out[i] = cov / var
    return out


# ---------------------------------------------------------------------------
# Aggressive / passive flow ratio
# ---------------------------------------------------------------------------
# ratio = aggressive_volume / total_volume   in rolling window
# Persistent ratio > 0.6 with rising price = real demand (not just spoofing).
@njit(cache=True, fastmath=True)
def aggressive_ratio(aggressive_vol: np.ndarray, total_vol: np.ndarray,
                     window: int = 20) -> np.ndarray:
    n = aggressive_vol.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    for i in range(window - 1, n):
        a = 0.0; t = 0.0
        for j in range(i - window + 1, i + 1):
            a += aggressive_vol[j]
            t += total_vol[j]
        if t > 0:
            out[i] = a / t
    return out


# ---------------------------------------------------------------------------
# Basis / premium z-score (perp vs spot or futures vs spot)
# ---------------------------------------------------------------------------
# basis = (perp - spot) / spot
# Rolling z-score; extremes mark crowded positioning that mean-reverts.
@njit(cache=True, fastmath=True)
def basis_zscore(perp: np.ndarray, spot: np.ndarray,
                 window: int = 168) -> np.ndarray:
    n = perp.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    basis = np.empty(n, dtype=np.float64)
    for i in range(n):
        if spot[i] > 0:
            basis[i] = (perp[i] - spot[i]) / spot[i]
        else:
            basis[i] = np.nan
    for i in range(window - 1, n):
        s = i - window + 1
        m = 0.0; cnt = 0
        for j in range(s, i + 1):
            if not np.isnan(basis[j]):
                m += basis[j]; cnt += 1
        if cnt < window // 2:
            continue
        m /= cnt
        v = 0.0
        for j in range(s, i + 1):
            if not np.isnan(basis[j]):
                d = basis[j] - m
                v += d * d
        v /= cnt
        if v <= 0:
            continue
        if not np.isnan(basis[i]):
            out[i] = (basis[i] - m) / np.sqrt(v)
    return out
