"""
Wyckoff phases: accumulation, distribution, spring, upthrust.

Approximation — a real Wyckoff schematic is subjective. We detect:
  - Trading range (TR) via volatility contraction
  - Spring: false breakdown of TR low followed by reclaim
  - Upthrust After Distribution (UTAD): false breakout of TR high then close back
  - Sign of Strength (SOS) / Sign of Weakness (SOW)
"""
from __future__ import annotations
import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def detect_trading_range(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                         atr: np.ndarray, lookback: int = 30,
                         compression: float = 0.6):
    """Mark indices where price is in a contraction TR.

    Returns boolean mask. A bar is in TR if range over last `lookback`
    is < compression * average range over the period before that.
    """
    n = high.shape[0]
    out = np.zeros(n, dtype=np.bool_)
    for i in range(2 * lookback, n):
        recent = 0.0
        for j in range(i - lookback, i):
            recent += high[j] - low[j]
        prior = 0.0
        for j in range(i - 2 * lookback, i - lookback):
            prior += high[j] - low[j]
        if prior > 0 and (recent / prior) < compression:
            out[i] = True
    return out


@njit(cache=True, fastmath=True)
def detect_springs_upthrusts(high: np.ndarray, low: np.ndarray,
                             close: np.ndarray, in_tr: np.ndarray,
                             lookback: int = 30):
    """Spring (bull): break TR low intrabar, close back inside.
    Upthrust (bear): break TR high intrabar, close back inside.

    Returns (idx, type): 1=spring, -1=upthrust.
    """
    n = high.shape[0]
    out_i = np.empty(n, dtype=np.int64)
    out_t = np.empty(n, dtype=np.int8)
    cnt = 0
    for i in range(lookback, n):
        if not in_tr[i]:
            continue
        tr_low = low[i - lookback]
        tr_high = high[i - lookback]
        for j in range(i - lookback + 1, i):
            if low[j] < tr_low:
                tr_low = low[j]
            if high[j] > tr_high:
                tr_high = high[j]
        if low[i] < tr_low and close[i] > tr_low:
            out_i[cnt] = i; out_t[cnt] = 1; cnt += 1
        elif high[i] > tr_high and close[i] < tr_high:
            out_i[cnt] = i; out_t[cnt] = -1; cnt += 1
    return out_i[:cnt], out_t[:cnt]


@njit(cache=True, fastmath=True)
def sign_of_strength_weakness(open_: np.ndarray, close: np.ndarray,
                              volume: np.ndarray,
                              atr: np.ndarray, vol_ma: np.ndarray):
    """SOS: large bull bar (close - open) on rising volume after spring.
    SOW: large bear bar on rising volume after upthrust.
    Returns (idx, type): 1=SOS, -1=SOW.
    """
    n = close.shape[0]
    out_i = np.empty(n, dtype=np.int64)
    out_t = np.empty(n, dtype=np.int8)
    cnt = 0
    for i in range(1, n):
        if np.isnan(atr[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or atr[i] <= 0:
            continue
        body = close[i] - open_[i]
        if body > 1.5 * atr[i] and volume[i] > 1.5 * vol_ma[i]:
            out_i[cnt] = i; out_t[cnt] = 1; cnt += 1
        elif body < -1.5 * atr[i] and volume[i] > 1.5 * vol_ma[i]:
            out_i[cnt] = i; out_t[cnt] = -1; cnt += 1
    return out_i[:cnt], out_t[:cnt]
