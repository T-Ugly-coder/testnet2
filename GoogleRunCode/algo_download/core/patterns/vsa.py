"""
Volume Spread Analysis (VSA) — Tom Williams style.

Detects:
  - No Demand bar: narrow up-bar on low volume in uptrend (weakness)
  - No Supply bar: narrow down-bar on low volume in downtrend (strength)
  - Stopping Volume: wide-spread down-bar on ultra-high volume closing off lows
  - Climactic Volume: wide-spread up-bar on ultra-high volume closing off highs
  - Effort vs Result divergence (large vol, tiny range = absorption)
"""
from __future__ import annotations
import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def vsa_signals(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                close: np.ndarray, volume: np.ndarray,
                atr: np.ndarray, vol_ma: np.ndarray):
    """
    Returns (idx, code).
    code: 1=no_demand, -1=no_supply, 2=stopping_volume_bull,
          -2=climactic_volume_bear, 3=absorption_bull, -3=absorption_bear
    """
    n = close.shape[0]
    out_i = np.empty(n, dtype=np.int64)
    out_c = np.empty(n, dtype=np.int8)
    cnt = 0
    for i in range(1, n):
        a = atr[i]; vm = vol_ma[i]
        if np.isnan(a) or np.isnan(vm) or a <= 0 or vm <= 0:
            continue
        spread = high[i] - low[i]
        bull = close[i] > open_[i]
        bear = close[i] < open_[i]

        # No Demand: narrow up-bar low volume
        if bull and spread < 0.5 * a and volume[i] < 0.7 * vm and close[i] > close[i - 1]:
            out_i[cnt] = i; out_c[cnt] = 1; cnt += 1
        # No Supply
        if bear and spread < 0.5 * a and volume[i] < 0.7 * vm and close[i] < close[i - 1]:
            out_i[cnt] = i; out_c[cnt] = -1; cnt += 1
        # Stopping volume (bullish): wide down-bar, huge vol, close near high
        if bear and spread > 1.5 * a and volume[i] > 2.0 * vm and \
                (close[i] - low[i]) > 0.6 * spread:
            out_i[cnt] = i; out_c[cnt] = 2; cnt += 1
        # Climactic (bearish): wide up-bar, huge vol, close near low
        if bull and spread > 1.5 * a and volume[i] > 2.0 * vm and \
                (high[i] - close[i]) > 0.6 * spread:
            out_i[cnt] = i; out_c[cnt] = -2; cnt += 1
        # Absorption: huge vol, tiny range
        if spread < 0.4 * a and volume[i] > 2.0 * vm:
            if close[i] > open_[i]:
                out_i[cnt] = i; out_c[cnt] = 3; cnt += 1
            else:
                out_i[cnt] = i; out_c[cnt] = -3; cnt += 1
    return out_i[:cnt], out_c[:cnt]
