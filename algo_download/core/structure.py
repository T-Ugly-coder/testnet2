"""
Market structure: vectorized swing detection, BOS, CHoCH.

Tick-precision: when fed tick-aggregated bars, swings reflect true microstructure.
For tick-bars instead of time-bars, see data/bars.py.
"""
from __future__ import annotations
import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def detect_swings(high: np.ndarray, low: np.ndarray,
                  left: int = 5, right: int = 5):
    """Fractal swing detection.

    Returns (swing_high_idx, swing_high_px, swing_low_idx, swing_low_px) arrays.
    A swing high at i requires high[i] strictly > all highs in [i-left, i+right] (excluding i).
    Symmetric for lows.
    """
    n = high.shape[0]
    sh_idx = np.empty(n, dtype=np.int64)
    sh_px = np.empty(n, dtype=np.float64)
    sl_idx = np.empty(n, dtype=np.int64)
    sl_px = np.empty(n, dtype=np.float64)
    nh = 0
    nl = 0
    for i in range(left, n - right):
        is_h = True
        is_l = True
        hi = high[i]
        lo = low[i]
        for j in range(i - left, i + right + 1):
            if j == i:
                continue
            if high[j] >= hi:
                is_h = False
            if low[j] <= lo:
                is_l = False
            if not is_h and not is_l:
                break
        if is_h:
            sh_idx[nh] = i
            sh_px[nh] = hi
            nh += 1
        if is_l:
            sl_idx[nl] = i
            sl_px[nl] = lo
            nl += 1
    return sh_idx[:nh], sh_px[:nh], sl_idx[:nl], sl_px[:nl]


def classify_structure(sh_px: np.ndarray, sl_px: np.ndarray) -> str:
    if len(sh_px) < 2 or len(sl_px) < 2:
        return "RANGE"
    hh = sh_px[-1] > sh_px[-2]
    hl = sl_px[-1] > sl_px[-2]
    lh = sh_px[-1] < sh_px[-2]
    ll = sl_px[-1] < sl_px[-2]
    if hh and hl:
        return "BULLISH"
    if lh and ll:
        return "BEARISH"
    return "RANGE"


@njit(cache=True, fastmath=True)
def bos_choch(close: np.ndarray, sh_idx: np.ndarray, sh_px: np.ndarray,
              sl_idx: np.ndarray, sl_px: np.ndarray,
              right_bars: int = 5):
    """Compute BOS and CHoCH events without look-ahead.

    A swing at index j is only KNOWN at index j + right_bars (because
    fractal detection needs `right` future bars to confirm). We therefore
    only consider a swing as available once i >= sh_idx[k] + right_bars.

    Returns:
      (events_idx, events_type) — type:
        1=bullish_bos, -1=bearish_bos, 2=bullish_choch, -2=bearish_choch
    """
    n = close.shape[0]
    events_idx = np.empty(n, dtype=np.int64)
    events_type = np.empty(n, dtype=np.int8)
    cnt = 0
    trend = 0  # 1 bull, -1 bear, 0 unknown
    last_sh_i = 0
    last_sl_i = 0
    nh = sh_px.shape[0]
    nl = sl_px.shape[0]
    for i in range(n):
        # advance only to swings already CONFIRMED (i.e. swing_idx + right <= i)
        while last_sh_i < nh and sh_idx[last_sh_i] + right_bars <= i:
            last_sh_i += 1
        while last_sl_i < nl and sl_idx[last_sl_i] + right_bars <= i:
            last_sl_i += 1
        if last_sh_i < 1 or last_sl_i < 1:
            continue
        recent_high = sh_px[last_sh_i - 1]
        recent_low = sl_px[last_sl_i - 1]
        if close[i] > recent_high:
            if trend == -1:
                events_idx[cnt] = i
                events_type[cnt] = 2  # bullish CHoCH
                cnt += 1
            else:
                events_idx[cnt] = i
                events_type[cnt] = 1  # bullish BOS
                cnt += 1
            trend = 1
        elif close[i] < recent_low:
            if trend == 1:
                events_idx[cnt] = i
                events_type[cnt] = -2
                cnt += 1
            else:
                events_idx[cnt] = i
                events_type[cnt] = -1
                cnt += 1
            trend = -1
    return events_idx[:cnt], events_type[:cnt]


@njit(cache=True, fastmath=True)
def displacement(open_: np.ndarray, close: np.ndarray, atr: np.ndarray,
                 mult: float = 2.0):
    """Detect displacement candles (body > mult * ATR)."""
    n = close.shape[0]
    idx = np.empty(n, dtype=np.int64)
    direction = np.empty(n, dtype=np.int8)
    strength = np.empty(n, dtype=np.float64)
    cnt = 0
    for i in range(n):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        body = abs(close[i] - open_[i])
        if body > mult * atr[i]:
            idx[cnt] = i
            direction[cnt] = 1 if close[i] > open_[i] else -1
            strength[cnt] = body / atr[i]
            cnt += 1
    return idx[:cnt], direction[:cnt], strength[:cnt]
