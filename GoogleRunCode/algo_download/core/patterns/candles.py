"""
Candlestick patterns — corrected from spec.

Fixes:
- Three White Soldiers / Three Black Crows: opens must be within prior body.
- Strict gap conditions for kicker.
- Tier 1 vs Tier 2 retained.
- All return arrays of (index, signal).
"""
from __future__ import annotations
import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def detect_candles(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                   close: np.ndarray, atr: np.ndarray):
    """
    Returns flat arrays: (idx, code, signal, strength).
    code:
       1=bull_engulf, -1=bear_engulf
       2=morning_star, -2=evening_star
       3=bull_kicker, -3=bear_kicker
       4=hammer, -4=shooting_star
       5=piercing, -5=dark_cloud
       6=dragonfly_doji, -6=gravestone_doji
       7=three_white_soldiers, -7=three_black_crows
       8=tweezer_bottom, -8=tweezer_top
    signal: 1 long, -1 short
    """
    n = close.shape[0]
    # Up to 16 independent pattern checks may fire on a single bar.
    cap = n * 16
    out_i = np.empty(cap, dtype=np.int64)
    out_c = np.empty(cap, dtype=np.int8)
    out_s = np.empty(cap, dtype=np.int8)
    out_g = np.empty(cap, dtype=np.float64)
    cnt = 0
    for i in range(2, n):
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue
        o0 = open_[i - 2]; h0 = high[i - 2]; l0 = low[i - 2]; c0 = close[i - 2]
        o1 = open_[i - 1]; h1 = high[i - 1]; l1 = low[i - 1]; c1 = close[i - 1]
        o2 = open_[i];     h2 = high[i];     l2 = low[i];     c2 = close[i]
        b0 = abs(c0 - o0); b1 = abs(c1 - o1); b2 = abs(c2 - o2)
        r2 = h2 - l2
        bull0 = c0 > o0; bull1 = c1 > o1; bull2 = c2 > o2

        # Bullish engulfing
        if (not bull1) and bull2 and o2 <= c1 and c2 >= o1 and \
                b2 > 1.2 * b1 and b2 > 0.6 * a:
            out_i[cnt] = i; out_c[cnt] = 1; out_s[cnt] = 1; out_g[cnt] = b2 / a; cnt += 1
        # Bearish engulfing
        if bull1 and (not bull2) and o2 >= c1 and c2 <= o1 and \
                b2 > 1.2 * b1 and b2 > 0.6 * a:
            out_i[cnt] = i; out_c[cnt] = -1; out_s[cnt] = -1; out_g[cnt] = b2 / a; cnt += 1
        # Morning star
        if (not bull0) and b0 > 0.5 * a and b1 < 0.3 * a and bull2 and b2 > 0.5 * a \
                and c2 > 0.5 * (o0 + c0):
            out_i[cnt] = i; out_c[cnt] = 2; out_s[cnt] = 1; out_g[cnt] = b2 / a; cnt += 1
        # Evening star
        if bull0 and b0 > 0.5 * a and b1 < 0.3 * a and (not bull2) and b2 > 0.5 * a \
                and c2 < 0.5 * (o0 + c0):
            out_i[cnt] = i; out_c[cnt] = -2; out_s[cnt] = -1; out_g[cnt] = b2 / a; cnt += 1
        # Bullish kicker (gap above prior open after bearish c1)
        if (not bull1) and bull2 and o2 > h1 and b2 > 0.8 * a:
            out_i[cnt] = i; out_c[cnt] = 3; out_s[cnt] = 1; out_g[cnt] = b2 / a; cnt += 1
        # Bearish kicker
        if bull1 and (not bull2) and o2 < l1 and b2 > 0.8 * a:
            out_i[cnt] = i; out_c[cnt] = -3; out_s[cnt] = -1; out_g[cnt] = b2 / a; cnt += 1
        # Hammer
        if r2 > 0 and bull2 and b2 < 0.35 * r2 and (o2 - l2) > 2 * b2 and (h2 - c2) < 0.5 * b2:
            out_i[cnt] = i; out_c[cnt] = 4; out_s[cnt] = 1; out_g[cnt] = (o2 - l2) / a; cnt += 1
        # Shooting star
        if r2 > 0 and (not bull2) and b2 < 0.35 * r2 and (h2 - o2) > 2 * b2 and (c2 - l2) < 0.5 * b2:
            out_i[cnt] = i; out_c[cnt] = -4; out_s[cnt] = -1; out_g[cnt] = (h2 - o2) / a; cnt += 1
        # Piercing
        if (not bull1) and bull2 and o2 < l1 and c2 > 0.5 * (o1 + c1) and c2 < o1:
            out_i[cnt] = i; out_c[cnt] = 5; out_s[cnt] = 1; out_g[cnt] = b2 / a; cnt += 1
        # Dark cloud
        if bull1 and (not bull2) and o2 > h1 and c2 < 0.5 * (o1 + c1) and c2 > o1:
            out_i[cnt] = i; out_c[cnt] = -5; out_s[cnt] = -1; out_g[cnt] = b2 / a; cnt += 1
        # Dragonfly doji
        if r2 > 0 and b2 < 0.1 * r2 and r2 > 0.5 * a and \
                (min(o2, c2) - l2) > 0.7 * r2:
            out_i[cnt] = i; out_c[cnt] = 6; out_s[cnt] = 1; out_g[cnt] = r2 / a; cnt += 1
        # Gravestone doji
        if r2 > 0 and b2 < 0.1 * r2 and r2 > 0.5 * a and \
                (h2 - max(o2, c2)) > 0.7 * r2:
            out_i[cnt] = i; out_c[cnt] = -6; out_s[cnt] = -1; out_g[cnt] = r2 / a; cnt += 1
        # Three white soldiers — opens within prior body
        if bull0 and bull1 and bull2 and \
                b0 > 0.3 * a and b1 > 0.3 * a and b2 > 0.3 * a and \
                (o0 < o1) and (o1 < c0) and (o1 < o2) and (o2 < c1) and \
                c1 > c0 and c2 > c1:
            out_i[cnt] = i; out_c[cnt] = 7; out_s[cnt] = 1
            out_g[cnt] = (b0 + b1 + b2) / (3.0 * a); cnt += 1
        # Three black crows — opens within prior body
        if (not bull0) and (not bull1) and (not bull2) and \
                b0 > 0.3 * a and b1 > 0.3 * a and b2 > 0.3 * a and \
                (c0 < o1) and (o1 < o0) and (c1 < o2) and (o2 < o1) and \
                c1 < c0 and c2 < c1:
            out_i[cnt] = i; out_c[cnt] = -7; out_s[cnt] = -1
            out_g[cnt] = (b0 + b1 + b2) / (3.0 * a); cnt += 1
        # Tweezer bottom
        if (not bull1) and bull2 and abs(l1 - l2) < 0.05 * a:
            out_i[cnt] = i; out_c[cnt] = 8; out_s[cnt] = 1; out_g[cnt] = 1.0; cnt += 1
        # Tweezer top
        if bull1 and (not bull2) and abs(h1 - h2) < 0.05 * a:
            out_i[cnt] = i; out_c[cnt] = -8; out_s[cnt] = -1; out_g[cnt] = 1.0; cnt += 1

    return out_i[:cnt], out_c[:cnt], out_s[:cnt], out_g[:cnt]


def filter_at_poi(idx: np.ndarray, signal: np.ndarray,
                  bar_low: np.ndarray, bar_high: np.ndarray,
                  pois: list[tuple[int, float, float]]) -> np.ndarray:
    """Keep only patterns whose bar overlaps an active POI zone.

    pois: list of (start_idx, low, high) — POI active from start_idx onward.
    A pattern at bar i is kept if any POI with start <= i has its [low,high]
    overlapping the pattern bar's [low,high].
    """
    keep = np.zeros(idx.shape[0], dtype=bool)
    for k in range(idx.shape[0]):
        i = int(idx[k])
        bl = bar_low[i]
        bh = bar_high[i]
        for start, lo, hi in pois:
            if i < start:
                continue
            # overlap test
            if bh >= lo and bl <= hi:
                keep[k] = True
                break
    return keep
