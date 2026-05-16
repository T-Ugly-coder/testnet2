"""
Swing Failure Pattern (SFP) — Tom Dante / TopStepTrader style.

An SFP forms when price wicks beyond a key swing level then closes back inside,
trapping breakout traders. Stronger than a generic sweep when accompanied by
divergence (CVD or momentum) and timeframe context.
"""
from __future__ import annotations
import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def detect_sfp(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               sh_idx: np.ndarray, sh_px: np.ndarray,
               sl_idx: np.ndarray, sl_px: np.ndarray,
               wick_min_atr: float, atr: np.ndarray):
    """
    Returns (idx, side, level, wick_size).
    side: 1 = bearish SFP (wick above resistance), -1 = bullish SFP.

    Each swing level is consumed (marked swept) the first time it is
    pierced, so subsequent bars do not re-emit on the same level.
    """
    n = high.shape[0]
    out_i = np.empty(n, dtype=np.int64)
    out_s = np.empty(n, dtype=np.int8)
    out_l = np.empty(n, dtype=np.float64)
    out_w = np.empty(n, dtype=np.float64)
    cnt = 0
    nh = sh_px.shape[0]
    nl = sl_px.shape[0]
    swept_h = np.zeros(nh, dtype=np.bool_)
    swept_l = np.zeros(nl, dtype=np.bool_)
    for i in range(n):
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue
        for kh in range(nh - 1, -1, -1):
            if sh_idx[kh] >= i or swept_h[kh]:
                continue
            lvl = sh_px[kh]
            if high[i] > lvl:
                if close[i] < lvl:
                    wick = high[i] - lvl
                    if wick > wick_min_atr * a:
                        out_i[cnt] = i; out_s[cnt] = 1
                        out_l[cnt] = lvl; out_w[cnt] = wick; cnt += 1
                swept_h[kh] = True
            break
        for kl in range(nl - 1, -1, -1):
            if sl_idx[kl] >= i or swept_l[kl]:
                continue
            lvl = sl_px[kl]
            if low[i] < lvl:
                if close[i] > lvl:
                    wick = lvl - low[i]
                    if wick > wick_min_atr * a:
                        out_i[cnt] = i; out_s[cnt] = -1
                        out_l[cnt] = lvl; out_w[cnt] = wick; cnt += 1
                swept_l[kl] = True
            break
    return out_i[:cnt], out_s[:cnt], out_l[:cnt], out_w[:cnt]
