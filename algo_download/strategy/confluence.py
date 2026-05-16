"""
Confluence strategy — bias + sweep + trigger.

Causality contract:
    signal[i], sl[i], tp[i] are computed using ONLY bars 0..i. The engine
    fills at open[i+1] using these values, so there is no look-ahead.

Premise (intentionally simple — extend with additional confluence layers):
  1. Bias from the latest confirmed BOS/CHoCH (structure trend).
  2. Trigger on a bias-aligned Swing Failure Pattern (SFP) at bar i AND a
     bias-aligned candle pattern at bar i.
  3. SL beyond the wick by `sl_buffer_atr * ATR(i)`; TP at fixed R-multiple.

Look-ahead handling for swings:
    detect_swings produces swing arrays whose index `j` is only KNOWN at
    bar `j + swing_right` (need `right` future bars to confirm a fractal).
    bos_choch gates this internally. detect_sfp does NOT, so we shift swing
    indices by swing_right before calling detect_sfp. After the shift, the
    "swing available" condition `sh_idx_known[k] >= i` correctly excludes
    swings that have not yet been confirmed at bar i.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.indicators import atr_wilder
from ..core.structure import detect_swings, bos_choch
from ..core.patterns.candles import detect_candles
from ..core.patterns.sfp import detect_sfp


def _trend_from_events(n: int, ev_idx: np.ndarray,
                       ev_type: np.ndarray) -> np.ndarray:
    """Forward-fill structural trend (+1 bull, -1 bear, 0 unknown).

    ev_type from bos_choch: 1 = bull BOS, 2 = bull CHoCH, -1 = bear BOS,
    -2 = bear CHoCH. We collapse to sign.
    """
    out = np.zeros(n, dtype=np.int8)
    j = 0
    cur = np.int8(0)
    for i in range(n):
        while j < ev_idx.shape[0] and ev_idx[j] <= i:
            cur = np.int8(1) if ev_type[j] > 0 else np.int8(-1)
            j += 1
        out[i] = cur
    return out


def _bool_mask(idx: np.ndarray, side: np.ndarray, n: int,
               want_side: int) -> np.ndarray:
    """Return length-n boolean mask: True at indices where side == want_side."""
    out = np.zeros(n, dtype=np.bool_)
    for k in range(idx.shape[0]):
        if side[k] == want_side:
            out[idx[k]] = True
    return out


def build_signals(bars: pd.DataFrame, *,
                  atr_period: int = 14,
                  swing_left: int = 5,
                  swing_right: int = 5,
                  rr: float = 2.0,
                  sl_buffer_atr: float = 0.5,
                  wick_min_atr: float = 0.5) -> dict:
    """Build causal (signal, sl, tp) arrays from OHLC bars.

    Parameters
    ----------
    bars : DataFrame with columns open, high, low, close.
    atr_period : Wilder ATR period.
    swing_left/right : fractal swing detection windows.
    rr : reward:risk multiple for TP.
    sl_buffer_atr : extra ATR buffer placed beyond the wick for SL.
    wick_min_atr : minimum SFP wick size (in ATRs) to qualify.

    Returns
    -------
    dict with keys:
        signal : np.int8 array, +1 long / -1 short / 0 nothing
        sl, tp : np.float64 arrays (NaN where signal == 0)
    """
    o = bars["open"].to_numpy(np.float64)
    h = bars["high"].to_numpy(np.float64)
    l = bars["low"].to_numpy(np.float64)
    c = bars["close"].to_numpy(np.float64)
    n = c.shape[0]

    if n == 0:
        return {
            "signal": np.zeros(0, dtype=np.int8),
            "sl": np.zeros(0, dtype=np.float64),
            "tp": np.zeros(0, dtype=np.float64),
            "rr": np.zeros(0, dtype=np.float64),
        }

    atr = atr_wilder(h, l, c, atr_period)

    sh_idx, sh_px, sl_idx, sl_px = detect_swings(h, l, swing_left, swing_right)

    # Structural trend from BOS/CHoCH (bos_choch is already causal).
    ev_idx, ev_type = bos_choch(c, sh_idx, sh_px, sl_idx, sl_px,
                                right_bars=swing_right)
    trend = _trend_from_events(n, ev_idx, ev_type)

    # Causal SFP: shift swing indices by swing_right so a swing only becomes
    # available for matching once it has been confirmed.
    sh_idx_known = sh_idx + swing_right
    sl_idx_known = sl_idx + swing_right

    sfp_idx, sfp_side, sfp_lvl, sfp_wick = detect_sfp(
        h, l, c,
        sh_idx_known, sh_px, sl_idx_known, sl_px,
        wick_min_atr, atr,
    )

    # Candle patterns at bar i use only bars i-2, i-1, i — already causal.
    cd_idx, cd_code, cd_signal, cd_strength = detect_candles(o, h, l, c, atr)

    bull_pat = _bool_mask(cd_idx, cd_signal, n, 1)
    bear_pat = _bool_mask(cd_idx, cd_signal, n, -1)

    # detect_sfp.side: -1 = bullish SFP (sweep low + close back above),
    #                   +1 = bearish SFP (sweep high + close back below).
    bull_sfp = _bool_mask(sfp_idx, sfp_side, n, -1)
    bear_sfp = _bool_mask(sfp_idx, sfp_side, n, 1)

    signal = np.zeros(n, dtype=np.int8)
    sl_out = np.full(n, np.nan, dtype=np.float64)
    tp_out = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        a = atr[i]
        if not np.isfinite(a) or a <= 0.0:
            continue

        if trend[i] == 1 and bull_sfp[i] and bull_pat[i]:
            entry_est = c[i]
            sl_v = l[i] - sl_buffer_atr * a
            if not (sl_v < entry_est):
                continue
            tp_v = entry_est + rr * (entry_est - sl_v)
            signal[i] = 1
            sl_out[i] = sl_v
            tp_out[i] = tp_v

        elif trend[i] == -1 and bear_sfp[i] and bear_pat[i]:
            entry_est = c[i]
            sl_v = h[i] + sl_buffer_atr * a
            if not (sl_v > entry_est):
                continue
            tp_v = entry_est - rr * (sl_v - entry_est)
            signal[i] = -1
            sl_out[i] = sl_v
            tp_out[i] = tp_v

    return {"signal": signal, "sl": sl_out, "tp": tp_out,
            "rr": np.full(n, float(rr), dtype=np.float64)}
