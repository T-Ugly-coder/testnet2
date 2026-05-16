"""
SMC primitives — Order Blocks, FVGs, Sweeps, Inducements.
Vectorized with numba; works on tick-aggregated bars for tick precision.
"""
from __future__ import annotations
import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Fair Value Gap detection (3-candle imbalance)
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def detect_fvg(high: np.ndarray, low: np.ndarray):
    """Return (idx, type, top, bottom). type: 1=bull, -1=bear."""
    n = high.shape[0]
    idx = np.empty(n, dtype=np.int64)
    typ = np.empty(n, dtype=np.int8)
    top = np.empty(n, dtype=np.float64)
    bot = np.empty(n, dtype=np.float64)
    cnt = 0
    for i in range(2, n):
        if low[i] > high[i - 2]:
            idx[cnt] = i - 1
            typ[cnt] = 1
            top[cnt] = low[i]
            bot[cnt] = high[i - 2]
            cnt += 1
        elif high[i] < low[i - 2]:
            idx[cnt] = i - 1
            typ[cnt] = -1
            top[cnt] = low[i - 2]
            bot[cnt] = high[i]
            cnt += 1
    return idx[:cnt], typ[:cnt], top[:cnt], bot[:cnt]


@njit(cache=True, fastmath=True)
def fvg_filled_mask(fvg_idx: np.ndarray, fvg_typ: np.ndarray,
                    fvg_top: np.ndarray, fvg_bot: np.ndarray,
                    high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Mark each FVG as fully filled by subsequent price action."""
    m = fvg_idx.shape[0]
    out = np.zeros(m, dtype=np.bool_)
    n = high.shape[0]
    for k in range(m):
        i0 = fvg_idx[k] + 2  # FVG confirmed on candle i+2 (third candle)
        for j in range(i0, n):
            if fvg_typ[k] == 1:  # bull FVG, filled if low <= bot
                if low[j] <= fvg_bot[k]:
                    out[k] = True
                    break
            else:  # bear FVG, filled if high >= top
                if high[j] >= fvg_top[k]:
                    out[k] = True
                    break
    return out


# ---------------------------------------------------------------------------
# Order Blocks: last opposing candle before displacement
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def detect_order_blocks(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                        close: np.ndarray, atr: np.ndarray,
                        disp_mult: float = 2.0,
                        min_body_atr: float = 0.2):
    """Return (idx, type, ob_high, ob_low). type: 1 bullish OB, -1 bearish OB.

    Requires the opposing candle to have a body >= min_body_atr * ATR so that
    a tiny doji isn't promoted to an OB.
    """
    n = close.shape[0]
    idx = np.empty(n, dtype=np.int64)
    typ = np.empty(n, dtype=np.int8)
    ob_h = np.empty(n, dtype=np.float64)
    ob_l = np.empty(n, dtype=np.float64)
    cnt = 0
    for i in range(1, n):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        body = abs(close[i] - open_[i])
        if body <= disp_mult * atr[i]:
            continue
        bullish_disp = close[i] > open_[i]
        # find last opposing candle in the prior 10 with sufficient body
        for j in range(i - 1, max(i - 11, -1), -1):
            opp_body = abs(close[j] - open_[j])
            if opp_body < min_body_atr * atr[i]:
                continue
            opp_bull = close[j] > open_[j]
            if bullish_disp and not opp_bull:
                idx[cnt] = j
                typ[cnt] = 1
                ob_h[cnt] = high[j]
                ob_l[cnt] = low[j]
                cnt += 1
                break
            if (not bullish_disp) and opp_bull:
                idx[cnt] = j
                typ[cnt] = -1
                ob_h[cnt] = high[j]
                ob_l[cnt] = low[j]
                cnt += 1
                break
    return idx[:cnt], typ[:cnt], ob_h[:cnt], ob_l[:cnt]


# ---------------------------------------------------------------------------
# Liquidity sweeps: wick beyond a swing level then close back
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def detect_sweeps(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  sh_idx: np.ndarray, sh_px: np.ndarray,
                  sl_idx: np.ndarray, sl_px: np.ndarray):
    """Return (bar_idx, side, level). side: 1=buy-side sweep (bearish),
    -1=sell-side sweep (bullish).

    Walks all swing levels prior to bar ``i`` and reports a sweep on the
    *first still-unswept* level whose wick gets pierced and reclaimed.
    A swing is marked swept once it has been pierced on either side, so we
    do not double-emit on a level that has already been taken out.
    """
    n = high.shape[0]
    out_i = np.empty(n, dtype=np.int64)
    out_s = np.empty(n, dtype=np.int8)
    out_l = np.empty(n, dtype=np.float64)
    cnt = 0
    nh = sh_px.shape[0]
    nl = sl_px.shape[0]
    swept_h = np.zeros(nh, dtype=np.bool_)
    swept_l = np.zeros(nl, dtype=np.bool_)
    for i in range(n):
        # Buy-side: most recent unswept swing high pierced and closed back below.
        for kh in range(nh - 1, -1, -1):
            if sh_idx[kh] >= i or swept_h[kh]:
                continue
            lvl = sh_px[kh]
            if high[i] > lvl:
                if close[i] < lvl:
                    out_i[cnt] = i
                    out_s[cnt] = 1
                    out_l[cnt] = lvl
                    cnt += 1
                swept_h[kh] = True
            break
        # Sell-side: most recent unswept swing low pierced and closed back above.
        for kl in range(nl - 1, -1, -1):
            if sl_idx[kl] >= i or swept_l[kl]:
                continue
            lvl = sl_px[kl]
            if low[i] < lvl:
                if close[i] > lvl:
                    out_i[cnt] = i
                    out_s[cnt] = -1
                    out_l[cnt] = lvl
                    cnt += 1
                swept_l[kl] = True
            break
    return out_i[:cnt], out_s[:cnt], out_l[:cnt]


# ---------------------------------------------------------------------------
# Inducement detection — corrected to use time-ordered swing sequence
# ---------------------------------------------------------------------------
def detect_inducement(swings_time_ordered, atr_at: np.ndarray,
                      max_depth_atr: float = 0.5):
    """
    swings_time_ordered: list of dicts {'type': 'H'|'L', 'idx': i, 'px': p}
    Returns list of {'idx', 'side', 'level'}.
    """
    out = []
    n = len(swings_time_ordered)
    for k in range(2, n):
        prev = swings_time_ordered[k - 2]
        cur = swings_time_ordered[k - 1]
        nxt = swings_time_ordered[k]
        if prev["type"] == "L" and cur["type"] == "L" and nxt["type"] == "H":
            depth = prev["px"] - cur["px"]
            if 0 < depth < max_depth_atr * atr_at[cur["idx"]]:
                if nxt["px"] > prev["px"]:
                    out.append({"idx": cur["idx"], "side": "sell_side",
                                "level": prev["px"]})
        elif prev["type"] == "H" and cur["type"] == "H" and nxt["type"] == "L":
            depth = cur["px"] - prev["px"]
            if 0 < depth < max_depth_atr * atr_at[cur["idx"]]:
                if nxt["px"] < prev["px"]:
                    out.append({"idx": cur["idx"], "side": "buy_side",
                                "level": prev["px"]})
    return out


# ---------------------------------------------------------------------------
# Equal highs/lows — liquidity pools
# ---------------------------------------------------------------------------
def equal_levels(prices: np.ndarray, tol: float):
    """Cluster nearly-equal swing prices."""
    if prices.size == 0:
        return []
    sorted_idx = np.argsort(prices)
    sp = prices[sorted_idx]
    clusters = []
    cur = [sp[0]]
    for p in sp[1:]:
        if p - cur[-1] <= tol:
            cur.append(p)
        else:
            if len(cur) >= 2:
                clusters.append({"level": float(np.mean(cur)),
                                 "count": len(cur)})
            cur = [p]
    if len(cur) >= 2:
        clusters.append({"level": float(np.mean(cur)), "count": len(cur)})
    return clusters
