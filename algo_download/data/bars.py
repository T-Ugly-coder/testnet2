"""
Tick-to-bar conversion: time, volume, dollar, range, and renko bars.

Tick precision unlocks proper bar types beyond time-bars:
  - Volume bars: each bar contains exactly N units of base volume
  - Dollar bars: each bar contains exactly $N notional traded
  - Range bars: each bar moves exactly R price units before closing
  - Renko: filtered noise via brick size

Reference: López de Prado, "Advances in Financial ML" — these reduce
heteroskedasticity and produce more iid samples for ML/backtest.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True, fastmath=True)
def _agg(ts: np.ndarray, px: np.ndarray, qty: np.ndarray,
         is_buyer_maker: np.ndarray, threshold: np.ndarray, mode: int):
    """Generic bar aggregator.

    mode:
        0 = volume bars (threshold = base qty)
        1 = dollar bars (threshold = price * qty)
        2 = range bars (threshold = price range)
    """
    n = ts.shape[0]
    out_ts = np.empty(n, dtype=np.int64)
    out_o = np.empty(n, dtype=np.float64)
    out_h = np.empty(n, dtype=np.float64)
    out_l = np.empty(n, dtype=np.float64)
    out_c = np.empty(n, dtype=np.float64)
    out_v = np.empty(n, dtype=np.float64)
    out_buy = np.empty(n, dtype=np.float64)
    out_sell = np.empty(n, dtype=np.float64)
    cnt = 0
    cur_o = px[0]; cur_h = px[0]; cur_l = px[0]; cur_c = px[0]
    cur_v = 0.0; cur_buy = 0.0; cur_sell = 0.0; cur_dollar = 0.0
    cur_ts = ts[0]
    thr_idx = 0
    for i in range(n):
        p = px[i]; q = qty[i]
        if p > cur_h:
            cur_h = p
        if p < cur_l:
            cur_l = p
        cur_c = p
        cur_v += q
        cur_dollar += p * q
        if is_buyer_maker[i]:
            cur_sell += q  # taker is seller
        else:
            cur_buy += q
        thr = threshold[thr_idx] if thr_idx < threshold.shape[0] else threshold[-1]
        crossed = False
        if mode == 0 and cur_v >= thr:
            crossed = True
        elif mode == 1 and cur_dollar >= thr:
            crossed = True
        elif mode == 2 and (cur_h - cur_l) >= thr:
            crossed = True
        if crossed:
            out_ts[cnt] = ts[i]; out_o[cnt] = cur_o; out_h[cnt] = cur_h
            out_l[cnt] = cur_l; out_c[cnt] = cur_c; out_v[cnt] = cur_v
            out_buy[cnt] = cur_buy; out_sell[cnt] = cur_sell
            cnt += 1
            cur_o = p; cur_h = p; cur_l = p; cur_c = p
            cur_v = 0.0; cur_buy = 0.0; cur_sell = 0.0; cur_dollar = 0.0
    return (out_ts[:cnt], out_o[:cnt], out_h[:cnt], out_l[:cnt],
            out_c[:cnt], out_v[:cnt], out_buy[:cnt], out_sell[:cnt])


def make_bars(ticks: pd.DataFrame, mode: str = "volume",
              threshold: float = 100.0) -> pd.DataFrame:
    """ticks columns: ts_ms, price, qty, is_buyer_maker."""
    ts = ticks["ts_ms"].to_numpy(np.int64)
    px = ticks["price"].to_numpy(np.float64)
    qty = ticks["qty"].to_numpy(np.float64)
    bm = ticks["is_buyer_maker"].to_numpy(np.bool_)
    mode_i = {"volume": 0, "dollar": 1, "range": 2}[mode]
    thr = np.array([threshold], dtype=np.float64)
    t, o, h, l, c, v, b, s = _agg(ts, px, qty, bm, thr, mode_i)
    return pd.DataFrame({
        "ts_ms": t, "open": o, "high": h, "low": l, "close": c,
        "volume": v, "buy_vol": b, "sell_vol": s,
    })


def time_bars(ticks: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    df = ticks.copy()
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df["dollar"] = df["price"] * df["qty"]
    df["buy_vol"] = np.where(~df["is_buyer_maker"], df["qty"], 0.0)
    df["sell_vol"] = np.where(df["is_buyer_maker"], df["qty"], 0.0)
    g = df.set_index("ts").groupby(pd.Grouper(freq=freq))
    out = pd.DataFrame({
        "open": g["price"].first(),
        "high": g["price"].max(),
        "low": g["price"].min(),
        "close": g["price"].last(),
        "volume": g["qty"].sum(),
        "dollar_volume": g["dollar"].sum(),
        "buy_vol": g["buy_vol"].sum(),
        "sell_vol": g["sell_vol"].sum(),
        "n_trades": g.size(),
    }).dropna()
    out["ts_ms"] = (out.index.astype("int64") // 1_000_000).astype(np.int64)
    return out.reset_index(drop=True)
