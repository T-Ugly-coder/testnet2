"""
Auction Market Theory: POC, VAH/VAL, Initial Balance, balance/imbalance.

Tick-precision: feed price-bin data from raw ticks for true volume profile.
"""
from __future__ import annotations
import numpy as np
from numba import njit


@njit(cache=True, fastmath=True, parallel=False)
def volume_profile_bars(low: np.ndarray, high: np.ndarray, volume: np.ndarray,
                        n_bins: int = 100):
    """Aggregate volume into price bins from OHLCV.

    Approximation: distribute candle volume uniformly across its [low, high].
    For tick precision use volume_profile_ticks instead.
    """
    pmin = low.min()
    pmax = high.max()
    if pmax <= pmin:
        return np.full(n_bins, 0.0), pmin, pmax
    bins = np.zeros(n_bins, dtype=np.float64)
    bin_size = (pmax - pmin) / n_bins
    for i in range(low.shape[0]):
        rng = high[i] - low[i]
        if rng <= 0 or volume[i] <= 0:
            continue
        b_lo = int((low[i] - pmin) / bin_size)
        b_hi = int((high[i] - pmin) / bin_size)
        if b_hi >= n_bins:
            b_hi = n_bins - 1
        if b_lo < 0:
            b_lo = 0
        nbins = b_hi - b_lo + 1
        per = volume[i] / nbins
        for b in range(b_lo, b_hi + 1):
            bins[b] += per
    return bins, pmin, pmax


@njit(cache=True, fastmath=True)
def volume_profile_ticks(prices: np.ndarray, sizes: np.ndarray,
                         n_bins: int = 200):
    """True volume profile from tick data."""
    pmin = prices.min()
    pmax = prices.max()
    if pmax <= pmin:
        return np.zeros(n_bins), pmin, pmax
    bins = np.zeros(n_bins, dtype=np.float64)
    bin_size = (pmax - pmin) / n_bins
    for i in range(prices.shape[0]):
        b = int((prices[i] - pmin) / bin_size)
        if b >= n_bins:
            b = n_bins - 1
        if b < 0:
            b = 0
        bins[b] += sizes[i]
    return bins, pmin, pmax


def value_area(bins: np.ndarray, pmin: float, pmax: float,
               coverage: float = 0.70):
    """Compute POC, VAH, VAL from a binned profile."""
    n_bins = bins.shape[0]
    bin_size = (pmax - pmin) / n_bins
    poc_bin = int(np.argmax(bins))
    poc_price = pmin + (poc_bin + 0.5) * bin_size
    total = bins.sum()
    target = total * coverage
    lo, hi = poc_bin, poc_bin
    acc = bins[poc_bin]
    while acc < target and (lo > 0 or hi < n_bins - 1):
        up = bins[hi + 1] if hi + 1 < n_bins else -1
        dn = bins[lo - 1] if lo - 1 >= 0 else -1
        if up >= dn and hi + 1 < n_bins:
            hi += 1; acc += bins[hi]
        elif lo - 1 >= 0:
            lo -= 1; acc += bins[lo]
        else:
            break
    return {
        "poc": poc_price,
        "val": pmin + (lo + 0.5) * bin_size,
        "vah": pmin + (hi + 0.5) * bin_size,
        "coverage": acc / total if total > 0 else 0.0,
    }


def initial_balance(high: np.ndarray, low: np.ndarray, ts: np.ndarray,
                    session_start_hour_utc: int = 13,
                    ib_minutes: int = 60):
    """Compute IB (first ``ib_minutes`` of session) high/low PER session day.

    ts: array of unix-ms timestamps (sorted ascending).
    The session day rolls at ``session_start_hour_utc`` UTC, so sessions that
    cross UTC midnight are kept contiguous.
    Returns list of per-session dicts.
    """
    n = ts.shape[0]
    if n == 0:
        return []
    ms_per_day = 86_400_000
    ms_per_hour = 3_600_000
    shifted = ts - int(session_start_hour_utc) * ms_per_hour
    sess_day = shifted // ms_per_day

    # Find session-day boundaries.
    boundaries = np.empty(n, dtype=np.bool_)
    boundaries[0] = True
    if n > 1:
        boundaries[1:] = sess_day[1:] != sess_day[:-1]
    starts = np.where(boundaries)[0]

    out = []
    for k in range(starts.shape[0]):
        s = int(starts[k])
        e_session = int(starts[k + 1]) if k + 1 < starts.shape[0] else n
        # IB window: first ib_minutes of this session.
        end_ts = ts[s] + ib_minutes * 60_000
        e = s
        while e < e_session and ts[e] <= end_ts:
            e += 1
        if e == s:
            continue
        ib_high = float(high[s:e].max())
        ib_low = float(low[s:e].min())
        if e < e_session:
            ext_up = bool((high[e:e_session] > ib_high).any())
            ext_dn = bool((low[e:e_session] < ib_low).any())
        else:
            ext_up = False
            ext_dn = False
        out.append({
            "session_start_ms": int(ts[s]),
            "session_day": int(sess_day[s]),
            "ib_high": ib_high,
            "ib_low": ib_low,
            "ib_range": ib_high - ib_low,
            "extended_up": ext_up,
            "extended_down": ext_dn,
            "ib_failure": ext_up and ext_dn,
        })
    return out
