"""
Regime classification — what professionals run BEFORE any pattern fires.

Implements:
  - Hurst exponent (rolling, R/S analysis)         [Mandelbrot 1972]
  - Variance Ratio test (Lo–MacKinlay 1988)
  - Lag-1 autocorrelation of returns (rolling)
  - ATR percentile rank (volatility regime)
  - Realized volatility (Parkinson 1980 high-low estimator)
  - Trend strength via linear-regression R^2 of log price
  - Composite regime classifier

Math references inline. All windowed functions return arrays length == input
with NaN-padding for the warm-up period.
"""
from __future__ import annotations
import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Hurst exponent — Rescaled Range (R/S) analysis
# ---------------------------------------------------------------------------
# For a series x of length N, partition into k non-overlapping sub-series of
# length n. For each sub-series:
#   y_i = x_i - mean
#   Z_i = cumulative sum of y
#   R   = max(Z) - min(Z)
#   S   = std(x) (population)
#   (R/S)_n averaged over k sub-series
# Then log(R/S) vs log(n) regression slope == H.
#   H ≈ 0.5  random walk
#   H >  0.5 persistent / trending
#   H <  0.5 anti-persistent / mean-reverting
@njit(cache=True, fastmath=True)
def _rs_at_lag(returns: np.ndarray, lag: int) -> float:
    """Average R/S statistic over non-overlapping windows of size `lag`."""
    n = returns.shape[0]
    k = n // lag
    if k < 1:
        return np.nan
    rs_sum = 0.0
    valid = 0
    for w in range(k):
        s = w * lag
        e = s + lag
        # mean
        m = 0.0
        for i in range(s, e):
            m += returns[i]
        m /= lag
        # mean-adjusted cumulative deviations + variance
        z = 0.0
        zmin = 0.0
        zmax = 0.0
        var = 0.0
        for i in range(s, e):
            d = returns[i] - m
            z += d
            if z > zmax:
                zmax = z
            if z < zmin:
                zmin = z
            var += d * d
        var /= lag
        if var <= 0:
            continue
        sd = np.sqrt(var)
        rng = zmax - zmin
        if rng <= 0:
            continue
        rs_sum += rng / sd
        valid += 1
    if valid == 0:
        return np.nan
    return rs_sum / valid


@njit(cache=True, fastmath=True)
def hurst_rolling(log_returns: np.ndarray, window: int = 128) -> np.ndarray:
    """Rolling Hurst exponent via R/S analysis.

    Uses sub-series sizes [window/16, window/8, window/4, window/2].
    H is the slope of log(R/S) vs log(lag).
    """
    n = log_returns.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 16 or n < window:
        return out
    lags = np.array([window // 16, window // 8, window // 4, window // 2],
                    dtype=np.int64)
    nl = lags.shape[0]
    log_lags = np.log(lags.astype(np.float64))
    # Pre-compute sums for slope calc
    mean_log_lag = 0.0
    for j in range(nl):
        mean_log_lag += log_lags[j]
    mean_log_lag /= nl
    denom = 0.0
    for j in range(nl):
        d = log_lags[j] - mean_log_lag
        denom += d * d
    if denom <= 0:
        return out
    for i in range(window - 1, n):
        seg = log_returns[i - window + 1: i + 1]
        # Compute R/S at each lag
        log_rs = np.empty(nl, dtype=np.float64)
        ok = True
        mean_log_rs = 0.0
        for j in range(nl):
            rs = _rs_at_lag(seg, int(lags[j]))
            if np.isnan(rs) or rs <= 0:
                ok = False
                break
            log_rs[j] = np.log(rs)
            mean_log_rs += log_rs[j]
        if not ok:
            continue
        mean_log_rs /= nl
        num = 0.0
        for j in range(nl):
            num += (log_lags[j] - mean_log_lag) * (log_rs[j] - mean_log_rs)
        out[i] = num / denom
    return out


# ---------------------------------------------------------------------------
# Variance Ratio test (Lo–MacKinlay 1988)
# ---------------------------------------------------------------------------
# VR(q) = Var(r_t(q)) / (q * Var(r_t))    where r_t(q) = sum_{j=0}^{q-1} r_{t-j}
# Under random walk null:  VR(q) = 1
#   VR > 1  positive serial correlation (trending / momentum)
#   VR < 1  negative serial correlation (mean-reversion)
# Standard form uses overlapping q-period returns for efficiency.
@njit(cache=True, fastmath=True)
def variance_ratio_rolling(log_returns: np.ndarray, q: int = 4,
                           window: int = 128) -> np.ndarray:
    n = log_returns.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if q < 2 or window < q * 4 or n < window:
        return out
    for i in range(window - 1, n):
        s = i - window + 1
        # Var(r_1) — population variance of single-period returns
        m1 = 0.0
        for j in range(s, i + 1):
            m1 += log_returns[j]
        m1 /= window
        v1 = 0.0
        for j in range(s, i + 1):
            d = log_returns[j] - m1
            v1 += d * d
        v1 /= window
        if v1 <= 0:
            continue
        # Var(r_q) — overlapping q-period returns, centered at q * μ̂ (the null
        # mean, per Lo–MacKinlay; this is intentional, NOT the empirical mean
        # of q-sums).
        nq = window - q + 1
        mq = q * m1
        vq = 0.0
        # rolling q-sum
        rq = 0.0
        for j in range(s, s + q):
            rq += log_returns[j]
        d = rq - mq
        vq += d * d
        for j in range(s + q, i + 1):
            rq += log_returns[j] - log_returns[j - q]
            d = rq - mq
            vq += d * d
        # Lo–MacKinlay (1988) unbiased estimator σ̂²_c(q):
        #   σ̂²_c(q) = (1 / m) · Σ (R_t(q) − q·μ̂)²
        #   m = q · (T − q + 1) · (1 − q/T)
        # The (1 − q/T) factor corrects for finite-sample overlap bias.
        # σ̂²_a = v1 (already normalized by T = window).
        # VR(q) = σ̂²_c(q) / σ̂²_a.
        m_lm = q * nq * (1.0 - q / window)
        if m_lm <= 0:
            continue
        out[i] = (vq / m_lm) / v1
    return out


# ---------------------------------------------------------------------------
# Rolling lag-1 autocorrelation of returns
# ---------------------------------------------------------------------------
# rho(1) = Cov(r_t, r_{t-1}) / Var(r_t)
@njit(cache=True, fastmath=True)
def autocorr_lag1_rolling(returns: np.ndarray, window: int = 64) -> np.ndarray:
    n = returns.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 4 or n < window + 1:
        return out
    for i in range(window, n):
        s = i - window + 1
        m = 0.0
        for j in range(s, i + 1):
            m += returns[j]
        m /= window
        cov = 0.0
        var = 0.0
        for j in range(s, i + 1):
            d = returns[j] - m
            var += d * d
            if j > s:
                cov += d * (returns[j - 1] - m)
        if var <= 0:
            continue
        out[i] = cov / var
    return out


# ---------------------------------------------------------------------------
# ATR percentile rank — current ATR's percentile in a rolling history window
# ---------------------------------------------------------------------------
# Returns value in [0, 1]. Used as volatility regime gate:
#   < 0.2  compression (mean-revert / breakout setup)
#   > 0.8  expansion (trend-follow regime)
@njit(cache=True, fastmath=True)
def atr_percentile_rank(atr: np.ndarray, window: int = 252) -> np.ndarray:
    n = atr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    for i in range(window - 1, n):
        a = atr[i]
        if np.isnan(a):
            continue
        cnt = 0
        below = 0
        for j in range(i - window + 1, i + 1):
            v = atr[j]
            if np.isnan(v):
                continue
            cnt += 1
            if v < a:
                below += 1
        if cnt > 0:
            out[i] = below / cnt
    return out


# ---------------------------------------------------------------------------
# Parkinson realized volatility (high-low estimator, 1980)
# ---------------------------------------------------------------------------
# σ²_P = (1 / (4 * ln 2)) * mean( ln(H/L)^2 )       per bar
# Annualized: multiply by sqrt(bars_per_year). Caller scales as needed.
# More efficient than close-to-close vol since it uses intra-bar range.
_PARKINSON_K = 1.0 / (4.0 * np.log(2.0))


@njit(cache=True, fastmath=True)
def parkinson_vol(high: np.ndarray, low: np.ndarray,
                  window: int = 20) -> np.ndarray:
    n = high.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    for i in range(window - 1, n):
        s = 0.0
        valid = 0
        for j in range(i - window + 1, i + 1):
            if high[j] > 0 and low[j] > 0 and high[j] > low[j]:
                lr = np.log(high[j] / low[j])
                s += lr * lr
                valid += 1
        if valid > 0:
            out[i] = np.sqrt(_PARKINSON_K * s / valid)
    return out


# ---------------------------------------------------------------------------
# Trend strength — R^2 of log price regressed on time index, rolling
# ---------------------------------------------------------------------------
# y = a + b*t + e   over rolling window
# R^2 in [0,1]. High R^2 + positive slope == clean uptrend.
@njit(cache=True, fastmath=True)
def trend_r2_rolling(log_price: np.ndarray, window: int = 50):
    n = log_price.shape[0]
    r2 = np.full(n, np.nan, dtype=np.float64)
    slope = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return r2, slope
    # Pre-compute t mean and SS_t (constant for fixed window)
    tm = (window - 1) / 2.0
    ss_t = 0.0
    for k in range(window):
        d = k - tm
        ss_t += d * d
    if ss_t <= 0:
        return r2, slope
    for i in range(window - 1, n):
        s = i - window + 1
        ym = 0.0
        for j in range(s, i + 1):
            ym += log_price[j]
        ym /= window
        sxy = 0.0
        ss_y = 0.0
        for k in range(window):
            dy = log_price[s + k] - ym
            dt = k - tm
            sxy += dt * dy
            ss_y += dy * dy
        if ss_y <= 0:
            r2[i] = 0.0
            slope[i] = 0.0
            continue
        b = sxy / ss_t
        # SS_explained = b^2 * SS_t ;  R^2 = SS_explained / SS_y
        r2[i] = (b * b * ss_t) / ss_y
        slope[i] = b
    return r2, slope


# ---------------------------------------------------------------------------
# Composite regime classifier
# ---------------------------------------------------------------------------
# State codes:
#    1 = trend up        (Hurst > 0.55, slope > 0,  R^2 > 0.5, vol regime > 0.5)
#   -1 = trend down      (Hurst > 0.55, slope < 0, ...)
#    2 = mean-revert     (Hurst < 0.45 OR autocorr < -0.1, vol regime < 0.5)
#    0 = chop / undefined
def classify_regime(hurst: np.ndarray, slope: np.ndarray, r2: np.ndarray,
                    autocorr: np.ndarray, atr_pct: np.ndarray) -> np.ndarray:
    n = hurst.shape[0]
    out = np.zeros(n, dtype=np.int8)
    for i in range(n):
        h = hurst[i]; sl = slope[i]; r = r2[i]
        ac = autocorr[i]; ap = atr_pct[i]
        if np.isnan(h) or np.isnan(sl) or np.isnan(r) or np.isnan(ap):
            continue
        if h > 0.55 and r > 0.5 and ap > 0.5:
            out[i] = 1 if sl > 0 else -1
            continue
        if (h < 0.45 or (not np.isnan(ac) and ac < -0.1)) and ap < 0.5:
            out[i] = 2
    return out
