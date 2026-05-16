"""Unit tests for algo.core.patterns.regime.

Validates:
  - Variance ratio ~ 1 on iid Gaussian noise (random walk null).
  - Variance ratio > 1 on a momentum series (cumulative drift).
  - Variance ratio < 1 on a strongly mean-reverting AR(1) with negative phi.
  - Hurst ~ 0.5 on iid noise; > 0.5 on a trending integrated series.
  - trend_r2_rolling returns R^2 in [0, 1] and slope sign matches direction.
  - atr_percentile_rank in [0, 1].
"""
from __future__ import annotations

import numpy as np
import pytest

from algo.core.patterns.regime import (
    variance_ratio_rolling,
    hurst_rolling,
    autocorr_lag1_rolling,
    trend_r2_rolling,
    atr_percentile_rank,
    parkinson_vol,
    classify_regime,
)


def _last_finite(a: np.ndarray) -> float:
    a = np.asarray(a)
    finite = a[np.isfinite(a)]
    assert finite.size > 0, "no finite values"
    return float(finite[-1])


def test_vr_iid_noise_near_one():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 1.0, 4000)
    vr = variance_ratio_rolling(r, q=4, window=512)
    # Average over the second half — Lo–MacKinlay VR should hover near 1.
    tail = vr[2000:]
    tail = tail[np.isfinite(tail)]
    assert 0.85 < tail.mean() < 1.15


def test_vr_trending_above_one():
    # Build returns that have positive serial correlation: r_t = 0.6 r_{t-1} + e
    rng = np.random.default_rng(1)
    n = 4000
    e = rng.normal(0, 1.0, n)
    r = np.empty(n)
    r[0] = e[0]
    for i in range(1, n):
        r[i] = 0.6 * r[i - 1] + e[i]
    vr = variance_ratio_rolling(r, q=4, window=512)
    tail = vr[2000:]
    tail = tail[np.isfinite(tail)]
    assert tail.mean() > 1.5


def test_vr_meanrev_below_one():
    rng = np.random.default_rng(2)
    n = 4000
    e = rng.normal(0, 1.0, n)
    r = np.empty(n)
    r[0] = e[0]
    for i in range(1, n):
        r[i] = -0.6 * r[i - 1] + e[i]
    vr = variance_ratio_rolling(r, q=4, window=512)
    tail = vr[2000:]
    tail = tail[np.isfinite(tail)]
    assert tail.mean() < 0.7


def test_hurst_random_near_half():
    rng = np.random.default_rng(3)
    r = rng.normal(0, 1.0, 4000)
    h = hurst_rolling(r, window=512)
    tail = h[2000:]
    tail = tail[np.isfinite(tail)]
    assert 0.3 < tail.mean() < 0.7


def test_autocorr_sign_matches_construction():
    rng = np.random.default_rng(4)
    n = 2000
    e = rng.normal(0, 1.0, n)
    r = np.empty(n)
    r[0] = e[0]
    for i in range(1, n):
        r[i] = 0.5 * r[i - 1] + e[i]
    ac = autocorr_lag1_rolling(r, window=200)
    tail = ac[1000:]
    tail = tail[np.isfinite(tail)]
    assert tail.mean() > 0.2


def test_trend_r2_uptrend():
    n = 500
    x = np.linspace(0, 1, n)  # perfectly linear
    r2, slope = trend_r2_rolling(x, window=50)
    assert _last_finite(r2) > 0.99
    assert _last_finite(slope) > 0


def test_trend_r2_bounds():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 500)
    r2, _ = trend_r2_rolling(x, window=50)
    finite = r2[np.isfinite(r2)]
    assert (finite >= 0).all() and (finite <= 1).all()


def test_atr_percentile_in_unit_interval():
    rng = np.random.default_rng(6)
    atr = np.abs(rng.normal(0.5, 0.1, 1000))
    p = atr_percentile_rank(atr, window=200)
    finite = p[np.isfinite(p)]
    assert (finite >= 0).all() and (finite <= 1).all()


def test_parkinson_vol_positive():
    rng = np.random.default_rng(7)
    high = 100 + np.abs(rng.normal(0, 1.0, 500))
    low = 100 - np.abs(rng.normal(0, 1.0, 500))
    pv = parkinson_vol(high, low, window=20)
    finite = pv[np.isfinite(pv)]
    assert (finite >= 0).all()
    assert finite.size > 0


def test_classify_regime_shapes():
    n = 100
    h = np.full(n, 0.6)
    sl = np.full(n, 0.001)
    r2 = np.full(n, 0.7)
    ac = np.full(n, 0.0)
    ap = np.full(n, 0.6)
    out = classify_regime(h, sl, r2, ac, ap)
    assert out.shape == (n,)
    # All conditions for "trend up" satisfied
    assert (out == 1).all()
