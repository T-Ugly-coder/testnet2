"""Causality tests for algo.strategy.confluence.build_signals.

The contract: signal[i], sl[i], tp[i] must be computable using only bars
0..i. We check by truncating the input at progressively earlier prefixes and
verifying that the prefix's outputs match the corresponding leading slice of
the full-series outputs.

Note: numba-jitted functions are deterministic, so equality (not just
approximation) should hold for finite values; we still use np.allclose for
NaN-safe comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algo.strategy.confluence import build_signals


def _synth_bars(n: int, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 0.5, n))
    o = c + rng.normal(0, 0.1, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.3, n))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.3, n))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c})


def _equal_with_nans(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.asarray(a); b = np.asarray(b)
    if a.shape != b.shape:
        return False
    eq = (a == b) | (np.isnan(a) & np.isnan(b))
    return bool(np.all(eq))


def test_build_signals_shape():
    bars = _synth_bars(300)
    out = build_signals(bars)
    n = len(bars)
    assert out["signal"].shape == (n,)
    assert out["sl"].shape == (n,)
    assert out["tp"].shape == (n,)


def test_build_signals_empty():
    out = build_signals(pd.DataFrame({"open": [], "high": [], "low": [], "close": []}))
    assert out["signal"].shape == (0,)
    assert out["sl"].shape == (0,)
    assert out["tp"].shape == (0,)


def test_build_signals_causality_prefix_invariance():
    """Truncating bars at K must not change outputs at indices < K (modulo
    the right-bar fractal warm-up at the very tail)."""
    bars = _synth_bars(400)
    full = build_signals(bars)
    # Choose K well inside the series, leaving a margin >= swing_right (5).
    K = 250
    margin = 5
    pref = build_signals(bars.iloc[:K].reset_index(drop=True))
    # Compare indices 0..K-margin-1
    end = K - margin
    assert _equal_with_nans(full["signal"][:end], pref["signal"][:end])
    assert _equal_with_nans(full["sl"][:end], pref["sl"][:end])
    assert _equal_with_nans(full["tp"][:end], pref["tp"][:end])


def test_build_signals_direction_sanity():
    bars = _synth_bars(500)
    out = build_signals(bars)
    sig = out["signal"]; sl = out["sl"]; tp = out["tp"]
    c = bars["close"].to_numpy()
    longs = np.where(sig == 1)[0]
    shorts = np.where(sig == -1)[0]
    for i in longs:
        # SL below close (entry estimate); TP above close
        assert sl[i] < c[i] < tp[i]
    for i in shorts:
        assert tp[i] < c[i] < sl[i]


def test_build_signals_nan_where_no_signal():
    bars = _synth_bars(200)
    out = build_signals(bars)
    no_sig = out["signal"] == 0
    # SL/TP must be NaN where there's no signal
    assert np.all(np.isnan(out["sl"][no_sig]))
    assert np.all(np.isnan(out["tp"][no_sig]))
