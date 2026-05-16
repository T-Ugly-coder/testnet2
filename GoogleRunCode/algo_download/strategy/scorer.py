"""
Pluggable confluence scorer.

Idea
----
Each *scorer* is a stateless function that, given the bar context, returns
a (possibly time-varying) score in some interval. The `ConfluenceScorer`
class:

  1. Calls each registered scorer once over the full bar series.
  2. Linearly weights and sums their per-bar outputs into bull/bear scores.
  3. Emits a +1/-1 entry signal when the dominant side's score exceeds
     `entry_threshold` AND the dominant side's score exceeds the opposite
     side by `min_dominance`.
  4. Computes SL/TP from ATR (same convention as `confluence.build_signals`).

Each scorer must satisfy the causality contract:
    scorer(bars, ctx) -> (bull: np.ndarray, bear: np.ndarray)
where bull[i], bear[i] depend only on bars[0..i].

This module ships with thin wrappers that adapt the existing detectors:
    - trend_scorer          (BOS/CHoCH structural bias)
    - sfp_scorer            (bias-aligned Swing Failure Pattern)
    - candle_scorer         (bias-aligned candle pattern strength)
    - regime_scorer         (Hurst + R^2 + ATR percentile)

Add more by registering callables via `ConfluenceScorer.register(name, fn,
weight)`. The system is intentionally additive: you can drop scorers in or
out without changing the engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

from ..core.indicators import atr_wilder, sma, anchored_vwap, vwap_bands
from ..core.structure import detect_swings, bos_choch
from ..core.patterns.candles import detect_candles
from ..core.patterns.sfp import detect_sfp
from ..core.patterns.regime import (
    hurst_rolling, trend_r2_rolling, atr_percentile_rank,
)
from ..core.patterns.vsa import vsa_signals
from ..core.patterns.wyckoff import (
    detect_trading_range, detect_springs_upthrusts,
    sign_of_strength_weakness,
)


# A scorer returns (bull_score, bear_score), both length-n float arrays in
# [0, 1] (or any consistent scale; weights compensate).
ScorerFn = Callable[[pd.DataFrame, dict], Tuple[np.ndarray, np.ndarray]]


@dataclass
class ScorerSpec:
    name: str
    fn: ScorerFn
    weight: float = 1.0


# ---------------------------------------------------------------------------
# Built-in scorers
# ---------------------------------------------------------------------------

def _ctx_or_compute(ctx: dict, bars: pd.DataFrame) -> dict:
    """Cache common arrays across scorers in the shared `ctx` dict."""
    if "h" not in ctx:
        ctx["o"] = bars["open"].to_numpy(np.float64)
        ctx["h"] = bars["high"].to_numpy(np.float64)
        ctx["l"] = bars["low"].to_numpy(np.float64)
        ctx["c"] = bars["close"].to_numpy(np.float64)
        ctx["n"] = ctx["c"].shape[0]
        if "volume" in bars.columns:
            ctx["v"] = bars["volume"].to_numpy(np.float64)
        else:
            ctx["v"] = None
        if "ts_ms" in bars.columns:
            ctx["ts_ms"] = bars["ts_ms"].to_numpy(np.int64)
        else:
            ctx["ts_ms"] = None
    if "atr" not in ctx:
        ctx["atr"] = atr_wilder(ctx["h"], ctx["l"], ctx["c"],
                                 ctx.get("atr_period", 14))
    if "swings" not in ctx:
        sl_arg = ctx.get("swing_left", 5)
        sr = ctx.get("swing_right", 5)
        sh_idx, sh_px, sl_idx, sl_px = detect_swings(ctx["h"], ctx["l"],
                                                      sl_arg, sr)
        ctx["swings"] = (sh_idx, sh_px, sl_idx, sl_px)
        ctx["swing_right"] = sr
    if "vol_ma" not in ctx and ctx["v"] is not None:
        ctx["vol_ma"] = sma(ctx["v"], ctx.get("vol_ma_period", 20))
    return ctx


def trend_scorer(bars: pd.DataFrame, ctx: dict) -> Tuple[np.ndarray, np.ndarray]:
    """+1 to bull/bear when BOS/CHoCH gives a bullish/bearish bias."""
    ctx = _ctx_or_compute(ctx, bars)
    n = ctx["n"]
    sh_idx, sh_px, sl_idx, sl_px = ctx["swings"]
    ev_idx, ev_type = bos_choch(ctx["c"], sh_idx, sh_px, sl_idx, sl_px,
                                 right_bars=ctx["swing_right"])
    bull = np.zeros(n, dtype=np.float64)
    bear = np.zeros(n, dtype=np.float64)
    j = 0
    cur = 0  # +1 bull, -1 bear, 0 unknown
    strength = 0.0
    for i in range(n):
        while j < ev_idx.shape[0] and ev_idx[j] <= i:
            cur = 1 if ev_type[j] > 0 else -1
            # CHoCH (|2|) is a stronger signal than BOS (|1|)
            strength = 1.0 if abs(ev_type[j]) == 2 else 0.7
            j += 1
        if cur == 1:
            bull[i] = strength
        elif cur == -1:
            bear[i] = strength
    return bull, bear


def sfp_scorer(bars: pd.DataFrame, ctx: dict) -> Tuple[np.ndarray, np.ndarray]:
    ctx = _ctx_or_compute(ctx, bars)
    n = ctx["n"]
    sh_idx, sh_px, sl_idx, sl_px = ctx["swings"]
    sr = ctx["swing_right"]
    sfp_idx, sfp_side, _, sfp_wick = detect_sfp(
        ctx["h"], ctx["l"], ctx["c"],
        sh_idx + sr, sh_px, sl_idx + sr, sl_px,
        ctx.get("wick_min_atr", 0.5), ctx["atr"],
    )
    bull = np.zeros(n, dtype=np.float64)
    bear = np.zeros(n, dtype=np.float64)
    for k in range(sfp_idx.shape[0]):
        i = int(sfp_idx[k])
        # Wick magnitude (in ATRs) doubles as confidence proxy, capped at 2.0.
        atr_i = ctx["atr"][i] if i < n else np.nan
        w = sfp_wick[k] / atr_i if (atr_i and atr_i > 0) else 0.0
        w = min(max(w, 0.0), 2.0) / 2.0
        if sfp_side[k] == -1:
            bull[i] = max(bull[i], w)
        elif sfp_side[k] == 1:
            bear[i] = max(bear[i], w)
    return bull, bear


def candle_scorer(bars: pd.DataFrame, ctx: dict) -> Tuple[np.ndarray, np.ndarray]:
    ctx = _ctx_or_compute(ctx, bars)
    n = ctx["n"]
    cd_idx, _, cd_signal, cd_strength = detect_candles(
        ctx["o"], ctx["h"], ctx["l"], ctx["c"], ctx["atr"]
    )
    bull = np.zeros(n, dtype=np.float64)
    bear = np.zeros(n, dtype=np.float64)
    for k in range(cd_idx.shape[0]):
        i = int(cd_idx[k])
        s = float(cd_strength[k])
        # Strength normalized to [0,1] via tanh-like cap at 2.0
        s = min(max(s, 0.0), 2.0) / 2.0
        if cd_signal[k] == 1:
            bull[i] = max(bull[i], s)
        elif cd_signal[k] == -1:
            bear[i] = max(bear[i], s)
    return bull, bear


def regime_scorer(bars: pd.DataFrame, ctx: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Boost both sides equally when regime is trending; suppress when chop."""
    ctx = _ctx_or_compute(ctx, bars)
    n = ctx["n"]
    log_p = np.log(np.maximum(ctx["c"], 1e-12))
    log_r = np.diff(log_p, prepend=log_p[0])
    h_w = ctx.get("hurst_window", 128)
    r2_w = ctx.get("trend_r2_window", 50)
    apr_w = ctx.get("atr_pct_window", 252)
    h = hurst_rolling(log_r, window=h_w)
    r2, slope = trend_r2_rolling(log_p, window=r2_w)
    apr = atr_percentile_rank(ctx["atr"], window=apr_w)
    bull = np.zeros(n, dtype=np.float64)
    bear = np.zeros(n, dtype=np.float64)
    for i in range(n):
        hi = h[i]; ri = r2[i]; si = slope[i]; ai = apr[i]
        if not (np.isfinite(hi) and np.isfinite(ri) and np.isfinite(ai)):
            continue
        # Trending regime: H>0.5 + R^2 high + ATR percentile high.
        trend_strength = max(0.0, hi - 0.5) * 2.0   # 0..1
        score = trend_strength * ri * ai            # all in [0,1]
        if si > 0:
            bull[i] = score
        elif si < 0:
            bear[i] = score
    return bull, bear


def vsa_scorer(bars: pd.DataFrame, ctx: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Volume Spread Analysis confluence.

    Maps VSA codes from `vsa_signals` to bull/bear scores:
        no_supply (-1)            -> bull (continuation strength)
        no_demand (+1)            -> bear
        stopping_volume_bull (+2) -> bull (climactic absorption at low)
        climactic_volume_bear(-2) -> bear
        absorption_bull (+3)      -> bull
        absorption_bear (-3)      -> bear

    No-op (zeros) when the bars frame has no `volume` column. Causal: each
    bar uses only its own and prior values via vol_ma SMA.
    """
    ctx = _ctx_or_compute(ctx, bars)
    n = ctx["n"]
    bull = np.zeros(n, dtype=np.float64)
    bear = np.zeros(n, dtype=np.float64)
    if ctx["v"] is None:
        return bull, bear
    idx, code = vsa_signals(
        ctx["o"], ctx["h"], ctx["l"], ctx["c"],
        ctx["v"], ctx["atr"], ctx["vol_ma"],
    )
    # Per-code confidence (subjective but consistent in scale).
    bull_w = {-1: 0.5, 2: 1.0, 3: 0.7}
    bear_w = {1: 0.5, -2: 1.0, -3: 0.7}
    for k in range(idx.shape[0]):
        i = int(idx[k])
        cd = int(code[k])
        if cd in bull_w:
            bull[i] = max(bull[i], bull_w[cd])
        elif cd in bear_w:
            bear[i] = max(bear[i], bear_w[cd])
    return bull, bear


def wyckoff_scorer(bars: pd.DataFrame, ctx: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Wyckoff phase confluence: spring/upthrust + SOS/SOW.

    Springs (bull) and upthrusts (bear) are emitted only when the bar sits
    inside a contracted trading range, so they're already regime-filtered.
    SOS/SOW require volume; we OR them in if a volume column is available.
    """
    ctx = _ctx_or_compute(ctx, bars)
    n = ctx["n"]
    bull = np.zeros(n, dtype=np.float64)
    bear = np.zeros(n, dtype=np.float64)
    lookback = int(ctx.get("wyckoff_lookback", 30))
    in_tr = detect_trading_range(ctx["h"], ctx["l"], ctx["c"], ctx["atr"],
                                  lookback=lookback,
                                  compression=float(ctx.get("wyckoff_compression", 0.6)))
    su_idx, su_t = detect_springs_upthrusts(ctx["h"], ctx["l"], ctx["c"],
                                             in_tr, lookback=lookback)
    for k in range(su_idx.shape[0]):
        i = int(su_idx[k])
        if su_t[k] == 1:
            bull[i] = max(bull[i], 1.0)
        elif su_t[k] == -1:
            bear[i] = max(bear[i], 1.0)
    if ctx["v"] is not None:
        sw_idx, sw_t = sign_of_strength_weakness(
            ctx["o"], ctx["c"], ctx["v"], ctx["atr"], ctx["vol_ma"],
        )
        for k in range(sw_idx.shape[0]):
            i = int(sw_idx[k])
            if sw_t[k] == 1:
                bull[i] = max(bull[i], 0.7)
            elif sw_t[k] == -1:
                bear[i] = max(bear[i], 0.7)
    return bull, bear


def vwap_dev_scorer(bars: pd.DataFrame, ctx: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Anchored-VWAP deviation as a mean-reversion / location bias.

    Anchor at start of the series (cheap; no calendar dependency). At each
    bar i, compute z = (close - vwap) / band_sd. We map:
        z <= -k  -> bull (price stretched below VWAP, mean-revert long)
        z >=  k  -> bear (price stretched above VWAP, mean-revert short)
    where k = `vwap_z_threshold`. Magnitude beyond k is squashed by tanh.

    No-op when the bars frame has no `volume` column. Anchored VWAP is
    strictly causal (uses bars 0..i only).
    """
    ctx = _ctx_or_compute(ctx, bars)
    n = ctx["n"]
    bull = np.zeros(n, dtype=np.float64)
    bear = np.zeros(n, dtype=np.float64)
    if ctx["v"] is None:
        return bull, bear
    c = ctx["c"]; v = ctx["v"]
    vwap_mult = float(ctx.get("vwap_band_mult", 1.0))
    warmup = int(ctx.get("vwap_warmup_bars", 30))
    vw, up, _lo = vwap_bands(c, v, anchor_idx=0, mult=vwap_mult)
    k = float(ctx.get("vwap_z_threshold", 1.0))
    for i in range(n):
        if i < warmup:
            continue
        if not (np.isfinite(vw[i]) and np.isfinite(up[i])):
            continue
        # Recover sd regardless of `vwap_band_mult` value.
        sd = (up[i] - vw[i]) / vwap_mult if vwap_mult > 0 else 0.0
        if sd <= 0.0:
            continue
        z = (c[i] - vw[i]) / sd
        if z <= -k:
            bull[i] = float(np.tanh(-z - k))
        elif z >= k:
            bear[i] = float(np.tanh(z - k))
    return bull, bear


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------
@dataclass
class ConfluenceScorer:
    entry_threshold: float = 1.0
    min_dominance: float = 0.5
    rr: float = 2.0
    sl_buffer_atr: float = 0.5
    atr_period: int = 14
    swing_left: int = 5
    swing_right: int = 5
    wick_min_atr: float = 0.5
    scorers: list[ScorerSpec] = field(default_factory=list)

    @classmethod
    def default(cls, **overrides) -> "ConfluenceScorer":
        s = cls(**overrides)
        s.register("trend", trend_scorer, weight=1.5)
        s.register("sfp", sfp_scorer, weight=1.0)
        s.register("candle", candle_scorer, weight=0.8)
        s.register("regime", regime_scorer, weight=0.7)
        s.register("vsa", vsa_scorer, weight=0.6)
        s.register("wyckoff", wyckoff_scorer, weight=0.6)
        s.register("vwap_dev", vwap_dev_scorer, weight=0.4)
        return s

    def register(self, name: str, fn: ScorerFn, weight: float = 1.0) -> None:
        self.scorers.append(ScorerSpec(name=name, fn=fn, weight=weight))

    def build_signals(self, bars: pd.DataFrame) -> dict:
        """Same return shape as confluence.build_signals."""
        n = len(bars)
        if n == 0:
            return {
                "signal": np.zeros(0, dtype=np.int8),
                "sl": np.zeros(0, dtype=np.float64),
                "tp": np.zeros(0, dtype=np.float64),
                "rr": np.zeros(0, dtype=np.float64),
                "bull_score": np.zeros(0, dtype=np.float64),
                "bear_score": np.zeros(0, dtype=np.float64),
            }
        ctx: dict = {
            "atr_period": self.atr_period,
            "swing_left": self.swing_left,
            "swing_right": self.swing_right,
            "wick_min_atr": self.wick_min_atr,
        }
        # Merge any extra ctx knobs stashed by the runner adapter (e.g.
        # vol_ma_period, wyckoff_lookback, wyckoff_compression, vwap_z_threshold).
        extra = getattr(self, "_extra_ctx", None)
        if extra:
            for k, v in extra.items():
                ctx.setdefault(k, v)
        bull_total = np.zeros(n, dtype=np.float64)
        bear_total = np.zeros(n, dtype=np.float64)
        for spec in self.scorers:
            bu, be = spec.fn(bars, ctx)
            bull_total += spec.weight * bu
            bear_total += spec.weight * be

        atr = ctx["atr"]
        c = ctx["c"]; h = ctx["h"]; l = ctx["l"]
        signal = np.zeros(n, dtype=np.int8)
        sl_out = np.full(n, np.nan, dtype=np.float64)
        tp_out = np.full(n, np.nan, dtype=np.float64)

        for i in range(n):
            a = atr[i]
            if not np.isfinite(a) or a <= 0.0:
                continue
            bs = bull_total[i]; ss = bear_total[i]
            if bs >= self.entry_threshold and (bs - ss) >= self.min_dominance:
                entry_est = c[i]
                sl_v = l[i] - self.sl_buffer_atr * a
                if not (sl_v < entry_est):
                    continue
                tp_v = entry_est + self.rr * (entry_est - sl_v)
                signal[i] = 1
                sl_out[i] = sl_v
                tp_out[i] = tp_v
            elif ss >= self.entry_threshold and (ss - bs) >= self.min_dominance:
                entry_est = c[i]
                sl_v = h[i] + self.sl_buffer_atr * a
                if not (sl_v > entry_est):
                    continue
                tp_v = entry_est - self.rr * (sl_v - entry_est)
                signal[i] = -1
                sl_out[i] = sl_v
                tp_out[i] = tp_v

        return {
            "signal": signal, "sl": sl_out, "tp": tp_out,
            "rr": np.full(n, float(self.rr), dtype=np.float64),
            "bull_score": bull_total, "bear_score": bear_total,
        }
