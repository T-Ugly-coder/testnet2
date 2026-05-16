"""End-to-end runner: build causal signals -> run backtest engine.

Two strategy "kinds" are supported:
  * "confluence" (default): rule-based bias+sweep+trigger via
    `confluence.build_signals`. Few knobs.
  * "scorer": pluggable weighted scorer system via `scorer.ConfluenceScorer`.
    Many knobs => suitable target for the auto-optimizer in algo.optimize.

`backtest_strategy` keeps backward compatibility: callers that pass only
`confluence.build_signals` kwargs continue to work. Callers that pass any
scorer-only kwarg, or `strategy_kind="scorer"`, get the scorer pipeline.
"""
from __future__ import annotations

import inspect
from typing import Any
import pandas as pd

from .confluence import build_signals as _confluence_build
from .scorer import (
    ConfluenceScorer,
    trend_scorer, sfp_scorer, candle_scorer, regime_scorer,
    vsa_scorer, wyckoff_scorer, vwap_dev_scorer,
)
from ..backtest.engine import run_backtest


# Public alias kept so `from algo.strategy import build_signals` keeps working.
build_signals = _confluence_build

_BUILD_PARAMS = set(inspect.signature(_confluence_build).parameters) - {"bars"}
_BT_PARAMS = set(inspect.signature(run_backtest).parameters) - {"bars", "signals"}

_SCORER_FIELDS = {
    "entry_threshold", "min_dominance", "rr", "sl_buffer_atr",
    "atr_period", "swing_left", "swing_right", "wick_min_atr",
    "vol_ma_period", "wyckoff_lookback", "wyckoff_compression",
    "vwap_z_threshold", "vwap_band_mult", "vwap_warmup_bars",
}
_SCORER_WEIGHT_FIELDS = {
    "w_trend", "w_sfp", "w_candle", "w_regime",
    "w_vsa", "w_wyckoff", "w_vwap_dev",
}
_SCORER_ALL = _SCORER_FIELDS | _SCORER_WEIGHT_FIELDS


def _build_scorer_signals(bars: pd.DataFrame, **kw: Any) -> dict:
    """Adapter: instantiate a ConfluenceScorer from flat kwargs."""
    weights = {
        "trend": float(kw.pop("w_trend", 1.5)),
        "sfp": float(kw.pop("w_sfp", 1.0)),
        "candle": float(kw.pop("w_candle", 0.8)),
        "regime": float(kw.pop("w_regime", 0.7)),
        "vsa": float(kw.pop("w_vsa", 0.6)),
        "wyckoff": float(kw.pop("w_wyckoff", 0.6)),
        "vwap_dev": float(kw.pop("w_vwap_dev", 0.4)),
    }
    spec_kwargs = {k: v for k, v in kw.items() if k in _SCORER_FIELDS}
    # Pop scorer-shared ctx-only params that aren't ConfluenceScorer fields.
    ctx_only = {"vol_ma_period", "wyckoff_lookback", "wyckoff_compression",
                "vwap_z_threshold", "vwap_band_mult", "vwap_warmup_bars"}
    cs_kwargs = {k: v for k, v in spec_kwargs.items() if k not in ctx_only}
    cs = ConfluenceScorer(**cs_kwargs)
    # Stash ctx-only knobs on the instance for forwarding via build_signals.
    cs._extra_ctx = {k: spec_kwargs[k] for k in ctx_only if k in spec_kwargs}
    cs.register("trend", trend_scorer, weight=weights["trend"])
    cs.register("sfp", sfp_scorer, weight=weights["sfp"])
    cs.register("candle", candle_scorer, weight=weights["candle"])
    cs.register("regime", regime_scorer, weight=weights["regime"])
    cs.register("vsa", vsa_scorer, weight=weights["vsa"])
    cs.register("wyckoff", wyckoff_scorer, weight=weights["wyckoff"])
    cs.register("vwap_dev", vwap_dev_scorer, weight=weights["vwap_dev"])
    return cs.build_signals(bars)


def backtest_strategy(bars: pd.DataFrame,
                      strategy_kind: str = "confluence",
                      **kwargs) -> dict:
    """Run a strategy end-to-end on `bars`.

    Parameters
    ----------
    strategy_kind : "confluence" | "scorer"
        Which signal-building pipeline to use. Auto-detects "scorer" if any
        scorer-only kwarg (e.g. ``entry_threshold``, ``w_trend``) is supplied
        and ``strategy_kind`` was left at its default.
    **kwargs
        Forwarded to the chosen build_signals() and to run_backtest().
    """
    has_scorer_kw = any(
        (k in _SCORER_ALL) and (k not in _BUILD_PARAMS) for k in kwargs
    )
    if has_scorer_kw and strategy_kind == "confluence":
        strategy_kind = "scorer"

    if strategy_kind == "confluence":
        sig_params = _BUILD_PARAMS
        build_fn = _confluence_build
    elif strategy_kind == "scorer":
        sig_params = _SCORER_ALL
        build_fn = _build_scorer_signals
    else:
        raise ValueError(f"unknown strategy_kind: {strategy_kind!r}")

    sig_kwargs: dict = {}
    bt_kwargs: dict = {}
    for k, v in kwargs.items():
        if k in sig_params:
            sig_kwargs[k] = v
        elif k in _BT_PARAMS:
            bt_kwargs[k] = v
        else:
            raise TypeError(f"backtest_strategy: unknown kwarg {k!r}")

    signals = build_fn(bars, **sig_kwargs)
    return run_backtest(bars, signals, **bt_kwargs)
