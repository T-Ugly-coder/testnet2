"""Example plugin component for the strategy framework.

Load with:

    python -m strategy_framework find ... --component-module strategy_framework.examples.ema_slope_component
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_framework.registry import ParamSpec, register_component


def _factory(params):
    period = int(params["ema_slope_period"])
    lookback = int(params["ema_slope_lookback"])

    def scorer(bars: pd.DataFrame, ctx: dict):
        close = bars["close"].to_numpy(float)
        ema = pd.Series(close).ewm(span=period, adjust=False).mean().to_numpy()
        prev = pd.Series(ema).shift(lookback).to_numpy()
        slope = ema - prev
        bull = ((close > ema) & (slope > 0.0)).astype(float)
        bear = ((close < ema) & (slope < 0.0)).astype(float)
        bull[~np.isfinite(slope)] = 0.0
        bear[~np.isfinite(slope)] = 0.0
        return bull, bear

    return scorer


register_component(
    "ema_slope",
    factory=_factory,
    params=(
        ParamSpec.integer("ema_slope_period", 20, 160, default=80),
        ParamSpec.integer("ema_slope_lookback", 3, 30, default=8),
    ),
    default_weight=0.0,
)
