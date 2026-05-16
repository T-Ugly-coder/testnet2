"""Strategy sub-package — orchestrates indicators + structure + patterns into
causal (signal, sl, tp) arrays consumed by backtest/engine.run_backtest.
"""
from .confluence import build_signals
from .runner import backtest_strategy

__all__ = ["build_signals", "backtest_strategy"]
