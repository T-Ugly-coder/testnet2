"""Auto-optimizer package.

Public API:
    from algo.optimize import optimize, OptimizeConfig, ObjectiveConfig
    result = optimize(bars, OptimizeConfig(timeframe="1h", n_trials=200))
    print(result["best_params"])
    print(result["holdout_score"])
"""
from .objective import ObjectiveConfig, score
from .auto import OptimizeConfig, optimize, rank_top_strategies

__all__ = [
    "ObjectiveConfig", "score",
    "OptimizeConfig", "optimize", "rank_top_strategies",
]
