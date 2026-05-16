"""
Composite objective for the auto-optimizer.

The objective collapses a multi-metric performance vector into a single scalar
that the optimizer (Optuna) maximizes. It encodes the user's "Wall-Street
quality" preferences:

  * Risk-adjusted return (Sharpe is primary, Sortino secondary).
  * Trade-frequency band (target ~10-15 trades/month by default; harshly
    penalize both droughts and over-trading) — penalty is C0-continuous.
  * Drawdown control (penalty grows quadratically beyond a soft cap).
  * Profitability (positive expectancy & profit factor) using TRADE-WEIGHTED
    aggregation across folds rather than naive fold-mean.
  * Stability across walk-forward folds (mean - lambda * std), with lambda
    shrunk when the number of folds is too small to estimate std reliably.

All inputs are computed on out-of-sample (test) folds via
`algo.backtest.walkforward.walk_forward`.

Notes
-----
* `bars_per_month` should match the timeframe of the bars passed to the
  walk-forward (e.g. 1m crypto -> ~43_800; 1h -> ~720; 1d -> ~21).
* Folds with < 5 trades are not penalized for std (insufficient sample),
  but they DO contribute to the trade-frequency penalty.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ObjectiveConfig:
    target_trades_per_month: float = 12.0      # midpoint of 10-15
    min_trades_per_month: float = 8.0
    max_trades_per_month: float = 20.0
    soft_dd_cap: float = 0.20                  # 20% before DD penalty bites
    hard_dd_cap: float = 0.40                  # 40% => effectively disqualify
    min_win_rate: float = 0.45
    sharpe_weight: float = 1.0
    sortino_weight: float = 0.3
    pf_weight: float = 0.3
    expectancy_weight: float = 0.2
    win_rate_weight: float = 0.4
    stability_lambda: float = 0.5              # subtract lambda*std(sharpe)
    freq_penalty_weight: float = 1.0
    dd_penalty_weight: float = 2.0
    # Expectancy is fed to tanh(expectancy * scale).
    # If `expectancy_R` is available (preferred: dimensionless R-multiple),
    # `expectancy_R_scale` operates on it and `expectancy_scale` is unused.
    # If only currency `expectancy` is available, we fall back to that and
    # divide by `expectancy_currency_norm` first to make the scale meaningful.
    expectancy_scale: float = 20.0
    expectancy_R_scale: float = 1.0            # tanh(1*R) ~= 0.76 at 1R, ~0.96 at 2R
    expectancy_currency_norm: float = 1.0      # divide currency expectancy by this
    # Symmetric outside-band slopes for freq penalty.
    freq_under_slope: float = 1.0
    freq_over_slope: float = 1.0
    nan_score: float = -1e9


_REQUIRED_FOLD_COLUMNS = (
    "sharpe", "sortino", "profit_factor", "expectancy",
    "win_rate", "max_drawdown", "n_trades", "test_bars",
)


def _safe(v, default=0.0):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if np.isfinite(f) else default


def _clean_series(s: pd.Series, replace_pos_inf: float = 5.0,
                  replace_neg_inf: float = 0.0) -> pd.Series:
    """Replace ±inf and drop NaN. Keeps the original index for weighting."""
    return (s.replace([np.inf, -np.inf], [replace_pos_inf, replace_neg_inf])
             .dropna())


def _weighted_mean(values: pd.Series, weights: pd.Series,
                   default: float = 0.0) -> float:
    """Trade-weighted mean (or any-weight). Single fall-back path: simple mean
    of the finite values if all weights are zero/NaN, default if even that is
    not finite.
    """
    if values is None or len(values) == 0:
        return default
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    w = (pd.to_numeric(weights, errors="coerce")
           .reindex(values.index)
           .fillna(0.0)
           .clip(lower=0.0)
           .to_numpy(dtype=np.float64))
    finite = np.isfinite(v)
    if not finite.any():
        return default
    vv = v[finite]
    ww = w[finite]
    s = float(ww.sum())
    if s <= 0:
        m = float(vv.mean())
        return m if np.isfinite(m) else default
    return float(np.sum(vv * ww) / s)


def _weighted_mean_std(v: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted mean and (unbiased) weighted std for a 1-D array.

    Falls back gracefully when weights collapse or the array is too small.
    """
    finite = np.isfinite(v) & np.isfinite(w)
    if not finite.any():
        return 0.0, 0.0
    vv = v[finite]
    ww = w[finite].clip(min=0.0)
    if vv.size == 0:
        return 0.0, 0.0
    if ww.sum() <= 0:
        m = float(vv.mean())
        sd = float(vv.std(ddof=1)) if vv.size > 1 else 0.0
        return m, sd
    m = float(np.sum(vv * ww) / np.sum(ww))
    if vv.size < 2:
        return m, 0.0
    # Reliability-weighted variance with Bessel-style correction.
    sw = float(np.sum(ww))
    sw2 = float(np.sum(ww * ww))
    denom = sw - (sw2 / sw) if sw > 0 else 0.0
    if denom <= 0:
        return m, float(vv.std(ddof=1))
    var = float(np.sum(ww * (vv - m) ** 2) / denom)
    return m, float(np.sqrt(max(var, 0.0)))


def trades_per_month(folds_df: pd.DataFrame, bars_per_month: float) -> float:
    """Average trades per month across folds (weighted by fold bar-count)."""
    if folds_df is None or len(folds_df) == 0:
        return 0.0
    if "n_trades" not in folds_df.columns or "test_bars" not in folds_df.columns:
        return 0.0
    total_trades = float(folds_df["n_trades"].fillna(0).sum())
    total_bars = float(folds_df["test_bars"].fillna(0).sum())
    if total_bars <= 0 or bars_per_month <= 0:
        return 0.0
    return total_trades / (total_bars / bars_per_month)


def _freq_penalty(tpm: float, cfg: ObjectiveConfig) -> float:
    """Piecewise-linear penalty around the target band, C0-continuous at lo/hi.

    Inside the band: 0.5 * |tpm - tgt| / tgt.
    Outside: continues linearly from the boundary value (no jump).
    """
    lo = cfg.min_trades_per_month
    hi = cfg.max_trades_per_month
    tgt = max(cfg.target_trades_per_month, 1e-9)
    if lo <= tpm <= hi:
        return 0.5 * abs(tpm - tgt) / tgt
    if tpm < lo:
        boundary = 0.5 * abs(lo - tgt) / tgt
        return boundary + cfg.freq_under_slope * (lo - tpm) / tgt
    boundary = 0.5 * abs(hi - tgt) / tgt
    return boundary + cfg.freq_over_slope * (tpm - hi) / tgt


def _dd_penalty(max_dd_abs: float, cfg: ObjectiveConfig) -> float:
    """Quadratic above the soft cap; saturates at the hard cap."""
    if max_dd_abs <= cfg.soft_dd_cap:
        return 0.0
    if max_dd_abs >= cfg.hard_dd_cap:
        return 10.0
    excess = (max_dd_abs - cfg.soft_dd_cap) / max(cfg.hard_dd_cap - cfg.soft_dd_cap, 1e-9)
    return 10.0 * (excess ** 2)


def _check_required_columns(folds_df: pd.DataFrame) -> Optional[str]:
    missing = [c for c in _REQUIRED_FOLD_COLUMNS if c not in folds_df.columns]
    if missing:
        return f"missing_columns:{','.join(missing)}"
    return None


def score(folds_df: pd.DataFrame, bars_per_month: float,
          cfg: Optional[ObjectiveConfig] = None) -> dict:
    """Return a dict with `score` and component metrics.

    `score` is the scalar to maximize.
    """
    cfg = cfg or ObjectiveConfig()
    if folds_df is None or len(folds_df) == 0:
        return {"score": cfg.nan_score, "reason": "no_folds"}

    miss = _check_required_columns(folds_df)
    if miss is not None:
        return {"score": cfg.nan_score, "reason": miss}

    sharpe_raw = pd.to_numeric(folds_df["sharpe"], errors="coerce").to_numpy(dtype=np.float64)
    bars_w = pd.to_numeric(folds_df["test_bars"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    finite_sh = np.isfinite(sharpe_raw)
    if not finite_sh.any():
        return {"score": cfg.nan_score, "reason": "no_sharpe"}

    n_trades_w = pd.to_numeric(folds_df["n_trades"], errors="coerce").fillna(0)

    # Sortino: weight by trades when available (otherwise fold-mean fallback).
    sortino = _weighted_mean(folds_df["sortino"], n_trades_w)
    # Profit factor: cap +inf at 5.0 (huge but finite); -inf shouldn't occur but
    # guard anyway by mapping to 0.0. Trade-weighted across folds.
    pf_clean = folds_df["profit_factor"].replace(
        [np.inf, -np.inf], [5.0, 0.0])
    pf = _weighted_mean(pf_clean, n_trades_w)
    # Expectancy currency value is always computed once for the report.
    expectancy = _weighted_mean(folds_df["expectancy"], n_trades_w)
    # Prefer the dimensionless R-multiple if folds carry it.
    if "expectancy_R" in folds_df.columns:
        expectancy_R = _weighted_mean(folds_df["expectancy_R"], n_trades_w)
        expectancy_term = float(np.tanh(expectancy_R * cfg.expectancy_R_scale))
    else:
        expectancy_R = 0.0
        norm = max(cfg.expectancy_currency_norm, 1e-9)
        expectancy_term = float(np.tanh((expectancy / norm) * cfg.expectancy_scale))
    win_rate = _weighted_mean(folds_df["win_rate"], n_trades_w)
    # Max drawdown: take the worst across folds regardless of sign convention.
    dd_series = folds_df["max_drawdown"].replace(
        [np.inf, -np.inf], np.nan).dropna()
    max_dd = float(dd_series.abs().max()) if not dd_series.empty else 0.0

    # Sharpe is bar-weighted across folds (folds with more test bars are more
    # informationally credible than tiny folds).
    sharpe_mean, sharpe_std = _weighted_mean_std(sharpe_raw, bars_w)
    n_eff = int(finite_sh.sum())
    # Smooth shrinkage of the stability penalty as a function of fold count.
    eff_lambda = cfg.stability_lambda * min(1.0, max(n_eff - 1, 0) / 3.0)
    stability = sharpe_mean - eff_lambda * sharpe_std

    tpm = trades_per_month(folds_df, bars_per_month)
    fp = _freq_penalty(tpm, cfg)
    ddp = _dd_penalty(max_dd, cfg)

    raw = (
        cfg.sharpe_weight * stability
        + cfg.sortino_weight * sortino
        + cfg.pf_weight * (min(max(pf, 0.0), 5.0) / 5.0)
        + cfg.expectancy_weight * expectancy_term
        + cfg.win_rate_weight * max(0.0, win_rate - cfg.min_win_rate)
        - cfg.freq_penalty_weight * fp
        - cfg.dd_penalty_weight * ddp
    )
    final = float(raw) if np.isfinite(raw) else cfg.nan_score

    return {
        "score": final,
        "sharpe_mean": sharpe_mean,
        "sharpe_std": sharpe_std,
        "stability": stability,
        "sortino_mean": sortino,
        "profit_factor_mean": pf,
        "expectancy_mean": expectancy,
        "expectancy_R_mean": expectancy_R,
        "win_rate_mean": win_rate,
        "max_drawdown_abs": max_dd,
        "trades_per_month": tpm,
        "freq_penalty": fp,
        "dd_penalty": ddp,
        "n_folds": int(len(folds_df)),
        "config": asdict(cfg),
    }
