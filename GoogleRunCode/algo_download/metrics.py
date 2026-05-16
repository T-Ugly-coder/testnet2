"""
Performance analytics for an equity curve and a trade ledger.

All functions are pure and operate on numpy arrays / pandas DataFrames produced
by `algo.backtest.engine.run_backtest`.

Conventions
-----------
- `equity` is a 1-D array of mark-to-market account value, one entry per bar.
- `returns` are simple per-bar returns: r_i = eq_i / eq_{i-1} - 1.
- `bars_per_year` is the annualization factor (e.g. 252*24*60 for 1-minute
  bars, 252 for daily). Caller supplies it; we never guess.
- All ratios use simple returns, not log-returns, to match industry usage.
- Sharpe / Sortino are zero-rf by default. Pass `rf_per_bar` if needed.

Definitions
-----------
Sharpe       = sqrt(N) * mean(r) / std(r, ddof=1)
Sortino      = sqrt(N) * mean(r) / downside_std(r)        (downside vs MAR=0)
Calmar       = CAGR / |MaxDD|
MaxDD        = min(eq / cummax(eq) - 1)            (negative number)
ProfitFactor = sum(wins) / sum(|losses|)
Expectancy   = mean(pnl)
WinRate      = wins / total
CAGR         = (eq[-1]/eq[0]) ** (bars_per_year / N) - 1
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PerfStats:
    n_bars: int
    n_trades: int
    final_balance: float
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_drawdown_duration: int
    volatility_ann: float
    win_rate: float
    profit_factor: float
    expectancy: float           # currency units (mean per-trade pnl as traders use)
    expectancy_R: float         # R-multiple (mean pnl / risk_amt). 1.0 = avg trade earned 1R.
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float

    def to_dict(self) -> dict:
        return asdict(self)


def equity_returns(equity: np.ndarray) -> np.ndarray:
    """Per-bar simple returns from an equity curve. Vectorized."""
    eq = np.asarray(equity, dtype=np.float64)
    if eq.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    prev = eq[:-1]
    nxt = eq[1:]
    out = np.zeros_like(prev)
    safe = prev > 0
    out[safe] = nxt[safe] / prev[safe] - 1.0
    return out


def sharpe_ratio(returns: np.ndarray, bars_per_year: float,
                 rf_per_bar: float = 0.0) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if r.shape[0] < 2:
        return 0.0
    excess = r - rf_per_bar
    sd = np.std(excess, ddof=1)
    if sd <= 0:
        return 0.0
    return float(np.sqrt(bars_per_year) * np.mean(excess) / sd)


def sortino_ratio(returns: np.ndarray, bars_per_year: float,
                  mar_per_bar: float = 0.0) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if r.shape[0] < 2:
        return 0.0
    downside = r[r < mar_per_bar] - mar_per_bar
    if downside.shape[0] == 0:
        return 0.0
    # population sqrt of mean square (semi-deviation)
    dd = np.sqrt(np.mean(downside * downside))
    if dd <= 0:
        return 0.0
    return float(np.sqrt(bars_per_year) * (np.mean(r) - mar_per_bar) / dd)


def max_drawdown(equity: np.ndarray) -> tuple[float, int]:
    """Return (max_dd_pct, max_dd_duration_bars). Drawdown is negative or 0.

    Vectorized: O(n) two-pass with numpy.
    """
    eq = np.asarray(equity, dtype=np.float64)
    n = eq.shape[0]
    if n == 0:
        return 0.0, 0
    peaks = np.maximum.accumulate(eq)
    safe = peaks > 0
    dd = np.zeros(n, dtype=np.float64)
    dd[safe] = eq[safe] / peaks[safe] - 1.0
    max_dd = float(dd.min()) if dd.size else 0.0

    # Drawdown duration = longest stretch of bars since last new high.
    # peak_idx[i] = index of last new equity high at or before i.
    is_new_peak = np.concatenate(([True], eq[1:] > peaks[:-1]))
    peak_idx = np.maximum.accumulate(np.where(is_new_peak, np.arange(n), -1))
    durations = np.arange(n) - peak_idx
    # Only count durations while we're in drawdown (eq < peak).
    durations = np.where(eq < peaks, durations, 0)
    max_dur = int(durations.max()) if durations.size else 0
    return float(max_dd), max_dur


def cagr(equity: np.ndarray, bars_per_year: float) -> float:
    eq = np.asarray(equity, dtype=np.float64)
    n = eq.shape[0]
    if n < 2 or eq[0] <= 0 or eq[-1] <= 0:
        return 0.0
    years = n / bars_per_year
    if years <= 0:
        return 0.0
    return float((eq[-1] / eq[0]) ** (1.0 / years) - 1.0)


def profit_factor(pnls: np.ndarray) -> float:
    p = np.asarray(pnls, dtype=np.float64)
    if p.shape[0] == 0:
        return 0.0
    wins = p[p > 0].sum()
    losses = -p[p < 0].sum()
    if losses <= 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def trade_stats(trades: pd.DataFrame) -> dict:
    """Per-trade aggregates from the engine's trades DataFrame.

    Computes both:
      - `expectancy` : mean(pnl) in account currency (what traders see in $).
      - `expectancy_R` : mean(pnl / risk_amt), the R-multiple. Requires the
         engine to populate a `risk_amt` column. Falls back to 0.0 if missing
         or all zero.
    """
    if trades is None or len(trades) == 0:
        return {
            "n_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "expectancy": 0.0, "expectancy_R": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0,
            "largest_win": 0.0, "largest_loss": 0.0,
        }
    p = trades["pnl"].to_numpy(np.float64)
    wins = p[p > 0]
    losses = p[p < 0]
    # R-multiple: prefer a per-trade `risk_amt` column when available.
    if "risk_amt" in trades.columns:
        ra = pd.to_numeric(trades["risk_amt"], errors="coerce").to_numpy(np.float64)
        ok = np.isfinite(ra) & (ra > 0) & np.isfinite(p)
        exp_R = float(np.mean(p[ok] / ra[ok])) if ok.any() else 0.0
    else:
        exp_R = 0.0
    return {
        "n_trades": int(p.shape[0]),
        "win_rate": float(wins.shape[0] / p.shape[0]) if p.shape[0] else 0.0,
        "profit_factor": profit_factor(p),
        "expectancy": float(np.mean(p)),
        "expectancy_R": exp_R,
        "avg_win": float(wins.mean()) if wins.size else 0.0,
        "avg_loss": float(losses.mean()) if losses.size else 0.0,
        "largest_win": float(wins.max()) if wins.size else 0.0,
        "largest_loss": float(losses.min()) if losses.size else 0.0,
    }


def compute_perf(equity: np.ndarray, trades: pd.DataFrame,
                 bars_per_year: float,
                 rf_per_bar: float = 0.0) -> PerfStats:
    """One-shot bundle of all common metrics."""
    eq = np.asarray(equity, dtype=np.float64)
    n = eq.shape[0]
    r = equity_returns(eq)
    mdd, mdur = max_drawdown(eq)
    cg = cagr(eq, bars_per_year)
    vol = float(np.std(r, ddof=1) * np.sqrt(bars_per_year)) if r.shape[0] > 1 else 0.0
    calmar = (cg / abs(mdd)) if mdd < 0 else 0.0
    ts = trade_stats(trades)
    total = float(eq[-1] / eq[0] - 1.0) if n >= 2 and eq[0] > 0 else 0.0
    return PerfStats(
        n_bars=n,
        n_trades=ts["n_trades"],
        final_balance=float(eq[-1]) if n else 0.0,
        total_return=total,
        cagr=cg,
        sharpe=sharpe_ratio(r, bars_per_year, rf_per_bar),
        sortino=sortino_ratio(r, bars_per_year, rf_per_bar),
        calmar=float(calmar),
        max_drawdown=mdd,
        max_drawdown_duration=mdur,
        volatility_ann=vol,
        win_rate=ts["win_rate"],
        profit_factor=ts["profit_factor"],
        expectancy=ts["expectancy"],
        expectancy_R=ts["expectancy_R"],
        avg_win=ts["avg_win"],
        avg_loss=ts["avg_loss"],
        largest_win=ts["largest_win"],
        largest_loss=ts["largest_loss"],
    )


# ---------------------------------------------------------------------------
# Bars-per-year helpers — caller still supplies; this is just a registry.
# ---------------------------------------------------------------------------
BARS_PER_YEAR = {
    "1m": 365 * 24 * 60,        # 24/7 crypto
    "5m": 365 * 24 * 12,
    "15m": 365 * 24 * 4,
    "1h": 365 * 24,
    "4h": 365 * 6,
    "1d": 365,
    "1w": 52,
}

# Equity/FX: explicit per-timeframe table built from session-hour fundamentals,
# NOT a flat 252/365 rescale (which is dimensionally wrong for daily/weekly bars).
_TRADING_HOURS_PER_DAY = 6.5
_TRADING_DAYS_PER_YEAR = 252
BARS_PER_YEAR_MARKET_HOURS = {
    "1m":  int(_TRADING_DAYS_PER_YEAR * _TRADING_HOURS_PER_DAY * 60),   # 98_280
    "5m":  int(_TRADING_DAYS_PER_YEAR * _TRADING_HOURS_PER_DAY * 12),   # 19_656
    "15m": int(_TRADING_DAYS_PER_YEAR * _TRADING_HOURS_PER_DAY * 4),    # 6_552
    "1h":  int(_TRADING_DAYS_PER_YEAR * _TRADING_HOURS_PER_DAY),        # 1_638
    "4h":  int(_TRADING_DAYS_PER_YEAR * (_TRADING_HOURS_PER_DAY / 4)),  # ~409
    "1d":  _TRADING_DAYS_PER_YEAR,                                      # 252
    "1w":  _TRADING_DAYS_PER_YEAR // 5,                                 # 50
}


def bars_per_year(timeframe: str, market_hours: bool = False) -> float:
    """Return annualization factor for a given timeframe.

    `market_hours=True` uses an explicit per-TF table derived from a 6.5h
    trading session over 252 trading days/year (equities/FX). The crypto
    default assumes 24/7. Caller is responsible for choosing.
    """
    table = BARS_PER_YEAR_MARKET_HOURS if market_hours else BARS_PER_YEAR
    base = table.get(timeframe)
    if base is None:
        raise KeyError(f"unknown timeframe {timeframe!r}; pass float instead")
    return float(base)
