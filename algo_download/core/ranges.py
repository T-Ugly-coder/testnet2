"""
Dealing range, premium/discount, OTE — corrected math.

Fixes vs. spec:
- Removed buggy SHORT inversion formula.
- Removed non-standard 0.705 fib level.
- Single source of truth for OTE bounds in both directions.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class DealingRange:
    swing_high: float
    swing_low: float
    equilibrium: float
    range_size: float
    ote_long: tuple[float, float]   # for LONG entries (in discount)
    ote_short: tuple[float, float]  # for SHORT entries (in premium)
    fib_levels: dict[str, float]


def compute_dealing_range(swing_high: float, swing_low: float) -> DealingRange:
    rs = swing_high - swing_low
    eq = swing_low + 0.5 * rs
    fib = {
        "0.0": swing_low,
        "0.236": swing_low + 0.236 * rs,
        "0.382": swing_low + 0.382 * rs,
        "0.5": eq,
        "0.618": swing_low + 0.618 * rs,
        "0.786": swing_low + 0.786 * rs,
        "1.0": swing_high,
    }
    # LONG entries are taken in DISCOUNT (lower half) at retracement of an
    # upmove from swing_low; OTE_LONG = [0.618, 0.786] retrace from low's side
    # measured upward from swing_low gives [low+0.618*rs, low+0.786*rs]?
    # Actually OTE for LONG entries (after price rallied to high then pulled
    # back): retracement levels from high; 0.618 retrace = swing_high - 0.618*rs
    #                                       0.786 retrace = swing_high - 0.786*rs
    ote_long = (swing_high - 0.786 * rs, swing_high - 0.618 * rs)
    # OTE for SHORT entries (after price dropped to low then bounced):
    # retracement from low up: 0.618 retrace = swing_low + 0.618*rs
    ote_short = (swing_low + 0.618 * rs, swing_low + 0.786 * rs)
    return DealingRange(swing_high, swing_low, eq, rs,
                        ote_long, ote_short, fib)


def is_premium(price: float, dr: DealingRange) -> bool:
    return price > dr.equilibrium


def is_discount(price: float, dr: DealingRange) -> bool:
    return price < dr.equilibrium


def in_ote(price: float, dr: DealingRange, direction: str) -> bool:
    if direction == "LONG":
        lo, hi = dr.ote_long
    else:
        lo, hi = dr.ote_short
    return lo <= price <= hi


def validate_entry_zone(price: float, direction: str, dr: DealingRange):
    if direction == "LONG":
        if not is_discount(price, dr):
            return False, "REJECTED_BUYING_PREMIUM"
        return True, "OTE_LONG" if in_ote(price, dr, "LONG") else "DISCOUNT_LONG"
    if direction == "SHORT":
        if not is_premium(price, dr):
            return False, "REJECTED_SELLING_DISCOUNT"
        return True, "OTE_SHORT" if in_ote(price, dr, "SHORT") else "PREMIUM_SHORT"
    return False, "UNKNOWN_DIRECTION"
