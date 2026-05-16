"""
Anchored VWAP, session VWAP and dev bands as a pattern layer.

Calendar boundaries (week / month / session) are computed from real
``datetime64`` values, not naive ``ms // 86_400_000 // 30`` arithmetic, so
anchors do not drift across years.
"""
from __future__ import annotations
import numpy as np
from ..indicators import anchored_vwap, vwap_bands


_MS_PER_DAY = 86_400_000
_MS_PER_HOUR = 3_600_000


def _change_mask(keys: np.ndarray) -> np.ndarray:
    change = np.empty(keys.shape[0], dtype=np.bool_)
    if keys.shape[0] == 0:
        return change
    change[0] = True
    if keys.shape[0] > 1:
        change[1:] = keys[1:] != keys[:-1]
    return change


def session_anchors(ts_ms: np.ndarray, session_hour_utc: int = 0) -> np.ndarray:
    """Return indices marking the start of each session day.

    A "session day" rolls over at ``session_hour_utc`` UTC, so a session that
    starts at 22:00 UTC and runs through the next morning is treated as a
    single contiguous day.
    """
    if ts_ms.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    shifted = ts_ms - int(session_hour_utc) * _MS_PER_HOUR
    days = shifted // _MS_PER_DAY
    return np.where(_change_mask(days))[0]


def htf_anchored_vwaps(price: np.ndarray, volume: np.ndarray, ts_ms: np.ndarray):
    """Anchor at the most recent ISO-week start and calendar-month start.

    Uses ``numpy.datetime64`` so week/month boundaries match the real
    calendar (no 30-day drift, ISO weeks start on Monday).
    """
    if ts_ms.shape[0] == 0:
        return {"weekly_avwap": np.empty(0), "monthly_avwap": np.empty(0)}

    dt = ts_ms.astype("datetime64[ms]")

    # ISO week: floor to Monday.
    days_since_epoch = (dt.astype("datetime64[D]")
                        - np.datetime64("1970-01-01", "D")).astype(np.int64)
    # 1970-01-01 was a Thursday (weekday=3); shift so Monday=0.
    iso_week = (days_since_epoch + 3) // 7

    # Calendar month index = year*12 + month.
    months_dt = dt.astype("datetime64[M]")
    month_idx = (months_dt - np.datetime64("1970-01", "M")).astype(np.int64)

    w_change = _change_mask(iso_week)
    m_change = _change_mask(month_idx)

    last_w = int(np.where(w_change)[0][-1])
    last_m = int(np.where(m_change)[0][-1])

    return {
        "weekly_avwap": anchored_vwap(price, volume, last_w),
        "monthly_avwap": anchored_vwap(price, volume, last_m),
    }
