"""Trading-state buckets in America/New_York time (DST-aware), as an exact partition of a UTC window.

States are evaluated minute by minute and run-length encoded into intervals, so every instant
belongs to exactly one bucket. Holidays come from `exchange_calendars` (XNYS). US holidays
count as "closed" for the equity states. For CME we treat them as a CME-closed day as well,
which is an approximation (DECISIONS.md).

Per asset (the order of `BUCKETS`):
- WTI, GOLD (CME Globex): us_day | globex_extended | daily_break | weekend
- SNDK, MU (Nasdaq): regular | pre_market | after_hours | overnight | weekend
The `transition` flag marks minutes within ±10 min of 09:30, 16:00 or the Sunday 18:00 reopen.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")
TRANSITION_MIN = 10
CME_ASSETS = ("WTI", "GOLD")
EQUITY_ASSETS = ("SNDK", "MU")
_CME_STATES = ["us_day", "globex_extended", "daily_break", "weekend"]
_EQUITY_STATES = ["regular", "pre_market", "after_hours", "overnight", "weekend"]
BUCKETS = {**{a: _CME_STATES for a in CME_ASSETS},
           **{a: _EQUITY_STATES for a in EQUITY_ASSETS}}


@lru_cache(maxsize=1)
def _nyse_holidays() -> frozenset:
    import exchange_calendars as xc
    cal = xc.get_calendar("XNYS")
    days = cal.regular_holidays.holidays(pd.Timestamp("2024-01-01"), pd.Timestamp("2027-12-31"))
    return frozenset(d.date() for d in days)


def _et_minutes(t0: datetime, t1: datetime) -> pd.DatetimeIndex:
    if t0.tzinfo is None or t1.tzinfo is None:
        raise ValueError("use timezone-aware UTC datetimes")
    return pd.date_range(t0, t1, freq="1min", inclusive="left", tz="UTC").tz_convert(ET)


def equity_state(et: pd.DatetimeIndex) -> np.ndarray:
    hol = _nyse_holidays()
    mins = et.hour * 60 + et.minute
    dow = et.dayofweek                     # Mon=0 .. Sun=6
    is_hol = np.array([d in hol for d in et.date])
    weekday = (dow < 5) & ~is_hol
    st = np.full(len(et), "closed", dtype=object)
    st[weekday & (mins >= 4 * 60) & (mins < 9 * 60 + 30)] = "pre_market"
    st[weekday & (mins >= 9 * 60 + 30) & (mins < 16 * 60)] = "regular"
    st[weekday & (mins >= 16 * 60) & (mins < 20 * 60)] = "after_hours"
    return st


def cme_state(et: pd.DatetimeIndex) -> np.ndarray:
    """CME Globex (CL, GC): open Sun 18:00 -> Fri 17:00 ET, daily break 17:00-18:00 Mon-Thu."""
    hol = _nyse_holidays()
    mins = et.hour * 60 + et.minute
    dow = et.dayofweek
    st = np.full(len(et), "open", dtype=object)
    st[(dow == 5)] = "closed"                                   # Saturday
    st[(dow == 4) & (mins >= 17 * 60)] = "closed"               # Friday after 17:00
    st[(dow == 6) & (mins < 18 * 60)] = "closed"                # Sunday before 18:00
    st[(dow <= 3) & (mins >= 17 * 60) & (mins < 18 * 60)] = "break"
    is_hol = np.array([d in hol for d in et.date])
    st[is_hol & (mins < 17 * 60)] = "closed"                    # approximation: holiday session closed
    return st


def asset_bucket(asset: str, et: pd.DatetimeIndex) -> np.ndarray:
    eq, cme = equity_state(et), cme_state(et)
    mins = et.hour * 60 + et.minute
    weekday = et.dayofweek < 5
    if asset in CME_ASSETS:
        b = np.where(cme == "open", "globex_extended", np.where(cme == "break", "daily_break", "weekend"))
        b = np.where((cme == "open") & (eq == "regular"), "us_day", b)
        return b
    if asset in EQUITY_ASSETS:
        b = eq.copy()
        closed = eq == "closed"
        # nights between two trading days are "overnight"; anything touching a weekend/holiday gap is "weekend"
        et_prev = et - pd.Timedelta(hours=8)
        et_next = et + pd.Timedelta(hours=8)
        hol = _nyse_holidays()
        trade_prev = (et_prev.dayofweek < 5) & ~np.array([d in hol for d in et_prev.date])
        trade_next = (et_next.dayofweek < 5) & ~np.array([d in hol for d in et_next.date])
        b[closed & trade_prev & trade_next] = "overnight"
        b[closed & ~(trade_prev & trade_next)] = "weekend"
        return b
    raise KeyError(asset)


def _transition(et: pd.DatetimeIndex) -> np.ndarray:
    mins = et.hour * 60 + et.minute
    marks = [9 * 60 + 30, 16 * 60]
    near = np.zeros(len(et), dtype=bool)
    for m in marks:
        near |= np.abs(mins - m) < TRANSITION_MIN
    near |= (et.dayofweek == 6) & (np.abs(mins - 18 * 60) < TRANSITION_MIN)
    return near


def bucket_intervals(asset: str, t0: datetime, t1: datetime) -> pd.DataFrame:
    """Exact partition of [t0, t1) into (start_utc, end_utc, bucket, transition, et_date) intervals."""
    et = _et_minutes(t0, t1)
    b = asset_bucket(asset, et)
    tr = _transition(et)
    key = pd.Series(b).astype(str) + "|" + pd.Series(tr).astype(str)
    change = np.r_[True, key.values[1:] != key.values[:-1]]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(et)]
    utc = et.tz_convert("UTC")
    end_ts = [utc[e] if e < len(et) else pd.Timestamp(t1) for e in ends]
    return pd.DataFrame({"start_utc": utc[starts], "end_utc": pd.DatetimeIndex(end_ts).tz_convert("UTC"),
                         "bucket": b[starts], "transition": tr[starts],
                         "et_date": [d.date() for d in et[starts]]})
