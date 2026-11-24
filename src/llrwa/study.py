"""Study windows: the single place where sample dates live.

Select a study with the environment variable LLRWA_STUDY (default "oct26", the paper's sample):

    LLRWA_STUDY=oct26 bash scripts/asset_pipeline.sh WTI

"sep26" is the original September sample (Hyperliquid from Amberdata trades). It is no longer
in the paper (DECISIONS D41) but keeps its original paths (data/clean, data/curves/<ASSET>,
results/tables). Other studies write to their own subfolders: data/clean/<subdir>,
data/curves/<subdir>/<ASSET>, results/<subdir>/{tables,figures}.

exclude_states: (asset, start UTC, end UTC) intervals left out of the trading-state estimates
(not of the whole-window estimate), e.g. an earnings release.

hl_source: where Hyperliquid comes from.
- "amberdata_trades": Amberdata trades (trade prices; Amberdata has no Hyperliquid quotes).
- "ws_bbo": our own websocket capture of the per-block best bid/offer (collector/, DECISIONS D33).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Study:
    name: str
    t0: str          # UTC start of the analysis window (inclusive)
    t1: str          # UTC end (exclusive)
    ref_day: str     # ET day used for the feed test and the reference-day bootstrap
    hl_source: str
    subdir: str      # "" keeps the original paths
    label: str       # period as printed in figure titles
    days: tuple = () # explicit ET days to clean; empty = every ET day overlapping the window
    exclude_states: tuple = ()  # ((asset, start_utc, end_utc), ...), see module docstring

    @property
    def T0(self) -> pd.Timestamp:
        return pd.Timestamp(self.t0, tz="UTC")

    @property
    def T1(self) -> pd.Timestamp:
        return pd.Timestamp(self.t1, tz="UTC")

    @property
    def et_days(self) -> list[str]:
        """ET calendar days to clean: `days` if given, else those overlapping the window (ends partial)."""
        if self.days:
            return list(self.days)
        d0 = self.T0.tz_convert(ET).date()
        d1 = (self.T1 - pd.Timedelta(microseconds=1)).tz_convert(ET).date()
        return [str(d0 + timedelta(days=k)) for k in range((d1 - d0).days + 1)]


STUDIES = {
    "sep26": Study("sep26", "2026-09-18 20:00", "2026-09-28 08:00", "2026-09-24", "amberdata_trades", "",
                   "18-25 Sep 2026", tuple(f"2026-09-{d:02d}" for d in range(18, 26))),
    # live capture started 26 Oct 14:10 UTC; window = the data available on 2 Nov (D41).
    # SanDisk reported earnings on 29 Oct after the close: 16:00 ET to the next 09:30 ET open is
    # left out of the SanDisk trading-state estimates.
    "oct26": Study("oct26", "2026-10-26 14:15", "2026-11-02 12:00", "2026-10-28", "ws_bbo", "oct26.nosync",
                   "26 Oct-2 Nov 2026",
                   exclude_states=(("SNDK", "2026-10-29 20:00", "2026-10-30 13:30"),)),
    # oct26 extended by one week of capture: two weeks, two weekends (D42). Same reference day and exclusion.
    "nov09": Study("nov09", "2026-10-26 14:15", "2026-11-09 12:00", "2026-10-28", "ws_bbo", "nov09.nosync",
                   "26 Oct-9 Nov 2026",
                   exclude_states=(("SNDK", "2026-10-29 20:00", "2026-10-30 13:30"),)),
}


def current() -> Study:
    name = os.environ.get("LLRWA_STUDY", "nov09")
    if name not in STUDIES:
        raise KeyError(f"unknown LLRWA_STUDY={name!r}; known: {sorted(STUDIES)}")
    return STUDIES[name]


def et_day_bounds(day: str) -> tuple[datetime, datetime]:
    """[00:00, 24:00) America/New_York of `day`, in UTC (DST-aware: 23 h or 25 h on switch days)."""
    d = date.fromisoformat(day)
    lo = datetime.combine(d, time(0), ET).astimezone(timezone.utc)
    hi = datetime.combine(d + timedelta(days=1), time(0), ET).astimezone(timezone.utc)
    return lo, hi


def et_dates(starts: pd.DatetimeIndex):
    """ET calendar date of each UTC timestamp."""
    return starts.tz_convert(ET).date


def state_mask(study: Study, asset: str, starts: pd.DatetimeIndex):
    """True for block starts that may enter trading-state estimates (False inside `exclude_states`)."""
    keep = np.ones(len(starts), dtype=bool)
    for a, lo, hi in study.exclude_states:
        if a == asset:
            keep &= ~((starts >= pd.Timestamp(lo, tz="UTC")) & (starts < pd.Timestamp(hi, tz="UTC")))
    return keep
