from datetime import datetime, timezone

import pandas as pd
import pytest

from llrwa.calendars.buckets import BUCKETS, bucket_intervals
from llrwa.symbols import ASSETS

T0 = datetime(2026, 9, 18, 20, tzinfo=timezone.utc)
T1 = datetime(2026, 9, 28, 8, tzinfo=timezone.utc)


def _at(df, ts):
    t = pd.Timestamp(ts, tz="UTC")
    row = df[(df["start_utc"] <= t) & (df["end_utc"] > t)]
    assert len(row) == 1
    return row.iloc[0]["bucket"]


@pytest.mark.parametrize("asset", ASSETS)
def test_exact_partition(asset):
    df = bucket_intervals(asset, T0, T1)
    assert set(df["bucket"]) <= set(BUCKETS[asset])
    assert df["start_utc"].iloc[0] == pd.Timestamp(T0) and df["end_utc"].iloc[-1] == pd.Timestamp(T1)
    assert (df["start_utc"].iloc[1:].values == df["end_utc"].iloc[:-1].values).all()
    assert (df["end_utc"] > df["start_utc"]).all()


@pytest.mark.parametrize("asset", ["WTI", "GOLD"])
def test_cme_states(asset):
    df = bucket_intervals(asset, T0, T1)
    assert _at(df, "2026-09-24 14:00") == "us_day"            # 10:00 EDT
    assert _at(df, "2026-09-24 20:30") == "globex_extended"   # 16:30 EDT
    assert _at(df, "2026-09-24 21:30") == "daily_break"       # 17:30 EDT Thursday
    assert _at(df, "2026-09-25 21:30") == "weekend"           # 17:30 EDT Friday
    assert _at(df, "2026-09-27 21:59") == "weekend"           # Sunday 17:59 EDT
    assert _at(df, "2026-09-27 22:00") == "globex_extended"   # Sunday 18:00 EDT reopen


@pytest.mark.parametrize("asset", ["SNDK", "MU"])
def test_equity_states(asset):
    df = bucket_intervals(asset, T0, T1)
    assert _at(df, "2026-09-24 13:30") == "regular"
    assert _at(df, "2026-09-24 12:00") == "pre_market"
    assert _at(df, "2026-09-24 23:00") == "after_hours"
    assert _at(df, "2026-09-25 02:00") == "overnight"         # Thu 22:00 EDT
    assert _at(df, "2026-09-26 02:00") == "weekend"           # Fri 22:00 EDT
    assert _at(df, "2026-09-28 07:59") == "weekend"           # Mon 03:59 EDT


def test_dst_fall_back_partition():
    t0 = datetime(2026, 10, 30, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 11, 3, 0, tzinfo=timezone.utc)
    df = bucket_intervals("WTI", t0, t1)
    assert (df["start_utc"].iloc[1:].values == df["end_utc"].iloc[:-1].values).all()
    assert _at(df, "2026-11-01 22:59") == "weekend"           # Sun 17:59 EST (UTC-5)
    assert _at(df, "2026-11-01 23:00") == "globex_extended"   # Sun 18:00 EST
    assert _at(df, "2026-11-02 14:30") == "us_day"            # Mon 09:30 EST


def test_transition_flag_around_open():
    df = bucket_intervals("MU", T0, T1)
    t = pd.Timestamp("2026-09-24 13:25", tz="UTC")
    row = df[(df["start_utc"] <= t) & (df["end_utc"] > t)].iloc[0]
    assert row["transition"]
