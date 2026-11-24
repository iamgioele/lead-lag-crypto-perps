"""Study windows (DST-aware ET days) and the Hyperliquid websocket reader (union over connections)."""
import gzip
import json
from datetime import datetime, timezone

from llrwa.data.hl_ws import load_bbo
from llrwa.study import STUDIES, et_day_bounds


def test_et_day_bounds_dst():
    assert et_day_bounds("2026-09-24") == (datetime(2026, 9, 24, 4, tzinfo=timezone.utc),
                                           datetime(2026, 9, 25, 4, tzinfo=timezone.utc))
    lo, hi = et_day_bounds("2026-11-01")          # clocks go back: 25-hour day
    assert (hi - lo).total_seconds() == 25 * 3600
    assert et_day_bounds("2026-11-02")[0] == datetime(2026, 11, 2, 5, tzinfo=timezone.utc)


def test_sep26_keeps_original_days():
    assert STUDIES["sep26"].et_days == [f"2026-09-{d:02d}" for d in range(18, 26)]
    assert STUDIES["sep26"].subdir == ""


def _msg(t, bid, ask, coin="xyz:CL"):
    return {"channel": "bbo", "data": {"coin": coin, "time": t, "bbo": [{"px": str(bid), "sz": "1", "n": 1},
                                                                        {"px": str(ask), "sz": "2", "n": 1}]}}


def test_load_bbo_union_and_truncation(tmp_path):
    t0 = datetime(2026, 10, 26, 14, tzinfo=timezone.utc)
    base = int(t0.timestamp() * 1000) + 1000
    d = tmp_path / "20261026"
    d.mkdir()
    # conn0 misses update 2, conn1 misses update 3; a GOLD line and a trades line must be ignored
    c0 = [_msg(base, 100, 101), _msg(base + 100, 100, 102), _msg(base + 150, 99, 101, coin="xyz:GOLD")]
    c1 = [_msg(base, 100, 101), _msg(base + 50, 100.5, 101)]
    for k, msgs in ((0, c0), (1, c1)):
        with gzip.open(d / f"14_conn{k}_run.jsonl.gz", "wt") as f:
            for i, m in enumerate(msgs):
                f.write(json.dumps({"r": (base + 400 + i) * 10**6, "m": m}) + "\n")
            f.write(json.dumps({"r": 1, "m": {"channel": "trades", "data": [{"coin": "xyz:CL"}]}}) + "\n")
    # a file cut by a hard kill: valid gzip prefix only
    raw = gzip.compress((json.dumps({"r": 1, "m": _msg(base + 200, 100, 103)}) + "\n").encode())
    (d / "14_conn2_run.jsonl.gz").write_bytes(raw[:-8])
    df = load_bbo("WTI", t0, datetime(2026, 10, 26, 15, tzinfo=timezone.utc), root=tmp_path)
    assert list(df["ts_ns"] // 10**6) == [base, base + 50, base + 100, base + 200]
    assert list(df["n_conn"]) == [2, 1, 1, 1]
    assert list(df["ask"]) == [101, 101, 102, 103]


def test_state_mask_excludes_sandisk_earnings_only():
    import pandas as pd
    from llrwa.study import state_mask
    st = STUDIES["oct26"]
    starts = pd.DatetimeIndex(["2026-10-29 19:55", "2026-10-29 20:00", "2026-10-30 13:25", "2026-10-30 13:30"], tz="UTC")
    assert list(state_mask(st, "SNDK", starts)) == [True, False, False, True]
    assert state_mask(st, "MU", starts).all()


def test_load_trades_dedupes_replay(tmp_path):
    from llrwa.data.hl_ws import load_trades
    t0 = datetime(2026, 10, 26, 14, tzinfo=timezone.utc)
    base = int(t0.timestamp() * 1000) + 1000
    d = tmp_path / "20261026"
    d.mkdir()
    tr = lambda t, tid, coin="xyz:CL": {"coin": coin, "side": "B", "px": "100", "sz": "1", "hash": "h", "time": t,
                                         "tid": tid, "users": ["a", "b"]}
    for k, batch in ((0, [[tr(base, 1), tr(base + 50, 2)], [tr(base, 1), tr(base + 50, 2), tr(base + 90, 3)]]),
                     (1, [[tr(base + 50, 2), tr(base + 90, 3), tr(base + 95, 9, coin="xyz:GOLD")]])):
        with gzip.open(d / f"14_conn{k}_run.jsonl.gz", "wt") as f:
            for msgs in batch:            # the second conn0 message is a reconnect replay
                f.write(json.dumps({"r": 1, "m": {"channel": "trades", "data": msgs}}) + "\n")
    df = load_trades("WTI", t0, datetime(2026, 10, 26, 15, tzinfo=timezone.utc), root=tmp_path)
    assert list(df["tid"]) == ["1", "2", "3"]
