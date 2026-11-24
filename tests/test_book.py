import numpy as np
import pandas as pd

from llrwa.data.book import rebuild_bbo


def _snap(snap_id, ts, bids, asks):
    rows = [(snap_id, ts, 1, p, v) for p, v in bids] + [(snap_id, ts, -1, p, v) for p, v in asks]
    return pd.DataFrame(rows, columns=["snap_id", "ts_ns", "side", "price", "volume"])


def test_rebuild_top_of_book_and_snapshot_check():
    s1 = _snap(1, 1000, [(99.0, 1), (98.0, 2)], [(101.0, 1), (102.0, 3)])
    s2 = _snap(2, 5000, [(100.0, 4), (99.0, 1)], [(101.0, 1)])
    ev = pd.DataFrame({
        "ts_ns": [1500, 2000, 2500, 3000, 3500],
        "seq":   [1, 2, 3, 4, 5],
        "side":  [1, 1, -1, -1, 1],
        "price": [100.0, 100.0, 101.0, 102.0, 100.0],
        "volume": [4.0, 4.0, 0.0, 5.0, 4.0],   # new best bid; unchanged update; delete best ask; resize; no-op
    })
    bbo, checks = rebuild_bbo(pd.concat([s1, s2]), ev)
    assert list(bbo["ts_ns"]) == [1000, 1500, 2500, 3000, 5000]
    assert list(bbo["bid"]) == [99.0, 100.0, 100.0, 100.0, 100.0]
    assert list(bbo["ask"]) == [101.0, 101.0, 102.0, 102.0, 101.0]
    assert list(bbo["askVolume"])[2:4] == [3.0, 5.0]
    assert len(checks) == 1 and not checks["match"].iloc[0]       # snapshot 2 shows ask 101 again
    assert checks["rebuilt_ask"].iloc[0] == 102.0 and checks["snap_ask"].iloc[0] == 101.0


def test_events_before_first_snapshot_are_ignored():
    s1 = _snap(1, 1000, [(99.0, 1)], [(101.0, 1)])
    ev = pd.DataFrame({"ts_ns": [500, 1500], "seq": [1, 2], "side": [1, 1], "price": [99.5, 99.5],
                       "volume": [1.0, 2.0]})
    bbo, _ = rebuild_bbo(s1, ev)
    assert list(bbo["bid"]) == [99.0, 99.5] and np.isclose(bbo["bidVolume"].iloc[-1], 2.0)
