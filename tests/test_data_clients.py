import gzip
import json
from datetime import date, datetime, timezone

import pyarrow.parquet as pq
import pytest

from llrwa.data.amber import AmberClient, RateLimiter, pull_tickers
from llrwa.data.manifest import completed_paths, time_chunks
from llrwa.data.tardis import OUT_SCHEMA, csv_name, csv_to_parquet


class FakeResponse:
    def __init__(self, status, payload=None, headers=None):
        self.status_code = status
        self.headers = headers or {}
        self.content = json.dumps({"payload": payload} if payload is not None else {}).encode()
        self.text = self.content.decode()


class FakeSession:
    """Scripted responses; records every request."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self.script.pop(0)


def _row(ts, bid, ask, ns=0):
    return {"instrument": "CLUSDT", "exchange": "binance", "exchangeTimestamp": ts,
            "exchangeTimestampNanoseconds": ns, "bid": str(bid), "ask": ask, "bidVolume": "1.5",
            "askVolume": 2, "sequence": 123456789012345678901, "extraField": 1}


def test_pagination_retry_after_and_atomic_write(tmp_path):
    sleeps = []
    script = [
        FakeResponse(429, headers={"Retry-After": "0.5"}),
        FakeResponse(200, {"data": [_row(1000, 90.0, 90.01), _row(1002, 90.0, 90.02)],
                           "metadata": {"next": "https://api.amberdata.com/markets/next-page?cursor=abc"}}),
        FakeResponse(503),
        FakeResponse(200, {"data": [_row(1500, 90.01, 90.02, ns=250_000)], "metadata": {}}),
    ]
    sess = FakeSession(script)
    client = AmberClient("k", session=sess, sleep=sleeps.append, rate_per_s=1e9)
    t0 = datetime(2026, 9, 24, 14, tzinfo=timezone.utc)
    t1 = datetime(2026, 9, 24, 15, tzinfo=timezone.utc)
    out = tmp_path / "chunk.parquet"
    rec = client.pull_tickers_chunk(asset="WTI", venue="binance", exchange="binance", instrument="CLUSDT",
                                    t0=t0, t1=t1, out_path=out, log_path=tmp_path / "log.jsonl")
    assert rec.status == "ok" and rec.rows == 3 and rec.pages == 2 and rec.retries == 2
    assert 0.5 in sleeps                                        # Retry-After honoured
    assert sess.calls[0][1]["startDate"] == int(t0.timestamp() * 1000)
    assert sess.calls[2][1] is None                             # params only on the first request
    assert out.exists() and not (tmp_path / "chunk.parquet.partial").exists()
    tbl = pq.read_table(out)
    assert tbl.column("bid").to_pylist() == [90.0, 90.0, 90.01]           # strings coerced
    assert tbl.column("exchangeTimestampNanoseconds").to_pylist() == [0, 0, 250_000]
    assert tbl.column("sequence").to_pylist()[0] == "123456789012345678901"  # big ints kept exact
    assert rec.max_gap_ms == 498.0
    assert "extraField" in rec.error                            # unmapped fields reported, not dropped silently
    assert str(out) in completed_paths(tmp_path / "log.jsonl")


def test_failed_chunk_leaves_no_file_and_is_retried(tmp_path):
    sess = FakeSession([FakeResponse(400)])
    client = AmberClient("k", session=sess, sleep=lambda s: None, rate_per_s=1e9)
    t0 = datetime(2026, 9, 24, 14, tzinfo=timezone.utc)
    rec = client.pull_tickers_chunk(asset="WTI", venue="binance", exchange="binance", instrument="CLUSDT",
                                    t0=t0, t1=t0.replace(hour=15), out_path=tmp_path / "c.parquet",
                                    log_path=tmp_path / "log.jsonl")
    assert rec.status == "error" and not (tmp_path / "c.parquet").exists()
    assert completed_paths(tmp_path / "log.jsonl") == set()


def test_resume_skips_completed_chunks(tmp_path):
    t0 = datetime(2026, 9, 24, 14, tzinfo=timezone.utc)
    t1 = datetime(2026, 9, 24, 16, tzinfo=timezone.utc)
    page = lambda ts: FakeResponse(200, {"data": [_row(ts, 90.0, 90.01)], "metadata": {}})
    client = AmberClient("k", session=FakeSession([page(1), page(2)]), sleep=lambda s: None, rate_per_s=1e9)
    recs = pull_tickers(client, asset="WTI", venue="binance", exchange="binance", instrument="CLUSDT",
                        t0=t0, t1=t1, root=tmp_path, chunk_minutes=60, workers=1)
    assert [r.status for r in recs] == ["ok", "ok"]
    client2 = AmberClient("k", session=FakeSession([]), sleep=lambda s: None, rate_per_s=1e9)
    assert pull_tickers(client2, asset="WTI", venue="binance", exchange="binance", instrument="CLUSDT",
                        t0=t0, t1=t1, root=tmp_path, chunk_minutes=60, workers=1) == []


def test_time_chunks_half_open_and_utc_only():
    t0 = datetime(2026, 9, 18, 20, tzinfo=timezone.utc)
    t1 = datetime(2026, 9, 18, 21, 20, tzinfo=timezone.utc)
    ch = time_chunks(t0, t1, 30)
    assert ch[0][0] == t0 and ch[-1][1] == t1 and len(ch) == 3
    assert all(a < b for a, b in ch) and all(ch[i][1] == ch[i + 1][0] for i in range(len(ch) - 1))
    with pytest.raises(ValueError):
        time_chunks(datetime(2026, 1, 1), datetime(2026, 1, 2), 60)


def test_rate_limiter_spacing():
    now = [0.0]
    slept = []
    lim = RateLimiter(10.0, clock=lambda: now[0], sleep=slept.append)
    for _ in range(3):
        lim.acquire()
    assert slept == pytest.approx([0.1, 0.2])


def test_tardis_csv_conversion_to_ns(tmp_path):
    csv = tmp_path / csv_name("hyperliquid", "book_ticker", date(2026, 9, 1), "xyz:CL")
    assert csv.name == "hyperliquid_book_ticker_2026-09-01_xyz-CL.csv.gz"
    lines = ["exchange,symbol,timestamp,local_timestamp,ask_amount,ask_price,bid_price,bid_amount",
             "hyperliquid,XYZ:CL,1788220800123456,1788220800350000,5.0,70.12,70.10,3.0",
             "hyperliquid,XYZ:CL,1788220800254000,1788220800480000,4.0,70.13,70.11,2.0"]
    with gzip.open(csv, "wt") as f:
        f.write("\n".join(lines) + "\n")
    out = tmp_path / "day.parquet"
    st = csv_to_parquet(csv, out)
    tbl = pq.read_table(out)
    assert tbl.schema == OUT_SCHEMA and st["rows"] == 2
    assert tbl.column("ts_ns").to_pylist() == [1788220800123456000, 1788220800254000000]
    assert tbl.column("bid").to_pylist() == [70.10, 70.11]


def test_parse_json_keeps_big_integers_exact():
    from llrwa.data.amber import parse_json
    assert parse_json(b'{"a": 123456789012345678901}')["a"] == 123456789012345678901
    assert parse_json(b'{"a": 1790416300719}')["a"] == 1790416300719


def test_completed_paths_survive_moving_the_project(tmp_path):
    """Log written on another machine (absolute /root/project/... paths) still marks local files done."""
    f = tmp_path / "amberdata" / "tickers" / "WTI" / "binance" / "20260924" / "1400_1410.parquet"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x")
    rec = {"path": "/root/project/data/raw/amberdata/tickers/WTI/binance/20260924/1400_1410.parquet",
           "status": "ok"}
    missing = {"path": "/root/project/data/raw/amberdata/tickers/WTI/binance/20260924/1410_1420.parquet",
               "status": "ok"}
    (tmp_path / "pull_log.jsonl").write_text(json.dumps(rec) + "\n" + json.dumps(missing) + "\n")
    assert completed_paths(tmp_path / "pull_log.jsonl") == {str(f)}
