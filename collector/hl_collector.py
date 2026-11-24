"""Live capture of Hyperliquid websocket market data (bbo, trades, l2Book, activeAssetCtx).

Every message is stored as received, together with our receive time, in hourly
gzipped JSON-lines files:

    <out>/raw/<YYYYMMDD>/<HH>_conn<k>_<run>.jsonl.gz   one line: {"r": recv_ns, "m": <raw message>}
    <out>/events/<YYYYMMDD>.jsonl                   connects, disconnects, subscription acks

The exchange time inside each message ("time", block time) is kept untouched;
recv_ns is our wall clock (time.time_ns()) and is only used to measure delivery
delay and to detect gaps. Several connections run in parallel with identical
subscriptions, so a drop on one does not leave a hole; duplicates are removed
when converting to parquet. <run> is the process start time, so a restart never
appends to a file that a hard kill may have left truncated.

After every (re)connect the server first sends a snapshot: the current bbo, book
and context, and the last 30 trades (up to about a minute old). Deduplicate trades
on (coin, time, tid) and bbo on (coin, time, payload) when converting.

Only depends on `websockets`. Run:  python hl_collector.py --out /data/hl
"""

import argparse
import asyncio
import gzip
import json
import logging
import os
import random
import signal
import time

import websockets

WS_URL = "wss://api.hyperliquid.xyz/ws"
COINS = ["xyz:CL", "xyz:GOLD", "xyz:SNDK", "xyz:MU"]
CHANNELS = ["bbo", "trades", "l2Book", "activeAssetCtx"]
PING_EVERY = 30      # server closes after 60 s without a message from us
SILENCE_LIMIT = 60   # reconnect if nothing arrives for this long
FLUSH_EVERY = 5      # seconds between flushes of the open gzip file

log = logging.getLogger("hl_collector")


RUN_ID = time.strftime("%Y%m%dT%H%M%S", time.gmtime())


class HourlyWriter:
    """Write lines to <out>/raw/<day>/<hour>_conn<k>_<run>.jsonl.gz, rolling every UTC hour."""

    def __init__(self, out, conn_id):
        self.out, self.conn_id = out, conn_id
        self.key, self.fh, self.last_flush = None, None, 0.0

    def write(self, recv_ns, raw):
        t = time.gmtime(recv_ns // 1_000_000_000)
        key = (time.strftime("%Y%m%d", t), time.strftime("%H", t))
        if key != self.key:
            self.close()
            d = os.path.join(self.out, "raw", key[0])
            os.makedirs(d, exist_ok=True)
            # append mode: a reconnect within the hour of the same run continues the same file
            name = f"{key[1]}_conn{self.conn_id}_{RUN_ID}.jsonl.gz"
            self.fh = gzip.open(os.path.join(d, name), "at")
            self.key = key
        self.fh.write('{"r":%d,"m":%s}\n' % (recv_ns, raw))
        now = time.monotonic()
        if now - self.last_flush > FLUSH_EVERY:
            self.fh.flush()
            self.last_flush = now

    def close(self):
        if self.fh is not None:
            self.fh.close()
            self.fh = None


def event(out, **fields):
    """Append one bookkeeping event (connect, disconnect, ack) to the daily events file."""
    fields["t_ns"] = time.time_ns()
    d = os.path.join(out, "events")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, time.strftime("%Y%m%d", time.gmtime()) + ".jsonl"), "a") as f:
        f.write(json.dumps(fields) + "\n")


async def pinger(ws):
    """Send a ping every PING_EVERY seconds, whether or not data is flowing."""
    while True:
        await asyncio.sleep(PING_EVERY)
        await ws.send('{"method":"ping"}')


async def run_connection(conn_id, out, stop):
    writer = HourlyWriter(out, conn_id)
    backoff = 1
    try:
        while not stop.is_set():
            try:
                async with websockets.connect(WS_URL, ping_interval=None, max_size=2**24,
                                              open_timeout=20) as ws:
                    event(out, conn=conn_id, kind="connect")
                    for coin in COINS:
                        for ch in CHANNELS:
                            await ws.send(json.dumps({"method": "subscribe",
                                                      "subscription": {"type": ch, "coin": coin}}))
                    backoff = 1
                    ping_task = asyncio.create_task(pinger(ws))
                    last_data = time.monotonic()
                    try:
                        while not stop.is_set():
                            # silence counts data only: pongs keep arriving on a stalled feed
                            wait = last_data + SILENCE_LIMIT - time.monotonic()
                            try:
                                raw = await asyncio.wait_for(ws.recv(), timeout=max(wait, 0.1))
                            except asyncio.TimeoutError:
                                event(out, conn=conn_id, kind="silence", seconds=SILENCE_LIMIT)
                                break
                            recv_ns = time.time_ns()
                            if raw.startswith('{"channel":"pong"'):
                                pass
                            elif raw.startswith('{"channel":"subscriptionResponse"') or \
                                    raw.startswith('{"channel":"error"'):
                                event(out, conn=conn_id, kind="ack", msg=raw[:300])
                            else:
                                writer.write(recv_ns, raw)
                                last_data = time.monotonic()
                    finally:
                        ping_task.cancel()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # network errors and anything unexpected: log, reconnect
                event(out, conn=conn_id, kind="disconnect", error=repr(e)[:300])
                log.warning("conn %d: %r", conn_id, e)
            if not stop.is_set():
                await asyncio.sleep(backoff + random.random())
                backoff = min(backoff * 2, 60)
    finally:
        writer.close()
        event(out, conn=conn_id, kind="stop")


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--connections", type=int, default=3)
    p.add_argument("--duration", type=float, default=0, help="seconds; 0 = run until stopped")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(s, stop.set)
    if a.duration > 0:
        loop.call_later(a.duration, stop.set)

    event(a.out, kind="start", coins=COINS, channels=CHANNELS, connections=a.connections)
    tasks = [asyncio.create_task(run_connection(k, a.out, stop)) for k in range(a.connections)]
    await stop.wait()
    # a connection blocked in recv() notices the stop only at the next message; cancel it
    await asyncio.sleep(1)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
