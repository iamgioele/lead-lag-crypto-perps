"""Quick health report of the capture: files, sizes, message counts per channel, disk use.

    python health.py /home/hl/data            # all days
    python health.py /home/hl/data 20261027   # one day
"""

import collections
import gzip
import json
import os
import shutil
import sys

out = sys.argv[1]
days = sys.argv[2:] or sorted(os.listdir(os.path.join(out, "raw")))
for day in days:
    d = os.path.join(out, "raw", day)
    counts = collections.Counter()
    size = 0
    hours = set()
    for f in sorted(os.listdir(d)):
        size += os.path.getsize(os.path.join(d, f))
        hours.add(f[:2])
        try:
            for line in gzip.open(os.path.join(d, f), "rt"):
                m = json.loads(line)["m"]
                counts[(f.split("_")[1][:5], m["channel"])] += 1
        except (EOFError, OSError):
            counts[("open", "truncated")] += 1   # file still being written
    print(f"{day}: {len(hours)} hours, {size / 1e6:.1f} MB")
    for k, v in sorted(counts.items()):
        print(f"   {k[0]:6s} {k[1]:16s} {v}")

ev = os.path.join(out, "events")
bad = collections.Counter()
for f in sorted(os.listdir(ev)):
    for line in open(os.path.join(ev, f)):
        e = json.loads(line)
        if e["kind"] in ("disconnect", "silence", "start"):
            bad[(f[:8], e["kind"])] += 1
print("events:", dict(bad) or "none")
du = shutil.disk_usage(out)
print(f"disk: {du.used / du.total:.0%} used, {du.free / 1e9:.1f} GB free")
