"""Phase 7 simulation study. Usage: python scripts/sim_study.py [E1 E2 ...] [--reps 500]

Writes per-repetition results to results/sim/{exp}.parquet (git-ignored) and summaries to
results/tables/sim_{exp}.csv. All seeds are deterministic: seed = 10_000 * setting_index + rep.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from llrwa.config import RESULTS
from llrwa.simulate.study import BASE, one_rep, summarise

# Calibrated to Phase 1 (WTI, weekday 14:00Z; DECISIONS D14, D19): vol ~0.85 bp/sqrt(s)
BIN = ("poisson", 30.0)          # Binance distinct update rate, tick 0.01 on ~94 = 1.06 bp
KRK = ("poisson", 35.0)          # Kraken rebuilt-book update rate, tick 0.001 = 0.106 bp
OKX = ("grid", 100.0)            # OKX Amberdata: 100 ms grid, ~65 ms stale
HLT = ("poisson", 0.56)          # Hyperliquid trades: distinct trade timestamps per second

EXPERIMENTS: dict[str, list[dict]] = {
    "E1_consistency": [dict(T_min=T) for T in (10, 60, 360)]
                      + [dict(x=("poisson", 10.0 * k), y=("poisson", 5.0 * k)) for k in (0.5, 2, 4, 8)],
    "E2_heterogeneous": [dict(x=("poisson", 40.0), y=("poisson", ly)) for ly in (0.5, 1, 2, 5, 10, 40)],
    "E3_snapshot": [dict(x=("poisson", 40.0), y=("grid", d)) for d in (10, 50, 100, 200, 500)]
                   + [dict(x=("grid_locked", 100.0, 0.0), y=("grid_locked", 100.0, 0.0), theta_ms=t) for t in (0, 30, 50)]
                   + [dict(x=("poisson", 40.0), y=("grid", 100.0), stale_y_ms=st) for st in (25, 65)],
    "E4_noise_tick": [dict(x=("poisson", 30.0), y=("poisson", 30.0), tick_bp=tb, change_only=co)
                      for tb in (0.1, 1.0, 2.0) for co in (False, True)]
                     + [dict(x=("poisson", 30.0), y=("poisson", 30.0), noise_bp=nb) for nb in (0.1, 0.5, 1.0)],
    "E5_baseline": [dict(theta_ms=t, x=("poisson", 50.0), y=("grid", 100.0), xcorr_bars_ms=(1, 10, 100))
                    for t in (0.0, 50.0)]
                   + [dict(theta_ms=t, x=("poisson", 40.0), y=("poisson", 2.0), xcorr_bars_ms=(1, 10, 100))
                      for t in (0.0, 50.0)],
    "E6_calibrated": [dict(vol_bp_per_sqrt_s=0.85, x=BIN, y=OKX, tick_bp=1.06, stale_y_ms=65.0, theta_ms=20.0, T_min=T)
                      for T in (60, 360)]
                     + [dict(vol_bp_per_sqrt_s=0.85, x=BIN, y=KRK, tick_bp=1.06, theta_ms=-30.0, T_min=T) for T in (60, 360)]
                     + [dict(vol_bp_per_sqrt_s=0.85, x=BIN, y=HLT, tick_bp=1.06, noise_bp=0.5, theta_ms=200.0,
                             max_lag_ms=2500.0, T_min=T) for T in (60, 360)],
}


def _job(args):
    setting, seed = args
    return one_rep(setting, seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="*", default=list(EXPERIMENTS))
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    (RESULTS / "sim").mkdir(parents=True, exist_ok=True)
    for exp in a.experiments:
        t0 = time.monotonic()
        rows, summ = [], []
        for si, setting in enumerate(EXPERIMENTS[exp]):
            jobs = [(setting, 10_000 * si + r) for r in range(a.reps)]
            with ProcessPoolExecutor(a.workers) as ex:
                reps = list(ex.map(_job, jobs, chunksize=10))
            df = pd.DataFrame(reps)
            label = ", ".join(f"{k}={v}" for k, v in setting.items() if k != "xcorr_bars_ms")
            df["setting"] = label
            rows.append(df)
            full = {**BASE, **setting}
            s = summarise(df, full["theta_ms"])          # error vs the true lag (staleness shows as bias)
            s.update({"setting": label, "theta_true_ms": full["theta_ms"]})
            for c in [c for c in df.columns if c.startswith("xc")]:
                s[c + "_mean"] = float(df[c].mean())
            summ.append(s)
            print(f"{exp} [{si + 1}/{len(EXPERIMENTS[exp])}] {label}: bias={s['bias_ms']:+.2f} rmse={s['rmse_ms']:.2f} "
                  f"({time.monotonic() - t0:.0f}s)", flush=True)
        pd.concat(rows, ignore_index=True).to_parquet(RESULTS / "sim" / f"{exp}.parquet", index=False)
        pd.DataFrame(summ).to_csv(RESULTS / "tables" / f"sim_{exp}.csv", index=False)


if __name__ == "__main__":
    main()
