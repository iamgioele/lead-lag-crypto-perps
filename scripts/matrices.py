"""Phases 4 and 6: lead-lag estimates for every scope from the curve stores.

Scopes: whole window, each ET day, each UTC hour, and each trading-state bucket (pooled over
days, transitions separate). Output: results/tables/scopes_{asset}_{variant}.csv (long format).

OKX lags are also shown offset-adjusted for Amberdata's measured staleness (DECISIONS D21):
theta_adj = theta - 65 ms when OKX is the follower candidate, + 65 ms when OKX is the leader
candidate. Hyperliquid is trades-based and shown raw only (D20).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from llrwa.calendars.buckets import bucket_intervals
from llrwa.config import DATA_CURVES, RESULTS, STUDY
from llrwa.hry.peaks import DEFAULT_LLR_LAGSET_MS, far_band_z, llr_squared
from llrwa.hry.store import PairStore
from llrwa.study import et_dates, state_mask

OKX_STALENESS_MS = 65.0
SIGNAL_Z = 10.0  # ~99th percentile of the pedestal-robust far-band score under zero correlation (DECISIONS D23)
T0, T1 = STUDY.T0, STUDY.T1


def okx_adjust(leader: str, follower: str, theta: float) -> float:
    if follower == "okx":
        return theta - OKX_STALENESS_MS
    if leader == "okx":
        return theta + OKX_STALENESS_MS
    return theta


def estimate(st: PairStore, mask: np.ndarray, min_n: int) -> dict:
    nx, ny = int(st.nx[mask].sum()), int(st.ny[mask].sum())
    out = {"n_x": nx, "n_y": ny}
    if nx < min_n or ny < min_n:
        return {**out, "status": "insufficient"}
    pk, lv, s = st.peak(mask)
    lagset = DEFAULT_LLR_LAGSET_MS[DEFAULT_LLR_LAGSET_MS <= st.max_lag_ms]
    z = far_band_z(lv, s, pk.value)
    status = "censored" if pk.censored else ("ambiguous" if pk.ambiguous else "ok")
    if status != "censored" and not (np.isfinite(z) and z >= SIGNAL_Z):
        status = "no_signal"
    return {**out, "status": status,
            "theta_ms": pk.theta_ms, "peak_rho": pk.value, "second_ms": pk.second_ms,
            "llr_ha": llr_squared(lv, s, lagset), "far_z": z}


def scopes(asset: str, starts: pd.DatetimeIndex) -> list[tuple[str, str, np.ndarray]]:
    out = [("window", "all", np.ones(len(starts), dtype=bool))]
    et_date = np.asarray(et_dates(starts))
    for d in sorted(set(et_date)):
        out.append(("et_day", str(d), et_date == d))
    hours = starts.floor("h")
    for h in sorted(set(hours)):
        out.append(("utc_hour", h.strftime("%Y-%m-%d %H:00"), np.asarray(hours == h)))
    bk = bucket_intervals(asset, T0.to_pydatetime(), T1.to_pydatetime())
    idx = np.searchsorted(bk["start_utc"].values.astype("datetime64[ns]").astype(np.int64),
                          starts.values.astype("datetime64[ns]").astype(np.int64), side="right") - 1
    lab = np.where(bk["transition"].values[idx], "transition", bk["bucket"].values[idx])
    keep = state_mask(STUDY, asset, starts)            # e.g. an earnings evening (study.exclude_states)
    for b in sorted(set(lab)):
        out.append(("bucket", b, (lab == b) & keep))
    days = et_date
    for b in sorted(set(lab) - {"transition"}):
        for d in sorted(set(days[lab == b])):
            out.append(("bucket_day", f"{b}|{d}", (lab == b) & (days == d) & keep))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--variant", default="mid_all")
    ap.add_argument("--min-n", type=int, default=200)
    a = ap.parse_args()
    base = DATA_CURVES / a.asset / a.variant
    rows = []
    for f in sorted(base.glob("*.npz")):
        x, y = f.stem.split("__")
        st = PairStore.load(f)
        starts = pd.DatetimeIndex(pd.to_datetime(st.block_start_ns, utc=True))
        for kind, label, mask in scopes(a.asset, starts):
            r = estimate(st, mask, a.min_n if kind != "utc_hour" else max(50, a.min_n // 4))
            if "theta_ms" in r:
                r["theta_okx_adj_ms"] = okx_adjust(x, y, r["theta_ms"]) if "okx" in (x, y) else np.nan
            rows.append({"asset": a.asset, "variant": a.variant, "scope": kind, "label": label,
                         "leader_candidate": x, "follower_candidate": y, **r})
    df = pd.DataFrame(rows)
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    out = RESULTS / "tables" / f"scopes_{a.asset}_{a.variant}.csv"
    df.to_csv(out, index=False)
    ok = df[df["status"] == "ok"]
    hours = ok[ok["scope"] == "utc_hour"].groupby(["leader_candidate", "follower_candidate"])["theta_ms"]
    summ = df[df["scope"] == "window"][["leader_candidate", "follower_candidate", "status", "theta_ms", "peak_rho",
                                        "llr_ha", "far_z"]].copy()
    summ = summ.merge(hours.median().rename("hour_median_ms"), on=["leader_candidate", "follower_candidate"], how="left")
    summ = summ.merge((hours.quantile(0.75) - hours.quantile(0.25)).rename("hour_iqr_ms"),
                      on=["leader_candidate", "follower_candidate"], how="left")
    summ = summ.merge(hours.size().rename("n_hours"), on=["leader_candidate", "follower_candidate"], how="left")
    pd.set_option("display.width", 220)
    print(summ.round(2).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
