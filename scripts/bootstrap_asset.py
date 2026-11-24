"""Joint moving-block bootstrap for one asset: pair CIs, venue-offset CIs, bimodal mode shares.

Usage: python scripts/bootstrap_asset.py --asset WTI [--B 500]
Output: results/tables/bootstrap_{asset}.csv (pairs) and bootstrap_offsets_{asset}.csv.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from llrwa.calendars.buckets import bucket_intervals
from llrwa.config import DATA_CURVES, RESULTS, STUDY
from llrwa.hry.inference import bootstrap_pairs, mode_shares, summarise_bootstrap
from llrwa.hry.pairwise import venue_offsets
from llrwa.hry.store import PairStore
from llrwa.study import et_dates, state_mask

VENUES = ["binance", "okx", "kraken", "hyperliquid"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--variant", default="mid_all")
    ap.add_argument("--B", type=int, default=500)
    a = ap.parse_args()
    base = DATA_CURVES / a.asset / a.variant
    stores = {tuple(f.stem.split("__")): PairStore.load(f) for f in sorted(base.glob("*.npz"))}
    any_st = next(iter(stores.values()))
    starts = pd.to_datetime(any_st.block_start_ns, utc=True)
    et_date = np.asarray(et_dates(starts))
    scopes = {"window": np.ones(len(starts), dtype=bool),
              f"refday_{STUDY.ref_day}": et_date == pd.Timestamp(STUDY.ref_day).date()}
    bk = bucket_intervals(a.asset, starts[0].to_pydatetime(), (starts[-1] + pd.Timedelta(minutes=5)).to_pydatetime())
    idx = np.searchsorted(bk["start_utc"].values.astype("datetime64[ns]").astype(np.int64),
                          starts.values.astype("datetime64[ns]").astype(np.int64), side="right") - 1
    lab = np.where(bk["transition"].values[idx], "transition", bk["bucket"].values[idx])
    keep = state_mask(STUDY, a.asset, starts)
    for b in sorted(set(lab) - {"transition"}):
        scopes[f"bucket:{b}"] = (lab == b) & keep
    out, offs = [], []
    for name, mask in scopes.items():
        bs = bootstrap_pairs(stores, mask, B=a.B, seed=7)
        s = summarise_bootstrap(bs).assign(scope=name)
        ms = mode_shares(bs, split_ms=20.0).rename(columns={"share_below": "share_below_20ms",
                                                            "share_above": "share_above_20ms"})
        out.append(s.merge(ms, on=["leader_candidate", "follower_candidate"]))
        taus = []
        sc = pd.read_csv(RESULTS / "tables" / f"scopes_{a.asset}_{a.variant}.csv").query("scope == 'window'")
        good = set(zip(sc.loc[sc["status"].isin(["ok", "ambiguous"]), "leader_candidate"],
                       sc.loc[sc["status"].isin(["ok", "ambiguous"]), "follower_candidate"]))
        width = s.assign(w=s["ci_hi_ms"] - s["ci_lo_ms"])
        good -= set(zip(width.loc[width["w"] > 100, "leader_candidate"], width.loc[width["w"] > 100, "follower_candidate"]))
        for b, g in bs.groupby("rep"):
            ok = g.apply(lambda r: (r["leader_candidate"], r["follower_candidate"]) in good, axis=1)
            tau, _ = venue_offsets(g.assign(censored=g["censored"] | ~ok), VENUES, "binance")
            taus.append(tau)
        T = pd.DataFrame(taus)
        offs.append(pd.DataFrame({"scope": name, "venue": T.columns, "tau_ci_lo_ms": T.quantile(0.025).values,
                                  "tau_ci_hi_ms": T.quantile(0.975).values, "tau_boot_median_ms": T.median().values}))
    res = pd.concat(out, ignore_index=True)
    res.to_csv(RESULTS / "tables" / f"bootstrap_{a.asset}.csv", index=False)
    of = pd.concat(offs, ignore_index=True)
    of.to_csv(RESULTS / "tables" / f"bootstrap_offsets_{a.asset}.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.round(2).to_string(index=False))
    print(of.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
