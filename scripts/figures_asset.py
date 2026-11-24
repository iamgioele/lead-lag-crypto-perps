"""Figures and summary tables for one asset from the curve store and scopes table.

Usage: python scripts/figures_asset.py --asset WTI [--variant mid_all]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from llrwa.calendars.buckets import BUCKETS
from llrwa.config import DATA_CURVES, RESULTS, STUDY
from llrwa.hry.pairwise import venue_offsets
from llrwa.hry.store import PairStore
from llrwa.plots import contrast_panels, lag_across_buckets, lag_heatmap, robustness_panels
from llrwa.symbols import ASSET_NAMES

VENUES = ["binance", "okx", "kraken", "hyperliquid"]
NAMES = ASSET_NAMES


def poorly_located(asset: str, scope: str) -> set:
    f = RESULTS / "tables" / f"bootstrap_{asset}.csv"
    if not f.exists():
        return set()
    b = pd.read_csv(f).query("scope == @scope")
    wide = b[(b["ci_hi_ms"] - b["ci_lo_ms"]) > 100]
    return set(zip(wide["leader_candidate"], wide["follower_candidate"]))


def matrix(res: pd.DataFrame, col: str, exclude: set = frozenset()) -> pd.DataFrame:
    M = pd.DataFrame(np.nan, index=VENUES, columns=VENUES)
    for _, r in res.iterrows():
        if (r["leader_candidate"], r["follower_candidate"]) in exclude:
            continue
        if r["status"] in ("ok", "ambiguous") and np.isfinite(r[col]):
            M.loc[r["leader_candidate"], r["follower_candidate"]] = r[col]
            M.loc[r["follower_candidate"], r["leader_candidate"]] = -r[col]
    return M


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--variant", default="mid_all")
    ap.add_argument("--ref-day", default=STUDY.ref_day)
    a = ap.parse_args()
    df = pd.read_csv(RESULTS / "tables" / f"scopes_{a.asset}_{a.variant}.csv")
    df["theta_adj"] = np.where(df["theta_okx_adj_ms"].notna(), df["theta_okx_adj_ms"], df["theta_ms"])
    fig = RESULTS / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    hl = "Hyperliquid from trades." if STUDY.hl_source == "amberdata_trades" else "Hyperliquid from per-block quotes."
    name = NAMES[a.asset]
    for scope, label, tag in (("window", "all", "window"), ("et_day", a.ref_day, "refday")):
        res = df[(df["scope"] == scope) & (df["label"] == label)]
        for col, ctag, note in (("theta_ms", "raw", "OKX as measured on Amberdata; " + hl),
                                ("theta_adj", "okxadj", "OKX shifted by its measured 65 ms feed staleness; " + hl)):
            M = matrix(res, col, poorly_located(a.asset, "window" if scope == "window" else f"refday_{label}"))
            period = STUDY.label if scope == "window" else f"ET day {label}"
            lag_heatmap(M, f"{name}: lead-lag, {period}", fig / f"heatmap_{a.asset}_{tag}_{ctag}.pdf",
                        note=note + " n/a: no signal or 95% interval wider than 100 ms.", vmax=500)
        if scope == "window":
            bad = poorly_located(a.asset, "window")
            keep = res["status"].isin(["ok", "ambiguous"]) & ~res.apply(
                lambda r: (r["leader_candidate"], r["follower_candidate"]) in bad, axis=1)
            use = res.assign(censored=~keep)          # only well-located pairs enter the offset fit
            tau, resid = venue_offsets(use, VENUES, "binance")
            use2 = use.assign(theta_ms=use["theta_adj"])
            tau_adj, _ = venue_offsets(use2, VENUES, "binance")
            pd.DataFrame({"tau_raw_ms": tau, "tau_okx_adjusted_ms": tau_adj}).round(1).to_csv(
                RESULTS / "tables" / f"offsets_{a.asset}_{a.variant}.csv")
            print("venue offsets (ms vs Binance; larger reacts later):\n", pd.DataFrame({"raw": tau, "okx_adj": tau_adj}).round(1))
    curves, rows = {}, []
    for x_i, x in enumerate(VENUES):
        for y in VENUES[x_i + 1:]:
            f = DATA_CURVES / a.asset / a.variant / f"{x}__{y}.npz"
            if not f.exists():
                continue
            st = PairStore.load(f)
            pk, lv, s = st.peak()
            curves[(x, y)] = (lv, s)
            r = df[(df["scope"] == "window") & (df["leader_candidate"] == x) & (df["follower_candidate"] == y)].iloc[0]
            rows.append({"leader_candidate": x, "follower_candidate": y, "theta_ms": pk.theta_ms, "llr_ha": r["llr_ha"]})
    contrast_panels(curves, pd.DataFrame(rows), f"{name}: HRY contrast by venue pair, pooled {STUDY.label}",
                    fig / f"contrast_{a.asset}_window.pdf")
    pairs = [(x, y) for (x, y) in curves]
    lag_across_buckets(df, pairs, BUCKETS[a.asset], f"{name}: lag by trading state", fig / f"buckets_{a.asset}.pdf")
    rob = RESULTS / "tables" / f"robustness_{a.asset}.csv"
    if rob.exists():
        robustness_panels(pd.read_csv(rob), f"{name}: estimate under each setting", fig / f"robustness_{a.asset}.pdf")


if __name__ == "__main__":
    main()
