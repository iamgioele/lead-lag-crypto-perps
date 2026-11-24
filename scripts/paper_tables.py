"""Generate LaTeX tables for the paper from results/tables (no hand-copied numbers).

Writes paper/tables/*.tex for the selected study (LLRWA_STUDY), plus study_macros.tex with the
sample dates used in the text of Chapter 4. Run after the per-asset pipelines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from llrwa.calendars.buckets import BUCKETS
from llrwa.config import RESULTS, ROOT, STUDY
from llrwa.symbols import ASSET_NAMES_TEX as ANAME, ASSETS

T = RESULTS / "tables"
OUT = ROOT / "paper" / "tables"
LBL = {"binance": "Binance", "okx": "OKX", "kraken": "Kraken", "hyperliquid": "Hyperliquid"}
PERIOD = STUDY.label.replace("-", "--")
HL_SOURCE = ("Hyperliquid from per-block quotes (own recording)" if STUDY.hl_source == "ws_bbo"
             else "Hyperliquid lags are trade-based")
STATUS = {"ok": "", "ambiguous": "two peaks", "no_signal": "no signal", "censored": "censored", "insufficient": "too few"}


def _wrap(lines: list[str]) -> list[str]:
    """Scale any tabular wider than the text width down to fit (adjustbox max width)."""
    out = []
    for l in lines:
        if l.startswith(r"\begin{tabular}"):
            out.append(r"\begin{adjustbox}{max width=\textwidth}")
        out.append(l)
        if l.startswith(r"\end{tabular}"):
            out.append(r"\end{adjustbox}")
    return out


def fmt(x, d=1, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.lstrip("+-") if float(s) == 0 else s          # no "-0" after rounding


def assets():
    return [a for a in ASSETS if (T / f"scopes_{a}_mid_all.csv").exists()]


def window_table() -> None:
    lines = [r"\begin{table}[ht]", r"\centering\small", r"\begin{tabular}{llrrrl}", r"\hline",
             r"Asset & Pair (first leads if $>0$) & $\hat\theta$ (ms) & 95\% CI & $\hat\rho$ & Note \\", r"\hline"]
    for a in assets():
        d = pd.read_csv(T / f"scopes_{a}_mid_all.csv")
        w = d[d["scope"] == "window"]
        bs = pd.read_csv(T / f"bootstrap_{a}.csv").query("scope == 'window'") if (T / f"bootstrap_{a}.csv").exists() else None
        first = True
        for _, r in w.iterrows():
            ci = "--"
            if bs is not None:
                m = bs[(bs["leader_candidate"] == r["leader_candidate"]) & (bs["follower_candidate"] == r["follower_candidate"])]
                if len(m) and r["status"] in ("ok", "ambiguous"):
                    ci = f"[{m['ci_lo_ms'].iloc[0]:+.1f}, {m['ci_hi_ms'].iloc[0]:+.1f}]"
            show = r["status"] in ("ok", "ambiguous")
            note = STATUS.get(r["status"], r["status"])
            if r["status"] == "ambiguous" and np.isfinite(r.get("second_ms", np.nan)):
                note += f" (also {r['second_ms']:+.0f})"
            if ci != "--" and (m["ci_hi_ms"].iloc[0] - m["ci_lo_ms"].iloc[0]) > 100:
                note = (note + "; " if note else "") + "poorly located"
            lines.append(f"{ANAME[a] if first else ''} & {LBL[r['leader_candidate']]} -- {LBL[r['follower_candidate']]} & "
                         f"{fmt(r['theta_ms']) if show else '--'} & {ci} & "
                         f"{fmt(r['peak_rho'], 2, False) if show else '--'} & {note} \\\\")
            first = False
        lines.append(r"\hline")
    lines += [r"\end{tabular}",
              r"\caption{Lead--lag over the whole sample (" + PERIOD + r"). Binance, OKX and Kraken from Amberdata; "
              + HL_SOURCE + r". OKX lags are as measured and may include "
              r"staleness of its 100\,ms snapshot feed. CI: joint block bootstrap (30-min "
              r"blocks). ``No signal'': peak below the 99\% null quantile of the far-lag score. $\hat\rho$ is the normalised "
              r"contrast \eqref{eq:rhohat}; it is not bounded by one, because an increment can overlap several increments of the other series.}",
              r"\label{tab:window}", r"\end{table}"]
    (OUT / "window.tex").write_text("\n".join(_wrap(lines)) + "\n")


def offsets_table() -> None:
    rows = []
    for a in assets():
        f = T / f"bootstrap_offsets_{a}.csv"
        if not f.exists():
            continue
        o = pd.read_csv(f).query("scope == 'window'").set_index("venue")
        raw = pd.read_csv(T / f"offsets_{a}_mid_all.csv", index_col=0)
        cells = []
        for v in ("okx", "kraken", "hyperliquid"):
            cells.append(f"{fmt(raw.loc[v, 'tau_raw_ms'], 0)} [{o.loc[v, 'tau_ci_lo_ms']:+.0f}, {o.loc[v, 'tau_ci_hi_ms']:+.0f}]")
        rows.append(f"{ANAME[a]} & " + " & ".join(cells) + r" \\")
    lines = [r"\begin{table}[ht]", r"\centering\small", r"\begin{tabular}{lrrr}", r"\hline",
             r"Asset & OKX & Kraken & Hyperliquid \\", r"\hline", *rows, r"\hline", r"\end{tabular}",
             r"\caption{Venue offsets $\tau$ (ms, relative to Binance; larger means later) from least squares on "
             r"$\hat\theta_{ij}\approx\tau_j-\tau_i$, with joint-bootstrap 95\% intervals. OKX as measured (may include "
             r"staleness of its 100\,ms snapshot feed). Only well-located pairs enter the fit (signal score at least 10, bootstrap interval at most 100\,ms wide).}",
             r"\label{tab:offsets}", r"\end{table}"]
    (OUT / "offsets.tex").write_text("\n".join(_wrap(lines)) + "\n")


def buckets_table() -> None:
    for a in assets():
        d = pd.read_csv(T / f"scopes_{a}_mid_all.csv")
        b = d[d["scope"] == "bucket"]
        cols = [c for c in BUCKETS[a] if c in set(b["label"])]
        bsf = T / f"bootstrap_{a}.csv"
        bsb = pd.read_csv(bsf) if bsf.exists() else pd.DataFrame(columns=["scope"])
        lines = [r"\begin{table}[ht]", r"\centering\small", r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\hline",
                 "Pair & " + " & ".join(c.replace("_", " ") for c in cols) + r" \\", r"\hline"]
        for (x, y), g in b.groupby(["leader_candidate", "follower_candidate"], sort=False):
            cells = []
            for c in cols:
                r = g[g["label"] == c]
                if not len(r):
                    cells.append("--"); continue
                r = r.iloc[0]
                m = bsb[(bsb["scope"] == f"bucket:{c}") & (bsb.get("leader_candidate") == x) & (bsb.get("follower_candidate") == y)] if len(bsb) else bsb
                wide = len(m) > 0 and (m["ci_hi_ms"].iloc[0] - m["ci_lo_ms"].iloc[0]) > 100
                if r["status"] in ("ok", "ambiguous") and wide:
                    cells.append(r"\textit{poorly located}")
                elif r["status"] == "ok":
                    cells.append(fmt(r["theta_ms"], 0))
                elif r["status"] == "ambiguous":
                    cells.append(fmt(r["theta_ms"], 0) + r"$^{\ast}$")
                else:
                    cells.append(r"\textit{" + STATUS.get(r["status"], r["status"]) + "}")
            lines.append(f"{LBL[x]} -- {LBL[y]} & " + " & ".join(cells) + r" \\")
        lines += [r"\hline", r"\end{tabular}",
                  rf"\caption{{{ANAME[a]}: lead--lag (ms, first venue leads if $>0$) pooled within each trading state, "
                  r"transitions excluded. $^{\ast}$: two competing peaks. \textit{Poorly located}: bootstrap 95\% interval "
                  r"wider than 100\,ms. OKX raw." + EXCL.get(a, "") + "}",
                  rf"\label{{tab:buckets_{a}}}", r"\end{table}"]
        (OUT / f"buckets_{a}.tex").write_text("\n".join(_wrap(lines)) + "\n")


EXCL = {"SNDK": r" The evening and night after SanDisk's earnings release (29 Oct, 16:00 ET to 30 Oct, 09:30 ET) are "
                r"left out."} if STUDY.exclude_states else {}


def study_macros() -> None:
    """Sample dates for the text, so Chapter 4 never hard-codes them."""
    t0, t1 = STUDY.T0, STUDY.T1
    sat = pd.date_range(t0.normalize(), t1, freq="D")
    mac = {"StudyPeriod": f"{t0.day} {t0:%B} to {t1.day} {t1:%B %Y}",
           "StudyStart": f"{t0.day} {t0:%B %Y}, {t0:%H:%M} UTC", "StudyEnd": f"{t1.day} {t1:%B %Y}, {t1:%H:%M} UTC",
           "StudyDays": f"{(t1 - t0).total_seconds() / 86400:.1f}",
           "StudyWeekends": str(int(((sat.dayofweek == 5) & (sat + pd.Timedelta(days=2) <= t1)).sum())),
           "RefDay": f"{pd.Timestamp(STUDY.ref_day).day} {pd.Timestamp(STUDY.ref_day):%B %Y}"}
    (OUT / "study_macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    study_macros()
    window_table()
    offsets_table()
    buckets_table()
    print(sorted(p.name for p in OUT.glob("*.tex")))


if __name__ == "__main__":
    main()


def _setting_label(s: str) -> str:
    rep = {"x=('poisson', ": "X Poi ", "y=('poisson', ": "Y Poi ", "y=('grid', ": "Y grid ", "x=('grid_locked', 100.0, 0.0)": "locked grids",
           ", y=('grid_locked', 100.0, 0.0)": "", "theta_ms=": "θ=", "stale_y_ms=": "stale ", "tick_bp=": "tick ",
           "change_only=True": "change-only", "change_only=False": "all updates", "noise_bp=": "noise ", "T_min=": "T=",
           ")": "", "vol_bp_per_sqrt_s=0.85, ": "", "max_lag_ms=2500.0, ": ""}
    for k, v in rep.items():
        s = s.replace(k, v)
    return s.replace("θ", r"$\theta$").replace("_", r"\_")


def sim_tables() -> None:
    for exp, cap in (("E3_snapshot", "Snapshot grids, locked grids and staleness (true lag 50\\,ms unless stated; X Poisson 40/s)."),
                     ("E4_noise_tick", "Price discreteness and noise; increments from all updates vs price changes only (both feeds Poisson 30/s)."),
                     ("E6_calibrated", "Settings calibrated to the data (Binance-like X; OKX-like 100\\,ms grid with 65\\,ms staleness, "
                                       "Kraken-like, or Hyperliquid-trade-like Y).")):
        f = T / f"sim_{exp}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        lines = [r"\begin{table}[ht]", r"\centering\small", r"\begin{tabular}{lrrrr}", r"\hline",
                 r"Setting & Bias (ms) & RMSE (ms) & $P(|e|\le5\,\mathrm{ms})$ & Median LLR$_{\mathrm{HA}}$ \\", r"\hline"]
        for _, r in d.iterrows():
            lines.append(f"{_setting_label(r['setting'])} & {r['bias_ms']:+.1f} & {r['rmse_ms']:.1f} & "
                         f"{r['p_within_5ms']:.2f} & {r['llr_ha_median']:.2f} \\\\")
        lines += [r"\hline", r"\end{tabular}", rf"\caption{{{cap} 500 replications per setting.}}",
                  rf"\label{{tab:sim_{exp.split('_')[0]}}}", r"\end{table}"]
        (OUT / f"sim_{exp.split('_')[0]}.tex").write_text("\n".join(_wrap(lines)) + "\n")


if __name__ == "__main__":
    sim_tables()


def sim_e5_table() -> None:
    f = T / "sim_E5_baseline.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    lines = [r"\begin{table}[ht]", r"\centering\small", r"\begin{tabular}{lrrrrrr}", r"\hline",
             r"Design & $\theta$ & HRY mean & \multicolumn{2}{c}{Prev.-tick argmax} & Centroid & Mass ratio \\",
             r" & & & 1\,ms bars & 100\,ms bars & 1\,ms bars & 1\,ms bars \\", r"\hline"]
    for _, r in d.iterrows():
        design = "Poisson 50/s vs 100\\,ms grid" if "grid" in r["setting"] else "Poisson 40/s vs Poisson 2/s"
        lines.append(f"{design} & {r['theta_true_ms']:.0f} & {r['theta_true_ms'] + r['bias_ms']:+.1f} & "
                     f"{r['xc1_argmax_mean']:+.1f} & {r['xc100_argmax_mean']:+.1f} & {r['xc1_centroid_mean']:+.1f} & "
                     f"{r['xc1_mass_ratio_mean']:.1f} \\\\")
    lines += [r"\hline", r"\end{tabular}",
              r"\caption{HRY against previous-tick cross-correlation of bar returns (means over 500 replications, ms). "
              r"Centroid and positive/negative mass ratio refer to the cross-correlation curve over $\pm1$\,s.}",
              r"\label{tab:sim_E5}", r"\end{table}"]
    (OUT / "sim_E5.tex").write_text("\n".join(_wrap(lines)) + "\n")


if __name__ == "__main__":
    sim_e5_table()
