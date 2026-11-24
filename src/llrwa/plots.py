"""Figure style and shared plots. Vector PDF output for the paper.

Venue colours are fixed by entity, never by rank, and follow the reference categorical order:
binance blue, okx orange, kraken aqua, hyperliquid yellow. Text always uses ink colours.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

VENUE_COLOR = {"binance": "#2a78d6", "okx": "#eb6834", "kraken": "#1baf7a", "hyperliquid": "#eda100"}
VENUE_LABEL = {"binance": "Binance", "okx": "OKX", "kraken": "Kraken", "hyperliquid": "Hyperliquid"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#8a8984", "grid": "#e4e3df", "surface": "#fcfcfb"}


def use_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": INK["muted"], "axes.labelcolor": INK["secondary"], "axes.titlecolor": INK["primary"],
        "xtick.color": INK["secondary"], "ytick.color": INK["secondary"], "text.color": INK["primary"],
        "axes.grid": True, "grid.color": INK["grid"], "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 9.5,
        "axes.titleweight": "bold", "lines.linewidth": 2.0, "pdf.fonttype": 42,
    })


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)


def interarrival_panels(dt_by_venue: dict[str, np.ndarray], title: str, path: Path,
                        notes: dict[str, str] | None = None) -> None:
    """Small multiples: log-x histogram of inter-arrival times (ms) over distinct timestamps."""
    use_style()
    venues = [v for v in VENUE_COLOR if v in dt_by_venue]
    fig, axes = plt.subplots(1, len(venues), figsize=(2.7 * len(venues), 2.4), sharex=True)
    axes = np.atleast_1d(axes)
    bins = np.logspace(0, 5, 61)
    for ax, v in zip(axes, venues):
        dt = np.asarray(dt_by_venue[v], dtype=float)
        dt = dt[dt >= 1]
        w = np.full(len(dt), 100.0 / max(len(dt), 1))
        ax.hist(dt, bins=bins, weights=w, color=VENUE_COLOR[v], edgecolor="white", linewidth=0.4)
        for ref in (100, 1000):
            ax.axvline(ref, color=INK["muted"], lw=0.8, ls=(0, (3, 2)))
        ax.set_xscale("log")
        ax.set_title(VENUE_LABEL[v], loc="left", pad=16)
        if notes and v in notes:
            ax.text(0, 1.03, notes[v], transform=ax.transAxes, fontsize=7.5, color=INK["secondary"])
        ax.set_xlabel("gap between updates (ms)")
    axes[0].set_ylabel("share of gaps (%)")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, fontweight="bold", y=1.04)
    fig.tight_layout(w_pad=1.5)
    save(fig, path)


def contrast_panels(curves: dict, res, title: str, path: Path, xlim_ms: float = 1500.0) -> None:
    """Small multiples, one per pair: smoothed normalised contrast vs lag, with the estimate marked."""
    use_style()
    pairs = list(curves)
    ncol = 3
    nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.3 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (a, b) in zip(axes, pairs):
        lv, s = curves[(a, b)]
        r = res[(res["leader_candidate"] == a) & (res["follower_candidate"] == b)].iloc[0]
        ax.axhline(0, color=INK["muted"], lw=0.6)
        ax.axvline(0, color=INK["muted"], lw=0.6)
        ax.plot(lv, s, color=VENUE_COLOR["binance"], lw=1.4)
        ax.axvline(r["theta_ms"], color=INK["primary"], lw=0.9, ls=(0, (3, 2)))
        ax.text(0.02, 0.95, f"θ̂ = {r['theta_ms']:+.1f} ms\nLLR$_{{HA}}$ = {r['llr_ha']:.2f}",
                transform=ax.transAxes, va="top", fontsize=7.5, color=INK["secondary"])
        ax.set_title(f"{VENUE_LABEL[a]} vs {VENUE_LABEL[b]}", loc="left")
        ax.set_xlim(-xlim_ms, xlim_ms)
    for ax in axes[len(pairs):]:
        ax.set_visible(False)
    for ax in axes[-ncol:]:
        ax.set_xlabel("lag θ (ms); > 0: first venue leads")
    for r in range(nrow):
        axes[r * ncol].set_ylabel("normalised contrast")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout(w_pad=1.2, h_pad=1.0)
    save(fig, path)


from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Diverging: blue (row follows) <- gray midpoint -> red (row leads); equal steps per arm
DIVERGING = LinearSegmentedColormap.from_list(
    "lead_lag", ["#104281", "#3987e5", "#cde2fb", "#f0efec", "#f7c9c8", "#e34948", "#9e2a2a"])


def lag_heatmap(M, title: str, path: Path, note: str = "", vmax: float | None = None) -> None:
    """Antisymmetric lag matrix (ms): cell (row, col) > 0 means the row venue leads the column venue."""
    use_style()
    venues = list(M.index)
    vals = M.values.astype(float)
    if vmax is None:
        vmax = float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 1.0
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.grid(False)
    im = ax.imshow(vals, cmap=DIVERGING, vmin=-vmax, vmax=vmax)
    labels = [VENUE_LABEL.get(v, v) for v in venues]
    ax.set_xticks(range(len(venues)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(venues)), labels)
    for i in range(len(venues)):
        for j in range(len(venues)):
            if i == j:
                continue
            v = vals[i, j]
            txt = "n/a" if not np.isfinite(v) else f"{v:+.0f}"
            dark = np.isfinite(v) and abs(v) > 0.55 * vmax
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color="white" if dark else INK["primary"])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("lag (ms); > 0: row leads column", color=INK["secondary"])
    cb.outline.set_visible(False)
    ax.set_title(title, loc="left")
    if note:
        fig.text(0.01, -0.12, note, fontsize=7.5, color=INK["secondary"])
    save(fig, path)


def lag_across_buckets(df, pairs: list[tuple[str, str]], buckets: list[str], title: str, path: Path) -> None:
    """One panel per pair: pooled lag per bucket (dot) with the range of daily estimates (line)."""
    use_style()
    n = len(pairs)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow), sharex=True, squeeze=False)
    axes = axes.ravel()
    xs = np.arange(len(buckets))
    for ax, (a, b) in zip(axes, pairs):
        sub = df[(df["leader_candidate"] == a) & (df["follower_candidate"] == b)]
        for k, bk in enumerate(buckets):
            pooled = sub[(sub["scope"] == "bucket") & (sub["label"] == bk) & sub["status"].isin(["ok", "ambiguous"])]
            daily = sub[(sub["scope"] == "bucket_day") & sub["label"].str.startswith(bk + "|") & (sub["status"] == "ok")]
            if len(daily) >= 2:
                ax.plot([k, k], [daily["theta_ms"].quantile(0.25), daily["theta_ms"].quantile(0.75)],
                        color=VENUE_COLOR["binance"], lw=2.0, alpha=0.45, solid_capstyle="round")
            if len(pooled):
                amb = pooled["status"].iloc[0] == "ambiguous"
                ax.plot(k, pooled["theta_ms"].iloc[0], "o", ms=6, markerfacecolor="white" if amb else VENUE_COLOR["binance"],
                        markeredgecolor=VENUE_COLOR["binance"], markeredgewidth=1.5)
        ax.axhline(0, color=INK["muted"], lw=0.6)
        ax.set_title(f"{VENUE_LABEL[a]} vs {VENUE_LABEL[b]}", loc="left")
        ax.set_xticks(xs, [bk.replace("_", " ") for bk in buckets], rotation=30, ha="right")
    for ax in axes[n:]:
        ax.set_visible(False)
    for r in range(nrow):
        axes[r * ncol].set_ylabel("lag (ms); > 0: first leads")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, fontweight="bold", y=1.02)
    fig.text(0.01, -0.02, "Dot: pooled estimate over all days in the bucket (hollow: two competing peaks). Bar: interquartile range of daily estimates.",
             fontsize=7.5, color=INK["secondary"])
    fig.tight_layout()
    save(fig, path)


def robustness_panels(df, title: str, path: Path) -> None:
    """One panel per pair: median lag under each setting (rows), baseline marked by a vertical line."""
    use_style()
    df = df.copy()
    df["setting"] = df["knob"].str.replace("_", " ") + ": " + df["value"].astype(str)
    settings = list(dict.fromkeys(df["setting"]))
    pairs = list(dict.fromkeys(zip(df["leader_candidate"], df["follower_candidate"])))
    ncol = 3
    nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 0.19 * len(settings) * nrow + 1.2), sharey=True,
                             squeeze=False)
    axes = axes.ravel()
    ys = np.arange(len(settings))[::-1]
    for ax, (a, b) in zip(axes, pairs):
        sub = df[(df["leader_candidate"] == a) & (df["follower_candidate"] == b)].set_index("setting")
        base = sub.loc[sub.index.str.startswith("baseline"), "theta_median_ms"]
        if len(base):
            ax.axvline(base.iloc[0], color=INK["muted"], lw=0.8, ls=(0, (3, 2)))
        vals = [sub["theta_median_ms"].get(s, np.nan) for s in settings]
        ax.plot(vals, ys, "o", ms=4.5, color=VENUE_COLOR["binance"], markeredgecolor="white", markeredgewidth=1.0)
        ax.set_title(f"{VENUE_LABEL[a]} vs {VENUE_LABEL[b]}", loc="left")
        ax.set_xlabel("lag (ms); > 0: first leads")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys, settings, fontsize=7)
    for r in range(nrow):
        axes[r * ncol].set_yticks(ys, settings, fontsize=7)
    for ax in axes[len(pairs):]:
        ax.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, path)
