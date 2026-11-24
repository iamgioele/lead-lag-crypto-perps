"""Figures for the simulation chapter.

S0: mean contrast vs the closed-form kernel (Proposition) for three sampling designs.
S1-S5: from results/tables/sim_*.csv produced by scripts/sim_study.py.
Usage: python scripts/sim_figures.py [S0 S1 ...]
"""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llrwa.config import RESULTS
from llrwa.hry.engine import increments
from llrwa.hry.pairwise import pair_curve
from llrwa.plots import INK, VENUE_COLOR, save, use_style
from llrwa.simulate import prices, sampling, theory

F, T = RESULTS / "figures", RESULTS / "tables"
BLUE, ORANGE = VENUE_COLOR["binance"], VENUE_COLOR["okx"]


def s0_theory_overlay(reps: int = 30) -> None:
    use_style()
    designs = [("Poisson 20/s vs Poisson 5/s", lambda r, T: sampling.poisson(5.0, T, r),
                lambda h: theory.k_poisson(h, 20.0, 5.0)),
               ("Poisson 20/s vs 100 ms grid", lambda r, T: sampling.grid(100.0, T, r),
                lambda h: theory.k_poisson_grid(h, 20.0, 100.0)),
               ("Gamma(cv 1.8) 16.7/s vs 100 ms grid", lambda r, T: sampling.grid(100.0, T, r), None)]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.7), sharey=True)
    for ax, (name, ymaker, kfun) in zip(axes, designs):
        curves, kemp = [], []
        for r in range(reps):
            rng = np.random.default_rng(900 + r)
            T_ms = 1_800_000.0
            tx = sampling.gamma_renewal(16.7, 1.8, T_ms, rng) if kfun is None else sampling.poisson(20.0, T_ms, rng)
            ty = ymaker(rng, T_ms)
            x, y = prices.lead_lag_pair(tx, ty, 40.0, 0.8, 1e-4, rng)
            X, Y = increments(*prices.observe(tx, x), keep_zero=True), increments(*prices.observe(ty, y), keep_zero=True)
            lags, rho = pair_curve(X, Y, 300.0, 1.0, width=1)
            curves.append(rho)
            if kfun is None and r < 5:
                kemp.append(theory.k_empirical(tx, ty, 40.0, lags - 40.0, 20000, rng))
        m = np.mean(curves, axis=0)
        k = kfun(lags - 40.0) if kfun else np.mean(kemp, axis=0)
        ax.plot(lags, m, color=BLUE, lw=2.0, label="simulated mean")
        ax.plot(lags, 0.8 * k, color=INK["primary"], lw=1.0, ls=(0, (3, 2)), label="ρ·k(h), theory")
        ax.axvline(40.0, color=INK["muted"], lw=0.6)
        ax.set_title(name, loc="left", fontsize=8.5)
        ax.set_xlabel("lag (ms); true lag 40 ms")
        err = np.max(np.abs(m - 0.8 * k))
        ax.text(0.02, 0.05, f"max |error| = {err:.3f}", transform=ax.transAxes, fontsize=7.5, color=INK["secondary"])
    axes[0].set_ylabel("normalised contrast")
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle("Mean HRY contrast vs the expected-shape proposition (30 × 30 min each)", x=0.01, ha="left",
                 fontsize=10, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, F / "sim_S0_theory_overlay.pdf")


def _plain_log_ticks(ax, ticks) -> None:
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())


def s1_consistency() -> None:
    use_style()
    d = pd.read_csv(T / "sim_E1_consistency.csv")
    inten = d[d["setting"].str.startswith("x=")].copy()
    inten["lam_y"] = inten["setting"].str.extract(r"y=\('poisson', ([\d.]+)\)").astype(float)
    base = d[d["setting"].str.startswith("T_min=60")]
    inten = pd.concat([inten, base.assign(lam_y=5.0)]).sort_values("lam_y")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ax = axes[0]
    ax.loglog(inten["lam_y"], inten["rmse_ms"], "o-", color=BLUE, lw=1.6, ms=5, label="RMSE")
    ax.loglog(inten["lam_y"], 1000.0 / inten["lam_y"], color=INK["muted"], lw=1.0, ls=(0, (3, 2)),
              label="mean gap of slower feed")
    ax.axhline(0.5, color=INK["muted"], lw=0.8, ls=":")
    ax.text(inten["lam_y"].min(), 0.56, "half-step grid floor (0.5 ms)", fontsize=7, color=INK["secondary"])
    _plain_log_ticks(ax, [2.5, 5, 10, 20, 40])
    ax.set_xlabel("slower feed intensity λ_Y (1/s), λ_X = 2λ_Y")
    ax.set_ylabel("ms")
    ax.set_title("Error shrinks faster than the mesh", loc="left")
    ax.legend(fontsize=7.5, frameon=False)
    ax = axes[1]
    tt = d[d["setting"].str.startswith("T_min=")].copy()
    tt["T"] = tt["setting"].str.extract(r"T_min=([\d.]+)").astype(float)
    ax.loglog(tt["T"], tt["rmse_ms"], "o-", color=BLUE, lw=1.6, ms=5)
    _plain_log_ticks(ax, [10, 60, 360])
    ax.set_yticks([2, 3, 5, 10], ["2", "3", "5", "10"])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("window length T (minutes)")
    ax.set_title("Error vs window length", loc="left")
    fig.tight_layout()
    save(fig, F / "sim_S1_consistency.pdf")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["S0", "S1"]
    if "S0" in todo:
        s0_theory_overlay()
    if "S1" in todo:
        s1_consistency()
