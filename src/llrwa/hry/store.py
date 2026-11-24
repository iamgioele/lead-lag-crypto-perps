"""Curve store: per-time-block contrast curves, so days, hours and buckets are just block sums.

For a pair (X, Y) we store, per 5-minute block b:
- V[b, k]: the contrast contributions of X intervals ending in b, at lag k;
- sx2[b], sy2[b]: the sums of squared increments of X and Y ending in b.
Any union of blocks gives the exact contrast of the corresponding pooled sample (pairs are
attributed to the block of their X interval). Normalisation uses the pooled sums of squares.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .engine import Increments, LagGrid, NS_PER_MS, contrast
from .peaks import Peak, pick_peak

BLOCK_NS = 5 * 60 * 1_000_000_000


@dataclass
class PairStore:
    lags_ms: np.ndarray        # extended grid (includes smoothing margin)
    V: np.ndarray              # (n_blocks, n_lags) float32
    sx2: np.ndarray            # (n_blocks,)
    sy2: np.ndarray
    block_start_ns: np.ndarray # (n_blocks,) int64
    max_lag_ms: float
    width: int
    nx: np.ndarray | None = None   # increments of X / Y ending in each block
    ny: np.ndarray | None = None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, lags_ms=self.lags_ms, V=self.V, sx2=self.sx2, sy2=self.sy2,
                            block_start_ns=self.block_start_ns, max_lag_ms=self.max_lag_ms, width=self.width,
                            nx=self.nx, ny=self.ny)

    @classmethod
    def load(cls, path: Path) -> "PairStore":
        z = np.load(path)
        return cls(z["lags_ms"], z["V"], z["sx2"], z["sy2"], z["block_start_ns"], float(z["max_lag_ms"]), int(z["width"]),
                   z["nx"], z["ny"])

    def curve(self, mask: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        """Pooled normalised contrast over the selected blocks, and the pooled normaliser."""
        m = np.ones(len(self.sx2), dtype=bool) if mask is None else mask
        v = self.V[m].sum(axis=0, dtype=np.float64)
        norm = float(np.sqrt(self.sx2[m].sum() * self.sy2[m].sum()))
        return (v / norm if norm > 0 else v), norm

    def peak(self, mask: np.ndarray | None = None, **kw) -> tuple[Peak, np.ndarray, np.ndarray]:
        rho, _ = self.curve(mask)
        return pick_peak(self.lags_ms, rho, width=self.width, max_lag_ms=self.max_lag_ms, **kw)


def build_pair_store(X: Increments, Y: Increments, t0_ns: int, t1_ns: int, *, max_lag_ms: float = 2500.0,
                     step_ms: float = 1.0, width: int = 9, half_step: bool = True,
                     block_ns: int = BLOCK_NS) -> PairStore:
    margin = (width - 1) // 2 * step_ms
    grid = LagGrid.symmetric(int((max_lag_ms + margin) * NS_PER_MS), int(step_ms * NS_PER_MS), half_step=half_step)
    n_blocks = int(np.ceil((t1_ns - t0_ns) / block_ns))
    bx = np.clip((X.end - t0_ns) // block_ns, 0, n_blocks - 1).astype(np.int64)
    by = np.clip((Y.end - t0_ns) // block_ns, 0, n_blocks - 1).astype(np.int64)
    V = contrast(X, Y, grid, block_of_x=bx, n_blocks=n_blocks).astype(np.float32)
    sx2 = np.bincount(bx, weights=X.dx ** 2, minlength=n_blocks)
    sy2 = np.bincount(by, weights=Y.dx ** 2, minlength=n_blocks)
    starts = t0_ns + block_ns * np.arange(n_blocks, dtype=np.int64)
    nx = np.bincount(bx, minlength=n_blocks)
    ny = np.bincount(by, minlength=n_blocks)
    return PairStore(grid.lags_ms, V, sx2, sy2, starts, max_lag_ms, width, nx, ny)
