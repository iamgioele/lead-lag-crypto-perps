import numpy as np

from llrwa.hry.engine import LagGrid, NS_PER_MS, contrast, increments
from llrwa.hry.store import PairStore, build_pair_store


def test_store_blocks_sum_to_full_curve_and_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    T = 3_600 * 1_000 * NS_PER_MS
    tx = np.unique(rng.integers(0, T, 20000)).astype(np.int64)
    ty = np.unique(rng.integers(0, T, 8000)).astype(np.int64)
    X = increments(tx, np.cumsum(rng.normal(size=len(tx))))
    Y = increments(ty, np.cumsum(rng.normal(size=len(ty))))
    st = build_pair_store(X, Y, 0, T, max_lag_ms=200.0)
    grid = LagGrid.symmetric(int(204 * NS_PER_MS), NS_PER_MS, half_step=True)
    full = contrast(X, Y, grid) / np.sqrt(X.sum_sq * Y.sum_sq)
    rho, _ = st.curve()
    assert np.allclose(rho, full, atol=1e-5)                 # float32 storage
    st.save(tmp_path / "p.npz")
    st2 = PairStore.load(tmp_path / "p.npz")
    assert np.allclose(st2.curve()[0], rho)
    half = st.block_start_ns < T // 2
    assert np.isclose(st.sx2[half].sum() + st.sx2[~half].sum(), X.sum_sq)
