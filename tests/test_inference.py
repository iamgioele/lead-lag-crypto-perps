import numpy as np

from llrwa.hry.engine import NS_PER_MS
from llrwa.hry.inference import bootstrap_pairs, summarise_bootstrap
from llrwa.hry.pairwise import series_increments
from llrwa.hry.store import build_pair_store
from llrwa.simulate import prices, sampling
import pandas as pd


def test_bootstrap_interval_contains_truth_on_synthetic_pair():
    rng = np.random.default_rng(3)
    T = 3 * 3_600_000.0
    tx, ty = sampling.poisson(10.0, T, rng), sampling.poisson(4.0, T, rng)
    x, y = prices.lead_lag_pair(tx, ty, 60.0, 0.8, 1e-4, rng)
    ox, oy = prices.observe(tx, x), prices.observe(ty, y)
    X = series_increments(pd.DataFrame({"ts_ns": ox[0], "mid": np.exp(ox[1])}))
    Y = series_increments(pd.DataFrame({"ts_ns": oy[0], "mid": np.exp(oy[1])}))
    st = build_pair_store(X, Y, 0, int(T * NS_PER_MS), max_lag_ms=500.0)
    bs = bootstrap_pairs({("x", "y"): st}, np.ones(len(st.sx2), dtype=bool), B=200, seed=1)
    s = summarise_bootstrap(bs).iloc[0]
    assert s["ci_lo_ms"] <= 60.0 <= s["ci_hi_ms"]
    assert 0 < s["boot_sd_ms"] < 20
