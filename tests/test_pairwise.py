import numpy as np
import pandas as pd

from llrwa.hry.pairwise import estimate_pairs, lag_matrix, venue_offsets
from llrwa.simulate import prices, sampling

DELAYS = {"a": 0.0, "b": 30.0, "c": 80.0, "d": 400.0}      # ms after the common efficient price
RATES = {"a": 40.0, "b": 15.0, "c": 25.0, "d": 3.0}


def _venues(seed=0, T_ms=2_400_000):
    rng = np.random.default_rng(seed)
    obs = {v: sampling.poisson(RATES[v], T_ms, rng) for v in DELAYS}
    allt = np.concatenate([obs[v] - DELAYS[v] for v in DELAYS])
    w = prices._bm_at(allt, 1e-4, rng)
    out, k = {}, 0
    for v in DELAYS:
        n = len(obs[v])
        logp = w[k:k + n]
        k += n
        t_ns, lp = prices.observe(obs[v], logp)
        out[v] = pd.DataFrame({"ts_ns": t_ns, "mid": 100 * np.exp(lp), "segment": 0})
    return out


def test_synthetic_four_venues_recover_delays_and_offsets():
    series = _venues()
    res, _ = estimate_pairs(series, max_lag_ms=1000)
    for _, r in res.iterrows():
        truth = DELAYS[r["follower_candidate"]] - DELAYS[r["leader_candidate"]]
        # precision scales with the slower venue's intensity (smooth apex of width ~ 1/sqrt(lam mu))
        tol = 25 if "d" in (r["leader_candidate"], r["follower_candidate"]) else 10
        assert abs(r["theta_ms"] - truth) <= tol, (r["leader_candidate"], r["follower_candidate"], r["theta_ms"])
    M = lag_matrix(res, list(DELAYS))
    assert np.allclose(M.values, -M.values.T, equal_nan=True)
    tau, resid = venue_offsets(res, list(DELAYS), "a")
    assert np.allclose(tau.values, list(DELAYS.values()), atol=15)
    assert resid["residual_ms"].abs().max() < 20


def test_unconnected_venue_offset_is_nan():
    res = pd.DataFrame({"leader_candidate": ["a", "c"], "follower_candidate": ["b", "d"],
                        "theta_ms": [10.0, 5.0], "censored": [False, True]})
    tau, _ = venue_offsets(res, ["a", "b", "c", "d"], "a")
    assert abs(tau["a"]) < 1e-9 and abs(tau["b"] - 10) < 1e-9 and np.isnan(tau["c"]) and np.isnan(tau["d"])
