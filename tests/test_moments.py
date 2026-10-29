"""Moment code checked against price paths with known, planted structure."""

import numpy as np

from gate0.config import load_config
from gate0.moments import (momentum_profile, optimism_anomaly, paired_contrast, pead, profile_summary,
                           rank_interaction, turning_point, turning_point_contrast)

MC = load_config().moments
T, N = 200, 40


PAD = 17


def _paths(continuation, rng, disp=None):
    """Log prices where each shock keeps drifting for 6 rounds, then reverses over rounds 12-17."""
    eps = rng.normal(0, 0.02, (T + PAD, N))
    r = eps.copy()
    k = np.full((T + PAD, N), continuation)
    if disp is not None:
        k = k * (disp > np.median(disp, axis=1, keepdims=True))
    for lag in range(1, 7):
        r[lag:] += k[:-lag] * eps[:-lag]
    for lag in range(12, 18):
        r[lag:] -= k[:-lag] * eps[:-lag]
    return np.vstack([np.zeros(N), np.cumsum(r[PAD:], axis=0)])[: T + 1]


def test_profile_rises_then_reverses():
    rng = np.random.default_rng(0)
    L = _paths(0.3, rng)
    prof = momentum_profile(L, rng.random((T + 1, N)), MC)
    curve = np.asarray(prof["profile_all"])
    assert prof["momentum_h"] > 0.2 and prof["momentum_h"] == curve[MC.interaction_horizon - 1]
    assert curve[-1] < curve[5]  # reversal sets in after the drift stops


def test_turning_point_is_the_sign_change():
    assert turning_point([0.1, 0.3, 0.2, 0.1, -0.1, -0.3]) == 4.5
    assert np.isnan(turning_point([0.1, 0.2, 0.3]))          # never turns within the horizons
    assert np.isnan(turning_point([-0.1, -0.2, -0.1]))       # no momentum, nothing to turn
    assert turning_point([-0.1, 0.2, -0.2]) == 2.5           # the first positive stretch starts the clock


def test_turning_point_bootstrap_over_seeds():
    rng = np.random.default_rng(7)
    curves = {"placebo": np.array([0.2, 0.3, 0.2, 0.1, -0.1, -0.2, -0.3, -0.3]),   # turns at 4.5
              "dhs": np.array([0.1, 0.15, 0.2, 0.3, 0.2, 0.1, -0.1, -0.2])}       # turns at 6.5
    ms = [{"arm": a, "seed": s, "masked": False, "profile_all": c + rng.normal(0, 0.01, 8)}
          for s in range(10) for a, c in curves.items()]
    ps = profile_summary(ms, "placebo", "profile_all", MC)
    assert ps["n_seeds"] == 10 and 4 < ps["turning_point"] < 5 and ps["tp_lo"] <= ps["turning_point"] <= ps["tp_hi"]
    c = turning_point_contrast(ms, "dhs", "placebo", "profile_all", MC)
    assert 1.5 < c["mean"] < 2.5 and c["lo"] > 0 and c["found_share"] == 1.0


def test_no_structure_no_momentum():
    rng = np.random.default_rng(1)
    L = np.vstack([np.zeros(N), np.cumsum(rng.normal(0, 0.02, (T, N)), axis=0)])
    assert abs(momentum_profile(L, rng.random((T + 1, N)), MC)["profile_all"][0]) < 0.1


def test_interaction_detects_dispersion_dependent_momentum():
    rng = np.random.default_rng(2)
    disp = np.tile(rng.random(N), (T + PAD, 1))  # persistent dispersion ranks
    L = _paths(0.3, rng, disp)
    assert rank_interaction(L, disp[: T + 1], MC) > 0
    prof = momentum_profile(L, disp[: T + 1], MC)
    assert np.nanmax(prof["profile_high"]) > np.nanmax(prof["profile_low"])


def test_pead_gap_sign():
    rng = np.random.default_rng(3)
    L = np.vstack([np.zeros(N), np.cumsum(rng.normal(0, 0.01, (T, N)), axis=0)])
    report = np.full((T + 1, N), np.nan)
    for t in range(4, T - 4, 4):
        s = rng.normal(0, 0.05, N)
        report[t] = L[t - 1] + s
        L[t:] += 0.5 * s                                  # immediate response
        L[t + 1 :] += np.where(s > 0, 0.4 * s, 0.0)       # drift only after good news
    out = pead(L, report, L, MC)  # consensus forecast = price here
    assert out["pead_gap"] > 0.2 and abs(out["pead_neg"]) < 0.1


def test_pead_surprise_is_measured_against_consensus_not_price():
    """Prices trend away from the consensus before each report; drift depends only on report - consensus."""
    rng = np.random.default_rng(8)
    forecast = np.vstack([np.zeros(N), np.cumsum(rng.normal(0, 0.01, (T, N)), axis=0)])
    L = forecast.copy()
    report = np.full((T + 1, N), np.nan)
    for t in range(4, T - 4, 4):
        s = rng.normal(0, 0.05, N)
        report[t] = forecast[t - 1] + s
        L[t - 1] += rng.normal(0, 0.05, N)                # pre-announcement price momentum, unrelated to news
        L[t + 1 :] += np.where(s > 0, 0.4 * s, 0.0)
    good, bad = pead(L, report, forecast, MC), pead(L, report, L, MC)
    assert good["pead_gap"] > 0.2 and abs(good["pead_neg"]) < 0.1
    assert abs(bad["pead_gap"] - 0.4) > abs(good["pead_gap"] - 0.4)


def test_optimism_underperformance_sign():
    rng = np.random.default_rng(4)
    opt = rng.random(N)
    L = np.vstack([np.zeros(N), np.cumsum(rng.normal(0, 0.01, (T, N)) - 0.002 * (opt - 0.5), axis=0)])
    assert optimism_anomaly(L, opt, MC) < 0


def test_paired_contrast_uses_seeds_not_agents():
    ms = [{"arm": a, "seed": s, "masked": False, "x": v + (1.0 if a == "dhs" else 0.0)}
          for s, v in enumerate(np.random.default_rng(5).normal(size=10)) for a in ("dhs", "placebo")]
    c = paired_contrast(ms, "dhs", "placebo", "x", MC)
    assert c["n_seeds"] == 10 and c["mean"] == 1.0 and c["lo"] == c["hi"] == 1.0


def test_state_wide_drift_is_not_state_momentum():
    """Assets in the high state share a persistent drift but have no momentum within the state."""
    rng = np.random.default_rng(6)
    grp = np.r_[np.ones(N // 2), np.zeros(N // 2)]
    drift = np.zeros(T)
    for t in range(1, T):
        drift[t] = 0.9 * drift[t - 1] + rng.normal(0, 0.01)
    L = np.vstack([np.zeros(N), np.cumsum(rng.normal(0, 0.02, (T, N)) + drift[:, None] * grp, axis=0)])
    disp = np.tile(grp, (T + 1, 1)) + rng.random((T + 1, N)) * 0.01
    prof = momentum_profile(L, disp, MC)
    assert abs(prof["profile_high"][0]) < 0.05 and abs(prof["profile_low"][0]) < 0.05


def test_nan_seeds_are_not_counted():
    ms = [{"arm": a, "seed": s, "masked": False, "x": np.nan} for s in range(3) for a in ("dhs", "placebo")]
    assert paired_contrast(ms, "dhs", "placebo", "x", MC)["n_seeds"] == 0
