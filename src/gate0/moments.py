"""Pre-registered moments, computed per run; uncertainty comes only from paired seed resampling."""

from __future__ import annotations

import warnings

import numpy as np

from gate0.config import Moments
from gate0.stats import bootstrap_ci


def _demean(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=-1, keepdims=True)


def _rank(x: np.ndarray) -> np.ndarray:
    """Cross-sectional ranks scaled to [-0.5, 0.5]."""
    return np.argsort(np.argsort(x)) / (len(x) - 1) - 0.5


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    d = (x * x).sum()
    return float((x * y).sum() / d) if d > 0 else np.nan


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Slope with an intercept, so a common level shift within a subset does not leak in."""
    return _slope(x - x.mean(), y - y.mean()) if len(x) > 2 else np.nan


def momentum_profile(L: np.ndarray, disp: np.ndarray, mc: Moments) -> dict:
    """Slope of cumulative future return (h = 1..H) on past F-round return, pooled across rounds,
    overall and split by cross-sectional median analyst dispersion at formation."""
    T = L.shape[0] - 1
    F, H = mc.formation_window, mc.max_horizon
    out = {"all": [], "high": [], "low": []}
    for h in range(1, H + 1):
        xs = {k: [] for k in out}
        ys = {k: [] for k in out}
        for t in range(F, T - h + 1):
            past, fut = L[t] - L[t - F], L[t + h] - L[t]
            high = disp[t] > np.median(disp[t])
            for k, mask in (("all", np.ones_like(high)), ("high", high), ("low", ~high)):
                # Demean within the subset, so a state-wide drift is not read as momentum.
                xs[k].append(_demean(past[mask]))
                ys[k].append(_demean(fut[mask]))
        for k in out:
            out[k].append(_slope(np.concatenate(xs[k]), np.concatenate(ys[k])) if xs[k] else np.nan)
    res = {f"profile_{k}": v for k, v in out.items()}
    # Manipulation check at one pre-registered horizon (no max over horizons, so no upward bias).
    res["momentum_h"] = float(out["all"][mc.interaction_horizon - 1])
    return res


def turning_point(curve) -> float:
    """Horizon where a profile that started positive first changes sign (linear interpolation);
    NaN if it is never positive or never turns within the horizons measured."""
    c = np.asarray(curve, dtype=float)
    pos = np.flatnonzero(c > 0)
    if not len(pos):
        return np.nan
    for i in range(pos[0] + 1, len(c)):
        if c[i] <= 0:
            a, b = c[i - 1], c[i]
            return float(i + a / (a - b))  # index i-1 is horizon i
    return np.nan


def rank_interaction(L: np.ndarray, disp: np.ndarray, mc: Moments) -> float:
    """Coefficient on rank(past return) x rank(dispersion) for the next-H-round return."""
    T = L.shape[0] - 1
    F, H = mc.formation_window, mc.interaction_horizon
    X, y = [], []
    for t in range(F, T - H + 1):
        rp, rd = _rank(L[t] - L[t - F]), _rank(disp[t])
        X.append(np.column_stack([rp, rd, rp * rd]))
        y.append(_demean(L[t + H] - L[t]))
    coef, *_ = np.linalg.lstsq(np.vstack(X), np.concatenate(y), rcond=None)
    return float(coef[2])


def pead(L: np.ndarray, report: np.ndarray, forecast: np.ndarray, mc: Moments) -> dict:
    """Post-announcement drift slope on the surprise, by sign. The surprise is the report minus the
    consensus (mean agent forecast before the announcement), not the price, so earlier price momentum
    does not leak into it."""
    T = L.shape[0] - 1
    k = mc.pead_window
    s_all, d_all = [], []
    for t in range(1, T - k + 1):
        if not np.all(np.isfinite(report[t])):
            continue
        s_all.append(report[t] - forecast[t - 1])
        d_all.append(_demean(L[t + k] - L[t]))
    s, d = np.concatenate(s_all), np.concatenate(d_all)
    pos, neg = _ols_slope(s[s > 0], d[s > 0]), _ols_slope(s[s < 0], d[s < 0])
    # Level: average drift across good and bad news (underreaction); gap: the asymmetry between them.
    return {"pead_pos": pos, "pead_neg": neg, "pead_level": (pos + neg) / 2, "pead_gap": pos - neg}


def optimism_anomaly(L: np.ndarray, optimism: np.ndarray, mc: Moments) -> float:
    """Slope of next-H-round return on rank(analyst optimism); underperformance is negative."""
    T = L.shape[0] - 1
    H = mc.interaction_horizon
    r = _rank(optimism)
    xs = [r for t in range(T - H + 1)]
    ys = [_demean(L[t + H] - L[t]) for t in range(T - H + 1)]
    return _slope(np.concatenate(xs), np.concatenate(ys))


def compute(res: dict, mc: Moments) -> dict:
    # Drop round 0: it is the pre-market price, not an auction outcome.
    L = np.log(np.asarray(res["prices"], dtype=float))[1:]
    disp = np.asarray(res["dispersion"], dtype=float)[1:]
    report = np.asarray(res["report"], dtype=float)[1:]
    log_value = np.asarray(res["log_value"], dtype=float)[1:]
    forecast = np.asarray(res["forecast"], dtype=float)[1:]
    r = np.diff(L, axis=0)
    out = {"arm": res["arm"], "seed": res["seed"], "masked": res["masked"]}
    out |= momentum_profile(L, disp, mc)
    out["dispersion_interaction"] = rank_interaction(L, disp, mc)
    out |= pead(L, report, forecast, mc)
    out["optimism_anomaly"] = optimism_anomaly(L, np.asarray(res["optimism"]), mc)
    # Descriptive only.
    out["return_vol"] = float(r.std())
    out["autocorr_1"] = float(np.corrcoef(r[1:].ravel(), r[:-1].ravel())[0, 1])
    out["turnover"] = float(np.mean(res["volume"][1:]))
    out["mispricing_rmse"] = float(np.sqrt(np.mean((L - log_value) ** 2)))
    out["fallbacks"] = res["fallbacks"]
    out["agents"] = res["agents"]  # seating, for the realized spread of recovered intensity
    return out


SCALARS = ["momentum_h", "dispersion_interaction", "pead_pos", "pead_neg", "pead_level", "pead_gap", "optimism_anomaly",
           "return_vol", "autocorr_1", "turnover", "mispricing_rmse"]
PROFILES = ["profile_all", "profile_high", "profile_low"]


def paired_contrast(moments: list[dict], arm: str, base: str, key: str, mc: Moments,
                    masked: tuple[bool, bool] = (False, False)) -> dict:
    """Mean and seed-bootstrap CI of (arm - base) over seeds present in both."""
    a = {m["seed"]: m[key] for m in moments if m["arm"] == arm and m["masked"] == masked[0]}
    b = {m["seed"]: m[key] for m in moments if m["arm"] == base and m["masked"] == masked[1]}
    seeds = sorted(a.keys() & b.keys())
    diffs = [a[s] - b[s] for s in seeds
             if a[s] is not None and b[s] is not None and np.isfinite(a[s] - b[s])]
    mean, lo, hi = bootstrap_ci(diffs, mc.bootstrap_draws, mc.ci_level)
    return {"arm": arm, "base": base, "moment": key, "n_seeds": len(diffs), "mean": mean, "lo": lo, "hi": hi}


# ---- horizon profiles: bootstrap over seeds of the whole curve ----------------------------


def _curves(moments: list[dict], arm: str, key: str, masked: bool) -> dict[int, np.ndarray]:
    return {m["seed"]: np.asarray(m[key], dtype=float) for m in moments if m["arm"] == arm and m["masked"] == masked}


def _resampled_means(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(X[idx], axis=1)


def _tp_ci(tps: np.ndarray, mc: Moments) -> tuple[float, float, float]:
    """Percentile CI over draws where a turning point exists, plus the share of such draws."""
    ok = tps[np.isfinite(tps)]
    a = (1 - mc.ci_level) / 2
    if not len(ok):
        return np.nan, np.nan, 0.0
    return float(np.quantile(ok, a)), float(np.quantile(ok, 1 - a)), float(len(ok) / len(tps))


def profile_summary(moments: list[dict], arm: str, key: str, mc: Moments, masked: bool = False) -> dict:
    """Seed-mean profile with pointwise CIs, and its turning point with a seed-bootstrap CI."""
    curves = _curves(moments, arm, key, masked)
    if not curves:
        return {"n_seeds": 0}
    X = np.stack([curves[s] for s in sorted(curves)])
    idx = np.random.default_rng(0).integers(0, len(X), size=(mc.bootstrap_draws, len(X)))
    means = _resampled_means(X, idx)
    a = (1 - mc.ci_level) / 2
    mean = _resampled_means(X, np.arange(len(X))[None])[0]
    lo, hi, share = _tp_ci(np.array([turning_point(c) for c in means]), mc)
    return {"n_seeds": len(X), "mean": mean.tolist(),
            "lo": np.nanquantile(means, a, axis=0).tolist(), "hi": np.nanquantile(means, 1 - a, axis=0).tolist(),
            "turning_point": turning_point(mean), "tp_lo": lo, "tp_hi": hi, "tp_found_share": share}


def turning_point_contrast(moments: list[dict], arm: str, base: str, key: str, mc: Moments,
                           masked: tuple[bool, bool] = (False, False)) -> dict:
    """(arm - base) turning point, resampling common seeds jointly over the whole curve."""
    a, b = _curves(moments, arm, key, masked[0]), _curves(moments, base, key, masked[1])
    seeds = sorted(a.keys() & b.keys())
    if not seeds:
        return {"n_seeds": 0, "mean": np.nan, "lo": np.nan, "hi": np.nan, "found_share": 0.0}
    A, B = np.stack([a[s] for s in seeds]), np.stack([b[s] for s in seeds])
    idx = np.random.default_rng(0).integers(0, len(seeds), size=(mc.bootstrap_draws, len(seeds)))
    d = np.array([turning_point(x) - turning_point(y)
                  for x, y in zip(_resampled_means(A, idx), _resampled_means(B, idx))])
    lo, hi, share = _tp_ci(d, mc)
    full = np.arange(len(seeds))[None]
    point = turning_point(_resampled_means(A, full)[0]) - turning_point(_resampled_means(B, full)[0])
    return {"n_seeds": len(seeds), "mean": float(point), "lo": lo, "hi": hi, "found_share": share}
