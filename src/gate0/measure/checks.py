"""The three pre-registered validity checks that turn a prompted primitive into a usable treatment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from gate0.config import Config
from gate0.stats import bootstrap_ci


def item_means(df: pd.DataFrame) -> pd.DataFrame:
    """Average repeats within (condition, task, item): items are the unit of independence."""
    return df.dropna(subset=["score"]).groupby(["condition", "task", "item"], as_index=False)["score"].mean()


def _cell(im: pd.DataFrame, condition: str, task: str) -> np.ndarray:
    return im[(im.condition == condition) & (im.task == task)]["score"].to_numpy()


def _paired(im: pd.DataFrame, a: str, b: str, task: str) -> list[float]:
    x = im[(im.condition == a) & (im.task == task)].set_index("item")["score"]
    y = im[(im.condition == b) & (im.task == task)].set_index("item")["score"]
    return (x - y).dropna().tolist()


def own_conditions(primitive: str, levels: list[int]) -> list[str]:
    return ["neutral" if k == 0 else f"{primitive}@{k}" for k in levels]


def analyze(df: pd.DataFrame, cfg: Config) -> dict[str, dict]:
    m, th = cfg.measurement, cfg.measurement.thresholds
    im = item_means(df)
    top = f"@{max(m.intensities)}"

    # SD unit per primitive: pooled within-condition SD over its own intensity grid.
    sd = {}
    for p in m.primitives:
        variances = [np.var(_cell(im, c, p), ddof=1) for c in own_conditions(p, m.intensities)
                     if len(_cell(im, c, p)) > 1]
        sd[p] = float(np.sqrt(np.mean(variances))) if variances else np.nan
        if not sd[p] > 0:
            sd[p] = np.nan  # no usable spread: every check on this primitive fails below

    results = {}
    for p in m.primitives:
        conds = own_conditions(p, m.intensities)
        level_means = [float(np.mean(_cell(im, c, p))) if len(_cell(im, c, p)) else np.nan for c in conds]
        rho = spearmanr(m.intensities, level_means).statistic if np.all(np.isfinite(level_means)) else np.nan
        own = (level_means[-1] - level_means[0]) / sd[p]

        cross = {}
        for q in m.primitives:
            if q == p:
                continue
            delta = (np.mean(_cell(im, p + top, q)) - np.mean(_cell(im, "neutral", q))) / sd[q]
            cross[q] = float(abs(delta) / abs(own)) if own else np.inf
        # A missing or non-finite cross cell is a failure, never silently skipped.
        vals = list(cross.values())
        max_cross = float(max(vals)) if all(np.isfinite(vals)) else np.inf

        level = 1 - th.rational_null_alpha
        # Paired over items: the top intensity must beat neutral on the same items, beyond noise.
        top_diff = _paired(im, conds[-1], conds[0], p)
        _, diff_lo, _ = bootstrap_ci(top_diff, cfg.moments.bootstrap_draws, level)
        null_mean, null_lo, null_hi = bootstrap_ci(_cell(im, "rational", p), cfg.moments.bootstrap_draws, level)
        margin = th.rational_null_margin * abs(level_means[-1] - level_means[0])
        checks = {
            "monotonic": bool(np.isfinite(rho) and rho >= th.monotonic_spearman and own > 0
                              and np.isfinite(diff_lo) and diff_lo > 0),
            "near_diagonal": bool(max_cross < th.max_cross_loading),
            # Equivalence, not failure to reject: the whole CI must sit inside +/- margin.
            "rational_null": bool(np.isfinite(margin) and -margin <= null_lo and null_hi <= margin),
        }
        results[p] = {
            "passed": all(checks.values()),
            "checks": checks,
            "spearman": float(rho),
            "level_means": dict(zip(m.intensities, level_means)),
            "own_effect_sd": float(own),
            "cross_loadings": cross,
            "max_cross_loading": float(max_cross),
            "rational": {"mean": null_mean, "lo": null_lo, "hi": null_hi, "margin": float(margin)},
            "top_vs_neutral_lo": float(diff_lo),
            "sd_unit": sd[p],
            # Recovered intensity distribution used to seat agents in the market.
            "measured": {
                str(k): _cell(im, c, p).tolist() for k, c in zip(m.intensities, conds) if k > 0
            },
        }
        if p == "self_attribution":
            sa = df[(df.task == p) & df.condition.isin(conds)].groupby("level")[["beta", "gamma"]].mean()
            results[p]["beta_gamma_by_level"] = {int(k): r.to_dict() for k, r in sa.iterrows()}
    return results
