"""Resampling helpers. Callers pass the unit of independence (items, or seeds), never agents."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(values, draws: int = 5000, level: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """(mean, lo, hi) percentile bootstrap of the mean over the given independent units."""
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(axis=1)
    a = (1 - level) / 2
    return float(x.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))
