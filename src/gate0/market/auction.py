"""Uniform-price call auction for one asset.

Agent j's demand at price p is  x_j(p) = scale * confidence_j * log(value_j / p),
clipped to [-holdings_j, budget_j / p] (no shorting, no borrowing). Every x_j is continuous and
non-increasing in p, so the clearing price is found by bisection on log p.
"""

from __future__ import annotations

import numpy as np

ITERATIONS = 80


def demand(p: float, value, confidence, holdings, budget, scale: float) -> np.ndarray:
    x = scale * confidence * np.log(value / p)
    return np.clip(x, -holdings, budget / p)


def clear(value, confidence, holdings, budget, scale: float, p_prev: float) -> tuple[float, np.ndarray]:
    """Return (clearing price, per-agent share trades summing to zero)."""
    value, confidence = np.asarray(value, float), np.asarray(confidence, float)
    holdings, budget = np.asarray(holdings, float), np.asarray(budget, float)
    excess = lambda lp: demand(np.exp(lp), value, confidence, holdings, budget, scale).sum()  # noqa: E731

    lo, hi = np.log(value.min()) - 0.01, np.log(value.max()) + 0.01
    if excess(lo) <= 0 or excess(hi) >= 0:
        return p_prev, np.zeros_like(value)  # nobody can buy, or nobody can sell
    for _ in range(ITERATIONS):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if excess(mid) > 0 else (lo, mid)
    p = float(np.exp((lo + hi) / 2))

    x = demand(p, value, confidence, holdings, budget, scale)
    buys, sells = x[x > 0].sum(), -x[x < 0].sum()
    matched = min(buys, sells)
    if matched <= 0:
        return p, np.zeros_like(x)
    x = np.where(x > 0, x * matched / buys, x * matched / sells)
    return p, x
