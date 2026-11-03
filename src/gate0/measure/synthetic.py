"""Synthetic agents with known bias parameters, used to validate estimators and checks without API calls."""

from __future__ import annotations

import numpy as np

from gate0.llm import Call
from gate0.measure.items import REPORTS, SOURCES, Item


def answer(item: Item, v: int, b: dict[str, float], rng: np.random.Generator, noise: float = 0.01) -> dict:
    """A distorted benchmark weighting. b holds per-primitive true intensities (0 = rational on that
    dimension). Each distortion moves exactly the slope its scorer reads, so a noiseless agent
    scores its true intensity on its own task and 0 elsewhere."""
    c, r, o, a = (b.get(k, 0.0) for k in ("conservatism", "representativeness", "overconfidence", "self_attribution"))
    t, var = item.truth, item.variants[v]
    w = dict(t["w"])
    w_a = sum(w[k] for k in REPORTS) * (1 - np.exp(-c))  # weight moved from the reports to the last price
    for k in REPORTS:
        w[k] *= np.exp(-c)            # underreacts to earnings reports
    w["p"] *= np.exp(o)               # overweights the private estimate
    if var.get("feedback") == "confirmed":
        w["p"] += a * t["w"]["p"]     # credits itself after a confirmation; no blame after a miss
    log_v = sum(w[s] * var["obs"][s] for s in SOURCES) + w_a * var["obs"]["a"]
    if var.get("streak"):
        log_v += r * var["streak"] * t["span"]  # extrapolates the streak
    log_v += rng.normal(0, noise)
    return {t["symbol"]: {"value": float(np.exp(log_v)), "confidence": 5.0}}


class SyntheticClient:
    """Drop-in for LLMClient.run: maps each call's persona to true intensities via `params`."""

    def __init__(self, params, seed: int = 0, noise: float = 0.01):
        self.params, self.noise = params, noise
        self.rng = np.random.default_rng(seed)

    def run(self, calls: list[Call]) -> list[dict]:
        return [
            answer(c.meta["item"], c.meta["variant"], self.params(c.meta["condition"]), self.rng, self.noise)
            for c in calls
        ]


def clean_params(scale: float = 0.25):
    """Each prompted level k of primitive p yields true intensity scale*k on p only; rational is 0."""
    return lambda cond: {p: scale * k for p, k in cond.intensities.items()}
