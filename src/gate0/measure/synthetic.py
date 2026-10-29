"""Synthetic agents with known bias parameters, used to validate estimators and checks without API calls."""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logit, softmax

from gate0.llm import Call
from gate0.measure.items import Item


def answer(item: Item, v: int, b: dict[str, float], rng: np.random.Generator, noise: float = 0.1) -> dict:
    """A biased Bayesian. b holds per-primitive true intensities (0 = rational on that dimension)."""
    c, r, o, a = (b.get(k, 0.0) for k in ("conservatism", "representativeness", "overconfidence", "self_attribution"))
    t = item.truth
    e = lambda: rng.normal(0, noise)  # noqa: E731
    if item.task == "conservatism":
        return {"probability_A": 100 * expit(t["prior_logodds"] + (1 - c) * t["evidence"] + e())}
    if item.task == "representativeness":
        logits = (1 - c) * np.array(t["loglik"]) + r * np.array(t["gain"])
        p = softmax(logits + rng.normal(0, noise, 3))
        return dict(zip(("X", "Y", "Z"), 100 * p))
    if item.task == "overconfidence":
        half = t["rational_width"] * np.exp(-o + e()) / 2
        return {"lower": t["mean"] - half, "upper": t["mean"] + half}
    if item.task == "self_attribution":
        r_base, r_hit, r_miss = (logit(x) for x in t["rational"])
        base = r_base + 0.5 * o  # overconfidence shifts the level; the estimator must cancel it
        moved = [base, base + (1 - c) * (r_hit - r_base) + a / 2, base - (1 - c) * (r_base - r_miss) + a / 2]
        return {"probability_correct": 100 * expit(moved[v] + e())}
    raise ValueError(item.task)


class SyntheticClient:
    """Drop-in for LLMClient.run: maps each call's persona to true intensities via `params`."""

    def __init__(self, params, seed: int = 0, noise: float = 0.1):
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
