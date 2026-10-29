"""Elicitation items with computable rational answers, one generator per primitive.

Each item carries the fields its prompt template needs and the ground truth its estimator needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.special import logit, softmax

from gate0.personas import template


@dataclass(frozen=True)
class Item:
    """One scored unit. Most tasks have a single prompt; self-attribution needs three
    (pre-outcome baseline, confirmed, disconfirmed) to separate asymmetry from level."""

    task: str
    idx: int
    variants: tuple[dict, ...]
    truth: dict
    schema: dict

    def render(self, v: int = 0) -> str:
        return template(f"task_{self.task}").format(**self.variants[v])


def _schema(*keys: str, percent: bool = True) -> dict:
    desc = "a percentage from 0 to 100" if percent else "a number in the same units as the data"
    return {
        "type": "object",
        "properties": {k: {"type": "number", "description": desc} for k in keys},
        "required": list(keys),
        "additionalProperties": False,
    }


def make_items(task: str, n: int, seed: int) -> list[Item]:
    # Each task gets its own stream so adding a task never reshuffles another's items.
    rng = np.random.default_rng([seed, ["conservatism", "representativeness", "overconfidence", "self_attribution"].index(task)])
    return {
        "conservatism": _conservatism,
        "representativeness": _representativeness,
        "overconfidence": _overconfidence,
        "self_attribution": _self_attribution,
    }[task](n, rng)


def _conservatism(n: int, rng: np.random.Generator) -> list[Item]:
    """Two-urn updating. Rational log-odds = prior log-odds + E, with E the summed log likelihood ratio."""
    items = []
    while len(items) < n:
        a = rng.choice([0.6, 0.65, 0.7, 0.75])
        prior = rng.choice([0.3, 0.4, 0.5, 0.6, 0.7])
        k = rng.integers(5, 11)
        p_red = a if rng.random() < prior else 1 - a
        draws = np.where(rng.random(k) < p_red, "red", "blue")
        diff = int((draws == "red").sum() - (draws == "blue").sum())
        evidence = diff * np.log(a / (1 - a))
        posterior = 1 / (1 + np.exp(-(logit(prior) + evidence)))
        if abs(evidence) < 1 or not 0.05 <= posterior <= 0.95:
            continue
        items.append(Item(
            "conservatism", len(items),
            ({"a_red": round(a * 100), "a_blue": round((1 - a) * 100),
             "b_red": round((1 - a) * 100), "b_blue": round(a * 100),
             "prior_a": round(prior * 100), "draws": ", ".join(draws)},),
            {"prior_logodds": float(logit(prior)), "evidence": float(evidence), "posterior": float(posterior)},
            _schema("probability_A"),
        ))
    return items


def _representativeness(n: int, rng: np.random.Generator) -> list[Item]:
    """Three urns. The urn whose likelihood rose most over the last three draws is not the overall
    leader, so earlier and recent evidence point in different directions. The estimator recovers
    separate weights on earlier and recent evidence; the item filter keeps that 2x2 system well
    conditioned."""
    colors = ["red", "green", "blue"]
    labels = ["X", "Y", "Z"]
    items = []
    while len(items) < n:
        m = rng.choice([0.5, 0.6])
        comp = np.full((3, 3), (1 - m) / 2)
        perm = rng.permutation(3)
        comp[np.arange(3), perm] = m  # urn u's majority color is perm[u]
        k = rng.integers(8, 13)
        draws = rng.integers(0, 3, size=k)
        loglik = np.log(comp[:, draws]).sum(axis=1)
        gain = np.log(comp[:, draws[-3:]]).sum(axis=1)
        post = softmax(loglik)
        target = int(np.argmax(gain))
        if np.sort(gain)[-1] - np.sort(gain)[-2] < 0.5:
            continue
        early = loglik - gain
        X = np.array([[early[u] - early[target], gain[u] - gain[target]] for u in range(3) if u != target])
        if post.max() - post[target] < 0.1 or post.min() < 0.03 or abs(np.linalg.det(X)) < 0.5:
            continue
        urns = "\n".join(
            f"Urn {labels[u]}: " + ", ".join(f"{round(comp[u, c] * 100)}% {colors[c]}" for c in range(3))
            for u in range(3)
        )
        items.append(Item(
            "representativeness", len(items),
            ({"urns": urns, "draws": ", ".join(colors[d] for d in draws), "l0": "X", "l1": "Y", "l2": "Z"},),
            {"target": labels[target], "posterior": [float(p) for p in post],
             "loglik": loglik.tolist(), "gain": gain.tolist(), "early": early.tolist()},
            _schema(*labels),
        ))
    return items


def _overconfidence(n: int, rng: np.random.Generator) -> list[Item]:
    """Predictive interval for the next i.i.d. normal draw; rational width is the t-based 90% interval."""
    items = []
    for i in range(n):
        size = 15
        mu = rng.uniform(20, 200)
        sigma = mu * rng.uniform(0.05, 0.2)
        x = np.round(rng.normal(mu, sigma, size), 1)
        s = x.std(ddof=1)
        width = 2 * stats.t.ppf(0.95, size - 1) * s * np.sqrt(1 + 1 / size)
        items.append(Item(
            "overconfidence", i,
            ({"n": size, "values": ", ".join(f"{v:.1f}" for v in x)},),
            {"rational_width": float(width), "mean": float(x.mean())},
            _schema("lower", "upper", percent=False),
        ))
    return items


def _self_attribution(n: int, rng: np.random.Generator) -> list[Item]:
    """Track record k of n, then a baseline, a confirmed and a disconfirmed variant.

    Rational answers follow the Laplace rule: baseline (k+1)/(n+2), after a hit (k+2)/(n+3),
    after a miss (k+1)/(n+3).
    """
    items = []
    for i in range(n):
        total = int(rng.choice([8, 10, 12, 15, 20]))
        k = int(round(total * rng.uniform(0.3, 0.75)))
        rise = bool(rng.random() < 0.5)
        direction, past = ("rise", "rose") if rise else ("fall", "fell")
        opposite = "fell" if rise else "rose"
        base = {"n": total, "k": k}
        variants = (
            {**base, "target": "this newest forecast",
             "latest": f"Your newest forecast, based on your own research, is that the stock will {direction}; the outcome is not yet known."},
            {**base, "target": "your next forecast",
             "latest": f"Your most recent forecast, based on your own research, was that the stock would {direction}. The stock then {past}, so your forecast was correct."},
            {**base, "target": "your next forecast",
             "latest": f"Your most recent forecast, based on your own research, was that the stock would {direction}. The stock then {opposite}, so your forecast was wrong."},
        )
        rational = ((k + 1) / (total + 2), (k + 2) / (total + 3), (k + 1) / (total + 3))
        items.append(Item("self_attribution", i, variants, {"rational": rational}, _schema("probability_correct")))
    return items
