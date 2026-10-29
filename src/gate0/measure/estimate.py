"""Item-level bias scores. Each is oriented so the rational answer scores 0 and the bias scores > 0."""

from __future__ import annotations

import numpy as np
from scipy.special import logit

from gate0.measure.items import Item

EPS = 0.01


def _prob(x: float) -> float:
    """Percent -> probability. A value strictly between 0 and 1 is read as already a fraction
    (a stated 0.66% where 66% was meant); the schema asks for 0-100, so this is rare."""
    x = float(x)
    p = x if 0 < x < 1 else x / 100.0
    return float(np.clip(p, EPS, 1 - EPS))


def score(item: Item, outs: list[dict | None]) -> float | None:
    """Score one item from its per-variant outputs (aligned with `item.variants`)."""
    if any(o is None for o in outs):
        return None
    out = outs if item.task == "self_attribution" else outs[0]
    try:
        return {
            "conservatism": _conservatism,
            "representativeness": _representativeness,
            "overconfidence": _overconfidence,
            "self_attribution": _self_attribution,
        }[item.task](item, out)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def _conservatism(item: Item, out: dict) -> float:
    """1 - w, where w is the weight on the likelihood ratio (Grether-style)."""
    t = item.truth
    w = (logit(_prob(out["probability_A"])) - t["prior_logodds"]) / t["evidence"]
    return float(1 - w)


def _representativeness(item: Item, out: dict) -> float:
    """Weight on the last three draws minus weight on the earlier draws.

    The two independent log-ratios of the stated probabilities are solved exactly for
    log(p_u / p_ref) = w_early * d_early + w_recent * d_recent. A rational answer has both weights
    at 1. Uniform shrinkage toward the prior (conservatism, or ignoring the evidence) lowers both
    equally, so it scores 0; only overweighting what rose most recently scores > 0.
    """
    total = sum(float(out[k]) for k in ("X", "Y", "Z"))
    if total <= 0:
        return None
    p = np.clip([float(out[k]) / total for k in ("X", "Y", "Z")], EPS, 1)
    t = item.truth
    ref = ("X", "Y", "Z").index(t["target"])
    others = [u for u in range(3) if u != ref]
    X = np.array([[t["early"][u] - t["early"][ref], t["gain"][u] - t["gain"][ref]] for u in others])
    y = np.array([np.log(p[u] / p[ref]) for u in others])
    w_early, w_recent = np.linalg.solve(X, y)
    return float(w_recent - w_early)


def _overconfidence(item: Item, out: dict) -> float:
    """log(rational width / stated width): overprecision > 0."""
    width = float(out["upper"]) - float(out["lower"])
    if width <= 0:
        return None
    return float(np.log(item.truth["rational_width"] / width))


def _self_attribution(item: Item, outs: list[dict]) -> float:
    """(beta - beta_rational) - (gamma - gamma_rational), in log-odds of stated forecast accuracy.

    beta = rise from the agent's own baseline after a hit, gamma = fall after a miss. Measuring both
    from the agent's own baseline cancels any level shift (e.g. overconfidence).
    """
    base, hit, miss = (logit(_prob(o["probability_correct"])) for o in outs)
    r_base, r_hit, r_miss = (logit(r) for r in item.truth["rational"])
    beta, gamma = hit - base, base - miss
    return float((beta - (r_hit - r_base)) - (gamma - (r_base - r_miss)))


def beta_gamma(item: Item, outs: list[dict | None]) -> tuple[float, float] | None:
    """Raw (beta, gamma) for one self-attribution item; DHS requires beta > gamma."""
    if any(o is None for o in outs):
        return None
    base, hit, miss = (logit(_prob(o["probability_correct"])) for o in outs)
    return float(hit - base), float(base - miss)
