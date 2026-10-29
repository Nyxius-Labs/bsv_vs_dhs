"""Item-level bias scores from paired vignette variants. Rational answers score 0; the bias scores > 0.

A slope is the change in the stated log value per unit change in one input's log value, from a
+delta / -delta pair of variants. Rational slopes are the benchmark weights in `item.truth["w"]`.
"""

from __future__ import annotations

import numpy as np

from gate0.measure.items import Item


def _log_value(item: Item, out: dict) -> float:
    v = float(out[item.truth["symbol"]]["value"])
    if not v > 0:
        raise ValueError("non-positive value")
    return float(np.log(v))


def _slope(item: Item, lv: list[float], i: int) -> float:
    """Slope from variants i (+delta) and i+1 (-delta)."""
    return (lv[i] - lv[i + 1]) / (2 * item.truth["delta"])


def score(item: Item, outs: list[dict | None]) -> float | None:
    """Score one item from its per-variant outputs (aligned with `item.variants`)."""
    if any(o is None for o in outs):
        return None
    try:
        lv = [_log_value(item, o) for o in outs]
        return {
            "conservatism": _conservatism,
            "representativeness": _representativeness,
            "overconfidence": _overconfidence,
            "self_attribution": _self_attribution,
        }[item.task](item, lv)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def _conservatism(item: Item, lv: list[float]) -> float:
    """Underreaction to the newest report plus overreaction to the previous one, in weight units:
    (w_new - s_new) + (s_old - w_old). Uniform underreaction and anchoring on older news both score > 0."""
    w = item.truth["w"]
    return float((w["r0"] - _slope(item, lv, 0)) + (_slope(item, lv, 2) - w["r1"]))


def _representativeness(item: Item, lv: list[float]) -> float:
    """Extrapolation of a report streak: (streak - swapped) value beyond the rational difference,
    in the streak's direction, per unit of the streak's span."""
    t, w = item.truth, item.truth["w"]
    a, b = item.variants[0]["obs"], item.variants[1]["obs"]
    rational = sum(w[s] * (a[s] - b[s]) for s in ("r2", "r1"))
    return float(t["direction"] * ((lv[0] - lv[1]) - rational) / t["span"])


def _overconfidence(item: Item, lv: list[float]) -> float:
    """Weight on the private estimate relative to its rational weight, minus the same for the
    consensus (DHS overconfidence: private information overweighted relative to public)."""
    w = item.truth["w"]
    return float(_slope(item, lv, 0) / w["p"] - _slope(item, lv, 2) / w["c"])


def _self_attribution(item: Item, lv: list[float]) -> float:
    """beta - gamma: rise in the private-estimate weight after a confirmation minus its fall after a
    contradiction, both from the agent's own no-feedback weight, in units of the rational weight."""
    beta, gamma = _beta_gamma(item, lv)
    return beta - gamma


def _beta_gamma(item: Item, lv: list[float]) -> tuple[float, float]:
    s = {item.variants[i]["feedback"]: _slope(item, lv, i) for i in range(0, len(lv), 2)}
    wp = item.truth["w"]["p"]
    return (s["confirmed"] - s["neither"]) / wp, (s["neither"] - s["not confirmed"]) / wp


def beta_gamma(item: Item, outs: list[dict | None]) -> tuple[float, float] | None:
    """Raw (beta, gamma) for one self-attribution item; DHS requires beta > gamma."""
    if any(o is None for o in outs):
        return None
    try:
        return _beta_gamma(item, [_log_value(item, o) for o in outs])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
