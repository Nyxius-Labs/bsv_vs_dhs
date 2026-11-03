"""Item-level bias scores from paired vignette variants. Rational answers score 0; the bias scores > 0.

A slope is the change in the stated log value per unit change in one input's log value, from a
+delta / -delta pair of variants. Rational slopes are the benchmark weights in `item.truth["w"]`.
"""

from __future__ import annotations

import numpy as np

from gate0.measure.items import Item

SLOPE_FLOOR = 0.01  # an ignored source (slope <= 0) counts as weight 0.01, keeping log-ratios finite


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


def _log_ratio(num: float, den: float, w_num: float, w_den: float) -> float:
    """log(num/den) - log(w_num/w_den): how much more one source is weighted than another, relative
    to the benchmark. A proportional change in every other source's weight leaves it unchanged, so
    a bias acting on a third source does not load on it."""
    return float(np.log(max(num, SLOPE_FLOOR) / max(den, SLOPE_FLOOR)) - np.log(w_num / w_den))


def _conservatism(item: Item, lv: list[float]) -> float:
    """Earnings news underweighted relative to the market's prior (the price): -log of the reports'
    share of the weight on {reports, price}, log((s_R + s_a) / s_R). The benchmark gives the price no
    weight, so rational = 0. R moves every report together, so extrapolating the reports' trend does
    not load on it; overweighting the private estimate takes weight from reports and price alike, so
    it does not load either."""
    s_r, s_a = _slope(item, lv, 0), max(_slope(item, lv, 2), 0.0)
    return float(np.log((max(s_r, SLOPE_FLOOR) + s_a) / max(s_r, SLOPE_FLOOR)))


def _representativeness(item: Item, lv: list[float]) -> float:
    """Extrapolation of a report streak: (rising - falling) value beyond the rational difference,
    per unit of the two streaks' combined span."""
    t, w = item.truth, item.truth["w"]
    a, b = item.variants[0]["obs"], item.variants[1]["obs"]
    rational = sum(w[s] * (a[s] - b[s]) for s in ("r2", "r1"))
    return float(((lv[0] - lv[1]) - rational) / (2 * t["span"]))


def _overconfidence(item: Item, lv: list[float]) -> float:
    """Private estimate overweighted relative to the public consensus (DHS overconfidence):
    log(s_p / s_c) - log(w_p / w_c)."""
    w = item.truth["w"]
    return _log_ratio(_slope(item, lv, 0), _slope(item, lv, 2), w["p"], w["c"])


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
