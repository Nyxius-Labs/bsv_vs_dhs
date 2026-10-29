"""Measurement vignettes: one-stock market prompts, rendered with the market's own per-asset text.

Each item is one scenario rendered in several variants that change exactly one input (the newest
report, an older report, the private estimate, the consensus, the order of earlier reports, or the
confirmation line). A primitive is scored from how the stated value moves *between* variants, so an
agent that computes the Bayesian answer cannot hide or fake a bias by computing.

The Bayesian benchmark combines the sources whose precision the prompt states (reports, consensus,
private estimate) by generalized least squares under a diffuse prior, with reports linked to the
current value by the stated per-round drift. Prices are treated as uninformative; since prices are
identical across an item's variants, they drop out of every paired score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gate0.config import Config
from gate0.market.agents import (
    asset_block, confirmation, consensus_error, order_schema, private_error, stated_errors, task_prompt,
)
from gate0.personas import tickers

SOURCES = ("r2", "r1", "r0", "c", "p")  # reports oldest to newest, consensus, private estimate
MIN_WEIGHT = 0.15  # a slope divided by a small rational weight is too noisy to score
MIN_WEIGHT_SA = 0.25  # self-attribution combines six answers, so it needs a larger private-estimate weight


@dataclass(frozen=True)
class Item:
    task: str
    idx: int
    variants: tuple[dict, ...]  # per variant: log inputs and flags (the scorer's and synthetic agent's view)
    prompts: tuple[str, ...]    # per variant: the rendered user prompt
    truth: dict                 # rational weights, symbol and task constants
    schema: dict

    def render(self, v: int = 0) -> str:
        return self.prompts[v]


def make_items(cfg: Config, task: str, n: int, seed: int) -> list[Item]:
    """n items for one task. Every rendered prompt is unique across the returned items."""
    rng = np.random.default_rng([seed, ["conservatism", "representativeness", "overconfidence",
                                        "self_attribution"].index(task)])
    build = {"conservatism": _conservatism, "representativeness": _representativeness,
             "overconfidence": _overconfidence, "self_attribution": _self_attribution}[task]
    items, seen = [], set()
    while len(items) < n:
        item = build(cfg, rng, len(items))
        if item is None or len(set(item.prompts)) < len(item.prompts) or seen & set(item.prompts):
            continue
        seen |= set(item.prompts)
        items.append(item)
    return items


# ---- scenario and benchmark -------------------------------------------------------------------


def _scenario(cfg: Config, rng: np.random.Generator, offset: int) -> dict:
    """A plausible moment in the market: three reports, consensus, private estimate, prices."""
    m = cfg.market
    k = m.announcement_every
    err = stated_errors(m)
    e, d = err["report_err"] / 100, err["drift"] / 100
    sv = rng.choice([m.fundamentals.signal_vol_low, m.fundamentals.signal_vol_high])
    latest = k * int(rng.integers(m.reports_shown, (m.n_rounds - 3) // k + 1))
    t = latest + offset
    rounds = [latest - 2 * k, latest - k, latest]
    v_now = np.log(rng.uniform(40, 200))
    # Value at each report round, walking back from now.
    v_at = {r: v_now - rng.normal(0, d * np.sqrt(t - r)) for r in rounds}
    obs = {"r2": v_at[rounds[0]] + rng.normal(0, e), "r1": v_at[rounds[1]] + rng.normal(0, e),
           "r0": v_at[rounds[2]] + rng.normal(0, e)}
    c_err, p_err = consensus_error(m, sv) / 100, private_error(m, sv) / 100
    obs["c"] = v_now + rng.normal(0, c_err)
    obs["p"] = v_now + rng.normal(0, p_err)
    # Price history: a random walk ending near the current value (identical across an item's variants).
    steps = rng.normal(0, d / 2, m.price_window - 1)
    log_prices = v_now + rng.normal(0, 0.03) - np.r_[np.cumsum(steps[::-1])[::-1], 0.0]
    prices = np.exp(log_prices)
    return {
        "t": t, "rounds": rounds, "obs": obs, "prices": list(prices), "sv": sv,
        "errors": {"r": e, "d": d, "c": c_err, "p": p_err},
        "dispersion": 100 * sv / np.sqrt(2) * rng.uniform(0.8, 1.2),
        "holding": round(float(rng.uniform(8, 12)), 1), "cash": round(float(rng.uniform(900, 1100)), 2),
    }


def rational_weights(sc: dict) -> dict[str, float]:
    """GLS weights on each source's log value for the current log value (diffuse prior)."""
    e, d, ce, pe = (sc["errors"][x] for x in ("r", "d", "c", "p"))
    lags = [sc["t"] - r for r in sc["rounds"]]
    cov = np.zeros((5, 5))
    for i in range(3):
        for j in range(3):
            cov[i, j] = d**2 * min(lags[i], lags[j]) + (e**2 if i == j else 0.0)
    cov[3, 3], cov[4, 4] = ce**2, pe**2
    w = np.linalg.solve(cov, np.ones(5))
    return dict(zip(SOURCES, (w / w.sum()).tolist()))


def _render(cfg: Config, sc: dict, name: str, obs: dict, prices: list[float],
            prev: tuple[float, float, str] | None) -> str:
    m = cfg.market
    block = asset_block(
        False, name, prices, [(r, float(np.exp(obs[s]))) for r, s in zip(sc["rounds"], ("r2", "r1", "r0"))],
        float(np.exp(obs["c"])), consensus_error(m, sc["sv"]), sc["dispersion"],
        float(np.exp(obs["p"])), private_error(m, sc["sv"]), prev, sc["holding"],
    )
    return task_prompt(m, False, sc["t"], sc["cash"], [block])


def _item(cfg, task, idx, rng, sc, variants, prev_by_variant=None, prices_by_variant=None, **extra) -> Item:
    symbol, name = tickers(1, rng, False)[0]
    prompts = tuple(
        _render(cfg, sc, name, v["obs"],
                prices_by_variant[i] if prices_by_variant else sc["prices"],
                prev_by_variant[i] if prev_by_variant else None)
        for i, v in enumerate(variants)
    )
    truth = {"symbol": symbol, "w": rational_weights(sc), "anchor": float(np.log(sc["prices"][-1])), **extra}
    return Item(task, idx, tuple(variants), prompts, truth, order_schema([symbol]))


def _shifted(obs: dict, source: str, delta: float) -> dict:
    return {**obs, source: obs[source] + delta}


# ---- per-primitive designs --------------------------------------------------------------------


def _conservatism(cfg: Config, rng: np.random.Generator, idx: int) -> Item | None:
    """Newest report +/- delta and the previous report +/- delta, just after the newest is released."""
    sc, dl = _scenario(cfg, rng, offset=0), cfg.measurement.shift
    w = rational_weights(sc)
    if w["r0"] < MIN_WEIGHT:
        return None
    variants = [{"obs": _shifted(sc["obs"], s, sign * dl)} for s in ("r0", "r1") for sign in (1, -1)]
    return _item(cfg, "conservatism", idx, rng, sc, variants, delta=dl)


def _representativeness(cfg: Config, rng: np.random.Generator, idx: int) -> Item:
    """The same three report values as a streak (monotone) vs. with the two older ones swapped."""
    sc = _scenario(cfg, rng, offset=0)
    direction = float(rng.choice([-1.0, 1.0]))
    s1, s2 = rng.uniform(0.04, 0.10, 2)
    r0 = sc["obs"]["r0"]
    r1, r2 = r0 - direction * s2, r0 - direction * (s1 + s2)
    streak = {**sc["obs"], "r2": r2, "r1": r1}
    swapped = {**sc["obs"], "r2": r1, "r1": r2}
    variants = [{"obs": streak, "streak": True}, {"obs": swapped, "streak": False}]
    return _item(cfg, "representativeness", idx, rng, sc, variants, direction=direction, span=float(s1 + s2))


def _overconfidence(cfg: Config, rng: np.random.Generator, idx: int) -> Item | None:
    """Private estimate +/- delta and consensus +/- delta, between reports."""
    sc, dl = _scenario(cfg, rng, offset=int(rng.integers(1, 4))), cfg.measurement.shift
    w = rational_weights(sc)
    if min(w["p"], w["c"]) < MIN_WEIGHT:
        return None
    variants = [{"obs": _shifted(sc["obs"], s, sign * dl)} for s in ("p", "c") for sign in (1, -1)]
    return _item(cfg, "overconfidence", idx, rng, sc, variants, delta=dl)


FEEDBACK = ("neither", "confirmed", "not confirmed")


def _self_attribution(cfg: Config, rng: np.random.Generator, idx: int) -> Item | None:
    """{no move, confirmed, contradicted} previous estimate x private estimate +/- delta.

    Only the second-to-last price differs between feedback conditions; the last price, and so the
    current information, is identical. The previous estimate is on the same side in all three.
    """
    sc, dl = _scenario(cfg, rng, offset=int(rng.integers(1, 4))), cfg.measurement.shift
    if rational_weights(sc)["p"] < MIN_WEIGHT_SA:
        return None
    direction = float(rng.choice([-1.0, 1.0]))
    last, move = sc["prices"][-1], 0.02
    estimate = last * np.exp(direction * 2 * move)
    before = {"neither": last, "confirmed": last * np.exp(-direction * move),
              "not confirmed": last * np.exp(direction * move)}
    variants, prev, prices = [], [], []
    for fb in FEEDBACK:
        outcome = confirmation(estimate, before[fb], last)
        assert f"({fb})" in outcome
        for sign in (1, -1):
            variants.append({"obs": _shifted(sc["obs"], "p", sign * dl), "feedback": fb})
            prev.append((float(estimate), 5.0, outcome))
            prices.append([*sc["prices"][:-2], float(before[fb]), last])
    return _item(cfg, "self_attribution", idx, rng, sc, variants, prev, prices, delta=dl)
