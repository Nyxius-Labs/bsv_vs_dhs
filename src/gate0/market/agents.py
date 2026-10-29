"""Seating agents into arms, the rule-based agent, and the LLM agent's round prompt."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gate0.config import Config, Market
from gate0.market.fundamentals import World
from gate0.personas import PRIMITIVES, system_prompt, template

# ---- seating -------------------------------------------------------------------------------


@dataclass
class AgentSpec:
    j: int
    intensities: dict[str, int] = field(default_factory=dict)   # prompted level per primitive
    measured: dict[str, float] = field(default_factory=dict)    # mean recovered intensity at that level
    recovered: dict[str, float] = field(default_factory=dict)   # the item-level draw that chose the level
    rational: bool = False

    @property
    def system(self) -> str:
        return system_prompt(self.intensities, self.rational)


def arm_primitives(cfg: Config, arm: str, passed: set[str]) -> list[str]:
    """Primitives an arm switches on; only primitives that passed measurement may enter."""
    if arm in ("rational", "placebo"):
        return []
    wanted = sum(cfg.theories.values(), []) if arm == "joint" else cfg.theories[arm]
    return [p for p in wanted if p in passed]


def seat_agents(cfg: Config, arm: str, seed: int, measurement: dict | None) -> list[AgentSpec]:
    """Discrete prompted levels whose shares match the measured distribution, not set by fiat.

    For each agent and primitive: draw one recovered item-level intensity from the pooled
    nonzero-level measurements, then bin it to the prompted level whose mean recovery is closest.
    (A deviation from seating at a continuous measured intensity, which would need a calibration
    map from recovered intensity back to a prompt.)
    The RNG depends on (seed, agent, primitive) only, so the same agent gets the same draw
    in every arm that includes that primitive.
    """
    passed = set(PRIMITIVES) if measurement is None else {p for p, r in measurement.items() if r["passed"]}
    prims = arm_primitives(cfg, arm, passed)
    top = max(cfg.measurement.intensities)
    specs = []
    for j in range(cfg.market.n_agents):
        spec = AgentSpec(j, rational=(arm == "rational"))
        for p in prims:
            if measurement is None:
                spec.intensities[p] = top
                spec.measured[p] = spec.recovered[p] = RULE_INTENSITY[p]
                continue
            dist = measurement[p]["measured"]
            pool = [(int(k), s) for k, scores in dist.items() for s in scores]
            rng = np.random.default_rng([seed, j, PRIMITIVES.index(p)])
            draw = pool[rng.integers(len(pool))][1]
            means = {int(k): float(np.mean(v)) for k, v in dist.items() if v}
            level = min(means, key=lambda k: abs(means[k] - draw))
            spec.intensities[p], spec.measured[p], spec.recovered[p] = level, means[level], float(draw)
        specs.append(spec)
    return specs


# ---- rule-based agent (dry runs: exercises the market and moments with zero LLM calls) ----

RULE_INTENSITY = {"conservatism": 0.5, "representativeness": 0.5, "overconfidence": 1.5, "self_attribution": 0.3}


class RuleAgent:
    """Kalman-filter investor with switchable BSV/DHS distortions."""

    def __init__(self, spec: AgentSpec, world: World, m: Market):
        b = spec.measured
        self.c = b.get("conservatism", 0.0)          # shrinks every update gain
        self.rho = b.get("representativeness", 0.0)  # extrapolates a streak of report changes
        self.o = b.get("overconfidence", 0.0)        # inflates perceived private-signal precision
        self.sa = b.get("self_attribution", 0.0)     # beta; gamma is held at zero
        self.j, self.world = spec.j, world
        f = m.fundamentals
        self.q = f.factor_vol**2 + f.idio_vol**2
        self.r_e = f.earnings_noise**2
        n = m.n_assets
        self.mean = world.log_value[0] + world.common_noise[0] + world.agent_noise[0, :, spec.j]
        self.var = np.full(n, f.signal_vol_high**2)
        self.tau = np.ones(n)
        self.tilt = np.zeros(n)
        self.changes: list[np.ndarray] = []
        self.last_report = np.full(n, np.nan)

    def act(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        w = self.world
        self.var = self.var + self.q
        if np.isfinite(w.report[t, 0]):
            surprise = w.report[t] - self.mean
            k = (1 - self.c) * self.var / (self.var + self.r_e)
            self.mean = self.mean + k * surprise
            self.var = (1 - k) * self.var
            confirmed = np.sign(surprise) == np.sign(self.tilt)
            self.tau = np.clip(self.tau * np.where(confirmed, 1 + self.sa, 1.0), 0.2, 5.0)
            self.tilt[:] = 0
            if np.all(np.isfinite(self.last_report)):
                self.changes.append(w.report[t] - self.last_report)
                if len(self.changes) >= 2:
                    a, b = self.changes[-2:]
                    streak = np.sign(a) == np.sign(b)
                    self.mean = self.mean + np.where(streak, self.rho * (a + b) / 2, 0.0)
            self.last_report = w.report[t]
        s = w.signal(t, self.j)
        noise = w.signal_vol[t] ** 2 / ((1 + self.o) * self.tau)
        k = (1 - self.c) * self.var / (self.var + noise)
        self.tilt += k * (s - self.mean)
        self.mean = self.mean + k * (s - self.mean)
        self.var = (1 - k) * self.var
        confidence = np.clip(0.1 / np.sqrt(self.var), 1, 10)
        return np.exp(self.mean), confidence


# ---- LLM agent prompt ------------------------------------------------------------------------

VOCAB = {
    False: {"report": "earnings report", "consensus": "analyst consensus value",
            "dispersion": "analyst dispersion", "private": "your analyst's private estimate",
            "units": "shares", "money": "$"},
    True: {"report": "report", "consensus": "forecaster consensus value",
           "dispersion": "forecaster spread", "private": "your forecaster's estimate",
           "units": "units", "money": ""},
}


def order_schema(symbols: list[str]) -> dict:
    asset = {
        "type": "object",
        "properties": {"value": {"type": "number"}, "confidence": {"type": "number"}},
        "required": ["value", "confidence"],
        "additionalProperties": False,
    }
    return {"type": "object", "properties": {s: asset for s in symbols}, "required": symbols, "additionalProperties": False}


def round_prompt(
    m: Market, world: World, t: int, j: int, names: list[tuple[str, str]], masked: bool,
    prices: np.ndarray, holdings: np.ndarray, cash: float, prev_value: np.ndarray | None, prev_conf: np.ndarray | None,
) -> str:
    v = VOCAB[masked]
    money = v["money"]
    reports = [r for r in world.report_rounds if r <= t]
    signal = np.exp(world.signal(t, j))
    blocks = []
    for i, (_, name) in enumerate(names):
        hist = prices[max(0, t - m.price_window) : t, i]
        lines = [f"{name}", f"  recent prices (oldest to newest): " + ", ".join(f"{money}{p:.2f}" for p in hist)]
        if reports:
            last = reports[-1]
            line = f"  latest {v['report']} (round {last}): {money}{np.exp(world.report[last, i]):.2f}"
            if len(reports) > 1:
                prev = reports[-2]
                line += f"; previous (round {prev}): {money}{np.exp(world.report[prev, i]):.2f}"
            lines.append(line)
        else:
            lines.append(f"  no {v['report']} yet")
        lines.append(f"  {v['consensus']}: {money}{np.exp(world.consensus[t, i]):.2f}; "
                     f"{v['dispersion']}: ±{100 * world.dispersion[t, i]:.1f}%")
        lines.append(f"  {v['private']}: {money}{signal[i]:.2f}")
        if prev_value is not None and t >= 2:
            # Made in round t-1 against price p[t-2]; confirmed if the next price moved toward it.
            before, after = prices[t - 2, i], prices[t - 1, i]
            toward = np.sign(prev_value[i] - before) * np.sign(after - before)
            outcome = ("the price then moved toward it (confirmed)" if toward > 0 else
                       "the price then moved away from it (not confirmed)" if toward < 0 else
                       "the price then did not move relative to it (neither)")
            lines.append(f"  your previous value estimate: {money}{prev_value[i]:.2f} "
                         f"(confidence {prev_conf[i]:.0f}); {outcome}")
        lines.append(f"  you hold {holdings[i]:.1f} {v['units']}")
        blocks.append("\n".join(lines))
    name = "market_task_masked" if masked else "market_task"
    return template(name).format(round=t, n_rounds=m.n_rounds, cash=cash, assets="\n\n".join(blocks))
