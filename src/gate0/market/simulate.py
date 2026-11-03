"""Market runs, stepped in lockstep so each round across all LLM runs is one batch."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gate0.config import Config
from gate0.llm import Call, LLMClient
from gate0.market.agents import AgentSpec, RuleAgent, order_schema, round_prompt, seat_agents, track_record
from gate0.market.auction import clear
from gate0.market.fundamentals import World, make_world
from gate0.personas import tickers


@dataclass
class MarketRun:
    cfg: Config
    arm: str
    seed: int
    masked: bool
    agents: list[AgentSpec]
    kind: str = "llm"  # or "rule"
    model: str = ""
    world: World = field(init=False)

    def __post_init__(self):
        m = self.cfg.market
        T, N, J = m.n_rounds, m.n_assets, m.n_agents
        self.world = make_world(m, self.seed)
        self.names = tickers(N, np.random.default_rng([self.seed, 0x7EC]), self.masked)
        self.symbols = [s for s, _ in self.names]
        self.prices = np.full((T + 1, N), np.nan)
        # Round 0 is a pre-market price set from public information only (never the true value).
        self.prices[0] = np.exp(self.world.consensus[0])
        self.volume = np.zeros((T + 1, N))
        self.holdings = np.full((J, N), float(m.initial_shares))
        self.cash = np.full(J, m.initial_cash)
        self.value = np.exp(np.stack([self.world.signal(0, j) for j in range(J)]))  # fallback before any answer
        self.conf = np.ones((J, N))
        self.estimates = np.full((J, T + 1, N), np.nan)  # estimates[j, s]: agent j's values from round s
        # Mean agent forecast (log) after each round; the pre-announcement consensus for PEAD.
        self.forecast = np.full((T + 1, N), np.nan)
        self.forecast[0] = np.log(self.value).mean(axis=0)
        self.fallbacks = 0
        self.rule_agents = [RuleAgent(a, self.world, m) for a in self.agents] if self.kind == "rule" else None

    @property
    def tag(self) -> str:
        return f"{self.arm}{'_masked' if self.masked else ''}_s{self.seed}"

    def calls(self, t: int) -> list[Call]:
        m = self.cfg.market
        schema = order_schema(self.symbols)
        # A track record needs n estimates whose next price is known: rounds t-n..t-1, all >= 1.
        show = m.track_record and t > m.track_record
        return [
            Call(
                self.model, a.system,
                round_prompt(m, self.world, t, a.j, self.names, self.masked, self.prices, self.holdings[a.j],
                             self.cash[a.j],
                             track_record(self.estimates[a.j], self.prices, t, m.track_record) if show else None),
                schema, nonce=self.tag,
            )
            for a in self.agents
        ]

    def ingest(self, outputs: list[dict | None], t: int) -> None:
        for j, out in enumerate(outputs):
            if out is None:
                self.fallbacks += 1
                continue  # keep last round's value and confidence
            for i, s in enumerate(self.symbols):
                try:
                    v, c = float(out[s]["value"]), float(out[s]["confidence"])
                except (KeyError, TypeError, ValueError):
                    self.fallbacks += 1
                    continue
                last = self.prices[t - 1, i]
                self.value[j, i] = np.clip(v, 0.2 * last, 5 * last) if np.isfinite(v) and v > 0 else self.value[j, i]
                self.conf[j, i] = np.clip(c, 1, 10) if np.isfinite(c) else self.conf[j, i]

    def act_rule(self, t: int) -> None:
        for j, agent in enumerate(self.rule_agents):
            self.value[j], self.conf[j] = agent.act(t)

    def clear_round(self, t: int) -> None:
        m = self.cfg.market
        self.forecast[t] = np.log(self.value).mean(axis=0)
        self.estimates[:, t] = self.value
        budget = self.cash / m.n_assets  # m.budget_per_asset == "cash_over_n"
        for i in range(m.n_assets):
            p, x = clear(self.value[:, i], self.conf[:, i], self.holdings[:, i], budget,
                         m.demand_scale, self.prices[t - 1, i])
            self.prices[t, i] = p
            self.holdings[:, i] += x
            self.cash -= x * p
            self.volume[t, i] = np.abs(x).sum() / 2

    def result(self) -> dict:
        w = self.world
        return {
            "arm": self.arm, "seed": self.seed, "masked": self.masked, "kind": self.kind, "model": self.model,
            "symbols": self.symbols, "prices": self.prices.tolist(), "volume": self.volume.tolist(),
            "forecast": self.forecast.tolist(),
            "log_value": w.log_value.tolist(), "report": np.where(np.isfinite(w.report), w.report, None).tolist(),
            "dispersion": w.dispersion.tolist(), "regime": w.regime.tolist(), "optimism": w.optimism.tolist(),
            "fallbacks": self.fallbacks,
            "agents": [{"j": a.j, "intensities": a.intensities, "measured": a.measured, "recovered": a.recovered,
                        "rational": a.rational}
                       for a in self.agents],
        }


def build_runs(cfg: Config, arms: list[str], seeds: list[int], masked: bool, kind: str,
               model: str, measurement: dict | None) -> list[MarketRun]:
    runs = []
    for arm in arms:
        for seed in seeds:
            agents = seat_agents(cfg, arm, seed, measurement)
            if arm not in ("rational", "placebo") and not any(a.intensities for a in agents):
                print(f"[market] skipping {arm}: none of its primitives passed measurement")
                break
            runs.append(MarketRun(cfg, arm, seed, masked, agents, kind, model))
    return runs


def simulate(runs: list[MarketRun], client: LLMClient | None, out_dir: Path) -> list[dict]:
    """Step all runs round by round; LLM calls for one round across every run go out together."""
    out_dir.mkdir(parents=True, exist_ok=True)
    T = runs[0].cfg.market.n_rounds if runs else 0
    for t in range(1, T + 1):
        llm_runs = [r for r in runs if r.kind == "llm"]
        if llm_runs:
            per_run = [r.calls(t) for r in llm_runs]
            outs = client.run([c for calls in per_run for c in calls])
            k = 0
            for r, calls in zip(llm_runs, per_run):
                r.ingest(outs[k : k + len(calls)], t)
                k += len(calls)
        for r in runs:
            if r.kind == "rule":
                r.act_rule(t)
            r.clear_round(t)
        if llm_runs:
            print(f"[market] round {t}/{T} done")
    results = []
    for r in runs:
        res = r.result()
        (out_dir / f"{r.tag}.json").write_text(json.dumps(res))
        results.append(res)
    return results
