"""Measurement-stage design and execution."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from gate0.config import Config
from gate0.llm import Call, LLMClient
from gate0.measure.estimate import beta_gamma, score
from gate0.measure.items import make_items
from gate0.personas import system_prompt


@dataclass(frozen=True)
class Condition:
    name: str
    intensities: dict = field(default_factory=dict, hash=False)
    rational: bool = False
    toggled: str | None = None
    level: int = 0

    @property
    def system(self) -> str:
        return system_prompt(self.intensities, self.rational)


NEUTRAL = Condition("neutral")
RATIONAL = Condition("rational", rational=True)


def at(primitive: str, level: int) -> Condition:
    if level == 0:
        return NEUTRAL
    return Condition(f"{primitive}@{level}", {primitive: level}, toggled=primitive, level=level)


def base_design(cfg: Config) -> list[tuple[Condition, str]]:
    """Own-task intensity grid, max-intensity toggles on the other tasks, rational arm on all tasks."""
    m = cfg.measurement
    top = max(m.intensities)
    cells = [(at(p, k), p) for p in m.primitives for k in m.intensities]
    cells += [(at(p, top), q) for p in m.primitives for q in m.primitives if q != p]
    cells += [(RATIONAL, q) for q in m.primitives]
    return cells


def strong_design(cfg: Config) -> list[tuple[Condition, str]]:
    s = cfg.strong
    conds = [RATIONAL if c == "rational" else at(c, s.intensity) for c in s.conditions]
    return [(c, q) for c in conds for q in cfg.measurement.primitives]


def run_measurement(
    cfg: Config, client: LLMClient, model: str, cells: list[tuple[Condition, str]], repeats: int,
    dev: bool = False,
) -> pd.DataFrame:
    """`dev=True` draws development items from a different seed, keeping the registered items held out."""
    m = cfg.measurement
    seed = m.item_seed + 1 if dev else m.item_seed
    items = {q: make_items(q, m.n_items, seed) for q in m.primitives}
    calls, index = [], []
    for cond, task in cells:
        for item in items[task]:
            for rep in range(repeats):
                index.append((cond, task, item, rep, len(calls), len(item.variants)))
                calls.extend(
                    Call(model, cond.system, item.render(v), item.schema, nonce=f"rep{rep}",
                         meta={"condition": cond, "item": item, "variant": v, "rep": rep})
                    for v in range(len(item.variants))
                )
    print(f"[measure] {len(calls)} calls on {model}")
    outs = client.run(calls)
    rows = []
    for cond, task, item, rep, start, nv in index:
        o = outs[start : start + nv]
        bg = beta_gamma(item, o) if task == "self_attribution" else None
        rows.append({
            "model": model, "condition": cond.name, "toggled": cond.toggled, "level": cond.level,
            "rational": cond.rational, "task": task, "item": item.idx, "rep": rep,
            "score": score(item, o), "beta": bg[0] if bg else None, "gamma": bg[1] if bg else None,
        })
    return pd.DataFrame(rows)
