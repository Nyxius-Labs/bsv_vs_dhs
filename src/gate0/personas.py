"""System prompts built from primitive intensities, plus ticker naming (real-looking or masked)."""

from __future__ import annotations

from functools import cache

import numpy as np

from gate0.config import PROMPTS_DIR

PRIMITIVES = ("conservatism", "representativeness", "overconfidence", "self_attribution")


@cache
def template(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text().strip()


@cache
def bias_levels(primitive: str) -> dict[int, str]:
    levels = {}
    for line in template(primitive).splitlines():
        level, text = line.split(":", 1)
        levels[int(level)] = text.strip()
    return levels


def system_prompt(intensities: dict[str, int] | None = None, rational: bool = False) -> str:
    """Neutral persona, plus a rational instruction or one bias sentence per nonzero primitive.

    `intensities=None, rational=False` is the neutral/placebo persona: finance context, no bias text.
    """
    parts = [template("neutral")]
    if rational:
        parts.append(template("rational"))
    for p in PRIMITIVES:
        level = (intensities or {}).get(p, 0)
        if level:
            parts.append(bias_levels(p)[level])
    return "\n\n".join(parts)


_SYLLABLES = ["nor", "vel", "kai", "tor", "bri", "lum", "sar", "dex", "qua", "mon", "fen", "ost", "ral", "zen"]
_SUFFIXES = ["Industries", "Systems", "Holdings", "Group", "Labs", "Corp", "Partners", "Energy", "Logistics", "Foods"]


def tickers(n: int, rng: np.random.Generator, masked: bool) -> list[tuple[str, str]]:
    """(symbol, display name) per asset. Masked names carry no corporate or market cues."""
    if masked:
        codes = rng.choice(np.arange(10, 100), size=n, replace=False)
        return [(f"A{c}", f"Asset A{c}") for c in codes]
    out, seen = [], set()
    while len(out) < n:
        stem = "".join(rng.choice(_SYLLABLES, size=2)).capitalize()
        symbol = (stem[:3] + stem[-1]).upper()
        if symbol in seen:
            continue
        seen.add(symbol)
        out.append((symbol, f"{stem} {rng.choice(_SUFFIXES)} ({symbol})"))
    return out
