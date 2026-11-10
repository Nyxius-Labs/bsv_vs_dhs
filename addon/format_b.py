"""Gate 0 v3b: a second test format whose wording does not mirror the persona text.

Usage (from the repo root):
    uv run python addon/format_b.py register        # write registration_v3b.lock
    uv run python addon/format_b.py smoke           # dev items, a few calls: checks answers parse
    uv run python addon/format_b.py run [--mode M]  # the registered grid (refuses if the lock drifted)
    uv run python addon/format_b.py report          # data/registered_v3b/report.md

Scores follow the v3 convention: the Bayesian answer scores 0 and the bias scores > 0.
- conservatism:       (w - s) / w, where s is the slope of the stated log estimate on the log survey value
                      and w the Bayesian weight on the survey.
- representativeness: (log forecast after a rising run - after a falling run) / (2 x run span); rational 0,
                      because the series is stated to vary independently around a known average.
- overconfidence:     log(Bayesian 90% range / stated 90% range), ranges in log units.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gate0.config import CONFIG_PATH, DATA_DIR, ROOT, load_config, registration_hashes
from gate0.llm import Call, LLMClient
from gate0.measure.checks import analyze
from gate0.measure.run import RATIONAL, at
from gate0.personas import tickers

V3B_CONFIG = ROOT / "configs" / "registration_v3b.yaml"
LOCK = ROOT / "registration_v3b.lock"
OUT = DATA_DIR / "registered_v3b"
Z90 = 1.6448536269514722

VALUE = {"type": "object", "properties": {"estimate": {"type": "number"}},
         "required": ["estimate"], "additionalProperties": False}
RANGE = {"type": "object",
         "properties": {"estimate": {"type": "number"}, "low": {"type": "number"}, "high": {"type": "number"}},
         "required": ["estimate", "low", "high"], "additionalProperties": False}


@dataclass(frozen=True)
class Item:
    task: str
    idx: int
    prompts: tuple[str, ...]
    truth: dict
    schema: dict


def settings() -> dict:
    return yaml.safe_load(V3B_CONFIG.read_text())


# ---- items --------------------------------------------------------------------------------------


def _conservatism(rng, idx, dl) -> Item | None:
    a, b = rng.uniform(8, 15), rng.uniform(8, 15)  # typical % errors: earlier estimate, survey
    w = a**2 / (a**2 + b**2)
    if not 0.3 <= w <= 0.7:
        return None
    name = tickers(1, rng, False)[0][1]
    prior = float(np.exp(rng.uniform(np.log(40), np.log(400))))
    g = rng.choice([-1, 1]) * rng.uniform(0.08, 0.25)
    shown = [round(prior * np.exp(g + s * dl), 1) for s in (1, -1)]
    prompts = tuple(
        f"Company: {name}.\n\n"
        f"Three months ago you estimated this company's sustainable annual revenue at ${prior:.1f} million. "
        f"Estimates like yours have a typical error of about {a:.0f}%.\n\n"
        f"An independent industry survey, published today, puts the company's sustainable annual revenue at "
        f"${r:.1f} million. Surveys like this have a typical error of about {b:.0f}%. The survey and your earlier "
        f"estimate are independent of each other.\n\n"
        f"What is your estimate now of the company's sustainable annual revenue, in millions of dollars? "
        f'Respond with JSON: {{"estimate": <number>}}.'
        for r in shown
    )
    return Item("conservatism", idx, prompts, {"w": w, "log_gap": float(np.log(shown[0] / shown[1]))}, VALUE)


def _representativeness(rng, idx, dl) -> Item | None:
    name = tickers(1, rng, False)[0][1]
    avg = int(round(rng.uniform(800, 3000), -1))
    sd = rng.uniform(10, 20)
    last = avg * np.exp(rng.uniform(-0.08, 0.08))
    s1, s2 = rng.uniform(0.05, 0.12, 2)
    first = int(round(avg * np.exp(rng.normal(0, sd / 200))))
    runs = []
    for d in (1, -1):  # rising, then falling, to the same newest month
        runs.append([first] + [int(round(last * np.exp(-d * x))) for x in (s1 + s2, s2)] + [int(round(last))])
    if len({tuple(r) for r in runs}) < 2:
        return None
    prompts = tuple(
        f"Company: {name}.\n\n"
        f"Over many years, this company's monthly unit orders have varied independently from month to month "
        f"around a long-run average of {avg:,} units, with a typical deviation of about {sd:.0f}%.\n\n"
        f"Orders in the last four months, oldest first: {', '.join(f'{x:,}' for x in r)} units.\n\n"
        f"What is your forecast of next month's orders, in units? "
        f'Respond with JSON: {{"estimate": <number>}}.'
        for r in runs
    )
    return Item("representativeness", idx, prompts, {"span": float(s1 + s2), "avg": avg}, VALUE)


def _overconfidence(rng, idx, dl) -> Item | None:
    name = tickers(1, rng, False)[0][1]
    e = rng.uniform(6, 15)
    est = float(np.exp(rng.uniform(np.log(20), np.log(200))))
    prompt = (
        f"Company: {name}.\n\n"
        f"Your own research puts the fair value of one share at ${est:.2f}. Over many past estimates, research like "
        f"yours has typically been off by about {e:.0f}% (one standard deviation, roughly symmetric in percentage "
        f"terms).\n\n"
        f"Give your best estimate of the fair value per share, and a range you are 90% sure contains it. "
        f'Respond with JSON: {{"estimate": <number>, "low": <number>, "high": <number>}}.'
    )
    return Item("overconfidence", idx, (prompt,), {"bayes_width": 2 * Z90 * np.log(1 + e / 100)}, RANGE)


BUILD = {"conservatism": _conservatism, "representativeness": _representativeness, "overconfidence": _overconfidence}


def make_items(task: str, n: int, seed: int, shift: float) -> list[Item]:
    rng = np.random.default_rng([seed, list(BUILD).index(task)])
    items, seen = [], set()
    while len(items) < n:
        it = BUILD[task](rng, len(items), shift)
        if it is None or seen & set(it.prompts):
            continue
        seen |= set(it.prompts)
        items.append(it)
    return items


# ---- scoring ------------------------------------------------------------------------------------


def _pos(x) -> bool:
    return x is not None and np.isfinite(x) and x > 0


def score(item: Item, outs: list[dict | None]) -> float:
    if any(o is None for o in outs):
        return np.nan
    t = item.truth
    if item.task == "conservatism":
        hi, lo = outs[0]["estimate"], outs[1]["estimate"]
        if not (_pos(hi) and _pos(lo)):
            return np.nan
        s = np.log(hi / lo) / t["log_gap"]
        return float((t["w"] - s) / t["w"])
    if item.task == "representativeness":
        up, down = outs[0]["estimate"], outs[1]["estimate"]
        if not (_pos(up) and _pos(down)):
            return np.nan
        return float(np.log(up / down) / (2 * t["span"]))
    o = outs[0]
    if not (_pos(o["low"]) and _pos(o["high"])) or o["high"] <= o["low"]:
        return np.nan
    return float(np.log(t["bayes_width"] / np.log(o["high"] / o["low"])))


# ---- design and run -----------------------------------------------------------------------------


def design(prims: list[str], levels: list[int]):
    top = max(levels)
    cells = [(at(p, k), p) for p in prims for k in levels]
    cells += [(at(p, top), q) for p in prims for q in prims if q != p]
    cells += [(RATIONAL, q) for q in prims]
    return cells


def measure(client: LLMClient, model: str, cells, items: dict, repeats: int) -> pd.DataFrame:
    calls, index = [], []
    for cond, task in cells:
        for item in items[task]:
            for rep in range(repeats):
                index.append((cond, task, item, rep, len(calls), len(item.prompts)))
                calls.extend(Call(model, cond.system, p, item.schema, nonce=f"v3b-rep{rep}") for p in item.prompts)
    print(f"[v3b] {len(calls)} calls on {model}")
    outs = client.run(calls)
    rows = []
    for cond, task, item, rep, start, nv in index:
        o = outs[start:start + nv]
        rows.append({"model": model, "thinking": "off", "condition": cond.name, "toggled": cond.toggled,
                     "level": cond.level, "rational": cond.rational, "task": task, "item": item.idx, "rep": rep,
                     "score": score(item, o), "answers": json.dumps(o)})
    return pd.DataFrame(rows)


def v3b_config():
    """The v3 config with the v3b primitive list: same thresholds, intensities, bootstrap draws."""
    cfg = load_config(CONFIG_PATH)
    s = settings()
    cfg.measurement.primitives = list(s["primitives"])
    return cfg


def hashes() -> dict[str, str]:
    h = registration_hashes()  # v3 config, every prompt, src/gate0, model ids
    h.pop("registration")
    for p in (V3B_CONFIG, Path(__file__).resolve()):
        h[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    h["registration"] = hashlib.sha256(json.dumps(h, sort_keys=True).encode()).hexdigest()
    return h


def check_lock() -> str:
    if not LOCK.exists():
        raise SystemExit("no registration_v3b.lock: run `register` first")
    locked = json.loads(LOCK.read_text())
    locked.pop("registered_at", None)
    now = hashes()
    drift = sorted(k for k in locked.keys() | now.keys() if locked.get(k) != now.get(k))
    if drift:
        raise SystemExit(f"registered objects changed since lock: {', '.join(drift)}")
    return locked["registration"]


def client(mode: str) -> LLMClient:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    m = load_config(CONFIG_PATH).models
    return LLMClient(DATA_DIR / "cache.sqlite", mode, m.effort, m.max_tokens, m.thinking, m.thinking_off)


def report(df: pd.DataFrame, res: dict, reg: str) -> str:
    v3 = json.loads((ROOT / "results/gate0-v3/measure/analysis.json").read_text())
    s = settings()
    lines = ["# Gate 0 v3b: second test format", "", f"Registration: `v3b` `{reg}`", "",
             "Decision rule: a primitive transfers if it passes all three v3 checks on format B (v3 thresholds).", "",
             "| primitive | format A (v3) | format B: monotonic (rho) | near-diagonal (max) | rational - neutral [CI], +/-margin "
             "| level means 0/1/2/3 | rational vs Bayes | format B verdict |", "|---|---|---|---|---|---|---|---|"]
    for p in s["primitives"]:
        r, c = res[p], res[p]["checks"]
        lm = " / ".join(f"{v:+.2f}" for v in r["level_means"].values())
        rn, rb = r["rational"], r["rational_vs_bayes"]
        lines.append(
            f"| {p} | {'PASS' if v3[p]['passed'] else 'FAIL'} | {'PASS' if c['monotonic'] else 'FAIL'} ({r['spearman']:.2f}) "
            f"| {'PASS' if c['near_diagonal'] else 'FAIL'} ({r['max_cross_loading']:.2f}) "
            f"| {'PASS' if c['rational_null'] else 'FAIL'} ({rn['mean']:+.2f} [{rn['lo']:+.2f}, {rn['hi']:+.2f}], +/-{rn['margin']:.2f}) "
            f"| {lm} | {rb['mean']:+.2f} [{rb['lo']:+.2f}, {rb['hi']:+.2f}] | **{'PASS' if r['passed'] else 'FAIL'}** |")
    lines += ["", "Cross-loadings on format B (|effect of the column persona at level 3 on the row score| / own effect, SD units):", "",
              "| persona at level 3 | " + " | ".join(s["primitives"]) + " |", "|---|" + "---|" * len(s["primitives"])]
    for p in s["primitives"]:
        lines.append(f"| {p} | " + " | ".join("own" if q == p else f"{res[p]['cross_loadings'][q]:.2f}" for q in s["primitives"]) + " |")
    valid = df.score.notna().mean()
    lines += ["", f"Scored cells: {df.score.notna().sum()} of {len(df)} ({valid:.1%}).", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["register", "smoke", "run", "report"])
    ap.add_argument("--mode", default="live", choices=["live", "batch", "offline"])
    a = ap.parse_args()
    s, cfg = settings(), v3b_config()
    model = cfg.models.candidates[0]
    if a.cmd == "register":
        h = hashes()
        LOCK.write_text(json.dumps({**h, "registered_at": time.time()}, indent=2, sort_keys=True) + "\n")
        print("registered", h["registration"])
        return
    if a.cmd == "smoke":
        items = {t: make_items(t, 3, s["item_seed"] + 1, s["shift"]) for t in s["primitives"]}
        cells = [(c, t) for t in s["primitives"] for c in (at(t, 0), at(t, 3), RATIONAL)]
        cl = client(a.mode)
        df = measure(cl, model, cells, items, 1)
        cl.close()
        print(df[["condition", "task", "item", "score", "answers"]].to_string())
        return
    reg = check_lock()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "run":
        items = {t: make_items(t, s["n_items"], s["item_seed"], s["shift"]) for t in s["primitives"]}
        cl = client(a.mode)
        df = measure(cl, model, design(s["primitives"], s["intensities"]), items, s["repeats"])
        st = cl.stats
        print(f"new {st.new_calls} hits {st.cache_hits} failures {st.failures} in/out {st.input_tokens:,}/{st.output_tokens:,}")
        cl.close()
        df.to_csv(OUT / "scores.csv", index=False)
    df = pd.read_csv(OUT / "scores.csv")
    res = analyze(df, cfg)
    (OUT / "analysis.json").write_text(json.dumps(res, indent=2, default=float))
    (OUT / "report.md").write_text(report(df, res, reg))
    print((OUT / "report.md").read_text())


if __name__ == "__main__":
    sys.exit(main())
