"""Dev harness: overlay candidate persona files and run a slice of the measurement grid on dev item seeds (never registered seeds)."""
import argparse, json, shutil, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
from dotenv import load_dotenv

import gate0.personas as P
from gate0.config import DATA_DIR, ROOT, load_config
from gate0.llm import LLMClient
from gate0.measure.checks import analyze
from gate0.measure.run import NEUTRAL, RATIONAL, at, base_design, run_measurement

ap = argparse.ArgumentParser()
ap.add_argument("--personas", default=None)
ap.add_argument("--model", default="claude-haiku-5-5")
ap.add_argument("--thinking", default="off")
ap.add_argument("--seed", type=int, default=20261132)  # iteration items; 20261133 = validation; v3 registered 20261201
ap.add_argument("--items", type=int, default=6)
ap.add_argument("--reps", type=int, default=1)
ap.add_argument("--grid", default="full")  # full | own | neutral | custom "cond:task,..."
ap.add_argument("--tasks", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--shift", type=float, default=0.12)
ap.add_argument("--record", type=int, default=3)
a = ap.parse_args()
load_dotenv(ROOT / ".env")

if a.personas:
    tmp = Path(tempfile.mkdtemp())
    for f in (ROOT / "prompts").glob("*.txt"):
        shutil.copy(f, tmp / f.name)
    for f in Path(a.personas).glob("*.txt"):
        shutil.copy(f, tmp / f.name)
    P.PROMPTS_DIR = tmp
    P.template.cache_clear(); P.bias_levels.cache_clear()

cfg = load_config()
m = cfg.measurement
m.n_items = a.items
m.shift = a.shift
cfg.market.track_record = a.record
m.item_seed = a.seed - 1  # run_measurement(dev=True) adds 1
tasks = a.tasks.split(",") if a.tasks else m.primitives
if a.grid == "full":
    cells = base_design(cfg)
elif a.grid == "own":
    cells = [(at(p, k), p) for p in tasks for k in m.intensities] + [(RATIONAL, p) for p in tasks]
elif a.grid == "neutral":
    cells = [(NEUTRAL, p) for p in tasks] + [(RATIONAL, p) for p in tasks]
else:
    cells = []
    for spec in a.grid.split(","):
        c, t = spec.split(":")
        cond = NEUTRAL if c == "neutral" else RATIONAL if c == "rational" else at(c.split("@")[0], int(c.split("@")[1]))
        cells.append((cond, t))
cells = [(c, t) for c, t in cells if t in tasks or a.grid == "full"]

client = LLMClient(DATA_DIR / "cache.sqlite", "live", cfg.models.effort, cfg.models.max_tokens, a.thinking, cfg.models.thinking_off)
raw = []
orig = client.run
def run(calls):
    outs = orig(calls)
    for c, o in zip(calls, outs):
        mt = c.meta
        it = mt["item"]
        v = o[it.truth["symbol"]]["value"] if o else None
        raw.append({"condition": mt["condition"].name, "task": it.task, "item": it.idx, "rep": mt["rep"],
                    "variant": mt["variant"], "value": v, "conf": o[it.truth["symbol"]]["confidence"] if o else None})
    return outs
client.run = run
df = run_measurement(cfg, client, a.model, cells, a.reps, True, a.thinking)
s = client.stats
print(f"new {s.new_calls} hits {s.cache_hits} fail {s.failures} in/out {s.input_tokens:,}/{s.output_tokens:,}")
client.close()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "scores.csv", index=False)
pd.DataFrame(raw).to_csv(out / "raw.csv", index=False)
print(df.groupby(["task", "condition"])["score"].agg(["mean", "std", "count"]).round(3).to_string())
if a.grid == "full":
    res = analyze(df, cfg)
    (out / "analysis.json").write_text(json.dumps(res, indent=2, default=float))
    for p, r in res.items():
        print(f"{p:20s} {'PASS' if r['passed'] else 'FAIL'} {r['checks']} rho={r['spearman']:.2f} "
              f"cross={r['max_cross_loading']:.2f} null={r['rational']['mean']:+.2f}[{r['rational']['lo']:+.2f},{r['rational']['hi']:+.2f}]±{r['rational']['margin']:.2f} "
              f"levels={ {k: round(v, 2) for k, v in r['level_means'].items()} }")
