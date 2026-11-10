"""Exploratory: BSV - neutral and BSV - rational on the registered gate moments, on the freeze seeds (100-104).

The registered freeze test is the analyst-optimism anomaly (see report.md); these contrasts were not registered.
"""
import json
import numpy as np

m = json.load(open("results/gate0-v3/freeze/moments.json"))
d = {(r["arm"], r["seed"]): r for r in m if not r.get("masked")}
seeds = sorted({s for _, s in d})
rng = np.random.default_rng(0)
for mom in ("pead_level", "momentum_h"):
    for base in ("placebo", "rational"):
        x = np.array([d[("bsv", s)][mom] - d[(base, s)][mom] for s in seeds])
        bs = x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"{mom:11s} bsv - {base:8s} {x.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] (n={len(x)})")
