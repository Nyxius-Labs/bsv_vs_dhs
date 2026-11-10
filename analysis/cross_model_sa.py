"""Self-attribution control check (rational - neutral) on fresh dev items (seed 20261135), by model.

Outputs of analysis/hmodel.py, in results/gate0-v3-dev/cross_model/. Margin = 0.25 x (level 3 - neutral),
as registered. Dev items only: exploratory.
"""
from pathlib import Path
import numpy as np, pandas as pd

rng = np.random.default_rng(0)
for d in sorted(Path("results/gate0-v3-dev/cross_model").iterdir()):
    s = pd.read_csv(d / "scores.csv").groupby(["condition", "item"])["score"].mean().unstack(0)
    diff = (s["rational"] - s["neutral"]).dropna().to_numpy()
    margin = 0.25 * (s["self_attribution@3"] - s["neutral"]).mean()
    bs = diff[rng.integers(0, len(diff), (5000, len(diff)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    ok = -margin <= lo and hi <= margin
    print(f"{d.name:28s} rational - neutral {diff.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] vs +/-{margin:.2f}  {'PASS' if ok else 'FAIL'}")
