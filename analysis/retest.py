"""Test-retest: correlation between repeats of the same prompt at the same persona level (registered v3).

Level means are removed first, so the correlation reflects agreement within a level, not the dose response.
"""
import itertools
import numpy as np, pandas as pd

s = pd.read_csv("results/gate0-v3/measure/scores.csv")
pr = lambda d: np.mean([np.corrcoef(d[a], d[b])[0, 1] for a, b in itertools.combinations(d.columns, 2)])  # noqa: E731
for t in s.task.unique():
    g = s[(s.task == t) & s.condition.isin(["neutral", f"{t}@1", f"{t}@2", f"{t}@3"])]
    w = g.pivot_table(index=["condition", "item"], columns="rep", values="score").dropna()
    rw = pr(w.sub(w.mean(axis=1).groupby(level=0).transform("mean"), axis=0))
    print(f"{t:20s} cells {len(w):3d}  within-level r {rw:.2f}  reliability of 3-repeat mean {3 * rw / (1 + 2 * rw):.2f}")
