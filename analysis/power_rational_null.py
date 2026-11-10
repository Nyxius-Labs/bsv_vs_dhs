"""Power of the registered control check (rational - neutral within +/- margin) for self-attribution.

Resamples the registered v3 items: the paired differences are re-centred on a true gap (0, or the observed
gap), and the check is rerun on each resample. Not registered; reported as exploratory.
"""
import numpy as np, pandas as pd

from gate0.measure.checks import _paired, item_means

D = "results/gate0-v3/measure"
rng = np.random.default_rng(0)


def ci(x, B=1000):
    b = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return np.quantile(b, [0.025, 0.975])


im = item_means(pd.read_csv(f"{D}/scores.csv"))
margin = pd.read_json(f"{D}/analysis.json")["self_attribution"]["rational"]["margin"]
d = np.array(_paired(im, "rational", "neutral", "self_attribution"))
print(f"items {len(d)}, observed gap {d.mean():+.3f}, margin +/-{margin:.3f}")
for gap in (0.0, d.mean()):
    x0, ok, S = d - d.mean() + gap, 0, 1000
    for _ in range(S):
        lo, hi = ci(x0[rng.integers(0, len(d), len(d))])
        ok += -margin <= lo and hi <= margin
    print(f"true gap {gap:+.3f}: check passes in {ok / S:.0%} of resamples")
