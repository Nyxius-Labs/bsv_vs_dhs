"""Figure 1 of the extended abstract: registered v3 scores by prompted level, with the rational persona.
Run: uv run --with matplotlib python abstract/make_figure.py"""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "data/registered_v3/measure/claude-sonnet-5-5/thinking-off"
s = pd.read_csv(f"{D}/scores.csv"); a = json.load(open(f"{D}/analysis.json"))
names = {"conservatism": "Conservatism (BSV)", "representativeness": "Representativeness (BSV)",
         "overconfidence": "Overconfidence (DHS)", "self_attribution": "Self-attribution (DHS)"}
rng = np.random.default_rng(0)
def ci(x):
    b = x[rng.integers(0, len(x), (5000, len(x)))].mean(1)
    return x.mean(), *np.quantile(b, [0.025, 0.975])

plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.95))
ink, grey = "#1f3a5f", "#9aa5b1"
for ax, (p, title) in zip(axes, names.items()):
    t = s[s.task == p]; im = t.groupby(["condition", "item"]).score.mean()
    conds = ["neutral"] + [f"{p}@{k}" for k in (1, 2, 3)]
    m = np.array([ci(im[c].to_numpy()) for c in conds])
    rn = a[p]["rational"]; margin = rn["margin"]  # paired (rational - neutral) CI, as the check uses it
    r = (m[0, 0] + rn["mean"], m[0, 0] + rn["lo"], m[0, 0] + rn["hi"])
    ax.axhspan(m[0, 0] - margin, m[0, 0] + margin, xmin=0, xmax=1, color=grey, alpha=0.18, lw=0)
    ax.errorbar(range(4), m[:, 0], yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], color=ink, lw=1.6,
                marker="o", ms=3.5, capsize=0, elinewidth=1)
    ax.errorbar([4.4], [r[0]], yerr=[[r[0] - r[1]], [r[2] - r[0]]], color=ink, marker="o", ms=4,
                mfc="white", capsize=0, elinewidth=1, lw=0)
    ax.set_xticks([0, 1, 2, 3, 4.4], ["0", "1", "2", "3", "R"])
    ax.set_xlim(-0.4, 4.8); ax.set_title(title + ("" if a[p]["passed"] else "\n[rational check fails]"), fontsize=7.5)
    ax.axhline(0, color=grey, lw=0.6, zorder=0)
    ax.tick_params(length=2)
axes[0].set_ylabel("bias score (0 = Bayesian weights)")
fig.supxlabel("prompted level (0 = neutral persona, R = rational persona)", fontsize=7.5, y=0.0)
fig.tight_layout(pad=0.3, w_pad=0.8)
fig.savefig("abstract/fig_levels.pdf"); fig.savefig("abstract/fig_levels.png", dpi=200)
for p in names:
    print(p, a[p]["passed"], {k: round(v, 3) for k, v in a[p]["rational"].items()})
