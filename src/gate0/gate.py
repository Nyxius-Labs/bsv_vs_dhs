"""Gate logic and the Markdown report."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gate0.config import Config
from gate0.moments import PROFILES, SCALARS, paired_contrast, profile_summary, turning_point_contrast
from gate0.stats import bootstrap_ci


def _per_seed(moments: list[dict], arm: str, key: str) -> dict[int, float]:
    return {m["seed"]: m[key] for m in moments
            if m["arm"] == arm and not m["masked"] and m[key] is not None and np.isfinite(m[key])}


def placebo_share(moments: list[dict], theory: str, key: str) -> tuple[float, float, int]:
    """(theory - rational, placebo - rational) means over seeds present in all three arms."""
    t, p, r = (_per_seed(moments, a, key) for a in (theory, "placebo", "rational"))
    seeds = sorted(t.keys() & p.keys() & r.keys())
    if not seeds:
        return np.nan, np.nan, 0
    return (float(np.mean([t[s] - r[s] for s in seeds])), float(np.mean([p[s] - r[s] for s in seeds])), len(seeds))


def moment_verdict(cfg: Config, moments: list[dict], theory: str, key: str) -> dict:
    """One gate moment: the effect beats placebo (CI rule) and the placebo rule holds."""
    effect = paired_contrast(moments, theory, "placebo", key, cfg.moments)
    effect_ok = bool(np.isfinite(effect["lo"]) and effect["lo"] > 0)  # cfg.gate.ci_rule == "lower_above_zero"
    gain, placebo_gain, n = placebo_share(moments, theory, key)
    # Fails unless the theory beats rational and placebo carries less than the registered share of that.
    placebo_ok = bool(n and gain > 0 and placebo_gain < cfg.gate.placebo_share_max * gain)
    return {"effect_vs_placebo": effect, "effect_passed": effect_ok, "gain_over_rational": gain,
            "placebo_gain_over_rational": placebo_gain, "placebo_passed": placebo_ok,
            "passed": effect_ok and placebo_ok}


def verdicts(cfg: Config, analysis: dict, moments: list[dict]) -> dict:
    mc, g = cfg.moments, cfg.gate
    out = {"theories": {}, "notes": []}
    passed = {p for p, r in analysis.items() if r["passed"]}
    if not passed:
        out["notes"].append("No primitive passed measurement: prompted biases behave as instructions, not measurements.")

    for theory, prims in cfg.theories.items():
        prim_ok = all(p in passed for p in prims)
        by_moment = {k: moment_verdict(cfg, moments, theory, k) for k in g.moments_for(theory)}
        out["theories"][theory] = {
            "primitives_passed": prim_ok, "moments": by_moment,
            "advances": prim_ok and any(v["passed"] for v in by_moment.values()),
        }
        for k, v in by_moment.items():
            if v["effect_passed"] and not v["placebo_passed"]:
                out["notes"].append(
                    f"{theory.upper()} on {k}: placebo − rational = {v['placebo_gain_over_rational']:+.3f} vs "
                    f"{theory} − rational = {v['gain_over_rational']:+.3f}; the placebo reproduces most of the effect.")
    out["full_study"] = all(t["advances"] for t in out["theories"].values())

    # Manipulation check (reported, not gating): does each arm produce momentum at all?
    out["manipulation"] = {}
    for arm in cfg.market.arms:
        vals = list(_per_seed(moments, arm, g.manipulation_moment).values())
        mean, lo, hi = bootstrap_ci(vals, mc.bootstrap_draws, mc.ci_level)
        out["manipulation"][arm] = {"mean": mean, "lo": lo, "hi": hi, "n_seeds": len(vals),
                                    "passed": bool(np.isfinite(lo) and lo > 0)}

    # Masking: recall (masked - unmasked, per masked arm) and induced belief under masking (arm - placebo, both masked).
    out["masking"] = {}
    for key in g.moments_for("joint"):
        rows = {}
        for arm in cfg.market.masked_arms:
            rows[f"{arm}: masked − unmasked"] = paired_contrast(moments, arm, arm, key, mc, masked=(True, False))
            if arm != "placebo" and "placebo" in cfg.market.masked_arms:
                rows[f"{arm} − placebo, both masked"] = paired_contrast(moments, arm, "placebo", key, mc, masked=(True, True))
        out["masking"][key] = rows
    return out


def leading_arm(cfg: Config, analysis: dict | None, moments: list[dict]) -> str:
    """The theory-bearing arm with the largest standardized effect over placebo (mean / CI width, best gate
    moment), among arms that advanced (all theory arms if none did; the freeze step is only a feasibility test)."""
    candidates = [a for a in ("bsv", "dhs", "joint") if any(m["arm"] == a for m in moments)]
    if analysis is not None:
        v = verdicts(cfg, analysis, moments)["theories"]
        advanced = [a for a in candidates if v.get(a, {}).get("advances")]
        candidates = advanced or candidates

    def score(arm: str) -> float:
        zs = []
        for key in cfg.gate.moments_for(arm):
            c = paired_contrast(moments, arm, "placebo", key, cfg.moments)
            if np.isfinite(c["mean"]):
                zs.append(c["mean"] / max(c["hi"] - c["lo"], 1e-12))
        return max(zs) if zs else np.nan

    scored = {a: score(a) for a in candidates}
    scored = {a: v for a, v in scored.items() if np.isfinite(v)}
    if not scored:
        raise RuntimeError("no theory arm has market results")
    return max(scored, key=scored.get)


# ---- report ----------------------------------------------------------------------------------


def _ci(c: dict) -> str:
    if not c["n_seeds"]:
        return "n/a"
    return f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] (n={c['n_seeds']})"


def _tp(point: float, lo: float, hi: float, share: float, fmt: str) -> str:
    """Turning point with its CI over the bootstrap draws in which one exists."""
    head = fmt.format(point) if np.isfinite(point) else "none"
    ci = f"[{fmt.format(lo)}, {fmt.format(hi)}]" if np.isfinite(lo) else "[–]"
    return f"{head} {ci}, found in {share:.0%} of draws"


def _yn(b: bool) -> str:
    return "PASS" if b else "FAIL"


def _seating(moments: list[dict]):
    """(arm, primitive, prompted levels, recovered draws) pooled over agents and unmasked seeds."""
    pooled: dict[tuple[str, str], tuple[list, list]] = {}
    for m in moments:
        if m["masked"]:
            continue
        for a in m.get("agents", []):
            for p, level in a["intensities"].items():
                levels, rec = pooled.setdefault((m["arm"], p), ([], []))
                levels.append(level)
                rec.append(a.get("recovered", a["measured"])[p])
    return [(arm, p, levels, rec) for (arm, p), (levels, rec) in sorted(pooled.items())]


def market_model(cfg: Config, analyses: dict[str, dict]) -> str:
    """The registered market-model rule, over analyses measured at the registered thinking setting:
    most theories with every primitive passing, then most primitives passing, then candidate order."""
    order = cfg.models.candidates

    def key(model: str) -> tuple:
        passed = {p for p, r in analyses[model].items() if r["passed"]}
        theories = sum(all(p in passed for p in prims) for prims in cfg.theories.values())
        return theories, len(passed), -order.index(model) if model in order else -len(order)

    return max(analyses, key=key)


def _measurement_table(analysis: dict) -> list[str]:
    L = ["| primitive | monotonic (ρ) | near-diagonal (max) | rational null (mean [CI], ±margin) | own effect (SD) "
         "| level means | verdict |", "|---|---|---|---|---|---|---|"]
    for p, r in analysis.items():
        c, rn = r["checks"], r["rational"]
        means = ", ".join(f"{k}: {v:+.2f}" for k, v in r["level_means"].items())
        L.append(f"| {p} | {_yn(c['monotonic'])} ({r['spearman']:.2f}) | {_yn(c['near_diagonal'])} "
                 f"({r['max_cross_loading']:.2f}) | {_yn(c['rational_null'])} ({rn['mean']:+.2f} "
                 f"[{rn['lo']:+.2f}, {rn['hi']:+.2f}], ±{rn['margin']:.2f}) | {r['own_effect_sd']:+.2f} | {means} "
                 f"| **{_yn(r['passed'])}** |")
    sa = analysis.get("self_attribution", {}).get("beta_gamma_by_level")
    if sa:
        L += ["", "Self-attribution β (rise in the private-estimate weight after a confirmation) and γ (fall after a "
              "contradiction), in units of its rational weight, by prompted level: "
              + "; ".join(f"{k}: β={v['beta']:+.2f}, γ={v['gamma']:+.2f}" for k, v in sa.items())]
    return L


def report(cfg: Config, registration: str, analyses: dict[tuple[str, str], dict], chosen: str | None,
           moments: list[dict] | None, freeze: dict | None, usage: dict) -> str:
    """`analyses` maps (model, thinking) to a measurement analysis; `chosen` is the market model."""
    th = cfg.measurement.thresholds
    L = ["# Gate 0 report", "", f"Registration: `{cfg.registration}` `{registration}`", ""]

    L += ["## Measurement stage", "",
          f"Checks: Spearman ≥ {th.monotonic_spearman} plus a paired top-vs-neutral CI above 0; max cross-loading "
          f"< {th.max_cross_loading}; rational null: the rational arm's {1 - th.rational_null_alpha:.0%} CI lies within "
          f"±{th.rational_null_margin} × the primitive's own (top − neutral) effect.", "",
          "Verdicts by model and thinking setting (the registered setting is "
          f"`{cfg.models.thinking}`; `{cfg.models.control_thinking}` is a descriptive control and never gates):", "",
          "| primitive | " + " | ".join(f"{m} / {t}" for m, t in analyses) + " |", "|---" * (len(analyses) + 1) + "|"]
    for p in cfg.measurement.primitives:
        L.append(f"| {p} | " + " | ".join(_yn(a[p]["passed"]) if p in a else "n/a" for a in analyses.values()) + " |")
    if chosen:
        L += ["", f"Market model by the registered rule (`{cfg.models.market_model_rule}`): **{chosen}**, thinking "
              f"`{cfg.models.thinking}`."]
    for (model, thinking), a in analyses.items():
        L += ["", f"### {model}, thinking {thinking}", ""] + _measurement_table(a)

    # Gate verdicts use the market model's measurement at the registered thinking setting.
    analysis = analyses.get((chosen, cfg.models.thinking)) or next(iter(analyses.values()), {})
    if moments:
        v = verdicts(cfg, analysis, moments)
        df = pd.DataFrame(moments)
        theory_arms = [a for a in cfg.market.arms if a != "placebo"]
        contrast_keys = list(dict.fromkeys([*cfg.gate.moments_for("joint"), cfg.gate.manipulation_moment]))
        L += ["", "## Identification stage", "",
              "Per-arm means over seeds (unmasked runs).", "",
              df[~df.masked].groupby("arm")[SCALARS].mean().round(3).to_markdown(), "",
              "Paired contrasts vs placebo: mean [seed-bootstrap CI]. Reported as direction and ordering; no winner is declared.", "",
              "| arm | " + " | ".join(contrast_keys) + " |", "|---" * (len(contrast_keys) + 1) + "|"]
        for arm in theory_arms:
            L.append(f"| {arm} | " + " | ".join(_ci(paired_contrast(moments, arm, "placebo", k, cfg.moments))
                                                for k in contrast_keys) + " |")

        L += ["", f"Momentum manipulation check ({cfg.gate.manipulation_moment}: profile slope at horizon "
              f"{cfg.moments.interaction_horizon}, level CI > 0). Reported, not gating.", "",
              "| arm | mean [CI] | momentum present |", "|---|---|---|"]
        for arm, r in v["manipulation"].items():
            L.append(f"| {arm} | {_ci(r)} | {_yn(r['passed'])} |")

        L += ["", "Horizon profiles: turning point = horizon where the seed-mean profile first changes sign "
              f"(interpolated), 'none' if it does not turn within {cfg.moments.max_horizon} rounds. CI: seed bootstrap "
              "over the whole curve, among the draws in which a sign change occurs.", "",
              "| arm | " + " | ".join(PROFILES) + " |", "|---" * (len(PROFILES) + 1) + "|"]
        for arm in cfg.market.arms:
            cells = []
            for key in PROFILES:
                ps = profile_summary(moments, arm, key, cfg.moments)
                cells.append("n/a" if not ps["n_seeds"] else
                             _tp(ps["turning_point"], ps["tp_lo"], ps["tp_hi"], ps["tp_found_share"], "{:.1f}"))
            L.append(f"| {arm} | " + " | ".join(cells) + " |")
        L += ["", "Turning point, arm − placebo (rounds):", "",
              "| arm | " + " | ".join(PROFILES) + " |", "|---" * (len(PROFILES) + 1) + "|"]
        for arm in theory_arms:
            cells = []
            for key in PROFILES:
                c = turning_point_contrast(moments, arm, "placebo", key, cfg.moments)
                cells.append(_tp(c["mean"], c["lo"], c["hi"], c["found_share"], "{:+.1f}") if c["n_seeds"] else "n/a")
            L.append(f"| {arm} | " + " | ".join(cells) + " |")
        L += ["", "Mean profile (all assets), slope by horizon:", "",
              "| arm | " + " | ".join(f"h{h}" for h in range(1, cfg.moments.max_horizon + 1)) + " |",
              "|---" * (cfg.moments.max_horizon + 1) + "|"]
        for arm in cfg.market.arms:
            ps = profile_summary(moments, arm, "profile_all", cfg.moments)
            if ps["n_seeds"]:
                L.append(f"| {arm} | " + " | ".join(f"{x:+.2f}" for x in ps["mean"]) + " |")

        L += ["", "Seating: agents at discrete prompted levels whose shares match the measured distribution "
              "(each agent's recovered draw binned to the nearest level mean). Realized recovered intensity, "
              "unmasked runs:", "", "| arm | primitive | level counts (1/2/3) | recovered mean ± SD [min, max] |",
              "|---|---|---|---|"]
        for arm, prim, levels, rec in _seating(moments):
            counts = "/".join(str(levels.count(k)) for k in (1, 2, 3))
            L.append(f"| {arm} | {prim} | {counts} | {np.mean(rec):+.2f} ± {np.std(rec):.2f} "
                     f"[{np.min(rec):+.2f}, {np.max(rec):+.2f}] |")

        L += ["", "## Masking", ""]
        for key, rows in v["masking"].items():
            L.append(f"- {key}: " + "; ".join(f"{name} {_ci(c)}" for name, c in rows.items()))

        L += ["", "## Gate", "",
              "A theory advances if both its primitives pass measurement and, on at least one of its own moments ("
              + "; ".join(f"{t.upper()}: {', '.join(ks)}" for t, ks in cfg.gate.gate_moments.items())
              + "), its effect beats placebo (CI lower bound > 0) and the placebo rule holds: (placebo − rational) < "
              f"{cfg.gate.placebo_share_max} × (theory − rational) with theory − rational > 0, on common seeds. "
              "A BSV pass on momentum shows the induced bias produces momentum, not that BSV beats DHS.", "",
              "| theory | both primitives pass | moment | effect > placebo | theory − rational / placebo − rational "
              "| placebo rule | moment passes | advances |", "|---|---|---|---|---|---|---|---|"]
        for t, g in v["theories"].items():
            for i, (k, mv) in enumerate(g["moments"].items()):
                head = (f"{t.upper()} | {_yn(g['primitives_passed'])}" if i == 0 else " | ")
                tail = f"**{'YES' if g['advances'] else 'NO'}**" if i == 0 else ""
                L.append(f"| {head} | {k} | {_yn(mv['effect_passed'])} {_ci(mv['effect_vs_placebo'])} "
                         f"| {mv['gain_over_rational']:+.3f} / {mv['placebo_gain_over_rational']:+.3f} "
                         f"| {_yn(mv['placebo_passed'])} | {_yn(mv['passed'])} | {tail} |")
        L += ["", f"**Full study proceeds: {'YES' if v['full_study'] else 'NO'}**", ""]
        L += [f"- {n}" for n in v["notes"]]

    if freeze:
        L += ["", "## Freeze-and-evaluate (feasibility only)", "",
              f"Frozen configuration: `{freeze['arm']}` (hash `{freeze['hash'][:16]}`), held-out anomaly: {cfg.freeze.anomaly}.", ""]
        for base_arm in cfg.freeze.arms:
            c = paired_contrast(freeze["moments"], freeze["arm"], base_arm, "optimism_anomaly", cfg.moments)
            L.append(f"- {freeze['arm']} − {base_arm}: {_ci(c)}")

    L += ["", "## Usage", "", "API calls billed under this registration (since `gate0 register`; dev runs: whole cache).",
          "", "| model | calls | input tokens | output tokens |", "|---|---|---|---|"]
    for model, (n, i, o) in sorted(usage.items()):
        L.append(f"| {model} | {n:,} | {i:,} | {o:,} |")
    return "\n".join(L) + "\n"
