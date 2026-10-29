import numpy as np
import pytest

from gate0.config import load_config
from gate0.market.agents import seat_agents
from gate0.market.simulate import MarketRun
from gate0.measure.checks import analyze
from gate0.measure.estimate import beta_gamma, score
from gate0.measure.items import make_items
from gate0.measure.run import base_design, run_measurement
from gate0.measure.synthetic import SyntheticClient, answer, clean_params

CFG = load_config()
TASKS = CFG.measurement.primitives


def _score(item, b, rng, noise=0.0):
    return score(item, [answer(item, v, b, rng, noise) for v in range(len(item.variants))])


@pytest.mark.parametrize("task", TASKS)
def test_rational_answers_score_zero(task):
    rng = np.random.default_rng(0)
    for item in make_items(CFG, task, 10, 1):
        assert _score(item, {}, rng) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("task", TASKS)
def test_estimator_recovers_true_intensity_and_nothing_else(task):
    rng = np.random.default_rng(0)
    items = make_items(CFG, task, 10, 1)
    for truth in (0.25, 0.5, 0.75):
        assert np.mean([_score(i, {task: truth}, rng) for i in items]) == pytest.approx(truth, rel=1e-9)
        for other in TASKS:
            if other != task:
                assert np.mean([_score(i, {other: truth}, rng) for i in items]) == pytest.approx(0.0, abs=1e-9)


def test_self_attribution_cancels_level_shift():
    """A uniform change in the private-estimate weight (overconfidence) is not self-attribution."""
    rng = np.random.default_rng(0)
    for item in make_items(CFG, "self_attribution", 10, 1):
        assert _score(item, {"overconfidence": 1.0}, rng) == pytest.approx(0.0, abs=1e-9)
        outs = [answer(item, v, {"self_attribution": 0.4}, rng, 0.0) for v in range(len(item.variants))]
        beta, gamma = beta_gamma(item, outs)
        assert beta == pytest.approx(0.4) and gamma == pytest.approx(0.0, abs=1e-9)


def test_variants_differ_in_one_input_only():
    for task in TASKS:
        for item in make_items(CFG, task, 5, 1):
            base = item.variants[0]["obs"]
            for v in item.variants[1:]:
                changed = {k for k in base if v["obs"][k] != base[k]}
                assert len(changed) <= 2  # representativeness swaps two reports; others move one


def test_prompts_are_unique_and_deterministic():
    for task in TASKS:
        a, b = make_items(CFG, task, 20, 7), make_items(CFG, task, 20, 7)
        prompts = [p for i in a for p in i.prompts]
        assert len(prompts) == len(set(prompts))
        assert prompts == [p for i in b for p in i.prompts]


def test_registered_and_dev_items_do_not_overlap():
    m = CFG.measurement
    for task in TASKS:
        reg = {p for i in make_items(CFG, task, m.n_items, m.item_seed) for p in i.prompts}
        dev = {p for i in make_items(CFG, task, m.n_items, m.item_seed + 1) for p in i.prompts}
        assert not reg & dev


def test_vignette_uses_the_market_asset_text():
    """A vignette is a one-stock market prompt: same template and same per-asset line formats."""
    import re
    from gate0.market.agents import round_prompt
    item = make_items(CFG, "self_attribution", 1, 1)[0]
    run = MarketRun(CFG, "placebo", 0, False, seat_agents(CFG, "placebo", 0, None))
    run.prices[:] = run.prices[0]  # an unsimulated run has no prices after round 0
    market = round_prompt(CFG.market, run.world, 13, 0, run.names, False, run.prices, run.holdings[0], 1000.0,
                          run.prices[0], np.full(CFG.market.n_assets, 5.0))
    shape = lambda text: re.sub(r"[0-9.]+", "#", text)  # noqa: E731
    vignette_lines = shape(item.render(0)).splitlines()
    market_lines = shape(market).splitlines()
    asset = [ln for ln in vignette_lines if ln.startswith("  ")]
    assert asset == [ln for ln in market_lines if ln.startswith("  ")][: len(asset)]
    names = {shape(n) for _, n in run.names} | {shape(item.render(0)).split("For each stock:")[1].split()[0]}
    frame = lambda lines: [ln for ln in lines if ln and not ln.startswith("  ")  # noqa: E731
                           and not any(ln.startswith(n) for n in names)]
    assert frame(vignette_lines) == frame(market_lines)


def _measure(params, seed=3, noise=0.005):
    return run_measurement(CFG, SyntheticClient(params, seed=seed, noise=noise), "synthetic", base_design(CFG),
                           CFG.measurement.repeats)


def test_clean_agents_pass_all_checks():
    res = analyze(_measure(clean_params()), CFG)
    for p, r in res.items():
        assert r["passed"], (p, r["checks"], r["max_cross_loading"])


def test_rational_bias_fails_null_check():
    base = clean_params()
    res = analyze(_measure(lambda c: {**base(c), "conservatism": 0.4} if c.rational else base(c)), CFG)
    assert not res["conservatism"]["checks"]["rational_null"]


def test_entangled_prompt_fails_cross_loading():
    base = clean_params()

    def leaky(cond):
        b = base(cond)
        if "overconfidence" in cond.intensities:
            b["conservatism"] = b["overconfidence"]  # the overconfidence prompt also induces conservatism
        return b

    assert not analyze(_measure(leaky), CFG)["overconfidence"]["checks"]["near_diagonal"]


def test_flat_response_rarely_passes_monotonicity():
    """A null agent passes monotonicity at about the CI's false-positive rate, and never all checks."""
    passes = full = trials = 0
    for seed in range(25):
        res = analyze(_measure(lambda c: {}, seed=seed), CFG)
        passes += sum(r["checks"]["monotonic"] for r in res.values())
        full += sum(r["passed"] for r in res.values())
        trials += len(res)
    assert passes / trials < 0.06 and full == 0


def test_missing_cross_cell_fails_near_diagonal():
    df = _measure(clean_params())
    df.loc[(df.condition == "conservatism@3") & (df.task == "self_attribution"), "score"] = None
    assert not analyze(df, CFG)["conservatism"]["checks"]["near_diagonal"]


def test_whole_dollar_rational_agent_passes_null():
    """LLM agents state values in whole dollars; rounding noise must not fail the rational null."""
    class Rounding(SyntheticClient):
        def run(self, calls):
            return [{s: {**x, "value": round(x["value"])} for s, x in o.items()} for o in super().run(calls)]

    df = run_measurement(CFG, Rounding(clean_params(), seed=3, noise=0.0), "synthetic", base_design(CFG), 3)
    res = analyze(df, CFG)
    assert all(r["checks"]["rational_null"] for r in res.values()), {p: r["rational"] for p, r in res.items()}
