import numpy as np
import pytest

from gate0.config import load_config
from gate0.measure.checks import analyze
from gate0.measure.estimate import score
from gate0.measure.items import make_items
from gate0.measure.run import base_design, run_measurement
from gate0.measure.synthetic import SyntheticClient, answer, clean_params

CFG = load_config()
TASKS = CFG.measurement.primitives


@pytest.mark.parametrize("task", TASKS)
def test_rational_answers_score_zero(task):
    rng = np.random.default_rng(0)
    for item in make_items(task, 10, 1):
        outs = [answer(item, v, {}, rng, noise=0.0) for v in range(len(item.variants))]
        assert score(item, outs) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("task", TASKS)
def test_estimator_recovers_true_intensity(task):
    rng = np.random.default_rng(0)
    for truth in (0.25, 0.5, 0.75):
        scores = [
            score(item, [answer(item, v, {task: truth}, rng, noise=0.0) for v in range(len(item.variants))])
            for item in make_items(task, 10, 1)
        ]
        assert np.mean(scores) == pytest.approx(truth, rel=1e-6)


def test_self_attribution_cancels_level_shift():
    rng = np.random.default_rng(0)
    for item in make_items("self_attribution", 10, 1):
        outs = [answer(item, v, {"overconfidence": 1.0}, rng, noise=0.0) for v in range(3)]
        assert score(item, outs) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("seed", [20261029, 20261030, 1, 2])
def test_representativeness_ignores_shrinkage(seed):
    """Conservatism and evidence-ignoring answers must not read as representativeness."""
    rng = np.random.default_rng(0)
    for item in make_items("representativeness", 10, seed):
        for c in (0.25, 0.5, 0.75):
            assert score(item, [answer(item, 0, {"conservatism": c}, rng, noise=0.0)]) == pytest.approx(0, abs=1e-9)
        assert abs(score(item, [{"X": 33.3, "Y": 33.3, "Z": 33.3}])) < 1e-9


def test_tiny_percentages_are_not_reinterpreted():
    item = make_items("representativeness", 1, 1)[0]
    tgt = item.truth["target"]
    a = score(item, [{**{k: 40 for k in "XYZ"}, tgt: 0.5}])
    b = score(item, [{**{k: 40 for k in "XYZ"}, tgt: 1.0}])
    assert a <= b  # less weight on the target never scores as more representativeness


def test_items_are_deterministic():
    for task in TASKS:
        a, b = make_items(task, 10, 7), make_items(task, 10, 7)
        assert [i.render(0) for i in a] == [i.render(0) for i in b]


def _measure(params):
    cells = base_design(CFG)
    return run_measurement(CFG, SyntheticClient(params, seed=3), "synthetic", cells, CFG.measurement.repeats)


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

    res = analyze(_measure(leaky), CFG)
    assert not res["overconfidence"]["checks"]["near_diagonal"]


def test_flat_response_rarely_passes_monotonicity():
    """A null agent passes monotonicity at about the CI's false-positive rate, and never all checks."""
    passes = full = trials = 0
    for seed in range(40):
        df = run_measurement(CFG, SyntheticClient(lambda c: {}, seed=seed), "synthetic", base_design(CFG), 3)
        res = analyze(df, CFG)
        passes += sum(r["checks"]["monotonic"] for r in res.values())
        full += sum(r["passed"] for r in res.values())
        trials += len(res)
    assert passes / trials < 0.06 and full == 0


def test_missing_cross_cell_fails_near_diagonal():
    df = _measure(clean_params())
    df.loc[(df.condition == "conservatism@3") & (df.task == "self_attribution"), "score"] = None
    assert not analyze(df, CFG)["conservatism"]["checks"]["near_diagonal"]


def test_rounded_rational_agent_passes_null():
    class Rounding(SyntheticClient):
        def run(self, calls):
            return [{k: round(v) for k, v in o.items()} for o in super().run(calls)]

    df = run_measurement(CFG, Rounding(clean_params(), seed=3, noise=0.0), "synthetic", base_design(CFG), 3)
    res = analyze(df, CFG)
    assert all(r["checks"]["rational_null"] for p, r in res.items() if p != "overconfidence"), \
        {p: r["rational"] for p, r in res.items()}
