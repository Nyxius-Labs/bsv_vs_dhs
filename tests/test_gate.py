import numpy as np

from gate0.config import load_config
from gate0.gate import leading_arm, verdicts

CFG = load_config()
ALL_PASS = {p: {"passed": True} for p in CFG.measurement.primitives}


MANIP = CFG.gate.manipulation_moment
KEYS = list(dict.fromkeys([*CFG.gate.moments_for("joint"), MANIP]))


def _moments(effects: dict[str, float], noise=0.01, seeds=range(10), keys=None):
    """`effects` on `keys` (default: each theory's first gate moment); other moments flat (manipulation at 0.1)."""
    rng = np.random.default_rng(0)
    keys = keys or [ks[0] for ks in CFG.gate.gate_moments.values()]
    flat = lambda k: 0.1 if k == MANIP else 0.0  # noqa: E731
    return [{"arm": a, "seed": s, "masked": False,
             **{k: (v if k in keys else flat(k)) + rng.normal(0, noise) for k in KEYS}}
            for s in seeds for a, v in effects.items()]


def test_theory_advances_when_effect_beats_placebo():
    v = verdicts(CFG, ALL_PASS, _moments({"bsv": 0.3, "dhs": 0.3, "joint": 0.3, "placebo": 0.1, "rational": 0.1}))
    assert v["theories"]["bsv"]["advances"] and v["full_study"]


def test_placebo_rule_blocks():
    # Theory beats placebo a little, but placebo carries most of the gain over rational.
    v = verdicts(CFG, ALL_PASS, _moments({"bsv": 0.30, "dhs": 0.30, "placebo": 0.25, "rational": 0.0}))
    first = v["theories"]["bsv"]["moments"][CFG.gate.gate_moments["bsv"][0]]
    assert first["effect_passed"] and not first["placebo_passed"]
    assert not v["theories"]["bsv"]["advances"]


def test_negative_gain_over_rational_blocks_instead_of_flipping_sign():
    v = verdicts(CFG, ALL_PASS, _moments({"bsv": 0.0, "dhs": 0.0, "placebo": 0.5, "rational": 0.1}))
    assert not v["theories"]["bsv"]["moments"][CFG.gate.gate_moments["bsv"][0]]["placebo_passed"]


def test_failed_primitive_blocks_and_leading_arm_prefers_advancing():
    analysis = {**ALL_PASS, "overconfidence": {"passed": False}}
    ms = _moments({"bsv": 0.3, "dhs": 0.5, "joint": 0.2, "placebo": 0.1, "rational": 0.1})
    v = verdicts(CFG, analysis, ms)
    assert v["theories"]["bsv"]["advances"] and not v["theories"]["dhs"]["advances"]
    assert leading_arm(CFG, analysis, ms) == "bsv"


def test_one_gate_moment_is_enough():
    # Effect only on BSV's second gate moment.
    second = CFG.gate.gate_moments["bsv"][1]
    ms = _moments({"bsv": 0.3, "dhs": 0.0, "joint": 0.0, "placebo": 0.05, "rational": 0.0}, keys=[second])
    v = verdicts(CFG, ALL_PASS, ms)
    assert v["theories"]["bsv"]["advances"] and not v["theories"]["dhs"]["advances"]


def test_theories_are_gated_on_their_own_moments():
    # A large effect on a DHS-only moment does not advance BSV.
    dhs_only = [k for k in CFG.gate.gate_moments["dhs"] if k not in CFG.gate.gate_moments["bsv"]]
    ms = _moments({"bsv": 0.5, "dhs": 0.5, "joint": 0.5, "placebo": 0.0, "rational": 0.0}, keys=dhs_only)
    v = verdicts(CFG, ALL_PASS, ms)
    assert v["theories"]["dhs"]["advances"] and not v["theories"]["bsv"]["advances"]
    assert set(v["theories"]["bsv"]["moments"]) == set(CFG.gate.gate_moments["bsv"])


def test_market_model_rule():
    from gate0.gate import market_model
    cfg = CFG.model_copy(deep=True)  # the rule needs two candidates; v3 registers one
    cfg.models.candidates = haiku, sonnet = ["claude-haiku-5-5", "claude-sonnet-5-5"]
    passing = lambda *ps: {p: {"passed": p in ps} for p in CFG.measurement.primitives}  # noqa: E731
    # More theories with both primitives passing wins, even with fewer primitives overall.
    a = {haiku: passing("conservatism", "representativeness"),
         sonnet: passing("conservatism", "overconfidence", "self_attribution")}
    a[sonnet]["representativeness"]["passed"] = False
    assert market_model(cfg, a) == sonnet  # sonnet: DHS complete (1 theory), 3 primitives; haiku: BSV (1), 2
    # Ties on theories and primitives go to candidate order.
    assert market_model(cfg, {haiku: passing("conservatism"), sonnet: passing("overconfidence")}) == haiku
    assert market_model(cfg, {haiku: passing(), sonnet: passing("overconfidence")}) == sonnet
