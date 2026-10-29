import numpy as np

from gate0.config import load_config
from gate0.market.agents import round_prompt, seat_agents
from gate0.market.auction import clear, demand
from gate0.market.fundamentals import make_world
from gate0.market.simulate import MarketRun, build_runs

CFG = load_config()


def test_auction_clears_between_valuations_and_balances():
    value = np.array([90.0, 100.0, 110.0, 120.0])
    conf = np.array([5.0, 5.0, 5.0, 5.0])
    p, x = clear(value, conf, np.full(4, 10.0), np.full(4, 1e6), 2.0, 100.0)
    assert 100 < p < 110
    assert abs(x.sum()) < 1e-9
    assert x[0] < 0 and x[3] > 0


def test_auction_respects_constraints():
    value = np.array([50.0, 200.0])
    p, x = clear(value, np.array([10.0, 10.0]), np.array([1.0, 0.0]), np.array([0.0, 50.0]), 5.0, 100.0)
    assert x[0] >= -1.0 - 1e-9          # cannot sell more than held
    assert x[1] * p <= 50.0 + 1e-6      # cannot spend beyond budget
    assert abs(x.sum()) < 1e-9


def test_demand_is_monotone():
    args = (np.array([90.0, 110.0]), np.array([3.0, 7.0]), np.array([5.0, 5.0]), np.array([500.0, 500.0]), 2.0)
    totals = [demand(p, *args).sum() for p in np.linspace(80, 120, 50)]
    assert all(a >= b for a, b in zip(totals, totals[1:]))


def test_world_is_shared_across_arms():
    a, b = make_world(CFG.market, 3), make_world(CFG.market, 3)
    assert np.array_equal(a.log_value, b.log_value) and np.array_equal(a.agent_noise, b.agent_noise)
    assert not np.array_equal(a.log_value, make_world(CFG.market, 4).log_value)


def test_seating_is_paired_across_arms():
    measurement = {
        p: {"passed": True, "measured": {"1": [0.1, 0.2], "2": [0.4, 0.5], "3": [0.8, 0.9]}}
        for p in ("conservatism", "representativeness", "overconfidence", "self_attribution")
    }
    bsv = seat_agents(CFG, "bsv", 5, measurement)
    joint = seat_agents(CFG, "joint", 5, measurement)
    for a, b in zip(bsv, joint):
        assert a.intensities["conservatism"] == b.intensities["conservatism"]
        assert a.recovered["conservatism"] == b.recovered["conservatism"]
        # The draw is binned to the level whose mean recovery is nearest.
        assert a.intensities["conservatism"] == {0.1: 1, 0.2: 1, 0.4: 2, 0.5: 2, 0.8: 3, 0.9: 3}[a.recovered["conservatism"]]
    measurement["overconfidence"]["passed"] = False
    assert all("overconfidence" not in a.intensities for a in seat_agents(CFG, "dhs", 5, measurement))


def test_masked_prompt_has_no_market_cues():
    run = MarketRun(CFG, "joint", 0, True, seat_agents(CFG, "joint", 0, None))
    text = round_prompt(CFG.market, run.world, 5, 0, run.names, True, run.prices, run.holdings[0], 1000.0, None, None)
    for word in ("stock", "earnings", "analyst", "$", "momentum"):
        assert word not in text.lower()


def test_rule_market_runs():
    runs = build_runs(CFG, ["rational", "bsv"], [0], False, "rule", "", None)
    from gate0.market.simulate import simulate
    import tempfile, pathlib
    res = simulate(runs, None, pathlib.Path(tempfile.mkdtemp()))
    prices = np.array(res[0]["prices"], dtype=float)
    assert np.all(np.isfinite(prices)) and np.all(prices > 0)
    assert np.array(res[0]["volume"]).sum() > 0


def test_true_value_never_reaches_agents():
    run = MarketRun(CFG, "placebo", 0, False, seat_agents(CFG, "placebo", 0, None))
    v0 = np.exp(run.world.log_value[0])
    assert not np.allclose(run.prices[0], v0)
    assert not np.any(np.isclose(run.value, v0[None, :]))
    text = round_prompt(CFG.market, run.world, 1, 0, run.names, False, run.prices, run.holdings[0], 1000.0, None, None)
    for v in v0:
        assert f"{v:.2f}" not in text


def test_previous_estimate_and_confirmation_shown():
    run = MarketRun(CFG, "placebo", 0, False, seat_agents(CFG, "placebo", 0, None))
    run.prices[1] = run.prices[0] * 1.1                      # price rose after the estimate
    prev = run.prices[0] * np.where(np.arange(CFG.market.n_assets) == 0, 1.2, 0.8)
    text = round_prompt(CFG.market, run.world, 2, 0, run.names, False, run.prices, run.holdings[0], 1000.0,
                        prev, np.full(CFG.market.n_assets, 5.0))
    assert "(confirmed)" in text and "(not confirmed)" in text
    assert text.count("(confirmed)") == 1                    # only asset 0 was estimated above the old price


def test_forecast_recorded_for_pead():
    runs = build_runs(CFG, ["placebo"], [0], False, "rule", "", None)
    from gate0.market.simulate import simulate
    import tempfile, pathlib
    res = simulate(runs, None, pathlib.Path(tempfile.mkdtemp()))[0]
    f = np.array(res["forecast"], dtype=float)
    assert f.shape == (CFG.market.n_rounds + 1, CFG.market.n_assets) and np.all(np.isfinite(f))
