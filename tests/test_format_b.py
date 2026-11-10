import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("format_b", Path(__file__).parents[1] / "addon" / "format_b.py")
fb = importlib.util.module_from_spec(spec)
sys.modules["format_b"] = fb
spec.loader.exec_module(fb)
S = fb.settings()


def _answers(item, bias):
    """A synthetic agent: Bayesian, distorted by `bias` on its own task's channel."""
    t = item.truth
    if item.task == "conservatism":
        w = t["w"] * (1 - bias)
        return [{"estimate": float(np.exp(sgn * w * t["log_gap"] / 2) * 100)} for sgn in (1, -1)]
    if item.task == "representativeness":
        return [{"estimate": t["avg"] * float(np.exp(sgn * bias * t["span"]))} for sgn in (1, -1)]
    half = t["bayes_width"] / 2 * np.exp(-bias)
    return [{"estimate": 50.0, "low": 50 * float(np.exp(-half)), "high": 50 * float(np.exp(half))}]


@pytest.mark.parametrize("task", S["primitives"])
def test_bayesian_scores_zero_and_bias_is_recovered(task):
    items = fb.make_items(task, 10, 1, S["shift"])
    for it in items:
        assert fb.score(it, _answers(it, 0.0)) == pytest.approx(0.0, abs=1e-9)
        assert fb.score(it, _answers(it, 0.4)) == pytest.approx(0.4, rel=1e-6)


def test_prompts_unique_and_registered_dev_disjoint():
    for task in S["primitives"]:
        reg = {p for i in fb.make_items(task, S["n_items"], S["item_seed"], S["shift"]) for p in i.prompts}
        dev = {p for i in fb.make_items(task, S["n_items"], S["item_seed"] + 1, S["shift"]) for p in i.prompts}
        assert len(reg) == S["n_items"] * (1 if task == "overconfidence" else 2) and not reg & dev


def test_format_b_wording_avoids_persona_vocabulary():
    """Format B never mentions earnings reports, market prices, analysts or the consensus."""
    for task in S["primitives"]:
        for it in fb.make_items(task, S["n_items"], S["item_seed"], S["shift"]):
            for p in it.prompts:
                low = p.lower()
                assert not any(w in low for w in ("earnings", "report", "analyst", "consensus", "market", "price", "streak", "trend"))
