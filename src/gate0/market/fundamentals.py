"""The seeded world: fundamentals, signals, reports and analyst data.

Everything here depends only on the seed, so every arm at a given seed sees identical
fundamentals and signal draws and arm contrasts are paired.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gate0.config import Market


@dataclass
class World:
    seed: int
    log_value: np.ndarray      # (T+1, N)
    regime: np.ndarray         # (T+1, N) 1 = high-dispersion state
    common_noise: np.ndarray   # (T+1, N) signal error shared by all analysts
    agent_noise: np.ndarray    # (T+1, N, J) idiosyncratic analyst error
    report: np.ndarray         # (T+1, N) log earnings report, NaN off report rounds
    consensus: np.ndarray      # (T+1, N) log analyst consensus (optimistically biased)
    dispersion: np.ndarray     # (T+1, N) public cross-analyst spread of estimates
    optimism: np.ndarray       # (N,)
    signal_vol: np.ndarray     # (T+1, N) total signal s.d. in the current regime

    @property
    def report_rounds(self) -> list[int]:
        return [t for t in range(self.report.shape[0]) if np.isfinite(self.report[t, 0])]

    def signal(self, t: int, j: int) -> np.ndarray:
        """Agent j's private log value estimate for every asset at round t."""
        return self.log_value[t] + self.common_noise[t] + self.agent_noise[t, :, j]


def make_world(m: Market, seed: int) -> World:
    f = m.fundamentals
    rng = np.random.default_rng([seed, 0xF00D])
    T, N, J = m.n_rounds, m.n_assets, m.n_agents

    loadings = rng.uniform(0.5, 1.5, N)
    factor = rng.normal(0, f.factor_vol, T + 1)
    shocks = loadings * factor[:, None] + rng.normal(0, f.idio_vol, (T + 1, N))
    shocks[0] = 0
    log_value = np.log(100) + rng.normal(0, 0.3, N) + np.cumsum(shocks, axis=0)

    regime = np.zeros((T + 1, N), dtype=int)
    regime[0] = rng.random(N) < f.p_high_dispersion
    for t in range(1, T + 1):
        stay = rng.random(N) < f.regime_persistence
        regime[t] = np.where(stay, regime[t - 1], rng.random(N) < f.p_high_dispersion)
    signal_vol = np.where(regime == 1, f.signal_vol_high, f.signal_vol_low)

    # Split signal error equally (in variance) between a common and an idiosyncratic part.
    part = signal_vol / np.sqrt(2)
    # The shared error is persistent (a common misreading of the firm) with stationary s.d. `part`.
    rho = f.common_persistence
    z = rng.normal(0, 1, (T + 1, N))
    common = np.zeros((T + 1, N))
    common[0] = z[0]
    for t in range(1, T + 1):
        common[t] = rho * common[t - 1] + np.sqrt(1 - rho**2) * z[t]
    common = common * part
    agent = rng.normal(0, 1, (T + 1, N, J)) * part[:, :, None]

    report = np.full((T + 1, N), np.nan)
    for t in range(m.announcement_every, T + 1, m.announcement_every):
        report[t] = log_value[t] + rng.normal(0, f.earnings_noise, N)

    optimism = rng.uniform(0, f.analyst_optimism_max, N)
    consensus = log_value + optimism + common + agent.mean(axis=2)
    dispersion = agent.std(axis=2, ddof=1)

    return World(seed, log_value, regime, common, agent, report, consensus, dispersion, optimism, signal_vol)
