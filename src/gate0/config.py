"""Registration config: schema, loading, and the hash lock that freezes it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "registration.yaml"
PROMPTS_DIR = ROOT / "prompts"
LOCK_PATH = ROOT / "registration.lock"
DATA_DIR = ROOT / "data"


class Models(BaseModel):
    base: str
    strong: str
    effort: str
    max_tokens: int


class Thresholds(BaseModel):
    monotonic_spearman: float
    max_cross_loading: float
    rational_null_alpha: float
    rational_null_margin: float


class Measurement(BaseModel):
    primitives: list[str]
    intensities: list[int]
    n_items: int
    repeats: int
    item_seed: int
    thresholds: Thresholds


class Fundamentals(BaseModel):
    factor_vol: float
    idio_vol: float
    signal_vol_low: float
    signal_vol_high: float
    common_persistence: float
    p_high_dispersion: float
    regime_persistence: float
    earnings_noise: float
    analyst_optimism_max: float


class Market(BaseModel):
    n_assets: int
    n_agents: int
    n_rounds: int
    seeds: list[int]
    arms: list[str]
    masked_arms: list[str]
    masked_seeds: list[int]
    announcement_every: int
    price_window: int
    initial_cash: float
    initial_shares: int
    demand_scale: float
    budget_per_asset: Literal["cash_over_n"]
    show_previous_estimate: bool
    fundamentals: Fundamentals


class Moments(BaseModel):
    formation_window: int
    max_horizon: int
    interaction_horizon: int
    pead_window: int
    bootstrap_draws: int
    ci_level: float


class Gate(BaseModel):
    gate_moments: dict[str, list[str]]  # per theory

    def moments_for(self, arm: str) -> list[str]:
        """A theory's own gate moments; the union (in order) for the joint arm."""
        if arm in self.gate_moments:
            return self.gate_moments[arm]
        return list(dict.fromkeys(k for ks in self.gate_moments.values() for k in ks))
    manipulation_moment: str
    ci_rule: Literal["lower_above_zero"]
    placebo_share_max: float


class Freeze(BaseModel):
    seeds: list[int]
    arms: list[str]
    anomaly: str


class Strong(BaseModel):
    conditions: list[str]
    intensity: int
    repeats: int


class Config(BaseModel):
    models: Models
    measurement: Measurement
    theories: dict[str, list[str]]
    market: Market
    moments: Moments
    gate: Gate
    freeze: Freeze
    strong: Strong


def load_config(path: Path = CONFIG_PATH) -> Config:
    return Config.model_validate(yaml.safe_load(path.read_text()))


def registration_hashes(config_path: Path = CONFIG_PATH, prompts_dir: Path = PROMPTS_DIR) -> dict[str, str]:
    """SHA-256 of the config, every prompt template, and the model identifiers."""
    hashes = {"config": _sha(config_path.read_bytes())}
    for p in sorted(prompts_dir.glob("*.txt")):
        hashes[f"prompts/{p.name}"] = _sha(p.read_bytes())
    # Item text, the market prompt body, scoring, moments and gate logic live in code.
    src = Path(__file__).resolve().parent
    for p in sorted(src.rglob("*.py")):
        hashes[f"src/{p.relative_to(src)}"] = _sha(p.read_bytes())
    models = load_config(config_path).models
    hashes["models"] = _sha(json.dumps(models.model_dump(), sort_keys=True).encode())
    hashes["registration"] = _sha(json.dumps(hashes, sort_keys=True).encode())
    return hashes


def write_lock(path: Path = LOCK_PATH) -> dict[str, str]:
    hashes = registration_hashes()
    path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    return hashes


def check_lock(path: Path = LOCK_PATH) -> str:
    """Return the registration hash, or raise if the frozen objects have drifted."""
    if not path.exists():
        raise RuntimeError("No registration.lock; run `gate0 register` first (or pass --dev).")
    locked = json.loads(path.read_text())
    current = registration_hashes()
    drift = sorted(k for k in locked.keys() | current.keys() if locked.get(k) != current.get(k))
    if drift:
        raise RuntimeError(f"Registered objects changed since lock: {', '.join(drift)}")
    return locked["registration"]


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()
