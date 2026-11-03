"""gate0 register | measure | market | freeze | report"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import typer
from dotenv import load_dotenv

from gate0 import moments as M
from gate0.config import DATA_DIR, LOCK_PATH, REGISTERED_DIR, check_lock, load_config, registered_at, write_lock
from gate0.gate import leading_arm, market_model, report as render_report
from gate0.llm import LLMClient
from gate0.market.simulate import build_runs, simulate
from gate0.measure.checks import analyze
from gate0.measure.run import base_design, run_measurement
from gate0.measure.synthetic import SyntheticClient, clean_params

app = typer.Typer(no_args_is_help=True, add_completion=False)

Mode = typer.Option("live", help="live | batch | offline (live for the pilot: market rounds are sequential)")
Dev = typer.Option(False, help="Skip the registration check; write under data/dev/.")
Synthetic = typer.Option(False, help="Synthetic agents with known biases instead of the API (implies --dev).")


def _setup(dev: bool) -> tuple[str, Path]:
    load_dotenv()
    if dev:
        return "unregistered", DATA_DIR / "dev"
    return check_lock(), REGISTERED_DIR


def _client(cfg, mode: str, thinking: str) -> LLMClient:
    return LLMClient(DATA_DIR / "cache.sqlite", mode, cfg.models.effort, cfg.models.max_tokens,
                     thinking, cfg.models.thinking_off)


def _done(client) -> None:
    if not isinstance(client, LLMClient):
        return
    s = client.stats
    typer.echo(f"cache hits {s.cache_hits}, new calls {s.new_calls}, failures {s.failures} {s.errors or ''}, "
               f"tokens in/out {s.input_tokens:,}/{s.output_tokens:,}")
    client.close()


def _csv(s: str | None, cast=str):
    return [cast(x) for x in s.split(",")] if s else None


@app.command()
def register(force: bool = typer.Option(False, help="Overwrite an existing lock.")):
    """Hash config, prompts, model IDs and the gate0 source into the registration lock."""
    if LOCK_PATH.exists() and not force:
        raise typer.BadParameter(f"{LOCK_PATH.name} exists; re-registering changes the pre-registered object (use --force)")
    h = write_lock()
    typer.echo(f"registered {h['registration']}")


@app.command()
def measure(
    mode: str = Mode, dev: bool = Dev, synthetic: bool = Synthetic,
    items: int = typer.Option(None, help="Override items per task (smoke runs; implies --dev)."),
    repeats: int = typer.Option(None, help="Override repeats (smoke runs; implies --dev)."),
    models: str = typer.Option(None, help="Comma-separated subset of candidate models (implies --dev)."),
    control: bool = typer.Option(True, help="Also run the registered descriptive control grid, if any."),
):
    """Measurement stage on every candidate model: intensity grid, toggles, rational arm, then the checks.
    Runs the registered thinking setting, then the descriptive control setting."""
    cfg = load_config()
    dev = dev or synthetic or bool(items or repeats or models) or not control
    cfg.measurement.n_items = items or cfg.measurement.n_items
    _, out = _setup(dev)
    model_list = _csv(models) or cfg.models.candidates
    settings = [cfg.models.thinking] + ([cfg.models.control_thinking] if control and cfg.models.control_thinking else [])
    for thinking in settings:
        client = SyntheticClient(clean_params()) if synthetic else _client(cfg, mode, thinking)
        for model in model_list:
            df = run_measurement(cfg, client, model, base_design(cfg), repeats or cfg.measurement.repeats, dev, thinking)
            _save_measurement(cfg, df, _measure_dir(out, model, thinking))
        _done(client)


def _measure_dir(out: Path, model: str, thinking: str) -> Path:
    return out / "measure" / model / f"thinking-{thinking}"


def _save_measurement(cfg, df: pd.DataFrame, d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / "scores.csv", index=False)
    analysis = analyze(df, cfg)
    (d / "analysis.json").write_text(json.dumps(analysis, indent=2, default=float))
    for p, r in analysis.items():
        typer.echo(f"{p:20s} {'PASS' if r['passed'] else 'FAIL'}  {r['checks']}")


@app.command()
def market(
    agent: str = typer.Option("llm", help="llm | rule (rule-based dry run, no API calls)"),
    mode: str = Mode, dev: bool = Dev,
    arms: str = typer.Option(None, help="Comma-separated subset of arms."),
    seeds: str = typer.Option(None, help="Comma-separated subset of seeds."),
    rounds: int = typer.Option(None, help="Override rounds (pilot slices; implies --dev)."),
    no_masked: bool = typer.Option(False, help="Skip the masked-ticker runs."),
):
    """Identification stage: all arms at paired seeds, plus masked-ticker runs."""
    cfg = load_config()
    # Any deviation from the registered design is a dev run; it never touches registered outputs.
    dev = dev or bool(arms or seeds or rounds or no_masked)
    if rounds:
        cfg.market.n_rounds = rounds
    _, out = _setup(dev)
    arm_list = _csv(arms) or cfg.market.arms
    # Dev runs default to a seed outside every registered set.
    seed_list = _csv(seeds, int) or ([999] if dev and agent == "llm" else cfg.market.seeds)
    registered = set(cfg.market.seeds) | set(cfg.freeze.seeds)
    if dev and agent == "llm" and registered & set(seed_list):
        raise typer.BadParameter("dev LLM runs must not use registered seeds")
    model, measurement = _market_model(cfg, out, agent)
    runs = build_runs(cfg, arm_list, seed_list, False, agent, model, measurement)
    if not no_masked:
        masked_arms = [a for a in cfg.market.masked_arms if a in arm_list]
        masked = [s for s in seed_list if s in cfg.market.masked_seeds]
        runs += build_runs(cfg, masked_arms, masked, True, agent, model, measurement)
    client = _client(cfg, mode, cfg.models.thinking) if agent == "llm" else None
    d = out / "market" / agent
    results = simulate(runs, client, d / "runs")
    _write_moments(cfg, results, d)
    if client:
        _done(client)


def _write_moments(cfg, results: list[dict], d: Path) -> None:
    mc = cfg.moments
    if cfg.market.n_rounds < mc.formation_window + mc.max_horizon:
        typer.echo(f"runs too short for moments ({cfg.market.n_rounds} rounds); raw runs saved in {d / 'runs'}")
        return
    ms = [M.compute(r, mc) for r in results]
    (d / "moments.json").write_text(json.dumps(ms, default=float))
    typer.echo(pd.DataFrame(ms).groupby(["arm", "masked"])[list(dict.fromkeys([cfg.gate.manipulation_moment, *cfg.gate.moments_for('joint')]))]
               .mean().round(3).to_string())


def _load_analyses(cfg, out: Path, thinking: str | None = None) -> dict[tuple[str, str], dict]:
    """(model, thinking) -> analysis, for every measurement that has run."""
    found = {}
    for model in cfg.models.candidates:
        for t in [thinking] if thinking else [x for x in (cfg.models.thinking, cfg.models.control_thinking) if x]:
            p = _measure_dir(out, model, t) / "analysis.json"
            if p.exists():
                found[(model, t)] = json.loads(p.read_text())
    return found


def _market_model(cfg, out: Path, agent: str) -> tuple[str, dict | None]:
    """The market model by the registered rule, and its measurement (for seating)."""
    if agent == "rule":
        return cfg.models.candidates[0], None
    analyses = {m: a for (m, _), a in _load_analyses(cfg, out, cfg.models.thinking).items()}
    if not analyses:
        raise typer.BadParameter(f"no measurement under {out / 'measure'}; run `gate0 measure` first")
    model = market_model(cfg, analyses)
    typer.echo(f"market model: {model} (thinking {cfg.models.thinking})")
    return model, analyses[model]


def _load_moments(out: Path, agent: str = "llm") -> list[dict] | None:
    p = out / "market" / agent / "moments.json"
    return json.loads(p.read_text()) if p.exists() else None


@app.command()
def freeze(mode: str = Mode, dev: bool = Dev, agent: str = typer.Option("llm")):
    """Freeze the leading configuration and run it against rational and placebo on held-out seeds."""
    cfg = load_config()
    registration, out = _setup(dev)
    moments = _load_moments(out, agent)
    if not moments:
        raise typer.BadParameter("no market results; run `gate0 market` first")
    model, measurement = _market_model(cfg, out, agent)
    arm = leading_arm(cfg, measurement, moments)
    # Dev LLM runs never touch the held-out freeze seeds (rule agents make no calls and see nothing).
    seeds = [998] if dev and agent == "llm" else cfg.freeze.seeds
    frozen = {"arm": arm, "model": model, "registration": registration,
              "measurement_hash": hashlib.sha256(json.dumps(measurement, sort_keys=True).encode()).hexdigest()}
    frozen["hash"] = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    d = out / "freeze" / agent
    d.mkdir(parents=True, exist_ok=True)
    (d / "frozen.json").write_text(json.dumps(frozen, indent=2))
    typer.echo(f"frozen {arm} ({frozen['hash'][:16]})")
    runs = build_runs(cfg, [arm, *cfg.freeze.arms], seeds, False, agent, model, measurement)
    client = _client(cfg, mode, cfg.models.thinking) if agent == "llm" else None
    results = simulate(runs, client, d / "runs")
    _write_moments(cfg, results, d)
    if client:
        _done(client)


@app.command()
def report(dev: bool = Dev, agent: str = typer.Option("llm")):
    """Write data/<registered_v2|dev>/report.md from whatever stages have run."""
    cfg = load_config()
    registration, out = _setup(dev)
    analyses = _load_analyses(cfg, out)
    if not analyses:
        raise typer.BadParameter(f"no measurement under {out / 'measure'}; run `gate0 measure` first")
    registered = {m: a for (m, t), a in analyses.items() if t == cfg.models.thinking}
    chosen = market_model(cfg, registered) if registered else None
    moments = _load_moments(out, agent)
    fz = out / "freeze" / agent
    freeze_data = None
    if (fz / "frozen.json").exists() and (fz / "moments.json").exists():
        freeze_data = json.loads((fz / "frozen.json").read_text())
        freeze_data["moments"] = json.loads((fz / "moments.json").read_text())
    client = LLMClient(DATA_DIR / "cache.sqlite", "offline")
    usage = client.usage_by_model(since=0.0 if dev else registered_at())
    client.close()
    text = render_report(cfg, registration, analyses, chosen, moments, freeze_data, usage)
    (out / "report.md").write_text(text)
    typer.echo(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    app()
