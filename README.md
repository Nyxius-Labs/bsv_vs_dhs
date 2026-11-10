# Can a Prompt Give an LLM Agent a Measurable Bias?

Brandon Yee and Jacob Crainic

LLM agents are increasingly given behavioral traits through their prompts and used as traders in simulated markets. But telling an agent to be overconfident does not show that it holds overconfident beliefs. This repository holds a preregistered pilot ("Gate 0") that tests whether a prompted bias can be measured before it is used as a treatment.

The test bed is two theories of stock-price momentum:

- **BSV** (Barberis, Shleifer and Vishny, 1998): *conservatism* (underreacting to earnings news) and *representativeness* (reading a run of reports as a trend).
- **DHS** (Daniel, Hirshleifer and Subrahmanyam, 1998): *overconfidence* (overweighting one's own private information) and *biased self-attribution* (trusting that information more after the market proves it right, but not less after it proves it wrong).

## How it works

1. **Measure each bias.** Each test prompt shows one stock exactly as the market does: prices, earnings reports, analyst consensus, a private estimate, and each source's stated error. Versions of a prompt differ in one input, and a bias is scored by how the agent's valuation moves between versions, relative to the Bayesian weights. An agent cannot score high just by saying it is biased.
2. **Check it.** Each persona is written at three strengths, plus a neutral and a "rational" persona. A bias passes only if:
   - its score rises with strength;
   - it leaves the other three scores alone;
   - the rational persona matches the neutral one.
3. **Run the market.** A theory enters the market only if both its biases pass. The market has 12 agents, 10 stocks and 40 rounds of a call auction, on 10 shared simulated data sets. A theory advances if its agents produce more post-earnings drift or momentum than a market of neutral agents. A disguised version, with coded tickers and no finance vocabulary, checks that effects do not come from recalled finance theory.

Every threshold, prompt, model ID and source file is hashed into a lock file before a registered run. The code refuses to run if any of them have changed.

## Results

| Registration | Model | Conservatism | Representativeness | Overconfidence | Self-attribution |
|---|---|---|---|---|---|
| v1 | Haiku 5.5, reasoning on | pass | fail | fail | fail |
| v2 | Haiku and Sonnet 5.5, reasoning off | fail | fail | fail | fail |
| v3 | Sonnet 5.5, reasoning off | pass | pass | pass | fail (control check only) |

In v3, BSV entered the market. Each figure below is BSV's difference from the neutral market, with a 95% interval over data sets:

- **post-earnings drift:** +0.18 [0.07, 0.30];
- **momentum:** +0.09 [0.06, 0.12]. BSV prices keep moving in the same direction for about four rounds and then reverse; neutral prices reverse from the start.

In the disguised market, the drift persists. DHS did not pass measurement, so the registered rule stops the comparison of the two theories here.

The dev iteration behind v3 is logged in [`results/gate0-v3-dev/iteration_log.md`](results/gate0-v3-dev/iteration_log.md).

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) and, for real runs, an `ANTHROPIC_API_KEY` in `.env`.

```bash
uv sync
uv run pytest                      # tests, no API calls

# zero-cost dry run: synthetic measurement and rule-based agents -> data/dev/report.md
uv run gate0 measure --synthetic
uv run gate0 market --agent rule --dev && uv run gate0 freeze --agent rule --dev
uv run gate0 report --agent rule --dev

# registered pipeline (v3)
uv run gate0 measure               # 7,680 calls
uv run gate0 market --mode batch   # 70 runs
uv run gate0 freeze --mode batch   # leading arm on held-out data sets
uv run gate0 report                # data/registered_v3/report.md
```

API responses are cached in `data/cache.sqlite`, so rerunning a finished stage costs nothing.

## Layout

- `configs/registration_v3.yaml`: every threshold, arm, seed and moment parameter.
- `prompts/`: persona and market prompt text.
- `src/gate0/measure/`: test prompts, scoring and the three checks.
- `src/gate0/market/`: simulated companies, agents and the call auction.
- `src/gate0/moments.py`, `gate.py`: market outcomes and the advance rule.
- `registration*.lock`: one lock file for each registration.

Tags `gate0-registered` (v1), `gate0-v2-registered` and `gate0-v3-registered` mark each registered state; v1 and v2 are reproducible only from their tags.
