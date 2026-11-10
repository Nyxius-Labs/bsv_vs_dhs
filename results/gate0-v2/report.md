# Gate 0 report

Registration: `v2-attempt1` `3a64484198be43fb86e0bee5ce7f8e85cec3449fda0161ee08dd808cf93a2f91`

## Measurement stage

Checks: Spearman ≥ 0.8 plus a paired top-vs-neutral CI above 0; max cross-loading < 0.25; rational null: the 95% CI of (rational − neutral), paired over items, lies within ±0.25 × the primitive's own (top − neutral) effect. The last column is the rational arm's level against the Bayesian benchmark (descriptive).

Verdicts by model and thinking setting (the registered setting is `off`; `adaptive` is a descriptive control and never gates):

| primitive | claude-haiku-5-5 / off | claude-haiku-5-5 / adaptive | claude-sonnet-5-5 / off | claude-sonnet-5-5 / adaptive |
|---|---|---|---|---|
| conservatism | FAIL | FAIL | FAIL | FAIL |
| representativeness | FAIL | FAIL | FAIL | FAIL |
| overconfidence | FAIL | FAIL | FAIL | FAIL |
| self_attribution | FAIL | FAIL | FAIL | FAIL |

Market model by the registered rule (`most_theories_then_primitives_then_order`): **claude-haiku-5-5**, thinking `off`.

### claude-haiku-5-5, thinking off

| primitive | monotonic (ρ) | near-diagonal (max) | rational − neutral (mean [CI], ±margin) | own effect (SD) | level means | verdict | rational vs Bayes |
|---|---|---|---|---|---|---|---|
| conservatism | PASS (1.00) | FAIL (1.01) | FAIL (-0.02 [-0.06, +0.02], ±0.01) | +0.31 | 0: +0.41, 1: +0.42, 2: +0.46, 3: +0.47 | **FAIL** | +0.40 [+0.32, +0.47] |
| representativeness | FAIL (-1.00) | FAIL (0.65) | FAIL (-0.01 [-0.02, +0.01], ±0.00) | -0.36 | 0: +0.00, 1: +0.00, 2: -0.00, 3: -0.01 | **FAIL** | -0.00 [-0.01, +0.01] |
| overconfidence | PASS (1.00) | FAIL (0.37) | FAIL (-0.45 [-0.59, -0.33], ±0.14) | +0.77 | 0: +1.19, 1: +1.36, 2: +1.43, 3: +1.74 | **FAIL** | +0.74 [+0.54, +0.95] |
| self_attribution | FAIL (0.20) | FAIL (4.23) | FAIL (+0.05 [-0.18, +0.24], ±0.02) | +0.15 | 0: -0.04, 1: +0.05, 2: +0.04, 3: +0.02 | **FAIL** | +0.01 [-0.14, +0.17] |

Self-attribution β (rise in the private-estimate weight after a confirmation) and γ (fall after a contradiction), in units of its rational weight, by prompted level: 0: β=-0.02, γ=+0.02; 1: β=+0.05, γ=+0.00; 2: β=+0.08, γ=+0.04; 3: β=+0.06, γ=+0.04

### claude-haiku-5-5, thinking adaptive

| primitive | monotonic (ρ) | near-diagonal (max) | rational − neutral (mean [CI], ±margin) | own effect (SD) | level means | verdict | rational vs Bayes |
|---|---|---|---|---|---|---|---|
| conservatism | FAIL (0.40) | FAIL (0.66) | FAIL (-0.13 [-0.19, -0.07], ±0.02) | +0.37 | 0: +0.46, 1: +0.42, 2: +0.44, 3: +0.53 | **FAIL** | +0.33 [+0.25, +0.41] |
| representativeness | FAIL (0.40) | FAIL (1.32) | FAIL (+0.02 [+0.00, +0.03], ±0.00) | +0.24 | 0: -0.01, 1: +0.00, 2: -0.01, 3: +0.00 | **FAIL** | +0.01 [-0.01, +0.02] |
| overconfidence | PASS (1.00) | PASS (0.12) | FAIL (-0.46 [-0.73, -0.24], ±0.10) | +0.58 | 0: +1.08, 1: +1.16, 2: +1.34, 3: +1.49 | **FAIL** | +0.62 [+0.34, +0.84] |
| self_attribution | FAIL (0.60) | FAIL (8.77) | FAIL (-0.13 [-0.44, +0.14], ±0.01) | +0.05 | 0: -0.10, 1: -0.16, 2: -0.08, 3: -0.08 | **FAIL** | -0.23 [-0.53, +0.01] |

Self-attribution β (rise in the private-estimate weight after a confirmation) and γ (fall after a contradiction), in units of its rational weight, by prompted level: 0: β=+0.00, γ=+0.10; 1: β=-0.02, γ=+0.13; 2: β=-0.00, γ=+0.08; 3: β=-0.02, γ=+0.06

### claude-sonnet-5-5, thinking off

| primitive | monotonic (ρ) | near-diagonal (max) | rational − neutral (mean [CI], ±margin) | own effect (SD) | level means | verdict | rational vs Bayes |
|---|---|---|---|---|---|---|---|
| conservatism | PASS (1.00) | FAIL (0.47) | PASS (+0.01 [-0.02, +0.04], ±0.09) | +2.25 | 0: +0.25, 1: +0.29, 2: +0.49, 3: +0.59 | **FAIL** | +0.26 [+0.22, +0.29] |
| representativeness | FAIL (-0.40) | FAIL (25.69) | FAIL (-0.01 [-0.03, +0.00], ±0.00) | -0.07 | 0: +0.03, 1: +0.02, 2: +0.03, 3: +0.02 | **FAIL** | +0.01 [+0.00, +0.02] |
| overconfidence | FAIL (0.80) | FAIL (1.32) | FAIL (-0.05 [-0.13, +0.02], ±0.05) | +0.45 | 0: +0.97, 1: +1.11, 2: +1.10, 3: +1.17 | **FAIL** | +0.92 [+0.74, +1.10] |
| self_attribution | FAIL (-0.60) | FAIL (10.11) | FAIL (+0.07 [-0.01, +0.16], ±0.00) | -0.06 | 0: -0.02, 1: +0.05, 2: -0.05, 3: -0.04 | **FAIL** | +0.05 [-0.05, +0.15] |

Self-attribution β (rise in the private-estimate weight after a confirmation) and γ (fall after a contradiction), in units of its rational weight, by prompted level: 0: β=+0.00, γ=+0.02; 1: β=+0.04, γ=-0.01; 2: β=-0.01, γ=+0.04; 3: β=+0.02, γ=+0.05

### claude-sonnet-5-5, thinking adaptive

| primitive | monotonic (ρ) | near-diagonal (max) | rational − neutral (mean [CI], ±margin) | own effect (SD) | level means | verdict | rational vs Bayes |
|---|---|---|---|---|---|---|---|
| conservatism | PASS (1.00) | FAIL (0.40) | PASS (-0.02 [-0.05, +0.01], ±0.09) | +2.38 | 0: +0.24, 1: +0.31, 2: +0.50, 3: +0.60 | **FAIL** | +0.22 [+0.19, +0.25] |
| representativeness | PASS (1.00) | FAIL (4.05) | FAIL (-0.00 [-0.02, +0.01], ±0.01) | +0.50 | 0: +0.01, 1: +0.01, 2: +0.01, 3: +0.03 | **FAIL** | +0.01 [-0.00, +0.02] |
| overconfidence | FAIL (0.80) | FAIL (0.90) | FAIL (-0.07 [-0.18, +0.04], ±0.05) | +0.45 | 0: +0.94, 1: +1.01, 2: +1.26, 3: +1.13 | **FAIL** | +0.87 [+0.71, +1.02] |
| self_attribution | FAIL (-0.80) | FAIL (2.41) | FAIL (+0.04 [-0.09, +0.17], ±0.04) | -0.51 | 0: -0.06, 1: +0.05, 2: -0.08, 3: -0.20 | **FAIL** | -0.02 [-0.13, +0.09] |

Self-attribution β (rise in the private-estimate weight after a confirmation) and γ (fall after a contradiction), in units of its rational weight, by prompted level: 0: β=-0.01, γ=+0.05; 1: β=+0.04, γ=-0.01; 2: β=-0.01, γ=+0.07; 3: β=-0.14, γ=+0.07

## Usage

API calls billed under this registration (since `gate0 register`; dev runs: whole cache).

| model | calls | input tokens | output tokens |
|---|---|---|---|
| claude-haiku-5-5 | 15,360 | 13,154,256 | 689,426 |
| claude-sonnet-5-5 | 15,360 | 13,154,256 | 368,994 |
