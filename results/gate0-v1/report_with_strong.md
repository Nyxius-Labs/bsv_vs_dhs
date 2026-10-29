# Gate 0 report

Registration: `6b633568023a8b56586f051e2ea55f05f9caa2d3492e6a6bc9c5de280174b262`

## Measurement stage

Thresholds: Spearman ≥ 0.8, max cross-loading < 0.25, rational 95% CI covers 0.

| primitive | monotonic (ρ) | near-diagonal (max) | rational null (mean [CI]) | own effect (SD) | level means | verdict |
|---|---|---|---|---|---|---|
| conservatism | PASS (1.00) | PASS (0.03) | PASS (+0.00 [-0.00, +0.00]) | +2.92 | 0: +0.00, 1: +0.11, 2: +0.37, 3: +0.47 | **PASS** |
| representativeness | FAIL (-0.80) | PASS (0.08) | PASS (+0.00 [-0.00, +0.00]) | -0.59 | 0: -0.00, 1: +0.00, 2: -0.04, 3: -0.10 | **FAIL** |
| overconfidence | FAIL (0.40) | PASS (0.17) | PASS (-0.00 [-0.00, -0.00]) | +0.42 | 0: -0.02, 1: +0.04, 2: +0.05, 3: -0.00 | **FAIL** |
| self_attribution | FAIL (0.20) | FAIL (20.19) | FAIL (+0.10 [-0.03, +0.29]) | +0.02 | 0: +0.10, 1: +0.08, 2: -0.05, 3: +0.10 | **FAIL** |

Self-attribution β (credit after a hit) and γ (blame after a miss), log-odds, by prompted level: 0: β=+0.05, γ=-0.04; 1: β=+0.07, γ=-0.01; 2: β=-0.02, γ=+0.03; 3: β=+0.12, γ=+0.02

## Stronger-model repeat

Mean item score by condition and task (rational = 0 on every task).

|                                                |   claude-haiku-5-5 |   claude-sonnet-5-5 |
|:-----------------------------------------------|-------------------:|--------------------:|
| ('conservatism@3', 'conservatism')             |              0.472 |               0.805 |
| ('conservatism@3', 'overconfidence')           |             -0.022 |              -0.094 |
| ('conservatism@3', 'representativeness')       |             -0.014 |               0.007 |
| ('conservatism@3', 'self_attribution')         |              0.086 |               0     |
| ('overconfidence@3', 'conservatism')           |              0     |               0.08  |
| ('overconfidence@3', 'overconfidence')         |             -0.003 |               0.432 |
| ('overconfidence@3', 'representativeness')     |             -0     |              -0.481 |
| ('overconfidence@3', 'self_attribution')       |              0.08  |               0.025 |
| ('rational', 'conservatism')                   |              0.001 |               0.065 |
| ('rational', 'overconfidence')                 |             -0.001 |               0.114 |
| ('rational', 'representativeness')             |              0     |              -0.107 |
| ('rational', 'self_attribution')               |              0.097 |              -0.058 |
| ('representativeness@3', 'conservatism')       |              0.001 |               0.233 |
| ('representativeness@3', 'overconfidence')     |             -0.018 |              -0.07  |
| ('representativeness@3', 'representativeness') |             -0.1   |               0.158 |
| ('representativeness@3', 'self_attribution')   |              0.108 |               0.25  |
| ('self_attribution@3', 'conservatism')         |              0     |               0.301 |
| ('self_attribution@3', 'overconfidence')       |             -0.004 |               0.047 |
| ('self_attribution@3', 'representativeness')   |             -0     |               0.027 |
| ('self_attribution@3', 'self_attribution')     |              0.101 |               0.58  |

## Usage

| model | input tokens | output tokens |
|---|---|---|
| claude-haiku-5-5 | 1,300,362 | 477,156 |
| claude-sonnet-5-5 | 726,413 | 273,880 |
