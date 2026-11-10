# Gate 0 report

Registration: `v3` `084503d94ffd190a72d69d25e9368c06b8e1b7b824224cec558b8365fe5b24f7`

## Measurement stage

Checks: Spearman ≥ 0.8 plus a paired top-vs-neutral CI above 0; max cross-loading < 0.25; rational null: the 95% CI of (rational − neutral), paired over items, lies within ±0.25 × the primitive's own (top − neutral) effect. The last column is the rational arm's level against the Bayesian benchmark (descriptive).

Verdicts by model and thinking setting (the registered setting is `off`):

| primitive | claude-sonnet-5-5 / off |
|---|---|
| conservatism | PASS |
| representativeness | PASS |
| overconfidence | PASS |
| self_attribution | FAIL |

Market model by the registered rule (`most_theories_then_primitives_then_order`): **claude-sonnet-5-5**, thinking `off`.

### claude-sonnet-5-5, thinking off

| primitive | monotonic (ρ) | near-diagonal (max) | rational − neutral (mean [CI], ±margin) | own effect (SD) | level means | verdict | rational vs Bayes |
|---|---|---|---|---|---|---|---|
| conservatism | PASS (1.00) | PASS (0.03) | PASS (+0.01 [-0.02, +0.04], ±0.82) | +8.93 | 0: +0.05, 1: +0.58, 2: +1.43, 3: +3.35 | **PASS** | +0.06 [+0.04, +0.09] |
| representativeness | PASS (1.00) | PASS (0.12) | PASS (+0.00 [-0.00, +0.01], ±0.12) | +7.22 | 0: +0.03, 1: +0.04, 2: +0.21, 3: +0.52 | **PASS** | +0.03 [+0.02, +0.04] |
| overconfidence | PASS (1.00) | PASS (0.07) | PASS (+0.22 [+0.08, +0.36], ±0.75) | +4.70 | 0: +1.04, 1: +1.30, 2: +3.00, 3: +4.05 | **PASS** | +1.25 [+0.97, +1.54] |
| self_attribution | PASS (1.00) | PASS (0.01) | FAIL (-0.20 [-0.33, -0.06], ±0.28) | +3.51 | 0: -0.01, 1: +0.17, 2: +0.39, 3: +1.10 | **FAIL** | -0.21 [-0.32, -0.09] |

Self-attribution β (rise in the private-estimate weight after a confirmation) and γ (fall after a contradiction), in units of its rational weight, by prompted level: 0: β=+0.12, γ=+0.13; 1: β=+0.24, γ=+0.07; 2: β=+0.44, γ=+0.05; 3: β=+1.11, γ=+0.01

## Identification stage

Per-arm means over seeds (unmasked runs).

| arm      |   momentum_h |   dispersion_interaction |   pead_pos |   pead_neg |   pead_level |   pead_gap |   optimism_anomaly |   return_vol |   autocorr_1 |   turnover |   mispricing_rmse |
|:---------|-------------:|-------------------------:|-----------:|-----------:|-------------:|-----------:|-------------------:|-------------:|-------------:|-----------:|------------------:|
| bsv      |        0.028 |                   -0.016 |      0.378 |      0.233 |        0.305 |      0.144 |             -0.006 |        0.045 |        0.367 |      1.368 |             0.066 |
| dhs      |       -0.12  |                   -0.041 |      0.266 |      0.052 |        0.159 |      0.214 |             -0.006 |        0.06  |       -0.044 |      1.65  |             0.047 |
| joint    |       -0.056 |                   -0.038 |      0.316 |      0.088 |        0.202 |      0.228 |             -0.006 |        0.052 |        0.097 |      1.414 |             0.048 |
| placebo  |       -0.064 |                   -0.029 |      0.267 |     -0.022 |        0.122 |      0.289 |             -0.007 |        0.052 |        0.043 |      0.765 |             0.045 |
| rational |       -0.05  |                   -0.026 |      0.264 |     -0.043 |        0.11  |      0.307 |             -0.006 |        0.05  |        0.062 |      0.694 |             0.044 |

Paired contrasts vs placebo: mean [seed-bootstrap CI]. Reported as direction and ordering; no winner is declared.

| arm | pead_level | momentum_h | dispersion_interaction | pead_gap |
|---|---|---|---|---|
| bsv | +0.183 [+0.071, +0.299] (n=10) | +0.092 [+0.060, +0.121] (n=10) | +0.014 [-0.008, +0.035] (n=10) | -0.145 [-0.306, +0.037] (n=10) |
| dhs | +0.037 [-0.033, +0.120] (n=10) | -0.056 [-0.068, -0.044] (n=10) | -0.011 [-0.019, -0.003] (n=10) | -0.076 [-0.210, +0.046] (n=10) |
| joint | +0.080 [+0.017, +0.147] (n=10) | +0.008 [+0.000, +0.014] (n=10) | -0.008 [-0.019, +0.003] (n=10) | -0.061 [-0.168, +0.037] (n=10) |
| rational | -0.012 [-0.051, +0.029] (n=10) | +0.014 [+0.010, +0.017] (n=10) | +0.004 [+0.001, +0.006] (n=10) | +0.018 [-0.029, +0.070] (n=10) |

Momentum manipulation check (momentum_h: profile slope at horizon 4, level CI > 0). Reported, not gating.

| arm | mean [CI] | momentum present |
|---|---|---|
| bsv | +0.028 [-0.030, +0.087] (n=10) | FAIL |
| dhs | -0.120 [-0.171, -0.072] (n=10) | FAIL |
| joint | -0.056 [-0.103, -0.012] (n=10) | FAIL |
| rational | -0.050 [-0.100, -0.003] (n=10) | FAIL |
| placebo | -0.064 [-0.113, -0.018] (n=10) | FAIL |

Horizon profiles: turning point = horizon where the seed-mean profile first changes sign (interpolated), 'none' if it does not turn within 12 rounds. CI: seed bootstrap over the whole curve, among the draws in which a sign change occurs.

| arm | profile_all | profile_high | profile_low |
|---|---|---|---|
| bsv | 4.6 [3.4, 5.9], found in 100% of draws | 4.3 [3.2, 6.5], found in 100% of draws | 5.1 [3.3, 6.8], found in 100% of draws |
| dhs | none [–], found in 0% of draws | none [11.0, 11.0], found in 0% of draws | none [1.0, 8.0], found in 17% of draws |
| joint | 1.1 [1.0, 3.3], found in 61% of draws | none [1.0, 11.7], found in 2% of draws | 4.1 [1.2, 7.7], found in 99% of draws |
| rational | none [1.0, 4.0], found in 47% of draws | none [1.0, 11.9], found in 3% of draws | 3.0 [1.1, 9.3], found in 95% of draws |
| placebo | none [1.0, 2.8], found in 13% of draws | none [1.0, 11.9], found in 1% of draws | 2.0 [1.1, 8.6], found in 90% of draws |

Turning point, arm − placebo (rounds):

| arm | profile_all | profile_high | profile_low |
|---|---|---|---|
| bsv | none [+2.0, +4.8], found in 13% of draws | none [-7.9, +3.1], found in 1% of draws | +3.1 [-2.4, +4.0], found in 90% of draws |
| dhs | none [–], found in 0% of draws | none [–], found in 0% of draws | none [-6.3, +4.9], found in 17% of draws |
| joint | none [+0.6, +2.5], found in 13% of draws | none [+0.1, +0.5], found in 0% of draws | +2.1 [-0.9, +3.8], found in 90% of draws |
| rational | none [+0.4, +3.5], found in 13% of draws | none [-4.2, +0.5], found in 0% of draws | +1.1 [+0.2, +3.5], found in 90% of draws |

Mean profile (all assets), slope by horizon:

| arm | h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8 | h9 | h10 | h11 | h12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bsv | +0.07 | +0.08 | +0.07 | +0.03 | -0.02 | -0.07 | -0.11 | -0.15 | -0.17 | -0.19 | -0.21 | -0.22 |
| dhs | -0.05 | -0.09 | -0.11 | -0.12 | -0.13 | -0.14 | -0.16 | -0.17 | -0.18 | -0.18 | -0.19 | -0.18 |
| joint | +0.00 | -0.02 | -0.03 | -0.06 | -0.08 | -0.11 | -0.14 | -0.16 | -0.17 | -0.18 | -0.18 | -0.18 |
| rational | -0.00 | -0.02 | -0.04 | -0.05 | -0.06 | -0.08 | -0.09 | -0.10 | -0.11 | -0.11 | -0.11 | -0.11 |
| placebo | -0.01 | -0.04 | -0.05 | -0.06 | -0.07 | -0.09 | -0.10 | -0.11 | -0.12 | -0.13 | -0.13 | -0.13 |

Seating: agents at discrete prompted levels whose shares match the measured distribution (each agent's recovered draw binned to the nearest level mean). Realized recovered intensity, unmasked runs:

| arm | primitive | level counts (1/2/3) | recovered mean ± SD [min, max] |
|---|---|---|---|
| bsv | conservatism | 44/31/45 | +1.93 ± 1.25 [+0.36, +4.52] |
| bsv | representativeness | 37/44/39 | +0.26 ± 0.21 [-0.00, +0.67] |
| dhs | overconfidence | 34/55/31 | +2.94 ± 1.27 [+0.40, +5.11] |
| joint | conservatism | 44/31/45 | +1.93 ± 1.25 [+0.36, +4.52] |
| joint | overconfidence | 34/55/31 | +2.94 ± 1.27 [+0.40, +5.11] |
| joint | representativeness | 37/44/39 | +0.26 ± 0.21 [-0.00, +0.67] |

## Masking

- pead_level: joint: masked − unmasked +0.042 [-0.000, +0.090] (n=10); joint − placebo, both masked +0.112 [+0.050, +0.175] (n=10); placebo: masked − unmasked +0.009 [-0.026, +0.047] (n=10)
- momentum_h: joint: masked − unmasked -0.019 [-0.026, -0.013] (n=10); joint − placebo, both masked -0.006 [-0.014, +0.002] (n=10); placebo: masked − unmasked -0.005 [-0.013, +0.005] (n=10)
- dispersion_interaction: joint: masked − unmasked -0.002 [-0.012, +0.007] (n=10); joint − placebo, both masked -0.009 [-0.018, -0.000] (n=10); placebo: masked − unmasked -0.001 [-0.007, +0.005] (n=10)
- pead_gap: joint: masked − unmasked +0.072 [+0.015, +0.125] (n=10); joint − placebo, both masked -0.027 [-0.075, +0.029] (n=10); placebo: masked − unmasked +0.038 [-0.012, +0.077] (n=10)

## Gate

A theory advances if both its primitives pass measurement and, on at least one of its own moments (BSV: pead_level, momentum_h; DHS: dispersion_interaction, pead_gap), its effect beats placebo (CI lower bound > 0) and the placebo rule holds: (placebo − rational) < 0.5 × (theory − rational) with theory − rational > 0, on common seeds. A BSV pass on momentum shows the induced bias produces momentum, not that BSV beats DHS.

| theory | both primitives pass | moment | effect > placebo | theory − rational / placebo − rational | placebo rule | moment passes | advances |
|---|---|---|---|---|---|---|---|
| BSV | PASS | pead_level | PASS +0.183 [+0.071, +0.299] (n=10) | +0.195 / +0.012 | PASS | PASS | **YES** |
|  |  | momentum_h | PASS +0.092 [+0.060, +0.121] (n=10) | +0.078 / -0.014 | PASS | PASS |  |
| DHS | FAIL | dispersion_interaction | FAIL -0.011 [-0.019, -0.003] (n=10) | -0.015 / -0.004 | FAIL | FAIL | **NO** |
|  |  | pead_gap | FAIL -0.076 [-0.210, +0.046] (n=10) | -0.093 / -0.018 | FAIL | FAIL |  |

**Full study proceeds: NO**


## Freeze-and-evaluate (feasibility only)

Frozen configuration: `bsv` (hash `ccb3a6592ee44eec`), held-out anomaly: analyst_optimism.

- bsv − rational: +0.000 [-0.001, +0.002] (n=5)
- bsv − placebo: +0.000 [-0.001, +0.002] (n=5)

## Usage

API calls billed under this registration (since `gate0 register`; dev runs: whole cache).

| model | calls | input tokens | output tokens |
|---|---|---|---|
| claude-fable-5-1 | 432 | 381,600 | 48,920 |
| claude-haiku-5-5 | 432 | 381,168 | 11,056 |
| claude-opus-5-5 | 432 | 381,600 | 54,967 |
| claude-sonnet-5-5 | 130,680 | 161,162,253 | 8,599,754 |
