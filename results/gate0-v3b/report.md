# Gate 0 v3b: second test format

Registration: `v3b` `b1b43ef6b833aa60c165fbd6648b68a8ba758e36e6583424a074cc4295a19c2e`

Decision rule: a primitive transfers if it passes all three v3 checks on format B (v3 thresholds).

| primitive | format A (v3) | format B: monotonic (rho) | near-diagonal (max) | rational - neutral [CI], +/-margin | level means 0/1/2/3 | rational vs Bayes | format B verdict |
|---|---|---|---|---|---|---|---|
| conservatism | PASS | PASS (1.00) | PASS (0.04) | PASS (-0.04 [-0.08, -0.00], +/-0.18) | -0.02 / +0.03 / +0.21 / +0.68 | -0.06 [-0.13, -0.00] | **PASS** |
| representativeness | PASS | FAIL (0.80) | FAIL (8.93) | PASS (-0.00 [-0.00, +0.00], +/-0.00) | +0.00 / -0.02 / +0.01 / +0.02 | -0.00 [-0.00, +0.00] | **FAIL** |
| overconfidence | PASS | PASS (1.00) | FAIL (6.22) | FAIL (-0.01 [-0.02, +0.00], +/-0.02) | -0.06 / -0.05 / -0.05 / +0.02 | -0.06 [-0.08, -0.05] | **FAIL** |

Cross-loadings on format B (|effect of the column persona at level 3 on the row score| / own effect, SD units):

| persona at level 3 | conservatism | representativeness | overconfidence |
|---|---|---|---|
| conservatism | own | 0.04 | 0.01 |
| representativeness | 8.93 | own | 0.15 |
| overconfidence | 6.22 | 0.00 | own |

Scored cells: 1260 of 1260 (100.0%).
