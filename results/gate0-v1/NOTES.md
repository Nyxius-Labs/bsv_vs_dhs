# Gate 0 v1: registered result

- Registration: `6b633568023a8b56586f051e2ea55f05f9caa2d3492e6a6bc9c5de280174b262` (commit `407df6c`, tag `gate0-registered`).
- Stage run: measurement on `claude-haiku-5-5`, 2026-10-29. 1,368 new API calls, 0 failures.
- Files: `report.md`, `analysis.json` and `scores.csv` are copied unchanged from `data/registered/`, which is gitignored.

## Outcome

Only conservatism passes all three checks. Representativeness, overconfidence and self-attribution fail. Neither BSV nor DHS has both of its primitives, so neither theory advances, and per the registration the market, freeze and full-study stages were not run.

## Disclosed deviations and report errata

1. **Duplicate self-attribution item.** Self-attribution items 1 and 3 are identical (track record 5 of 12, same direction). The item generator draws from a small space and does not deduplicate, so there are 9 distinct items, and item 1 is counted twice in the item-level bootstrap. The 72 cache hits in this run are item 3 being served item 1's responses. Items were not regenerated after the result was seen. Re-running `analyze` with item 3 dropped changes no verdict: representativeness additionally fails near-diagonal, and all verdicts are unchanged.
2. **Wording of the rational-null threshold in `report.md`.** The report says "rational 95% CI covers 0". The registered rule, implemented in `checks.py` and applied here, is an equivalence test: the rational arm's 95% CI must lie within ±`rational_null_margin` (0.25) × the primitive's own top-vs-neutral effect. The `rational null` column reflects the registered rule.
3. **Usage table in `report.md`.** It sums every response in the local cache, including pre-registration dev smoke tests, not just this run.

## Stronger-model repeat (Sonnet 5.5), run after the v1 tag

- Files: `strong/scores.csv`, plus `report_with_strong.md`, which is the regenerated report with the Haiku-vs-Sonnet table.
- Calls: 1,467 new, 0 final failures. 41 calls hit `max_tokens` (4,096, adaptive thinking) and 1 was refused; all 42 were retried with a fresh nonce and answered, so no item is missing.
- Design limit: the registered strong design has only the rational arm and level 3, with no neutral persona and no levels 1–2, so the three checks cannot be run on Sonnet. It is descriptive only.

Own-task scores, level 3 vs rational (raw score units):

| task | Haiku L3 | Haiku rational | Sonnet L3 | Sonnet rational |
|---|---|---|---|---|
| conservatism | 0.472 | 0.001 | 0.805 | 0.065 |
| representativeness | −0.100 | 0.000 | 0.158 | −0.107 |
| overconfidence | −0.003 | −0.001 | 0.432 | 0.114 |
| self-attribution | 0.101 | 0.097 | 0.580 | −0.058 |

Sonnet moves every primitive's own task in the predicted direction at level 3. At level 3, Sonnet's self-attribution β = +0.47 and γ = −0.11.

Sonnet also shows large off-task shifts. Representativeness@3 and self-attribution@3 raise the conservatism score to 0.23 and 0.30. Overconfidence@3 moves the representativeness score to −0.48. Sonnet's rational arm is also not at 0 on three tasks.

## Hand check of the representativeness scoring

The scoring is not a sign error.
- On constructed answers for item 5, overweighting the last three draws ×2 scores +0.97, ignoring them scores −1.00, and the rational answer scores 0.00.
- Haiku gives the exact Bayesian posterior in 87 of the 90 representativeness-persona responses, all of which score 0.
- The negative level means come from three responses:
  - Item 3, level 3: the answer is 11/1.4/87.7 against a rational 80/10/10. That is roughly the rational answer with urns X and Z swapped, and the two-weight solve turns it into −3.00.
  - Item 5, level 2, two repeats with the same answer: 14.3/77.1/8.6 against a rational 3.2/87.1/9.7, i.e. less weight on recent draws, scoring −0.59.
- The Spearman of −0.8 is therefore three slips ranked across four near-zero level means, not a dose response.
