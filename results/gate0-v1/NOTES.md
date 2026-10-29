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
