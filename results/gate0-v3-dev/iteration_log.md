# Gate 0 v3: dev iteration log (before any v3 registration)

All runs used **dev items only**:
- seed 20261132 for iteration;
- seed 20261133 for validation, which was not looked at until the design was fixed.

The v3 registered items will use `item_seed` 20261201.

All runs had thinking off. Dev spend on 2026-11-03: 15,088 Sonnet 5.5 calls (13.1M input / 0.35M output tokens) and 7,536 Haiku 5.5 calls (6.5M / 0.19M).

## Why v2 attempt 1 failed (diagnosed on dev items)

- **Neutral agents ignore stated errors.** They answer with a rough average anchored on the recent price. Scores normalised by Bayes weights therefore start far from zero.
- **Representativeness items could not show the bias.** The "non-streak" variant had a *larger* final jump than the streak, and models ignore older reports. Answers were identical in both variants.
- **Self-attribution feedback was ignored.** A single confirmed/not-confirmed line changed neither the value nor the confidence.
- **Conservatism and overconfidence overlapped by construction.** Weighting the private estimate more lowers the report weight, which raised the conservatism score.

## v3 design changes (items and scores)

| primitive | variants | score (rational = 0) |
|---|---|---|
| conservatism | all three reports ±δ together, at a zigzag history (no streak, report marked "new this round"); whole price history ±δ | log((s_R + s_price) / s_R): news underweighted vs the market's prior |
| representativeness | same newest report reached by a rising vs a falling streak | (rising − falling − rational) / (2·span) |
| overconfidence | private estimate ±δ, consensus ±δ | log(s_p/s_c) − log(w_p/w_c) |
| self-attribution | no record / "3 of your last 3 estimates borne out by the next price" / "0 of 3" × private estimate ±δ | β − γ (unchanged) |

Other changes:
- δ = 0.12 (v2: 0.08).
- The market shows the same track-record line in place of v2's single previous-estimate line.
- A Spearman float fix: an exact ρ = 0.8 had computed as 0.7999.

## Persona wording (final candidate = `prompts/*.txt`)

Each persona's wording is tied to the channel it acts on:
- conservatism: new report vs the price the market had already set;
- representativeness: streaks only ("reports that go back and forth suggest nothing to you");
- overconfidence: own analyst vs the analyst consensus;
- self-attribution: credit when the price bears out your estimates, blame the market otherwise.

No wording contains a number or names a construct.

What was tried and rejected:
- Conservatism anchored on "analysts' view": it pulled the consensus and loaded on overconfidence.
- Self-attribution "lean on it more" or "trust it far more than its stated error": it raised the private weight everywhere (overconfidence spillover).
- An analyst-track-record framing: neutral agents learned from it symmetrically.
- A 5-estimate record: no gain.

## Results, Sonnet 5.5, final candidate (12 items × 2 repeats per seed)

| primitive | iteration seed | validation seed | pooled (24 items) |
|---|---|---|---|
| conservatism | PASS | PASS | PASS (levels 0.05 / 0.56 / 1.31 / 3.10, cross 0.04) |
| representativeness | PASS | PASS | PASS (0.02 / 0.02 / 0.17 / 0.52, cross 0.15) |
| overconfidence | PASS | FAIL (rational-null CI too wide) | PASS (1.16 / 1.37 / 2.80 / 3.94, cross 0.11) |
| self-attribution | PASS | FAIL (weaker effect; cross 0.35, null) | FAIL: rational null only (−0.15 [−0.29, 0.00] vs ±0.22); monotone (−0.10 / 0.10 / 0.34 / 0.79), cross 0.23 |

Haiku 5.5 on the validation seed passes no primitive. Only overconfidence responds in step with the persona level, and its answers are noisy and price-anchored.

## Round 2: self-attribution fix (same day)

Diagnosis: on both dev seeds the rational persona cut its trust in its analyst after a "0 of 3" record more than neutral did, so self-attribution failed the rational null. Its level-3 persona also raised the private weight when no record was shown (overconfidence cross-loading 0.33 on the validation seed).

Changes:
- The contradicted record is "1 of 3" (was 0 of 3). Neutral then learns symmetrically (β − γ ≈ 0).
- Every self-attribution level ends "When no track record is shown, you weigh your own analyst as you normally would." This removed the no-record leak (cross 0.33 → 0.02).
- Level 1 softened ("a little more credit"), level 2 "clearly more", level 3 "you now trust it far more than anything else you see" (effect 0.84 → 1.35).

Rejected:
- A shared prompt sentence saying stated errors don't change with prices: it widened the rational − neutral gap.
- A log-ratio self-attribution scorer: noisier.
- A share-based overconfidence scorer: it narrowed the CI but compressed the margin just as much.

Both earlier dev seeds were used for this round's tuning, so the final design was validated on a **fresh dev seed, 20261134** (full grid, Sonnet 5.5, 12 items × 2 reps, 3,072 calls, 0 failures):

| primitive | levels 0 / 1 / 2 / 3 (rational) | max cross | rational − neutral vs margin | verdict |
|---|---|---|---|---|
| conservatism | 0.10 / 0.43 / 1.36 / 3.29 (0.06) | 0.05 | −0.04 [−0.08, −0.01] vs ±0.80 | PASS |
| representativeness | 0.02 / 0.03 / 0.19 / 0.53 (0.02) | 0.07 | −0.01 [−0.02, 0.00] vs ±0.13 | PASS |
| overconfidence | 0.90 / 1.21 / 2.80 / 3.81 (1.10) | 0.03 | +0.20 [+0.08, +0.32] vs ±0.73 | PASS |
| self-attribution | −0.14 / 0.25 / 0.46 / 0.90 (−0.14) | 0.16 | +0.00 [−0.19, +0.18] vs ±0.26 | PASS |

Round-2 dev spend: about 6,500 further Sonnet calls.
