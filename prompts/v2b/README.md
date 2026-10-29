# Attempt-2 persona wording (reserved, unused)

These four files would replace `prompts/{conservatism,representativeness,overconfidence,self_attribution}.txt` in v2 attempt 2. They are used only if attempt 1 fails a primitive. They were written and committed before any attempt-1 result existed, so they cannot be tuned to it. The git history of this file is the evidence.

How they differ from the attempt-1 (v1) text:
- Each line refers only to the information channel its primitive acts on:
  - conservatism: earlier view vs. new reports;
  - representativeness: streaks of reports;
  - overconfidence: own analyst vs. consensus;
  - self-attribution: price moves after one's own estimate.
- This is meant to reduce the cross-task spillover Sonnet showed in v1.
- They describe a trait, never a calculation, and never name the construct.
- Attempt 2 may change only these files. Items, seeds, thresholds and models stay as registered in attempt 1.
