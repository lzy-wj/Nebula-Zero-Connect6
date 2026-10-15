# Three-generation evidence ledger

This note records the comparison rules behind the headline numbers in the Chinese and
English READMEs. Machine-specific benchmark logs and deployment topology are intentionally
excluded from the public repository.

## Derived comparisons

- Parameter reduction: `1 - 1,651,035 / 11,796,004 = 86.00%`.
- Parameter ratio: V1 has `11,796,004 / 1,651,035 = 7.14×` as many parameters as V3.
- The V2 candidate finished its strength gate at 49–51 and was not promoted.
- At the `through-88` milestone, V3 scored 98-28-2 against the V1 `best` checkpoint at
  400 simulations and 96-30-2 at 1200 simulations. Both tests used 64 common openings
  with color reversal.

## V3 training snapshot through round 452

The public release is `through-452`. The figures below were frozen from the append-only
training state. Generated totals include both committed rounds and seven quality-rejected
attempts; training-visible totals exclude rejected data.

| Aggregate | Value |
| --- | ---: |
| Completed / quality-rejected actor attempts | 453 / 7 |
| All generated games | 1,133,440 |
| All generated turn samples | 59,135,552 |
| Training-visible games / turns | 1,116,192 / 58,214,613 |
| Training-visible black / white / draw | 537,966 / 488,961 / 89,265 |

The released replay is the exact 12-generation training window consumed by the published
model: 29,568 games and 1,713,854 turn samples. Its manifest and hashes are published with
the dataset; release packaging is documented in [PAIR_NATIVE_V3_RELEASE.md](PAIR_NATIVE_V3_RELEASE.md).

## Interpretation limits

- V2 is a deployment study rather than a promoted playing-strength generation.
- The public V3 model is `through-452`, while the V1 comparison was run at the earlier
  `through-88` milestone. The README labels this explicitly.
- Match scores are conditional on the named checkpoints, search budgets, and opening
  protocol. They are not Elo ratings or guarantees against unrelated engines.
