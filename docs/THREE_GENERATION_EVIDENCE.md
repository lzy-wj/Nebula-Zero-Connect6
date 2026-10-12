# Three-generation evidence ledger

This note records the provenance and comparison rules behind the headline numbers in the
Chinese and English READMEs. It exists to keep unlike hardware, search budgets, and match
protocols from being silently mixed.

## Source map

| Claim | Primary repository evidence |
| --- | --- |
| V1 11.796M parameters | `experiments/pair_native/RESULTS.md`, historical ModelScope baseline tournament |
| V2 2.16× candidate inference | `experiments/inference_vnext/RESULTS.md`, trained-model gate |
| V2 13,200 games / 18.5 min / 49–51 gate | `experiments/inference_vnext/RESULTS.md`, bootstrap follow-up |
| V3 1.651M parameters and early throughput | `experiments/pair_native/RESULTS.md`, native-pair cold-start result |
| V3 optimized 4/6-H20 throughput | `experiments/pair_native/RESULTS.md`, CPU/GPU saturation follow-up |
| V3 paired matches against V1 | `experiments/pair_native/RESULTS.md`, final 100-round model versus historical baselines |
| Released replay window | `docs/PAIR_NATIVE_V3_RELEASE.md` and the [public dataset manifest](https://huggingface.co/datasets/Lazyshu/Nebula-Zero-Connect6-Replay/blob/main/dataset_manifest.json) |

## Derived comparisons

- Parameter reduction: `1 - 1,651,035 / 11,796,004 = 86.00%`.
- Parameter ratio: V1 has `11,796,004 / 1,651,035 = 7.14×` as many parameters as V3.
- V2 inference speedup: `6.331 ms / 2.932 ms = 2.16×`.
- V2 bootstrap self-play speedup: `11.9 / 6.3 = 1.89×`.
- V3 stored-turn throughput: `1,643.96 / 425.8 = 3.86×`. This comparison includes
  scaling from four to six H20 GPUs and is labeled with both GPU counts.

## V3 training snapshot through round 452

The public release is `through-452`. The figures below were frozen from the append-only
training state. Actual generation cost includes both `completed_rounds.round_index <= 452`
and `rejected_attempts.round_index <= 452`; learner and gate cost uses
`promotions.through_round <= 452`.

| Aggregate | Value |
| --- | ---: |
| Completed / quality-rejected actor attempts | 453 / 7 |
| All generated games | 1,133,440 |
| All generated turn samples | 59,135,552 |
| Training-visible games / turns | 1,116,192 / 58,214,613 |
| Training-visible black / white / draw | 537,966 / 488,961 / 89,265 |
| Actor active time, including rejected retries | 62.0939 h |
| Learner active time | 31.3042 h |
| Gate active time | 17.1247 h |
| Sum of active phases | 110.5228 h |
| Allocated GPU-hours, lower estimate | 464.4019 GPU·h |
| Calendar span including pauses | 293.2289 h |

The GPU-hour estimate is
`sum(all actor-attempt duration × actor worker count) + learner duration + gate duration`.
It includes the seven recorded quality-rejected attempts but remains a lower estimate because
it excludes failed attempts without complete state records and all earlier research runs. The
453 committed rounds used seven actor workers for 449 rounds, two workers for two rounds, and
one worker for two rounds. Their base-search distribution was 340 / 68 / 36 / 9 rounds at
400 / 600 / 800 / 1200 simulations.

The snapshot starts at 2026-09-28 06:57:55 CST and ends at 2026-10-10 12:11:39 CST. It is
the final continuation segment containing the release—not a claim about total project cost.

## Interpretation limits

- V2 did not pass its strength gate, so it is a performance study rather than a promoted
  playing-strength generation.
- The public V3 model is `through-452`, while the published V1 comparison was run at the
  earlier `through-88` milestone. The README labels this explicitly.
- The H20 64-simulation throughput study measures implementation efficiency. Production
  self-play at 400–1200 simulations is a different workload.
- Match scores are conditional on the named checkpoints, search budgets, and opening
  protocol. They are not Elo ratings or guarantees against unrelated engines.
