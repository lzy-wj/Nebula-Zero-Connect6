# Pair-native prototype results

These are screening results, not a production-strength or TensorRT claim.

## Strict audit of real self-play games

Input: six legacy generations (`gen_20.csv` through `gen_25.csv`), 300 games
per generation and 1,800 games total. Every move was replayed under the
Connect6 rules; no record was repaired.

| Metric | Result |
| --- | ---: |
| Rule-valid games with verified winner | 1,800 / 1,800 |
| Training-usable games | 1,799 / 1,800 |
| Black / white / draw | 1,045 / 753 / 2 |
| Mean / median stones | 44.92 / 41 |
| P90 / P95 stones | 69 / 83 |
| Duplicate complete games | 0 |
| Exact unique turn-state rate | 83.52% |
| D4-canonical unique turn-state rate | 79.34% |
| Valid / invalid per-stone policies | 80,854 / 1 |
| Turns ending after the first winning stone | 27 |

The one invalid label is in `gen_21.csv`, CSV row 237: at move index 8 the
played move is `p12` (index 224), but its saved search policy assigns no mass
to that move. The game itself and recorded winner are valid. This demonstrates
why game-rule validity and label validity must be separate gates.

Observed sparse-token coverage at turn boundaries:

| Capacity | Covered states |
| --- | ---: |
| K32 | 69.35% |
| K64 | 93.98% |
| K96 | 98.35% |

K64 is therefore not a complete standalone model even on these mature games.
The cold-start distribution may be longer and must be measured independently.

The legacy files contain sequential first- and second-stone policies. They do
not contain the joint unordered-pair visit distribution, so the audit reports
zero exact native-pair policy targets. They remain valid for rule analysis and
terminal value targets.

## H20 PyTorch architecture screen

Hardware/software: one NVIDIA H20, PyTorch 2.8.0+cu128, eager FP16. Model:
1.621M parameters, C192, K64, four sparse attention blocks, rank-32 symmetric
pair head. Timing includes all 361 queries, the complete unordered pair score
matrix, legality masking, GPU Top-64, and copying the selected pairs/scores and
value to CPU.

| Batch | Latency | Positions/s |
| ---: | ---: | ---: |
| 1 | 2.719 ms | 368 |
| 8 | 2.930 ms | 2,731 |
| 16 | 2.945 ms | 5,434 |
| 32 | 2.951 ms | 10,846 |
| 64 | 3.250 ms | 19,692 |
| 128 | 4.332 ms | 29,546 |

Without CPU output copies, batch 64/128 reached 21,933/34,201 positions/s.
Reducing pair rank from 32 to 16 or Top-K from 64 to 32 did not materially
change end-to-end latency. The next optimization target is the trunk/runtime
and a TensorRT graph, not further shrinking the pair factorization.

### Four-GPU scaling

The same batch-128 benchmark was launched concurrently on H20 GPU 0–3. Each
process owns one model and copies its compact Top-64 result to CPU.

| GPU | Positions/s |
| ---: | ---: |
| 0 | 29,275 |
| 1 | 29,519 |
| 2 | 29,507 |
| 3 | 29,378 |
| **Aggregate** | **117,679** |

Aggregate scaling is about 3.98× relative to the 29,546 positions/s single-GPU
screen. There is no material cross-GPU interference at this stage. This is
still inference-only; end-to-end self-play scaling depends on keeping every
GPU batch full and avoiding shared CPU/I/O bottlenecks.

## H20 Exact/Pair TensorRT rerun

The A800 numbers above are reference-only. `pyramid_wide_c256_d17` was rebuilt
from random weights on the target H20 with TensorRT 10.16.1, FP16, tactic level
5, three timing iterations, max batch 128, and all outputs copied to host.
Exact used zero auxiliary streams and Pair rank 16 used one.

| Batch | Exact latency | Exact pos/s | Pair latency | Pair pos/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.774 ms | 1,291 | 0.785 ms | 1,274 |
| 2 | 0.797 ms | 2,511 | 0.802 ms | 2,493 |
| 4 | 0.824 ms | 4,854 | 0.824 ms | 4,857 |
| 8 | 0.857 ms | 9,333 | 0.877 ms | 9,122 |
| 16 | 0.912 ms | 17,542 | 0.964 ms | 16,599 |
| 32 | 1.049 ms | 30,518 | 1.151 ms | 27,808 |
| 64 | 1.411 ms | 45,357 | 1.623 ms | 39,438 |
| 128 | 2.156 ms | 59,357 | 2.485 ms | 51,513 |

These are deployment-graph measurements, not strength measurements. The new
cold-start loop below uses the independently designed native unordered-Pair
model and does not consume the A800 replay or checkpoints. PyTorch/TensorRT
validation on 0/9/65/97/180-stone boards passed: Exact policy/value maximum
absolute errors were `4.2e-6`/`4.9e-5`; Pair second-policy Top-1 agreement was
100%, with factor and selected-value errors below `2.2e-3`/`2.5e-4`.

## Native-Pair MCTS and cold-start result

The C++ tactical scan originally checked every empty point separately for both
players. Replacing those four scans with two combined six-window scans and
removing an O(batch²) context lookup raised batch-128, 64-simulation H20 search
from about 19.4k to 26.3k simulations/s. Four-card generation-zero throughput
then reached 13.43 games/s and 793.7 stored turns/s, versus 6.21 games/s and
425.8 turns/s in the earlier implementation.

Generation zero used a random 1.651M-parameter checkpoint, central 9×9 uniform
first stones, 64 simulations, Top-16 joint Pair policies, Gumbel candidate
exploration, and `value_scale=0`.

| Data quality metric | Gen 0 | Gen 1 |
| --- | ---: | ---: |
| Games / turns | 1,024 / 60,504 | 1,024 / 66,377 |
| Black / white / draw | 526 / 498 / 0 | 511 / 513 / 0 |
| Mean / P95 stones | 117.2 / 163 | 128.6 / 199 |
| Unique searched states | 98.41% | 98.39% |
| D4-unique searched states | 98.30% | 97.96% |
| Decisive search-value targets | 5.17% | 4.74% |
| Four-H20 throughput | 13.43 games/s | 12.07 games/s |
| Stored-turn throughput | 793.7 turns/s | 782.7 turns/s |

All eight Parquet shards passed full rule, winner, board-payload, policy,
search-statistic, provenance-hash, and forced-opening audits. No draw or
post-terminal repair was used.

Short training used game-level held-out splits, D4 augmentation, full legal
joint-Pair cross entropy, phase-balanced sampling, and 10× weighting for
search-proven value targets. With exact sample-weighted aggregation, the
decisive-value sign accuracy on gen-0 held-out games rose from 64.8% to 75.5%,
but the decisive test is the paired arena:

| Paired color-swapped gate, 64 simulations, Top-16 | Result |
| --- | ---: |
| Gen-0 trained policy vs random initial, both value off | 79-49 (61.7%) |
| Gen-0 value on vs random initial value off | 68-60 (53.1%) |
| Gen-1 policy vs gen-0 policy, both value off | 164-92 (64.1%) |
| Gen-1 value on vs gen-0 value off | 171-85 (66.8%) |
| Gen-1 value on vs same Gen-1 weights value off | 145-110-1 (56.8%) |

The first random-data attempt that trained against terminal outcomes improved
search consistency but scored only 62-66 against initialization when value was
disabled. It was therefore rejected. Search-proven values plus decisive-label
weighting produced a policy that passed both generation-zero and generation-one
strength gates. The generation-zero value was still harmful and stayed off.
The generation-one direct on/off gate passed by 145-110 with one draw, so
`value_scale=1` is justified from generation one onward under this
64-simulation configuration.

A CPU random-candidate replacement produced 814.8 turns/s, 2.7% above Gumbel,
and beat initialization 77-51. It nevertheless lost 33-95 to the Gumbel-trained
model, so it is rejected for quality despite its small speed advantage.

## CPU/GPU saturation follow-up

Profiling found that the original worker consumed about 1.1 CPU cores while an
H20 was only 40–44% busy. Tactical scans are independent across games, so they
were moved to a deterministic OpenMP stage. Inference batches are now padded
to the next 16/32/64/128 bucket instead of always to 128.

| One-H20 configuration | GPU SM | Effective CPU | Games/s | Turns/s |
| --- | ---: | ---: | ---: | ---: |
| 1 worker, 1 CPU thread, fixed 128 | 40–44% | ~1.1 cores | 2.562 | 187.7 |
| 1 worker, 24 CPU threads, fixed 128 | 71–77% | ~20 cores | 3.464 | 253.7 |
| 1 worker, 24 CPU threads, 16/32/64/128 buckets | 70–77% | ~20 cores | 4.435 | 324.0 |
| 2 workers, 12 threads each, bucketed | 94–98% | ~19 cores | 3.159 | 235.7 |

The fixed-batch one-thread and 24-thread runs used the same 256 game seeds;
all 256 move sequences and winners were byte-for-byte identical. Parallelism
therefore changed wall time, not search semantics.

The final four-H20 run used 96 pinned physical cores and produced 1,024 fully
audited games in 66.17 seconds: **15.47 games/s and 1,145.5 turns/s**. Each GPU
held 70–77% SM utilization and each worker consumed about 19 CPU cores. This
is 46% more stored turns/s than the previous four-card gen-1 run. Forcing the
GPU counter to 95–98% with two workers reduced useful throughput because both
workers spent compute on small padded batches, so the one-worker/24-thread
layout is the measured optimum.

## Eight-H20 asynchronous layout

Training batch size was rerun on four isolated H20s against the same 59,025
training turns. Batch 256/512/1024/2048 reached 10.30k/11.48k/12.54k/12.09k
samples/s, respectively. Batch 1024 is therefore the measured learner default,
21.8% faster than 256. Candidate strength is still controlled by the arena;
training loss alone never promotes a model.

With GPUs 0–5 as actors, each assigned 256 games and 24 local physical CPU
cores, the production 64-simulation configuration generated 1,536 games and
113,245 turns in 68.89 seconds. All six shards passed the full audit.

| Six-H20 actor result | Value |
| --- | ---: |
| Games/s | 22.30 |
| Stored turns/s | 1,643.96 |
| Per-worker turns/s | 324–363 |
| GPU SM while searching | about 70–76% |
| Gain over four-H20 optimized run | 43.5% |

A seven-actor control used the same 1,536-game workload. It reached 1,648.9
turns/s versus 1,644.0 for six actors, only 0.3% faster, while GPU 6 became a
280 turns/s tail. It would also share NUMA-node-1 CPU capacity with the
learner in the asynchronous run. The measured production default therefore
remains six actors plus isolated learner and gate GPUs; maximizing nominal GPU
occupancy did not minimize iteration wall time.

Independent review rejected the original with-replacement opening sampler:
128 pairs from 81 central points created exact duplicate deterministic
observations and an invalidly narrow interval. The production gate now draws
64 unique central openings without replacement, uses 1.96 standard errors and
zero margin, and calculates standard error over paired scores rather than
treating both color-swapped games as independent.
As a symmetry check, the same checkpoint gated against itself on 64 unique
openings scored exactly 0.500 with paired standard error 0 and did not pass
the strict `lower_bound > 0.5` promotion rule.

The final post-review two-round 6+1+1 smoke generated 84 games and 10,374
turns, trained two candidates, ran two unique-opening paired gates, and
atomically promoted both under a deliberately permissive smoke-only threshold.
All 12 replay shards passed hashes, manifest/table counts, schemas, game
replays, winners, board payloads, policy targets, generation IDs, checkpoint
hashes, and engine/search provenance. The second actor round overlapped both
the first learner and gate. A completed-state resume re-audited the replay and
promotion hash chain and created no duplicate attempt.

## Current H20 400-simulation / Top-64 validation

The historical cold-start tables above used 64 simulations and Top-16. They
are retained for reproducibility only. The current production candidate uses
400 simulations and Top-64 for both actors and the paired gate.

On one H20, 128 contexts and the current sparse Pair model reached 28,858
opening simulations/s and 29,131 searched-pair simulations/s. Increasing the
MCTS CPU pool from 12 to 24 to 48 threads gave 28,686, 29,163, and 29,520
pair simulations/s respectively; 24 threads is retained because doubling the
CPU allocation for the final 1.2% would reduce room for the six concurrent
actors.

The six-H20 steady test used one worker per GPU, 128 games/GPU, 24 pinned CPU
threads, 16/32/64/128 inference buckets, Gumbel candidate exploration, and a
fixed accepted checkpoint:

| Metric | Result |
| --- | ---: |
| Games / turns | 768 / 55,319 |
| Wall time | 231.57 s |
| Aggregate games/s / turns/s | 3.317 / 238.89 |
| Valid shards / duplicate complete games | 6 / 0 |
| Mean searched simulations | 400.00 |
| Mean neural evaluations | 379.96 |
| Searched exact / D4 unique rate | 98.43% / 97.28% |
| Policy support mean / p90 | 32.81 / 64 |
| Black / white / draw | 420 / 317 / 31 |

The analyzer found 768 explicit forced-opening turns and 102 terminal singleton
turns; all remaining searched turns used exactly 400 simulations. Thus there
is no hidden early-stop-at-64 path in the current engine. The run is a data
quality and throughput baseline; model strength still requires the paired gate
after training.

## Historical ModelScope baseline tournament

The five public historical checkpoints were downloaded from
`Lazyshu/Nebula_Zero_Connect6`: supervised `model_gen_0`, intermediate
`model_gen_400`, late `model_gen_800` and `model_gen_801`, and `best`. They are
11.796M-parameter 17-plane C6TransNet models. Each was rebuilt as an H20 FP16
TensorRT engine with batch 64 and bound to its checkpoint by a SHA256 sidecar
manifest. The native checkpoint was the overnight accepted model from round
96, SHA256 prefix `82f914256865`.

All comparisons used the same 64 unique central openings, color swaps, no root
noise, and 128 games. Native MCTS searches one unordered Pair per turn; legacy
MCTS searches each stone separately. Therefore a nominal legacy budget of 400
or 1200 is approximately 800 or 2400 simulations per normal full turn. This is
intentionally conservative for the native model and matches how the historical
system was deployed.

| Legacy baseline | Native W-L-D at 400/400 | Native score (paired 95% CI) | Native W-L-D at 1200/1200 | Native score (paired 95% CI) |
| --- | ---: | ---: | ---: | ---: |
| model_gen_0 | 127-1-0 | 99.22% (97.69–100%) | 126-2-0 | 98.44% (96.29–100%) |
| model_gen_400 | 112-16-0 | 87.50% (82.15–92.85%) | 113-15-0 | 88.28% (83.05–93.51%) |
| model_gen_800 | 78-50-0 | 60.94% (54.59–67.29%) | 77-51-0 | 60.16% (54.31–66.00%) |
| model_gen_801 | 82-45-1 | 64.45% (58.50–70.41%) | 63-63-2 | 50.00% (42.84–57.16%) |
| best | 75-52-1 | 58.98% (52.59–65.38%) | 69-58-1 | 54.30% (47.98–60.62%) |

The strongest historical baseline was also evaluated with crossed budgets:

| Native per-turn | Legacy per-stone | Native W-L-D | Native score (paired 95% CI) |
| ---: | ---: | ---: | ---: |
| 400 | 400 | 75-52-1 | 58.98% (52.59–65.38%) |
| 1200 | 400 | 85-43-0 | 66.41% (59.84–72.97%) |
| 400 | 1200 | 57-70-1 | 44.92% (37.75–52.09%) |
| 1200 | 1200 | 69-58-1 | 54.30% (47.98–60.62%) |

The native model is decisively stronger at the 400 setting and remains clearly
ahead of gen-0/400/800 at 1200. The late large legacy models benefit more from
deep search: gen-801 and `best` are statistically tied with native at 1200.
Accordingly the defensible conclusion is higher low-budget efficiency, not an
unqualified win at every search depth. The complete JSON results are under
`$NEBULA_ARTIFACT_ROOT/baseline_tournament_20260923`.

## Completed 100-round replay distribution

The 400-simulation/Top-64 overnight run completed 100 actor rounds and produced
153,600 strictly audited games, 5,566,150 turns, and 10,963,560 stones. Mean
length was 71.38 stones (median 49, p90 132, p95 171); black/white/draw counts
were 92,331/57,822/3,447 (60.11/37.64/2.24%). The last ten rounds had already
moved to 54.60/42.26/3.14%, so the aggregate black skew was concentrated in the
middle of training rather than being constant.

Game counts alone hide the learner distribution:

| Length bucket | Games | Stored-turn mass | Black / white / draw |
| --- | ---: | ---: | ---: |
| <=40 stones | 35.97% | 16.86% | 41,682 / 13,571 / 0 |
| 41-80 | 36.63% | 28.87% | 31,579 / 24,677 / 0 |
| 81-140 | 18.91% | 27.59% | 14,327 / 14,712 / 0 |
| >140 | 8.50% | 26.68% | 4,743 / 4,862 / 3,447 |

In the final round, short games were 14.58% of games but only 5.40% of turns;
long plus very-long games supplied 66.61% of learner-visible turns. Short games
were also strongly black-skewed, while 81-140-stone games were almost exactly
color-balanced. Outcome and length therefore have to be controlled jointly.

## Natural-opening deep-search arena

The native and historical models searched their own black singleton from an
empty board. The first four complete turns used temperature 0.10 and all later
turns temperature zero. Each cell is eight seeds with color reversal (16 games),
so these are directional battlefield screens, not high-power estimates. Natural
color reversals do not share an opening; uncertainty uses the independent-game
binomial standard error rather than the paired promotion statistic.

| Opponent | Native / legacy budget | Native W-L-D | Score | Black / white wins | Mean stones |
| --- | ---: | ---: | ---: | ---: | ---: |
| best | 400 / 400 | 10-6-0 | 62.5% | 14 / 2 | 35.25 |
| best | 1200 / 1200 | 10-6-0 | 62.5% | 14 / 2 | 35.25 |
| best | 3000 / 3000 | 2-14-0 | 12.5% | 8 / 8 | 51.69 |
| best | 10000 / 10000 | 8-8-0 | 50.0% | 16 / 0 | 59.75 |
| gen801 | 400 / 400 | 10-6-0 | 62.5% | 14 / 2 | 40.38 |
| gen801 | 1200 / 1200 | 8-8-0 | 50.0% | 10 / 6 | 39.06 |
| gen801 | 3000 / 3000 | 6-10-0 | 37.5% | 10 / 6 | 51.25 |
| gen801 | 10000 / 10000 | 3-13-0 | 18.75% | 9 / 7 | 99.25 |

Legacy budgets are per stone, while native budgets are per full Pair turn. A
second screen approximately matched full-turn simulations by using half the
legacy number:

| Opponent | Native / legacy budget | Native W-L-D | Score |
| --- | ---: | ---: | ---: |
| best | 400 / 200 | 11-5-0 | 68.75% |
| best | 1200 / 600 | 11-5-0 | 68.75% |
| best | 3000 / 1500 | 8-8-0 | 50.0% |
| best | 10000 / 5000 | 8-8-0 | 50.0% |
| gen801 | 400 / 200 | 11-5-0 | 68.75% |
| gen801 | 1200 / 600 | 12-3-1 | 78.13% |
| gen801 | 3000 / 1500 | 11-5-0 | 68.75% |
| gen801 | 10000 / 5000 | 4-12-0 | 25.0% |

The defensible result is that native Pair is more efficient through the low and
middle search range, while historical gen801 remains stronger at the 10k deep
screen. The 10k best results are dominated by color and only three unique low-
temperature openings, so they do not support a fine-grained ranking.

## Dynamic replay-control smoke

With identical seeds and natural openings, a 64-game 400/400 control produced
37 black wins, 23 white wins, and four draws (61.7% black among decisive games)
in 124.0 seconds. Giving only white 526 simulations produced 29/33/2 (46.8%
black among decisive games) in 140.4 seconds. The 13.2% wall-time cost moved the
small sample into the requested 45-55% band; a three-generation feedback window
is used in production to reduce single-round overshoot.

On the latest three real generations (184,245 training rows), dynamic joint
sampling with a strict 4:1 maximum/minimum row-weight ratio changed
short/medium/long/very-long turn mass from 5.80/30.31/33.67/30.22% to
11.44/38.74/33.26/16.56%, and black/white/draw mass from 44.94/39.25/15.82%
to 45.81/46.39/7.80%. The safety cap deliberately takes priority over exactly
reaching the nominal 20/35/30/15 and 47.5/47.5/5 targets. Effective sample size
was 141,814 rows (77.0%). Validation remains unweighted.

## Opening-temperature collapse control (H20)

The accepted `1ef05b0da14c` checkpoint exposed a failure mode that aggregate
color control alone did not fix: at opening temperature 0.35, move 182 occupied
more than 90% of natural openings. Four 512-game, two-H20 shards used the same
game seeds, Top-64, 400 base simulations, early temperature 0.35 after the
singleton, and no forced opening:

| Opening temperature / budgets | B / W / D | Black decisive | Games/s | Opening Top-1 | Effective openings | Unique complete games |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.35, B400/W600 | 281 / 220 / 11 | 56.09% | 1.537 | 94.34% | 1.40 | 86.52% |
| 1.00, B400/W400 | 299 / 205 / 8 | 59.33% | 2.084 | 42.97% | 15.02 | 94.92% |
| 1.25, B400/W400 | 270 / 232 / 10 | 53.78% | 2.105 | 27.93% | 25.85 | 96.29% |
| 1.50, B400/W400 | 278 / 228 / 6 | 54.94% | 2.007 | 21.88% | 31.81 | 97.85% |

Increasing only white's budget helped color balance but retained the collapsed
opening and was 27.0% slower in wall time than the 1.25/equal-budget shard
(37.0% higher throughput for the latter). Opening temperature is therefore
the primary diversity/color actuator; per-color simulation budgets correct the
remaining residual.

An independent 1,536-game, eight-H20 confirmation at temperature 1.25 passed
the raw-manifest replay audit and the production guards. It produced
839/661/36 black/white/draw, 55.93% black among decisive games (95% normal
interval 53.42–58.45%), 29.43% opening Top-1, 24.94 effective openings, 93.23%
unique complete games, and 7.159 games/s. Production consequently uses base
temperature 1.25 with a dynamic 1.0–1.5 range, rejects Top-1 above 50% or
complete-game uniqueness below 90%, and quarantines/retries a rejected actor
round without exposing it to the learner.

## Final 100-round model versus historical baselines

The clean opening-controlled run completed all 100 actor/learner/gate rounds
and accepted the through-88 checkpoint, SHA256
`6dfdbb8911b5926cb49dd0a080273404aff6621c59009a4892b8b93e6aa66587`.
It generated 153,600 audited games with no rejected actor shard.  Across the
last 20 generations, decisive black/white results were 50.71/49.29%, opening
Top-1 was 8.36%, and 98.22% of complete games were unique.

The first battlefield screen used 64 color-swapped pairs (128 games), natural
model-selected openings, temperature 0.10 for the first four complete turns,
and no root noise.  Because the two colors do not share an opening, intervals
below use the independent-game binomial standard error.

| Historical opponent | Native / legacy budget | Native W-L-D | Native score (95% interval) | Black / white wins | Unique openings |
| --- | ---: | ---: | ---: | ---: | ---: |
| best | 400 / 400 | 97-29-2 | 76.56% (69.22–83.90%) | 91 / 35 | 11 |
| best | 1200 / 1200 | 117-11-0 | 91.41% (86.55–96.26%) | 71 / 57 | 8 |
| gen801 | 400 / 400 | 103-25-0 | 80.47% (73.60–87.34%) | 87 / 41 | 11 |
| gen801 | 1200 / 1200 | 67-61-0 | 52.34% (43.69–61.00%) | 117 / 11 | 8 |

The low-temperature gen801/1200 screen is inconclusive and extremely
black-dominated.  A second diagnostic therefore used 64 identical central
openings with color reversal; intervals use the paired common-opening standard
error.  This isolates play after the first stone rather than deployment-style
opening selection.

| Historical opponent | Native / legacy budget | Native W-L-D | Native score (paired 95% interval) | Black / white wins |
| --- | ---: | ---: | ---: | ---: |
| best | 400 / 400 | 98-28-2 | 77.34% (71.30–83.39%) | 92 / 34 |
| best | 1200 / 1200 | 96-30-2 | 75.78% (68.63–82.93%) | 82 / 44 |
| gen801 | 400 / 400 | 97-31-0 | 75.78% (69.61–81.95%) | 95 / 33 |
| gen801 | 1200 / 1200 | 95-33-0 | 74.22% (66.98–81.46%) | 83 / 45 |

Finally, the accepted model played the run's starting `1ef05b0da14c`
checkpoint directly under the same paired protocol.  At 400 simulations it
scored 97-30-1, or 76.17% (68.19–84.15%); at 1200 it scored 97-29-2, or
76.56% (69.59–83.54%).  Aggregate black/white wins were 64/63 and 57/69,
respectively, so the measured gain is not a first-player artifact.  The final
checkpoint is therefore decisively stronger than both its training starting
point and the historical ModelScope `best`/`gen801` models on common openings.
Natural deep-search opening play against gen801 remains the one setting where
the current sample does not establish superiority.

Raw results are stored under
`$NEBULA_ARTIFACT_ROOT/final_vs_modelscope_20260924`.

## Seven-actor shared-service continuation

The completed 100-round trace showed a median 278.6-second actor round versus
120.2 seconds for learning plus 160.3 seconds for the gate.  Training and gate
are causally serialized for a candidate, so reserving separate GPUs left one
service GPU idle without shortening the iteration.  The continuation therefore
uses GPUs 0–6 for self-play and shares GPU 7 between mutually exclusive learner
and gate jobs.  Each 1,792-game round divides exactly into 256 games per actor.

The first two production rounds took 263.3 and 273.4 seconds, or 6.85 and 6.59
games/s, compared with 5.44 games/s averaged over the preceding six-actor run.
Their decisive black rates were 50.03% and 50.41%, opening Top-1 shares were
9.88% and 9.82%, and complete-game uniqueness was 97.49% and 98.49%.
The first learner used 55.1 seconds and its 128-game gate used 197.2 seconds;
the gate correctly rejected a 50.39% raw score whose paired lower bound was
42.04%.  During both service phases, all seven actor GPUs continued the next
generation, validating the shared-GPU mutual exclusion and the full 7+1
pipeline.

## Public KataGomo B18 calibration

The latest accepted continuation checkpoint at the time of measurement was
`be7ecec0477b`, through round 18.  It was screened against the public
[2026-04-06 KataGomo Connect6 release](https://github.com/hzyhhzy/KataGomo/releases/tag/ConnectSix_20260406),
model `connect6_19x3-b18c384n-s923096832-d279788159`, using a locally built CUDA
engine from the `ConnectSix2024` branch at commit `b4d9e276add7`.  Native
budgets are per complete Pair turn; KataGomo budgets are per stone.

| Opening | Native / Kata visits | Games | Native W-L-D | Black / white wins | Mean stones |
| --- | ---: | ---: | ---: | ---: | ---: |
| natural | 400 / 400 | 32 | 0-32-0 | 16 / 16 | 78.84 |
| forced, 8 distinct shared openings | 400 / 400 | 16 | 0-16-0 | 8 / 8 | 68.25 |
| natural | 400 / 8 | 16 | 0-16-0 | 8 / 8 | 66.75 |
| natural | 1200 / 1 | 16 | 0-16-0 | 8 / 8 | 57.25 |
| natural | 1200 / 8 | 16 | 0-16-0 | 8 / 8 | 53.44 |
| natural | 1200 / 1200 | 8 | 0-8-0 | 4 / 4 | 44.00 |

Natural play produced only the two deterministic openings selected by the two
models, so it is a deployment-style battlefield screen rather than an
opening-diversity estimate.  The shared-opening run covers eight distinct
central openings and reaches the same qualitative result.  For an all-loss
natural sample, the Wilson 95% upper score bounds are 10.72% for 32 games,
19.36% for 16 games, and 32.44% for 8 games; they do not establish a precise
Elo gap.  KataGomo won equally often as black and white in every screen, so the
result is not explained by first-player advantage.

In the serial arena, native mean search latency was 0.259 seconds per complete
turn at 400 simulations and 0.587 seconds at 1200.  KataGomo used 0.184 and
0.410 seconds per stone at the corresponding budgets, or roughly 0.368 and
0.821 seconds per ordinary two-stone turn.  This is a latency comparison with
native evaluator batch one, not the many-game self-play throughput benchmark;
it shows that the compact Pair path remains computationally efficient even
though its learned strength is far behind this public model.

The one-visit setting was independently checked with `genmove_debug`, which
reported exactly `Root visits: 1`.  Thus increasing native search from 400 to
1200 did not close the gap even against an essentially policy-only B18.  The
dominant limitation is currently learned policy/value strength and model
capacity, not insufficient MCTS count.  A 10,000-simulation continuation would
consume much more compute without answering that diagnosis as cleanly.

The adapter disables resignation and pondering, uses freestyle/NOVC rules,
checks Python and native C++ state after every complete turn, and records every
move.  Provenance hashes are:

- native checkpoint: `be7ecec0477b5f49468e80cda3b8b2d5b539456f78994aafc18c3ff222d8b2cf`
- downloaded `KatagomoConnect6_20260406.7z`: `7604184fc337c1175f4ef90294ca090b2b272ba55c3c0307addd200a9c4dc170`
- KataGomo model: `92fee13526b4317b04c712caaac8768576a3bc5a6707b05d64262ba73ce155e5`
- KataGomo engine: `072e8b7e3ee8e3ee87d1d6a49c96a1a9fd0fce7a551d6b3bfdcc83ad050a8ecd`
- GTP configuration: `d655a7017fb9485a2c8fd47b0c00e0d07d8a3facff76ccdee4721a011d330215`

Raw results are stored under
`$NEBULA_ARTIFACT_ROOT/katagomo_solo_20260924`; the downloaded
release and source/build tree are under
`$NEBULA_EXTERNAL_ROOT/katagomo`.

## One-GPU self-play scheduling sweep

An isolated H20 sweep used the accepted through-49 checkpoint
`cd353128470992` with 256 games per variant, 400 simulations, Top-64 candidate
selection, and natural model-selected openings.  Results are performance-only:
changing concurrency also changes global random-number consumption, so the
variants are not a paired strength or outcome comparison.

| Variant | Concurrent games | CPU threads | Affinity | Turns/s | Mean eval batch | Relative |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| baseline | 128 | 24 | physical cores | 43.803 | 41.22 | 1.000x |
| wider queue | 192 | 24 | physical cores | 44.837 | 46.45 | 1.024x |
| wider queue + SMT | 192 | 32 | physical + siblings | 44.792 | 46.45 | 1.023x |

Concurrency 192 improved the stable per-turn throughput measure by only 2.36%.
Adding SMT was 0.10% slower than the otherwise identical physical-core run.
Neither change is adopted by the active production continuation: the small
queue gain does not justify interrupting it, and SMT is rejected.  The next
larger gain should target pipeline balance or the CPU MCTS implementation,
rather than additional evaluator queue pressure.

Raw requests, manifests, logs, and the aggregate summary are under
`$NEBULA_BENCHMARK_ROOT/selfplay_sweep_after50_20260924`.

### Pipeline-balance projection and queued validation

The last 20 completed 1,792-game rounds averaged 287.6 seconds for actors,
160.2 seconds for the learner, and 176.8 seconds for the gate.  Applying every
historical round's observed timing separately, scaling actor and learner work
with games while holding the fixed-size gate constant, gives the following
capacity projection.  It is a scheduling model, not yet a measured result.

| Games/round | Games/actor | Projected critical path | Projected games/s | Gain |
| ---: | ---: | ---: | ---: | ---: |
| 1,792 | 256 | 337.3 s | 5.314 | baseline |
| 2,240 | 320 | 381.0 s | 5.879 | 10.6% |
| 2,464 | 352 | 408.0 s | 6.039 | 13.7% |
| 2,688 | 384 | 436.7 s | 6.156 | 15.8% |
| 2,912 | 416 | 467.9 s | 6.224 | 17.1% |

The 2,464-game point is selected for the real probe because actor and service
are each the bottleneck in half of the historical rounds.  It captures most of
the projected gain while increasing policy-update latency by less than the
larger points.  An eight-round isolated continuation is queued to start only
after the active 150-round run exits cleanly, its final accepted hash verifies,
and no unrelated GPU compute process exists.  Its output will be under
`$NEBULA_ARTIFACT_ROOT/pair_native/seven_actor_shared_service_2464_probe_after150_20260924`.

The projection above excludes the second, orchestrator-side replay audit after
each actor subprocess.  In the seven observed rounds where the previous gate
had already finished, that strict audit added 24.17 seconds on average.  It is
usually hidden behind the service stage at 1,792 games, but would become part
of the actor critical path after increasing the generation size.

Two composable, opt-in replay changes are therefore included in the queued
probe.  First, the seven committed shards are still audited semantically in
full, but in seven independent processes; the parent then rehashes both files
and decodes only the small game table for cross-shard checks.  Second, the
learner retains the exact rows decoded by its audit instead of hashing and
decoding every Parquet shard a second time.  A crossed-order real-shard CPU
benchmark measured 3.674 seconds for the historical learner path and 3.002
seconds for the single-pass path, a 1.224x replay-loading speedup, with identical
content SHA256.  Raw output is under
`$NEBULA_BENCHMARK_ROOT/replay_load_one_shard_20260924`.

Serial-versus-seven-process full-round validation and three gate evaluator
variants (compiled bucketed, eager bucketed, and compiled fixed-batch) are
queued before the 2,464-game probe.  The gate sweep uses the same checkpoint,
openings, seeds, and 400 simulations and records full-game fingerprints.  No
gate variant will be adopted solely from games/s if the games differ.

## Paused-run H20 optimization and 2,464-game validation

The 150-round extension was stopped cleanly after 58 committed actor rounds;
its state remains resumable and its accepted checkpoint through round 54 has
SHA256 `c47f72b18cd5`. No partial actor replay or candidate was committed.
Isolated H20 sweeps then selected 256 concurrent games, dynamic
16/32/64/128/192/256 evaluator buckets, and 16 physical MCTS CPU threads per
actor. In a production-shaped 352-game single-GPU comparison, concurrency 128
and 256 delivered 49.269 and 52.056 turns/s respectively (+5.66%); 320 fell
back to 50.713 turns/s. Fixed padding and eager evaluation were rejected.

An eight-round 7+1 continuation used 2,464 games per round, 400 simulations,
Top-64, natural model-selected openings, and the full dynamic color/opening
quality controls. It completed without a replay retry:

- 19,712 games and 859,089 Pair turns; mean 43.58 turns/game;
- black/white/draw 9,747/9,151/814, or 51.58% black among decisive games;
- short/medium/long/very-long 10.74%/53.61%/25.25%/10.40%;
- mean complete-game uniqueness 98.80%; maximum first-move share 5.11%;
- mean raw actor rate 7.666 games/s and 333.87 turns/s;
- actor duration median 322.96 seconds; observed start-interval median 324.11
  seconds before the final learner optimization;
- one of eight candidates passed the paired 95% promotion gate. The accepted
  checkpoint is `fe075b9ebf7e`.

The last three replay rounds contain 7,392 games, 293,057 training turns and
33,473 validation turns. Seven-process strict learner audit plus parent-side
rehash/load reduced replay loading from 86.096 to 29.031 seconds (2.97x); both
paths produced canonical content SHA256
`6e1547ff4c81ffbc51baaae78507349e2f725cf2c5f63e9626c29027d8927d19`.
A real four-epoch learner using this path completed in 127.03 seconds. The
eight observed gates had 170.16-second mean and 175.19-second median latency,
so optimized learner plus median gate is 302.23 seconds, below the actor
median. This removes the service tail without introducing speculative
candidate state, making seven actors plus one shared learner/gate GPU the
preferred production layout.

Serial full-round strict replay validation took 19.788 seconds versus 4.774
seconds with seven processes (4.15x). The gate evaluator sweep retained
compiled dynamic buckets: 24.72 turns/s versus 8.57 for eager buckets and
17.66 for compiled fixed padding. All changes are opt-in or preserve prior
defaults. Two independent reviews found no P0/P1 issue, and the final suite is
`108 passed`.

Raw artifacts are under
`$NEBULA_BENCHMARK_ROOT/combined_paused58_20260924`,
`$NEBULA_BENCHMARK_ROOT/selfplay_variants_paused58_20260924`,
and
`$NEBULA_ARTIFACT_ROOT/pair_native/combined_2464_c256_t16_probe_paused58_20260924`.

## Replay-pool and search-budget study (2026-09-27)

A fixed-baseline replay experiment compared 7,392 recent games against 29,568
loaded games drawn from recent, approximately 98k-game, and approximately
197k-game pools. On one common future holdout the losses were 2.7769, 2.7637,
2.7629, and 2.7781 respectively. The paired 32-opening arenas were too small
to distinguish playing strength. The selected production design therefore
keeps a 40-generation (~98k-game) pool and loads 12 generations per update:
eight recent plus four evenly spaced historical generations, capped at 1.2M
weighted training turns.

On 48 audited positions, action agreement with a 4,000-simulation reference
at 400/600/800/1200 simulations was 77.1%/81.3%/83.3%/87.5%. For 12 positions
selected as difficult at 400, agreement with a 10,000-simulation reference was
33.3% at 400, 50.0% at 1200, and 83.3% at 4000. Budgets below 400 recovered the
top action on some high-confidence positions but produced materially worse
search-value targets. Production therefore retains a 400 floor and uses a
deterministic generation-level 400:75,600:15,800:8,1200:2 schedule (478 mean).
Adaptive per-position search remains available but disabled: a real 128-game
cap-800 run reduced the mean budget from 800 to 570.8 with valid replay, while
wall throughput improved only 3.6% under shared-GPU contention.

Artifacts are under
`$NEBULA_BENCHMARK_ROOT/replay_pool_sweep_20260927`,
`$NEBULA_BENCHMARK_ROOT/search_budget_sweep_20260927`, and
`$NEBULA_BENCHMARK_ROOT/adaptive_selfplay_20260927`. The full
suite passes 111 tests; an independent production review found no P0/P1 issue.
