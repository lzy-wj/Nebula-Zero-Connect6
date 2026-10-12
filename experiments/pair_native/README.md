# Pair-native Connect6 experiments

This directory is isolated from the production loop. It explores a native
turn-level policy where an MCTS edge is an unordered pair `(a, b)` rather than
two sequential single-stone edges.

## Interactive Web arena

The Web arena supports both native-Pair PyTorch checkpoints and the historical
ModelScope TensorRT engines.  Humans click one stone at a time; native search
still advances atomically by a complete Connect6 turn.  The page exposes model
selection, 400/1200/3000/10000-simulation presets, temperature, Pair Top-K,
C-PUCT, value scale, root noise, CPU threads and the legacy inference batch.
Numeric search budgets are labelled honestly: per turn for native-Pair and per
stone for legacy models.

```bash
./experiments/pair_native/run_web.sh
```

The defaults bind the private loopback address `127.0.0.1:18473` and expose
physical GPU 0 as `cuda:0`. Override these with `NEBULA_PAIR_WEB_GPU`,
`NEBULA_PAIR_WEB_PORT`, and optionally `NEBULA_PAIR_WEB_CPUSET`. Checkpoints are
discovered below `artifacts/pair_native` and `artifacts/legacy`; use
`NEBULA_PAIR_WEB_TRAINING_ROOT` and `NEBULA_PAIR_WEB_LEGACY_ROOT` to select other
administrator-controlled roots. The browser never submits filesystem paths.

`http://127.0.0.1:18473/` is the live self-play monitor.  It plots per-generation
color balance, game length, length buckets, throughput, dynamic color budgets,
gate confidence bounds, and learner sampling drift, with explicit warning and
critical indicators.  The human-play page remains at
`http://127.0.0.1:18473/play`.

## Strict game-data audit

The audit replays every game without repairing it. It checks move legality,
the actual winner, moves after a terminal state, policy alignment,
normalization, occupied-point probability, chosen-move support, duplicate
games, and exact/D4 turn-state uniqueness.

```bash
python -m experiments.pair_native.game_audit \
  /path/to/reinforcement_learning/data/raw \
  --output /tmp/connect6_game_audit.json \
  --fail-on-invalid
```

Legacy files contain one policy for each individual stone. They are useful for
auditing and value targets, but they do **not** contain the joint unordered-pair
visit distribution required by a native pair policy. New self-play must store
that distribution directly.

## H20 PyTorch prototype benchmark

The benchmark includes the sparse trunk, all 361 point queries, the complete
unordered pair score matrix, legality masking, GPU Top-K, and optionally the
small result copy to CPU.

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.pair_native.benchmark \
  --batches 1,8,16,32,64,128 \
  --copy-outputs
```

This is a PyTorch architecture screen, not a TensorRT production result.

Measured screening results are recorded in [RESULTS.md](RESULTS.md).

## Native Pair replay contract

`pair_replay.py` writes atomic Apache Parquet shards with a manifest commit
marker. Game rows preserve the complete move sequence, result, random seed,
and SHA256 provenance for model, engine, and search configuration. Turn rows
store packed board states, the joint unordered-Pair visit distribution, root
value, unique-node counts, neural evaluations, and search time.

The shard auditor rejects mismatched winners, illegal or post-terminal moves,
non-canonical pairs, invalid singleton actions, corrupt board payloads,
policies that omit the played Pair, invalid search statistics, orphan turns,
and file hash/schema mismatches. Forced openings are explicit zero-search
turns: the auditor verifies their configured region and complete uniform
policy instead of pretending that they came from MCTS. A shard is not
training-visible until its manifest is written last.

Audit and summarize native replay with:

```bash
python -m experiments.pair_native.analyze_replay \
  --data-dir /path/to/pair-replay \
  --output /path/to/analysis.json
```

The report includes rule validity, exact and D4 state diversity, game lengths,
color balance, policy entropy/effective support, chosen-action mass, search
coverage, and the fraction of nonzero or decisive root values.

## Legacy C6TransNet baseline arena

`legacy_arena.py` compares the native unordered-Pair model with the historical
17-plane C6TransNet checkpoints. The engines keep their native semantics:
Pair MCTS spends its budget once per full turn, while legacy MCTS spends the
configured budget on each stone. Thus equal numeric budgets give the legacy
model about twice as many simulations per normal Connect6 turn. Every legacy
TensorRT engine must have a sidecar provenance manifest binding its SHA256 to
the checkpoint SHA256.

```bash
python -m experiments.pair_native.legacy_provenance \
  --checkpoint /path/to/model_gen_800.pth \
  --engine /path/to/model_gen_800.engine \
  --onnx /path/to/model_gen_800.onnx

CUDA_VISIBLE_DEVICES=0 NEBULA_CUDA_GRAPH=0 \
python -m experiments.pair_native.legacy_arena \
  --native-checkpoint /path/to/native-best.pth \
  --legacy-checkpoint /path/to/model_gen_800.pth \
  --legacy-engine /path/to/model_gen_800.engine \
  --pairs 64 --native-simulations 400 --legacy-simulations 400 \
  --top-k 64 --opening-radius 4 --cpu-affinity 0-23 \
  --output /path/to/result.json
```

For a deployment-style arena, start from an empty board and let whichever
model is black search its own opening. A small early temperature can provide
limited seed-to-seed variation without turning the arena into high-temperature
self-play:

```bash
CUDA_VISIBLE_DEVICES=0 NEBULA_CUDA_GRAPH=0 \
python -m experiments.pair_native.legacy_arena \
  --native-checkpoint /path/to/native-best.pth \
  --legacy-checkpoint /path/to/legacy-best.pth \
  --legacy-engine /path/to/legacy-best.engine \
  --pairs 8 --native-simulations 1200 --legacy-simulations 1200 \
  --top-k 64 --opening-mode natural \
  --early-temperature 0.10 --temperature-turns 4 \
  --cpu-affinity 0-23 --output /path/to/natural-result.json
```

Use forced openings only as a separate opening-coverage or color-bias
diagnostic. Native-Pair self-play already supports fully searched openings via
`--forced-opening-radius -1`.

Natural color swaps generally do not share an opening because the model that
plays black changes. Their primary score uncertainty is therefore the
independent-game binomial standard error. The paired standard error is emitted
only for forced common-opening games. Do not feed a natural-mode score into the
paired promotion gate; use forced common openings for that specific decision.
`--temperature-turns` counts complete Connect6 turns: in natural mode the
searched black singleton is turn one, while a forced opening is not searched.

The arena replays both engines through independent contexts and checks stone
count, player-to-move, terminal state, and winner after every non-terminal
turn. Results for the ModelScope gen-0/400/800/801/best baselines are recorded
in [RESULTS.md](RESULTS.md).

## Public KataGomo baseline arena

`katagomo_arena.py` compares the native Pair engine with the public Connect6
branch of [KataGomo](https://github.com/hzyhhzy/KataGomo).  The two engines
retain their actual action semantics:
native MCTS searches one unordered Pair per complete turn, while KataGomo GTP
searches each stone separately.  Consequently, a KataGomo budget of 400 visits
is approximately 800 visits per ordinary two-stone turn and must not be
described as compute-matched to native 400.

The external executable and model are deliberately not vendored.  Build the
`ConnectSix2024` branch and download an official Connect6 weight, then run a
natural-opening, color-reversed screen with:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m experiments.pair_native.katagomo_arena \
  --native-checkpoint /path/to/native-best.pth \
  --katagomo-engine /path/to/katago \
  --katagomo-model /path/to/19x.bin.gz \
  --katagomo-config /path/to/gtp.cfg \
  --pairs 16 --native-simulations 400 --katagomo-visits 400 \
  --top-k 64 --opening-mode natural \
  --native-device cuda:0 --katagomo-gpu 1 \
  --mcts-cpu-threads 24 --katagomo-threads 32 \
  --output /path/to/result.json
```

Natural mode lets the black engine choose its own singleton opening.  Forced
mode samples distinct shared central openings and reverses colors, and is a
separate opening-coverage diagnostic.  Resignation and pondering are disabled;
the adapter fixes the visit cap, freestyle rules, NOVC, the 361-stone limit,
and first-pass-win off.  It also checks the Python rules engine against the
native C++ context after every complete turn.  Result JSON binds the native
checkpoint, KataGomo executable, public model, and configuration by SHA256.
Public-baseline measurements and exact local provenance are in
[RESULTS.md](RESULTS.md).

## Native-Pair cold-start loop

The native C++ MCTS owns one tree per game, batches one leaf from every active
tree, uses canonical unordered Pair edges, reuses the played subtree, and
handles immediate one- and two-stone wins exactly. The evaluator uses a fixed
H20 batch, compiled PyTorch inference, compact Top-K transfer, and no duplicate
scan by default.

Recommended generation-zero command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.pair_native.selfplay \
  --output-dir /path/to/gen0 \
  --checkpoint /path/to/random-initial.pth \
  --games 1024 \
  --gpus 0,1,2,3 \
  --concurrent-games 128 \
  --simulations 400 \
  --top-k 64 \
  --candidate-gumbel 0.25 \
  --random-candidates 0 \
  --mcts-cpu-threads 24 \
  --batch-buckets 16,32,64,128 \
  --value-scale 0 \
  --forced-opening-radius -1 \
  --early-temperature 0.35 \
  --temperature-turns 4 \
  --generation 0
```

Keep the untrained value head disabled. Gumbel candidate exploration was
slightly slower than replacing four candidates in C++, but the trained model
won the direct quality gate 95-33, so the faster approximation is not the
recommended default.

Train the first generations against the exact joint visit policy and the
sparse search-proven root values:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.pair_native.train \
  --data-dir /path/to/gen0 \
  --init-checkpoint /path/to/random-initial.pth \
  --output-dir /path/to/gen0/train \
  --epochs 12 \
  --batch-size 1024 \
  --value-target search \
  --value-weight 1 \
  --decisive-value-weight 10 \
  --sampling-mode joint \
  --length-targets 0.20,0.35,0.30,0.15 \
  --winner-targets 0.475,0.475,0.05
```

Repeat `--data-dir` to mix recent audited generations during longer runs; the
loader still splits by whole game and rejects duplicate game IDs.

Promotion must use `arena.py` with paired central openings and color swaps.
Do not enable a learned value merely because its held-out loss decreased;
compare the same checkpoint with value on and off first.

In the measured run, generation-zero value failed this gate, while
generation-one value passed 145-110 with one draw over 256 games. The next
large generation should therefore start from the gen-1 best checkpoint with
the same search settings and `--value-scale 1`; retain `--value-scale 0` for a
small control shard so calibration regressions remain visible.

On the measured dual-socket H20 host, GPUs 0–3 are local to NUMA node 0 and
the launcher assigns each worker 24 distinct physical cores. The production
default uses one worker per GPU, 24 OpenMP MCTS threads, and inference buckets
16/32/64/128. Two workers per GPU can show 95–98% SM utilization, but produced
only 235.7 useful turns/s versus 324.0 for the single-worker bucketed setup;
do not optimize for utilization percentage alone.

## Eight-GPU asynchronous loop

The conservative H20 production layout is six self-play GPUs, one learner GPU,
and one paired arena gate GPU. A higher-throughput continuation layout uses
seven self-play GPUs and one GPU shared by the serialized learner and gate.
Actor rounds pin the accepted checkpoint they
started with, so a promotion can never change weights in the middle of a game. At
most one completed actor round may be stale. Replay is training-visible only
after every shard manifest passes the strict audit; a candidate becomes
accepted only when its paired lower confidence bound exceeds 50%.

```bash
python -m experiments.pair_native.async_loop \
  --output-dir /path/to/pair-native-run \
  --initial-checkpoint /path/to/gen1/best.pth \
  --rounds 20 \
  --start-generation 2 \
  --games-per-round 1536 \
  --actor-gpus 0,1,2,3,4,5 \
  --learner-gpu 6 \
  --gate-gpu 7
```

Production defaults are 128 concurrent games/GPU, a 400-simulation floor and
1200 ceiling, Top-64, searched natural openings, opening temperature 1.25
(dynamically bounded to 1.0–1.5), early temperature 0.35,
Gumbel 0.25, value scale 1, 24 pinned physical CPU cores per actor,
16/32/64/128 inference buckets, learner batch 1024, a three-generation replay
window, and a 64-pair color-swapped gate sampled without replacement from the
81 central openings. GPU 0–3 use physical CPUs 0–95; GPU 4–7 use 96–191.
Learner and gate alternate while all six actors continue,
so seven GPUs are active during either phase without mixing candidate and
accepted weights. Because the gate depends on the learner's completed
candidate, reserving distinct learner and gate GPUs necessarily leaves one idle.
They may instead safely share a GPU: the orchestrator enforces mutual exclusion.
For seven actor GPUs, use a game count divisible by seven (for example 1,792,
or 256 games per worker) to avoid the tail imbalance seen with 1,536 games.

The actor controller measures decisive black/white wins over the recent replay
window. If either color leaves the 45–55% band, only the under-performing color
receives extra MCTS simulations; the stronger color stays at 400 and neither
color can exceed 1200. Every color-specific budget is stored in the search hash
and audited per turn. Opening-temperature control uses the maximum per-round
opening concentration in the rolling window (as well as the merged distribution),
so different top moves cannot hide a single-generation collapse. Every completed
round is rejected if its top opening exceeds 50% or its complete-game fingerprint
unique rate is below 90%. A quality-rejected round is never sent to the learner;
the controller records its audited summary, retries the same generation with a
hotter opening, and allows two retries before failing the run. The learner independently rakes the current replay window
toward equal opening/middle/end phases, 20/35/30/15% short/medium/long/very-long
turn mass, and 47.5/47.5/5% black/white/draw mass. Per-row importance is capped
so the largest row weight is at most 4x the smallest; targets are approached
subject to that anti-duplication cap and available support. The unbalanced
validation set remains untouched.

Resume by repeating the original command with `--resume`; only `--rounds` may
be extended. Resume verifies the accepted checkpoint hash and fully re-audits
every committed replay round. Incomplete actor, learner, or gate attempt
directories are never treated as committed state. States written before the
opening-temperature/diversity controls were added are intentionally not resumed;
start a new output directory so their old replay is not mixed with the new
control policy.

For the current high-quality run, append these search flags to the complete
overnight command so an old shell alias cannot silently restore the historical
64-simulation/Top-16 setup:

```bash
python -m experiments.pair_native.async_loop \
  --simulations 400 --minimum-selfplay-simulations 400 \
  --maximum-selfplay-simulations 1200 \
  --top-k 64 --gate-simulations 400 \
  --forced-opening-radius -1 --early-temperature 0.35 \
  --temperature-turns 4 --opening-temperature 1.25 \
  --opening-temperature-minimum 1.0 --opening-temperature-maximum 1.5 \
  --dynamic-opening-temperature --maximum-opening-top-share 0.50 \
  --minimum-unique-game-rate 0.90 --dynamic-color-balance \
  --train-sampling-mode joint \
  --max-stale-rounds 1
```

The C++ search consumes the full budget for every searched turn, including
terminal-singleton choices; only the explicitly audited forced opening records
have zero simulations. A six-H20 steady run with 768 games produced
55,319 turns in 231.57 s (3.317 games/s, 238.89 turns/s). All six shards were
valid, with no duplicate complete games; the replay analyzer reported mean
searched simulations exactly 400, mean neural evaluations 379.96, searched
exact-state uniqueness 98.43%, and D4 uniqueness 97.28%. The policy support
averaged 32.81 actions with a 64-action p90. These figures are a throughput
and data-integrity baseline for the overnight run, not a strength claim.
