# Nebula Zero Connect6

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-114%20passed-brightgreen)](tests)
[![ModelScope](https://img.shields.io/badge/ModelScope-model%20%2B%20data-blue)](https://modelscope.cn/models/Lazyshu/Nebula_Zero_Connect6)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-model%20%2B%20data-yellow)](https://huggingface.co/Lazyshu/Nebula-Zero-Connect6-Pair)
[中文](README.md)

Nebula Zero is an AlphaZero-style Connect6 project. V1 won the Connect6 track of the first
Beijing Collegiate “AI+” Innovation Contest. The current V3 starts from random weights and
trains directly through self-play.

## Three iterations

| Version | Main change | Measured result |
| --- | --- | --- |
| V1 · Competition | 11.80M-parameter C6TransNet, per-stone policy, C++ MCTS | Contest build and later experimental baseline; the public training record reaches generation 801 |
| V2 · Inference-first | TensorRT, batching, CUDA Graphs, and candidate-network work | The candidate gate was 49–51, so it was not promoted |
| V3 · Pair-native | Native unordered pairs, 1.65M sparse network, Top-64 MCTS, asynchronous self-play | 86.0% fewer parameters than V1; 77.34% paired score against V1 `best` at 400 simulations |

V2 improved speed but did not demonstrate higher playing strength. V3 did not inherit its
weights; it restarted in a native Pair action space. The strength comparison is therefore
V1 to V3.

## V3 versus V1

The `through-88` milestone used 64 common openings with color reversal, for 128 games:

| V1 opponent | V3 per-turn / V1 per-stone simulations | V3 W-L-D | V3 score (paired 95% interval) |
| --- | ---: | ---: | ---: |
| `best` | 400 / 400 | 98-28-2 | 77.34% (71.30–83.39%) |
| `best` | 1200 / 1200 | 96-30-2 | 75.78% (68.63–82.93%) |

V1 searches once for each of the two stones in a normal turn, so equal nominal budgets give
it about twice V3's simulation count. The public model is the later `through-452` snapshot;
see the [evidence record](docs/THREE_GENERATION_EVIDENCE.md) for the aggregation rules.

## Training scale

- V1: the public training record reaches generation 801; complete historical training time was not archived.
- V2: candidate architectures, deployment, and self-play were validated, but the strength gate failed.
- V3: the continuation segment containing `through-452` generated 1,133,440 games, including
  seven quality retries.

The public V3 replay is the exact 12-generation window consumed by the model: 29,568 games
and 1,713,854 turn samples. See the [evidence record](docs/THREE_GENERATION_EVIDENCE.md) for
the aggregation rules.

## Current V3

A Connect6 turn normally places two stones. V3 searches and learns an unordered `(a, b)`
action directly instead of splitting a turn into two independent decisions.

| Item | Configuration |
| --- | --- |
| Network | 1,651,035 parameters; sparse-token trunk and symmetric pair head |
| Search | native-pair C++ MCTS; Top-64; 400–1200 simulations per complete turn |
| Training | random initialization; Parquet replay; candidate gating; scalable asynchronous actors |
| Release | `through-452` model, exact replay window, configs, manifests, and SHA256 files |

## Quick start

```bash
git clone https://github.com/lzy-wj/Nebula-Zero-Connect6.git
cd Nebula-Zero-Connect6
pip install -r requirements.txt

python experiments/pair_native/compile_pair_mcts.py
python -m pytest -q
```

See the [Pair-native guide](experiments/pair_native/README.md) and the
[V3 release guide](docs/PAIR_NATIVE_V3_RELEASE.md).

Models: [ModelScope](https://modelscope.cn/models/Lazyshu/Nebula_Zero_Connect6) ·
[Hugging Face](https://huggingface.co/Lazyshu/Nebula-Zero-Connect6-Pair)

Data: [ModelScope](https://modelscope.cn/datasets/Lazyshu/Nebula_Zero_Connect6) ·
[Hugging Face](https://huggingface.co/datasets/Lazyshu/Nebula-Zero-Connect6-Replay)

## Main entry points

| Task | Command/directory |
| --- | --- |
| V3 self-play and training | `python -m experiments.pair_native.async_loop ...` |
| Web arena and monitor | `bash experiments/pair_native/run_web.sh` |
| Model release package | `python -m experiments.pair_native.export_release ...` |
| Replay release package | `python -m experiments.pair_native.prepare_public_dataset ...` |
| V1 reinforcement learning | `reinforcement_learning/` |
| Desktop client | `local/` |

Weights, replay, logs, and TensorRT engines are not tracked by Git. Rebuild TensorRT engines
on the target machine. Match scores apply only to the listed checkpoints, search budgets,
and opening protocol; they are not a public Elo rating.

## Team

Liu Zhaoyang (reinforcement learning and algorithms), Chen Tao (desktop client), Gong Feixue
(architecture research and support), and Hu Zihan (human game data). Thanks to ShaohonChen,
the School of Mathematics at BUPT, SwanLab, DeepMind AlphaZero, and ModelScope.

MIT License.
