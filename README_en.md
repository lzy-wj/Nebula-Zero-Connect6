# Nebula Zero Connect6

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-114%20passed-brightgreen)](tests)
[![ModelScope](https://img.shields.io/badge/ModelScope-model%20%2B%20data-blue)](https://modelscope.cn/models/Lazyshu/Nebula_Zero_Connect6)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-model%20%2B%20data-yellow)](https://huggingface.co/liuzhao123/Nebula-Zero-Connect6-Pair)
[中文](README.md)

An AlphaZero-style Connect6 system trained from random initialization through self-play.
The current V3 combines a native unordered-pair policy, a sparse network, C++ MCTS, and
an asynchronous multi-GPU pipeline.

> Champion of the first Beijing Collegiate “AI+” Innovation Contest, Connect6 track.

## Current: V3 Pair-native

A Connect6 turn normally places two stones. V3 searches and learns an unordered `(a, b)`
action directly instead of splitting one turn into two independent decisions.

| Item | V3 configuration |
| --- | --- |
| Network | 1,651,035 parameters; sparse-token trunk and symmetric pair head |
| Search | native-pair C++ MCTS, Top-64, 400–1200 simulations per complete turn |
| Training | random initialization, self-play, Parquet replay, gating, asynchronous multi-GPU |
| Released data | exact 12-generation promotion window: 29,568 games and 1,713,854 turn samples |

Internal promotion scores are gating evidence, not a public rating or a claim of tournament
strength. Model, data, and search settings are released together for reproducibility.

```bash
git clone https://github.com/lzy-wj/Nebula-Zero-Connect6.git
cd Nebula-Zero-Connect6
pip install -r requirements.txt

python experiments/pair_native/compile_pair_mcts.py
python -m pytest -q
```

See the [Pair-native guide](experiments/pair_native/README.md) and the complete
[results log](experiments/pair_native/RESULTS.md). Weights are hosted in the
[ModelScope model repository](https://modelscope.cn/models/Lazyshu/Nebula_Zero_Connect6),
and replay data in the
[ModelScope dataset repository](https://modelscope.cn/datasets/Lazyshu/Nebula_Zero_Connect6).
V3 is mirrored to the Hugging Face
[model repository](https://huggingface.co/liuzhao123/Nebula-Zero-Connect6-Pair) and
[dataset repository](https://huggingface.co/datasets/liuzhao123/Nebula-Zero-Connect6-Replay).
See the [V3 release guide](docs/PAIR_NATIVE_V3_RELEASE.md) to reproduce both packages.

## Three major iterations

<details open>
<summary><b>V3 · Pair-native sparse (current)</b></summary>

- A native joint two-stone policy and MCTS avoid per-stone semantic mismatch.
- A 1.65M-parameter sparse model targets self-play throughput rather than model scale alone.
- 400–1200 simulation schedules, color/opening monitoring, and strict replay audits.
- Asynchronous 1/2/4/8-GPU profiles keep actors running while training and evaluation use
  their assigned devices.

</details>

<details>
<summary><b>V2 · Inference-first</b></summary>

- Systematic TensorRT, batching, CUDA Graph, parallel-MCTS, and network-candidate studies.
- Candidate networks improved inference throughput, but their match gates remained near even;
  this iteration is retained as engineering research rather than a strength upgrade.
- Its performance measurement and gating methods became the foundation of V3.

</details>

<details>
<summary><b>V1 · Competition</b></summary>

- An 11.8M-parameter C6TransNet with an autoregressive per-stone policy and C++ MCTS.
- Supervised pretraining, self-play, TensorRT inference, desktop, and Web clients.
- Historical weights and games remain available as the compatibility baseline and contest build.

</details>

## Main entry points

| Task | Command/directory |
| --- | --- |
| V3 self-play and training | `python -m experiments.pair_native.async_loop ...` |
| V3 Web arena and monitor | `bash experiments/pair_native/run_web.sh` |
| V3 model release package | `python -m experiments.pair_native.export_release ...` |
| V3 replay release package | `python -m experiments.pair_native.prepare_public_dataset ...` |
| V1 reinforcement learning | `reinforcement_learning/` |
| Desktop client | `local/` |

Git tracks source, tests, and build scripts—not weights, replay, logs, or TensorRT engines.
TensorRT engines are hardware/software specific and should be rebuilt on the target machine.

## Team

Liu Zhaoyang (reinforcement learning and algorithms), Chen Tao (desktop client), Gong Feixue
(architecture research and support), and Hu Zihan (human game data). Thanks to ShaohonChen,
the School of Mathematics at BUPT, SwanLab, DeepMind AlphaZero, and ModelScope.

MIT License.
