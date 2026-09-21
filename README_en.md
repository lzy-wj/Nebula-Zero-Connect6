# Nebula Zero

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Models](https://img.shields.io/badge/ModelScope-models-blue)](https://modelscope.cn/models/Lazyshu/Nebula_Zero_Connect6)
[中文](README.md)

An AlphaZero-style training and inference system for Connect6. It includes policy-value
networks, C++ MCTS, TensorRT inference, self-play training, candidate gating, and desktop
and web clients.

> Champion of the first Beijing Collegiate “AI+” Innovation Contest, Connect6 track.

## Design

A Connect6 turn places two stones, giving a direct action space of roughly `361 × 360`.
Nebula Zero factorizes the policy as

\[
P(a_1,a_2\mid s)=P(a_1\mid s)P(a_2\mid s,a_1)
\]

and also includes an experimental native pair-policy head.

| Layer | Implementation |
| --- | --- |
| Network | C6TransNet, 2D relative attention, policy and value heads |
| Search | C++17 MCTS, OpenMP, progressive widening, leaf deduplication |
| Inference | TensorRT FP16, fused board encoding, CUDA Graphs, pinned buffers |
| Training | supervised warm start, self-play, replay, paired gating, recovery |
| Applications | PyQt5 client, Flask arena, historical-model tournaments |

## Setup

Recommended: Linux/WSL2, Python 3.10+, NVIDIA GPU, g++, and OpenMP. PyTorch and
TensorRT must match the local CUDA installation.

```bash
git clone https://github.com/lzy-wj/Nebula-Zero-Connect6.git
cd Nebula-Zero-Connect6
pip install -r requirements.txt
pip install -r requirements-dev.txt     # development only

python reinforcement_learning/core/compile_mcts.py
python -m pytest -q
python scripts_smoke_env.py             # requires CUDA/TensorRT
```

Weights are hosted on [ModelScope](https://modelscope.cn/models/Lazyshu/Nebula_Zero_Connect6).
TensorRT engines are tied to the GPU architecture and TensorRT version; build them on the
target machine:

```bash
cd reinforcement_learning
python pipeline/export_onnx.py checkpoints/best.pth checkpoints/model.onnx
python pipeline/build_engine.py checkpoints/model.onnx checkpoints/model.engine
```

## Entry points

| Task | Entry point | Notes |
| --- | --- | --- |
| Supervised pretraining | `python supervised_learning/train.py` | [docs](supervised_learning/README.md) |
| Reinforcement learning | `python reinforcement_learning/run_forever.py` | [docs](reinforcement_learning/README.md) |
| Pair policy | `python experiments/pair_policy/run_loop.py` | [docs](experiments/pair_policy/README.md) |
| Architecture research | `python experiments/nebula_v3/train.py` | [docs](experiments/nebula_v3/README.md) |
| Benchmarks | `python benchmarks/benchmark_mcts.py ...` | [docs](benchmarks/README.md) |
| Desktop client | `python local/main.py` | [docs](local/README.md) |
| Web arena | `python Competition/web/app.py` | [docs](Competition/README.md) |

## Repository policy

Source, tests, and build scripts are versioned. Weights, ONNX files, TensorRT engines,
self-play data, replay buffers, logs, and machine-specific launch configuration are not.
Performance results should report the model, search budget, batch size, CPU threads, GPU,
and software versions.

## Team

Liu Zhaoyang (reinforcement learning and algorithms), Chen Tao (desktop client), Gong Feixue
(architecture research), and Hu Zihan (human game data). Thanks to ShaohonChen, the School
of Mathematics at BUPT, SwanLab, DeepMind AlphaZero, and ModelScope.

MIT License.
