# Nebula Zero - Connect6 AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ModelScope](https://img.shields.io/badge/ModelScope-Download-blue)](https://modelscope.cn/models/Lazyshu/Nebula_Zero_Connect6)
[中文版](README.md)

**Nebula Zero** is an AlphaZero-style Connect6 stack: autoregressive policy factorization, C++ MCTS, TensorRT inference, and a resumable self-play training loop.

> Champion entry of the first Beijing Collegiate “AI+” Innovation Contest (Connect6 track).

## Contents

- [Features](#features)
- [What is in the repo](#what-is-in-the-repo)
- [Layout](#layout)
- [Quick start](#quick-start)
- [Training entry points](#training-entry-points)
- [Environment variables](#environment-variables)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Features

- **Action space**: Connect6 plays two stones per turn (~\(361 \times 360\)). The baseline uses \(P(a_1,a_2\mid s)=P(a_1\mid s)\,P(a_2\mid s,a_1)\). Experiments also explore a **native pair policy** head.
- **Search / inference**: C++ MCTS, TensorRT FP16, fused self-play graphs, pinned buffers, CUDA Graphs, within-batch leaf dedup (same simulation budget, fewer NN evals).
- **Training**: supervised warm-start → self-play RL → gated promotion; multi-GPU workers; optional **remote self-play** (train locally, generate games remotely). Configuration is **env-driven**, not tied to machine-specific shell scripts.
- **Engineering**: `benchmarks/`, `tests/`, `scripts_smoke_env.py`; large artifacts are gitignored by default.

## What is in the repo

| Included | Not included (build or download) |
|----------|-----------------------------------|
| Training / search source, benchmarks, tests | `*.pth` / `*.pt` weights |
| MCTS build scripts | `*.engine` / `*.onnx` (rebuild on the target GPU) |
| Experiment code (`pair_policy`, `nebula_v3`) | Self-play CSV, replay, swanlog, run dirs |
| Docs + generic Python entrypoints | Host-specific launch shells |

Weights and large datasets: [ModelScope Lazyshu/Nebula_Zero_Connect6](https://modelscope.cn/models/Lazyshu/Nebula_Zero_Connect6) → place under the relevant `checkpoints/`.

## Layout

```text
Nebula-Zero-Connect6/
├── supervised_learning/       # SL pretrain on human games
├── reinforcement_learning/    # C6TransNet + MCTS + self-play loop
├── experiments/
│   ├── pair_policy/           # native dual-move policy loop
│   └── nebula_v3/             # isolated architecture experiments
├── benchmarks/
├── tests/
├── Competition/               # web arena
├── local/                     # PyQt5 client
├── final/                     # tournament scripts
├── scripts_smoke_env.py
└── requirements.txt
```

## Quick start

**Requirements:** Linux/WSL2, NVIDIA GPU, matching PyTorch + TensorRT, Python 3.8+, g++/OpenMP.

```bash
git clone https://github.com/lzy-wj/Nebula-Zero-Connect6.git
cd Nebula-Zero-Connect6
pip install -r requirements.txt

cd reinforcement_learning/core
python compile_mcts.py
cd ../..

python scripts_smoke_env.py   # optional; needs GPU + libmcts.so
```

**Weights & engines**

1. Put `best.pth` / `initial.pth` under `reinforcement_learning/checkpoints/` (or as module docs say).
2. Engines are **GPU/TRT-specific** — rebuild on the machine that will run inference:

```bash
cd reinforcement_learning
python pipeline/export_onnx.py checkpoints/best.pth checkpoints/current_model.onnx
python pipeline/build_engine.py checkpoints/current_model.onnx checkpoints/current_model.engine
```

## Training entry points

### Supervised

```bash
cd supervised_learning && python train.py
```

### Reinforcement learning

```bash
cd reinforcement_learning
python run_loop.py
python run_forever.py --swanlab-mode disabled
```

Override devices/scale with env vars, e.g.:

```bash
export NEBULA_SELFPLAY_GPUS=0,1
export NEBULA_NUM_WORKERS=2
export NEBULA_MCTS_BATCH_SIZE=64
python run_forever.py --swanlab-mode disabled
```

Self-play only:

```bash
python pipeline/generate.py \
  --engine checkpoints/current_model.engine \
  --out data/raw/gen_data.csv --total 64
```

### Pair-policy experiment

See `experiments/pair_policy/README.md`:

```bash
python experiments/pair_policy/run_loop.py
python experiments/pair_policy/run_forever.py --swanlab-mode disabled
```

### Architecture experiment (v3)

Isolated from the production loop; see `experiments/nebula_v3/README.md`.

### Benchmarks / GUI

```bash
python benchmarks/benchmark_mcts.py --engine reinforcement_learning/checkpoints/current_model.engine
cd local/mcts && python compile_dll.py && cd .. && python main.py
```

## Environment variables

| Variable | Role |
|----------|------|
| `NEBULA_SELFPLAY_GPUS` | GPUs for self-play, e.g. `0,1` |
| `NEBULA_NUM_WORKERS` | Workers (usually one per GPU) |
| `NEBULA_MCTS_THREADS` | CPU threads per worker |
| `NEBULA_MCTS_BATCH_SIZE` | Infer batch; rebuild engine after change |
| `NEBULA_SIMULATIONS_BLACK` / `_WHITE` | Search budget |
| `NEBULA_TRT_FUSED_SELFPLAY` | Fused TRT self-play graph |
| `NEBULA_CUDA_GRAPH` | CUDA Graphs |
| `NEBULA_SELFPLAY_BACKEND` | `local` or `remote` |
| `NEBULA_REMOTE_SELFPLAY_HOST` | SSH target, e.g. `user@remote-host` |
| `NEBULA_REMOTE_PROJECT_DIR` | Remote project path (required for remote) |
| `NEBULA_REMOTE_PYTHON` | Remote Python binary |
| `NEBULA_TRAINING_GPU(S)` | Training devices |
| `NEBULA_TRAIN_PRECISION` | `bf16` / `fp16` / `fp32` |

Full notes: [reinforcement_learning/README.md](reinforcement_learning/README.md).

## Acknowledgments

**Team:** Liu Zhaoyang ([@lzy-wj](https://github.com/lzy-wj)), Chen Tao ([@Colin0v0](https://github.com/Colin0v0)), Gong Feixue ([@FeixueGong](https://github.com/FeixueGong)), Hu Zihan (human game data).

**Thanks:** ShaohonChen, School of Mathematics (BUPT), [SwanLab](https://swanlab.cn), DeepMind AlphaZero, ModelScope.

## License

MIT License
