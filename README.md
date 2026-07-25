# Nebula Zero - 六子棋 AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ModelScope](https://img.shields.io/badge/ModelScope-模型下载-blue)](https://modelscope.cn/models/Lazyshu/Nebula_Zero_Connect6)
[English Version](README_en.md)

**Nebula Zero** 是面向六子棋的 AlphaZero 风格实现：自回归策略分解、C++ MCTS、TensorRT 推理加速，以及可恢复的自我对弈训练环。

> 首届北京市大学生「人工智能+」创新大赛六子棋赛道冠军项目。

## 目录

- [核心特性](#核心特性)
- [仓库里有什么 / 没有什么](#仓库里有什么--没有什么)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [训练与实验入口](#训练与实验入口)
- [常用环境变量](#常用环境变量)
- [致谢](#致谢)
- [许可证](#许可证)

## 核心特性

- **动作空间**：六子棋一步两子，组合空间约 \(361 \times 360\)。基线用自回归分解 \(P(a_1,a_2\mid s)=P(a_1\mid s)\,P(a_2\mid s,a_1)\)；实验目录还提供**原生双落子条件头**（pair policy）。
- **搜索与推理**
  - C++ MCTS + OpenMP
  - TensorRT FP16、融合自对弈图、页锁定缓冲、CUDA Graph
  - batch 内叶子局面去重（模拟次数不变，减少重复评估）
- **训练**
  - 监督预训练 → 自我对弈 RL → 成套门禁晋升
  - 多卡 worker、可选**远端自对弈**（本机训练 / 远端只产棋谱）
  - 配置以**环境变量**为主，不绑定某一台机器的脚本
- **工程**
  - `benchmarks/` 固定搜索预算的吞吐对比
  - `tests/` 与 `scripts_smoke_env.py` 冒烟检查
  - 权重 / 引擎 / 棋谱默认 gitignore，避免误提交大文件

## 仓库里有什么 / 没有什么

| 有 | 没有（需本机生成或从 ModelScope 下载） |
|----|----------------------------------------|
| 训练与搜索源码、基准与测试 | `*.pth` / `*.pt` 权重 |
| 编译脚本（MCTS） | `*.engine` / `*.onnx`（须在目标 GPU 上重建） |
| 实验代码（pair / v3） | 自我对弈 CSV、replay、swanlog、runs |
| 文档与通用 Python 入口 | 机器专用 shell（本机路径 / 固定 GPU 绑定） |

权重与大数据请从 [ModelScope: Lazyshu/Nebula_Zero_Connect6](https://modelscope.cn/models/Lazyshu/Nebula_Zero_Connect6) 获取，放到对应 `checkpoints/` 目录。

## 目录结构

```text
Nebula-Zero-Connect6/
├── supervised_learning/       # 人类棋谱监督预训练
├── reinforcement_learning/    # C6TransNet + MCTS + 自对弈主循环
├── experiments/
│   ├── pair_policy/           # 原生双落子策略训练环（候选/生产实验）
│   └── nebula_v3/             # 网络结构实验（与主循环隔离）
├── benchmarks/                # MCTS / TensorRT / 训练吞吐基准
├── tests/                     # 单元与轻量集成测试
├── Competition/               # Web 对战前端
├── local/                     # PyQt5 本地客户端
├── final/                     # 历史模型循环赛
├── scripts_smoke_env.py       # 环境 + 规则 + libmcts 冒烟
└── requirements.txt
```

## 快速开始

### 环境

- Linux / WSL2（推荐）
- NVIDIA GPU + 与 CUDA 匹配的 **PyTorch**、**TensorRT**
- Python 3.8+、g++/OpenMP（编译 MCTS）

```bash
git clone https://github.com/lzy-wj/Nebula-Zero-Connect6.git
cd Nebula-Zero-Connect6
pip install -r requirements.txt

# 编译 MCTS（首次必须）
cd reinforcement_learning/core
python compile_mcts.py
cd ../..

# 可选冒烟（需 GPU 与已编译 libmcts.so）
python scripts_smoke_env.py
```

### 权重与引擎

1. 将 `best.pth` / `initial.pth` 等放入 `reinforcement_learning/checkpoints/`（或按各模块 README）。
2. **TensorRT engine 与 GPU/TRT 版本绑定**，换卡必须在本机重建，不要拷贝旧 `.engine`：

```bash
cd reinforcement_learning
python pipeline/export_onnx.py checkpoints/best.pth checkpoints/current_model.onnx
python pipeline/build_engine.py checkpoints/current_model.onnx checkpoints/current_model.engine
```

## 训练与实验入口

细节以各子目录 README 为准；此处只列通用命令。

### 监督学习

```bash
cd supervised_learning
python train.py
```

### 强化学习（经典主循环）

```bash
cd reinforcement_learning
# 单代 / 监督式循环
python run_loop.py
# 长期多代（导出 → 自对弈 → 训练 → 门禁）
python run_forever.py --swanlab-mode disabled   # 或 online
```

设备与规模用环境变量覆盖，例如：

```bash
export NEBULA_SELFPLAY_GPUS=0,1
export NEBULA_NUM_WORKERS=2
export NEBULA_MCTS_BATCH_SIZE=64
export NEBULA_SIMULATIONS_BLACK=400
export NEBULA_SIMULATIONS_WHITE=1200
python run_forever.py --swanlab-mode disabled
```

只生成自对弈棋谱：

```bash
cd reinforcement_learning
python pipeline/generate.py \
  --engine checkpoints/current_model.engine \
  --out data/raw/gen_data.csv --total 64
```

### 双落子 pair 实验

```bash
# 见 experiments/pair_policy/README.md
python experiments/pair_policy/run_loop.py
# 或
python experiments/pair_policy/run_forever.py --swanlab-mode disabled
```

### 网络结构实验（v3）

与生产 loop 隔离，不覆盖正式 checkpoint。见 `experiments/nebula_v3/README.md`。

### 基准与评估

```bash
python benchmarks/benchmark_mcts.py --engine reinforcement_learning/checkpoints/current_model.engine
cd final && python tournament_pro.py
```

### 本地 GUI / Web

```bash
cd local/mcts && python compile_dll.py && cd .. && python main.py
# Web：见 Competition/README.md
```

## 常用环境变量

| 变量 | 含义 |
|------|------|
| `NEBULA_SELFPLAY_GPUS` | 自对弈可见 GPU，如 `0,1` |
| `NEBULA_NUM_WORKERS` | worker 数（通常每卡 1 个） |
| `NEBULA_MCTS_THREADS` | 每 worker 的 MCTS CPU 线程 |
| `NEBULA_MCTS_BATCH_SIZE` | 推理 batch；改后需重建 engine |
| `NEBULA_SIMULATIONS_BLACK` / `_WHITE` | 黑/白搜索预算 |
| `NEBULA_TRT_FUSED_SELFPLAY` | 融合 TRT 自对弈图（默认开） |
| `NEBULA_CUDA_GRAPH` | CUDA Graph（默认开） |
| `NEBULA_SELFPLAY_BACKEND` | `local` 或 `remote` |
| `NEBULA_REMOTE_SELFPLAY_HOST` | 远端 SSH，如 `user@remote-host` |
| `NEBULA_REMOTE_PROJECT_DIR` | 远端项目路径（remote 时必填） |
| `NEBULA_REMOTE_PYTHON` | 远端 Python 可执行文件 |
| `NEBULA_TRAINING_GPU(S)` | 训练卡 |
| `NEBULA_TRAIN_PRECISION` | `bf16` / `fp16` / `fp32` |

更完整说明见 [reinforcement_learning/README.md](reinforcement_learning/README.md)。

## 致谢

### 团队成员

- **刘钊洋** ([@lzy-wj](https://github.com/lzy-wj)) — 强化学习、算法
- **陈涛** ([@Colin0v0](https://github.com/Colin0v0)) — 本地 GUI
- **龚飞雪** ([@FeixueGong](https://github.com/FeixueGong)) — 架构实验与支持
- **胡子涵** — 人类对局数据

### 特别感谢

- **ShaohonChen** ([@ShaohonChen](https://github.com/ShaohonChen)) — 指导与校对
- **北京邮电大学数学科学学院** — 算力支持
- **[SwanLab](https://swanlab.cn)** — 训练可视化
- **DeepMind AlphaZero** — 算法原型
- **ModelScope** — 模型托管

## 许可证

MIT License
