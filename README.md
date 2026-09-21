# Nebula Zero

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Models](https://img.shields.io/badge/ModelScope-models-blue)](https://modelscope.cn/models/Lazyshu/Nebula_Zero_Connect6)
[English](README_en.md)

面向六子棋的 AlphaZero 风格训练与推理系统。项目包含策略价值网络、C++ MCTS、
TensorRT 推理、自我对弈训练、候选门禁，以及本地和 Web 对弈客户端。

> 首届北京市大学生「人工智能+」创新大赛六子棋赛道冠军项目。

## 设计

六子棋每回合落两子，直接策略空间约为 `361 × 360`。Nebula Zero 使用自回归分解

\[
P(a_1,a_2\mid s)=P(a_1\mid s)P(a_2\mid s,a_1)
\]

并提供原生双落子条件头实验。主要工程路径如下：

| 层 | 实现 |
| --- | --- |
| 网络 | C6TransNet、二维相对位置注意力、策略/价值头 |
| 搜索 | C++17 MCTS、OpenMP、渐进展开、批内局面去重 |
| 推理 | TensorRT FP16、融合棋盘编码、CUDA Graph、页锁定缓冲 |
| 训练 | 监督预训练、自我对弈、replay、配对门禁、断点恢复 |
| 应用 | PyQt5 客户端、Flask Web 对战、历史模型循环赛 |

## 安装

推荐 Linux/WSL2、Python 3.10+、NVIDIA GPU、g++ 与 OpenMP。PyTorch 和
TensorRT 必须与本机 CUDA 匹配。

```bash
git clone https://github.com/lzy-wj/Nebula-Zero-Connect6.git
cd Nebula-Zero-Connect6
pip install -r requirements.txt
pip install -r requirements-dev.txt     # 开发环境

python reinforcement_learning/core/compile_mcts.py
python -m pytest -q
python scripts_smoke_env.py             # 需要 CUDA/TensorRT
```

模型权重可从 [ModelScope](https://modelscope.cn/models/Lazyshu/Nebula_Zero_Connect6)
下载。TensorRT engine 与 GPU 架构及 TensorRT 版本绑定，应在目标机器上构建：

```bash
cd reinforcement_learning
python pipeline/export_onnx.py checkpoints/best.pth checkpoints/model.onnx
python pipeline/build_engine.py checkpoints/model.onnx checkpoints/model.engine
```

## 入口

| 任务 | 入口 | 文档 |
| --- | --- | --- |
| 监督预训练 | `python supervised_learning/train.py` | [说明](supervised_learning/README.md) |
| 强化学习 | `python reinforcement_learning/run_forever.py` | [说明](reinforcement_learning/README.md) |
| 双落子实验 | `python experiments/pair_policy/run_loop.py` | [说明](experiments/pair_policy/README.md) |
| 网络结构实验 | `python experiments/nebula_v3/train.py` | [说明](experiments/nebula_v3/README.md) |
| 性能基准 | `python benchmarks/benchmark_mcts.py ...` | [说明](benchmarks/README.md) |
| 本地客户端 | `python local/main.py` | [说明](local/README.md) |
| Web 对战 | `python Competition/web/app.py` | [说明](Competition/README.md) |

核心目录：

```text
supervised_learning/       监督预训练
reinforcement_learning/    生产训练、搜索与推理
experiments/               隔离的模型与策略实验
benchmarks/                固定预算性能基准
tests/                     单元与轻量集成测试
Competition/               Web 对战
local/                     桌面客户端
final/                     模型循环赛
```

## 仓库边界

源码、测试和构建脚本纳入版本控制；权重、ONNX、TensorRT engine、棋谱、replay、
日志与机器专用启动配置不纳入。性能结论应同时报告模型、搜索预算、batch、线程数、
GPU 和软件版本。

## 团队

刘钊洋（强化学习与算法）、陈涛（桌面客户端）、龚飞雪（架构实验与支持）、
胡子涵（人类棋谱数据）。感谢 ShaohonChen、北京邮电大学数学科学学院、SwanLab、
DeepMind AlphaZero 与 ModelScope。

MIT License。
