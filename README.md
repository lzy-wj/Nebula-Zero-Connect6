# Nebula Zero Connect6

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-114%20passed-brightgreen)](tests)
[![ModelScope](https://img.shields.io/badge/ModelScope-model%20%2B%20data-blue)](https://modelscope.cn/models/Lazyshu/Nebula_Zero_Connect6)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-model%20%2B%20data-yellow)](https://huggingface.co/Lazyshu/Nebula-Zero-Connect6-Pair)
[English](README_en.md)

Nebula Zero 是一个六子棋 AlphaZero 项目。V1 曾获首届北京市大学生「人工智能+」
创新大赛六子棋赛道冠军；当前 V3 从随机参数开始，直接用自我对弈训练。

## 三次迭代

| 版本 | 主要变化 | 实测结果 |
| --- | --- | --- |
| V1 · Competition | 11.80M 参数 C6TransNet，逐子策略，C++ MCTS | 比赛版本，也是后续实验的基线；公开训练记录超过 800 代 |
| V2 · Inference-first | 集中优化 TensorRT、batch、CUDA Graph 和候选网络 | 候选门禁为 49–51，没有晋级 |
| V3 · Pair-native | 原生无序双落子、1.65M 稀疏网络、Top-64 MCTS、异步自我对弈 | 参数比 V1 少 86.0%；对 V1 `best` 的配对得分为 77.34%（400 次搜索） |

V2 解决的是速度，没有证明棋力提高。V3 没有继承 V2 权重，而是换成原生 Pair 动作空间
重新训练，所以这里的棋力提升指 V1 到 V3。

## V3 对 V1

`through-88` 使用 64 个相同开局并换色，共 128 局：

| V1 对手 | V3 每回合 / V1 每子搜索次数 | V3 胜-负-和 | V3 得分（95% 配对区间） |
| --- | ---: | ---: | ---: |
| `best` | 400 / 400 | 98-28-2 | 77.34%（71.30–83.39%） |
| `best` | 1200 / 1200 | 96-30-2 | 75.78%（68.63–82.93%） |

V1 在普通回合的两次落子中各搜索一次，因此相同名义预算下，V1 的搜索模拟次数约为
V3 的两倍。公开模型是更晚的 `through-452`；统计口径见
[证据记录](docs/THREE_GENERATION_EVIDENCE.md)。

## 训练规模

- V1：公开记录超过 800 代，早期完整训练时间没有保留下来。
- V2：完成候选网络、部署链路和自我对弈验证，但没有通过棋力门禁。
- V3：`through-452` 所在续跑段共生成 1,133,440 局，其中包含 7 次质量重试。

V3 的公开 replay 是该模型实际使用的 12 代窗口，共 29,568 局、1,713,854 个回合样本。
统计口径见[证据记录](docs/THREE_GENERATION_EVIDENCE.md)。

## 当前 V3

六子棋一回合通常落两子。V3 直接搜索和训练无序动作 `(a, b)`，不再把一回合拆成两次
独立决策。

| 项目 | 配置 |
| --- | --- |
| 网络 | 1,651,035 参数；稀疏 token trunk + 对称 pair head |
| 搜索 | 原生双落子 C++ MCTS；Top-64；生产预算 400–1200 次/完整回合 |
| 训练 | 随机初始化；Parquet replay；候选门禁；可扩展异步 actor |
| 发布 | `through-452` 模型、精确 replay 窗口、配置、清单和 SHA256 |

## 快速开始

```bash
git clone https://github.com/lzy-wj/Nebula-Zero-Connect6.git
cd Nebula-Zero-Connect6
pip install -r requirements.txt

python experiments/pair_native/compile_pair_mcts.py
python -m pytest -q
```

详细说明见 [Pair-native 文档](experiments/pair_native/README.md)和
[V3 发布说明](docs/PAIR_NATIVE_V3_RELEASE.md)。

模型：[ModelScope](https://modelscope.cn/models/Lazyshu/Nebula_Zero_Connect6) ·
[Hugging Face](https://huggingface.co/Lazyshu/Nebula-Zero-Connect6-Pair)

数据：[ModelScope](https://modelscope.cn/datasets/Lazyshu/Nebula_Zero_Connect6) ·
[Hugging Face](https://huggingface.co/datasets/Lazyshu/Nebula-Zero-Connect6-Replay)

## 主要入口

| 任务 | 命令/目录 |
| --- | --- |
| V3 自我对弈与训练 | `python -m experiments.pair_native.async_loop ...` |
| Web 对弈与训练监控 | `bash experiments/pair_native/run_web.sh` |
| 模型发布包 | `python -m experiments.pair_native.export_release ...` |
| 数据发布包 | `python -m experiments.pair_native.prepare_public_dataset ...` |
| V1 强化学习 | `reinforcement_learning/` |
| 桌面客户端 | `local/` |

权重、replay、日志和 TensorRT engine 不进入 Git。TensorRT engine 需要在目标机器上
重新构建。上面的胜率只对应表中模型、搜索预算和开局协议，不是公开 Elo。

## 团队

刘钊洋（强化学习与算法）、陈涛（桌面客户端）、龚飞雪（架构实验与支持）、
胡子涵（人类棋谱数据）。感谢 ShaohonChen、北京邮电大学数学科学学院、SwanLab、
DeepMind AlphaZero 与 ModelScope。

MIT License。
