# Nebula Zero Connect6

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-114%20passed-brightgreen)](tests)
[![ModelScope](https://img.shields.io/badge/ModelScope-model%20%2B%20data-blue)](https://modelscope.cn/models/Lazyshu/Nebula_Zero_Connect6)
[English](README_en.md)

从随机初始化开始、以自我对弈训练的六子棋 AlphaZero 系统。当前 V3 使用原生无序双落子
策略、稀疏网络、C++ MCTS 和异步多 GPU 流水线。

> 首届北京市大学生「人工智能+」创新大赛六子棋赛道冠军项目。

## 当前版本：V3 Pair-native

六子棋一回合通常落两子。V3 直接把 `(a, b)` 作为一个无序动作搜索和训练，不再把一回合
拆成两次独立决策。

| 项目 | V3 配置 |
| --- | --- |
| 网络 | 1,651,035 参数，稀疏 token trunk + 对称 pair head |
| 搜索 | 原生双落子 C++ MCTS，Top-64，生产预算 400–1200 次/完整回合 |
| 训练 | 随机初始化、自我对弈、Parquet replay、候选门禁、异步多 GPU |
| 发布数据 | 晋级模型实际使用的 12 代 replay：29,568 局、1,713,854 个回合样本 |

内部晋级分数只用于版本门禁，不等同于公开等级分或比赛棋力。模型、数据和搜索预算会配套
发布，确保结果可复核。

```bash
git clone https://github.com/lzy-wj/Nebula-Zero-Connect6.git
cd Nebula-Zero-Connect6
pip install -r requirements.txt

python experiments/pair_native/compile_pair_mcts.py
python -m pytest -q
```

详细使用、训练和 Web 对弈说明见 [Pair-native 文档](experiments/pair_native/README.md)，
完整实验记录见 [RESULTS](experiments/pair_native/RESULTS.md)。权重放在
[ModelScope 模型仓库](https://modelscope.cn/models/Lazyshu/Nebula_Zero_Connect6)，数据放在
[ModelScope 数据集仓库](https://modelscope.cn/datasets/Lazyshu/Nebula_Zero_Connect6)。
发布包复现流程见 [V3 发布说明](docs/PAIR_NATIVE_V3_RELEASE.md)。

## 三次主要迭代

<details open>
<summary><b>V3 · Pair-native sparse（当前）</b></summary>

- 原生联合双落子策略与 MCTS，避免逐子动作的语义错位。
- 1.65M 参数稀疏模型，面向高吞吐自我对弈，而非单纯扩大网络。
- 400–1200 次动态搜索、颜色与开局监控、严格 replay 审计。
- 支持 1/2/4/8 GPU 异步布局；训练、评测期间其余 GPU 可继续产出数据。

</details>

<details>
<summary><b>V2 · Inference-first</b></summary>

- 系统化测试 TensorRT、batch、CUDA Graph、并行 MCTS 和候选网络。
- 候选网络显著提高推理吞吐，但对局门禁约为五五开，因此作为工程研究保留，未宣称棋力升级。
- 这一阶段建立的性能测量和门禁方法成为 V3 异步流水线基础。

</details>

<details>
<summary><b>V1 · Competition</b></summary>

- 11.8M 参数 C6TransNet，自回归逐子策略与 C++ MCTS。
- 包含监督预训练、自我对弈、TensorRT 推理、桌面端和 Web 端。
- 历史权重与棋谱继续保留，作为兼容基线与项目比赛版本。

</details>

## 主要入口

| 任务 | 命令/目录 |
| --- | --- |
| V3 自我对弈与训练 | `python -m experiments.pair_native.async_loop ...` |
| V3 Web 对弈与训练监控 | `bash experiments/pair_native/run_web.sh` |
| V3 模型发布包 | `python -m experiments.pair_native.export_release ...` |
| V3 数据发布包 | `python -m experiments.pair_native.prepare_public_dataset ...` |
| V1 强化学习 | `reinforcement_learning/` |
| 桌面客户端 | `local/` |

源码、测试和构建脚本进入 Git；权重、replay、日志和 TensorRT engine 不进入 Git 历史。
TensorRT engine 与硬件和软件版本绑定，应在目标机器上重新构建。

## 团队

刘钊洋（强化学习与算法）、陈涛（桌面客户端）、龚飞雪（架构实验与支持）、
胡子涵（人类棋谱数据）。感谢 ShaohonChen、北京邮电大学数学科学学院、SwanLab、
DeepMind AlphaZero 与 ModelScope。

MIT License。
