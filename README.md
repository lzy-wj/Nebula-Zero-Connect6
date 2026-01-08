# Nebula Zero - 六子棋 AI (Connect6 AlphaZero)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ModelScope](https://img.shields.io/badge/ModelScope-模型下载-blue)](https://modelscope.cn/models/lzy-wj/NebulaZero)
[English Version](README_en.md)

**Nebula Zero** 是一个针对六子棋（Connect6）设计的最先进的 AlphaZero 算法实现。针对六子棋复杂的动作空间，本项目引入了自回归（AR）分解和 TensorRT 加速等优化技术，能够在消费级硬件上实现超人类的棋力。

> **注**: 本项目为首届北京市大学生“人工智能+”创新大赛六子棋赛道冠军项目。

## 核心特性 (Features)

- **先进建模**: 利用自回归逻辑 ($P(A, B|S) = P(A|S) \times P(B|S, A)$) 完美解决了 $361 \times 360$ 的动作空间爆炸问题。
- **极致性能**: 
  - **C++ MCTS 引擎**: 重写核心蒙特卡洛树搜索逻辑，大幅提升计算吞吐量。
  - **TensorRT 加速**: 神经网络推理采用 FP16 精度加速，相比原生 PyTorch 提速约 26 倍。
- **稳健训练**: 
  - 多阶段流水线：监督学习（人类数据冷启动）-> 强化学习（自我对弈进化）。
  - 解决了“黑棋必胜”偏差：通过动态门控（Gating）和温度控制机制，确保模型平衡发展。

## 目录结构 (Directory Structure)

```plaintext
connect6/
├── reinforcement_learning/  # 核心 RL 训练循环 (AlphaZero)
├── supervised_learning/     # 监督学习 (人类数据预训练)
├── Competition/             # 用于在线评测的 Web 对战平台
├── local/                   # 本地 GUI 客户端 (PyQt5 + C++ 引擎)
├── final/                   # 大乱斗脚本 (模型评估与排位)
└── README.md
```

## 快速开始 (Quick Start)

### 环境要求
- Linux / WSL2
- NVIDIA GPU (需要 TensorRT 支持)
- Python 3.8+
- CMake & C++ 编译器

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/lzy-wj/Nebula-Zero-Connect6-.git
   cd Nebula-Zero-Connect6-
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```
   *(注意: 请确保安装与您的 CUDA 版本匹配的 TensorRT)*

3. **下载模型与数据**
   由于文件较大，模型权重和自对弈数据托管在 ModelScope 平台。
   请访问 [ModelScope 项目页](https://modelscope.cn/models/YourName/NebulaZero) 手动下载模型文件 (`.pth`)，并放入对应的 `checkpoints/` 目录。

### 使用方法

**训练 (强化学习)**
```bash
cd reinforcement_learning
python run_loop.py
```

**评估 (大乱斗)**
在历史模型之间运行循环排位赛。
```bash
cd final
python tournament_pro.py
```

**本地人机对战**
启动 GUI 界面与 AI 对弈。
```bash
cd local
python main.py
```

## 致谢 (Acknowledgments)
- **AlphaZero**: DeepMind 的开创性工作。
- **ModelScope**: 提供模型托管服务。

## 许可证 (License)
MIT License
