# Nebula Zero - Connect6 AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ModelScope](https://img.shields.io/badge/ModelScope-Download-blue)](https://modelscope.cn/models/lzy-wj/NebulaZero)
[中文版 (Chinese Version)](README.md)

**Nebula Zero** represents a state-of-the-art implementation of the AlphaZero algorithm for Connect6 (六子棋). It features specific optimizations for the complex action space of Connect6, including Autoregressive (AR) decomposition and TensorRT acceleration, achieving superhuman performance on consumer-grade hardware.

> **Note**: This project was the champion of the University Connect6 AI Competition.

## Features

- **Advanced Modeling**: Solves the action space explosion ($361 \times 360$) using Autoregressive Logic ($P(A, B|S) = P(A|S) \times P(B|S, A)$).
- **High Performance**: 
  - **C++ MCTS Engine**: Rewritten MCTS logic in C++ for maximum throughput.
  - **TensorRT**: Neural network inference accelerated by TensorRT (FP16), achieving ~26x speedup over vanilla PyTorch.
- **Robust Training**: 
  - Multi-stage pipeline: Supervised Learning (Human Data) -> Reinforcement Learning (Self-Play).
  - Solves "Black Win Bias" with dynamic Gating and temperature control.

## Directory Structure

```plaintext
connect6/
├── reinforcement_learning/  # Core RL Training Loop (AlphaZero)
├── supervised_learning/     # SL Training (Pre-training on human data)
├── Competition/             # Web Arena used for online evaluation
├── local/                   # Local GUI Client (PyQt5 + C++ Engine)
├── final/                   # Battle Royale Tournament Scripts (Model Evaluation)
└── README.md
```

## Quick Start

### Prerequisites
- Linux / WSL2
- NVIDIA GPU (TensorRT required)
- Python 3.8+
- CMake & C++ Compiler

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/lzy-wj/Nebula-Zero-Connect6-.git
   cd NebulaZero
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: You need to install TensorRT matching your CUDA version)*

3. **Download Assets**
   Due to size limits, large models and datasets are hosted on ModelScope.
   Please visit [ModelScope Project Page](https://modelscope.cn/models/YourName/NebulaZero) to manually download model files (`.pth`) and place them in the corresponding `checkpoints/` directory.

### Usage

**Training (Reinforcement Learning)**
```bash
cd reinforcement_learning
python run_loop.py
```

**Evaluation (Battle Royale)**
Run a tournament between historical checkpoints.
```bash
cd final
python tournament_pro.py
```

**Local GUI**
Play against the AI.
```bash
cd local
python main.py
```

## Acknowledgments
- **AlphaZero**: DeepMind's pioneering work.
- **ModelScope**: For hosting the model artifacts.

## License
MIT License
