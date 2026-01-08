# Nebula Zero - 强化学习模块 (Reinforcement Learning)

这是 Nebula Zero 的核心训练模块，实现了针对六子棋优化的 AlphaZero 算法。

## 目录结构

- `core/`: 核心游戏逻辑、神经网络模型 (`C6TransNet`) 以及 MCTS 实现。
- `pipeline/`: 数据分析和 **TensorRT 编译** 脚本。
- `run_loop.py`: 训练主循环入口 (自我对弈 -> 训练 -> 门控评估)。
- `config.py`: 超参数配置文件 (学习率、模拟次数、缓冲区大小等)。

## 首次运行：编译 C++ MCTS 引擎

训练模块使用 C++ 编写的高性能 MCTS 引擎。首次运行前需要编译：

```bash
cd reinforcement_learning/core
python compile_mcts.py
```

该脚本会自动检测操作系统（Windows/Linux）并使用正确的编译器。

## TensorRT 编译指南

为了实现极高的自我对弈效率，我们使用 TensorRT 来加速推理。

### 环境要求
- NVIDIA GPU
- CUDA Toolkit (11.x 或 12.x)
- TensorRT (需匹配 CUDA 版本)
- `torch2trt` 或标准 `tensorrt` Python 绑定

### 如何编译模型
训练循环通常会自动处理编译，但你也可以手动执行以下步骤：

1. **导出为 ONNX**:
   ```bash
   python pipeline/export_onnx.py <模型路径.pth> <输出路径.onnx>
   ```

2. **构建 TensorRT 引擎**:
   ```bash
   python pipeline/build_engine.py <输入路径.onnx> <输出路径.engine>
   ```

**⚠️ 注意**: TensorRT 引擎文件 (`.engine`) 是硬件绑定的。你不能在不同架构的 GPU 之间混用引擎文件 (例如从 RTX 3090 复制到 RTX 4090 是无法运行的)，必须重新编译。

## 开始训练

启动 AlphaZero 训练循环：

```bash
python run_loop.py
```

该脚本将执行以下流程：
1. 加载最新模型。
2. 生成自我对弈数据 (保存至 `data/raw`)。
3. 使用缓冲区数据训练神经网络。
4. 将新模型与旧模型进行对战评估。
5. 如果胜率超过阈值 (Gating)，则更新最佳模型。

## 关键配置与初始化 (Configuration)

在启动训练前，即便是第一次运行，也请务必检查 `config.py` 的以下设置：

1. **设置初始模型 (Initial Model)**:
   修改 `INITIAL_MODEL_PTH` 参数。将其指向监督学习阶段训练出的模型文件（例如 `../supervised_learning/output/checkpoints/checkpoint_latest.pth`）。这是强化学习“冷启动”的关键，如果路径不正确，训练将无法开始。

2. **继续训练 (Resume Training)**:
   系统内置了自动断点续训功能：
   - 你可以从 ModelScope 下载我们提供的预训练模型（`.pth` 文件）。
   - 将下载的模型文件放入 `checkpoints/` 目录。
   - 再次运行 `python run_loop.py` 时，脚本会自动扫描该目录，加载代数最大（最新）的模型继续训练，无需额外配置。

## 其他参数
你也可以在 `config.py` 中调整性能参数：
- `SIMULATIONS`: MCTS 模拟次数 (次数越多越强，但速度越慢)。
- `NUM_WORKERS`: 并行数据生成的工作进程数。
- `BATCH_SIZE`: 显存利用率控制。
