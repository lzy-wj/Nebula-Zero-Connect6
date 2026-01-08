# Nebula Zero - 强化学习模块

这是 Nebula Zero 的核心训练模块，实现了针对六子棋优化的 AlphaZero 算法。

## 目录结构

- `core/`: 核心游戏逻辑、神经网络模型 (`C6TransNet`) 以及 MCTS 实现
- `pipeline/`: 数据分析和 TensorRT 编译脚本
- `run_loop.py`: 训练主循环入口
- `config.py`: 超参数配置文件

## 开始训练

### 1. 编译 MCTS 引擎（首次运行必须）

```bash
cd reinforcement_learning/core
python compile_mcts.py
```

脚本会自动检测操作系统（Windows/Linux）并使用正确的编译器。

### 2. 准备初始模型

系统会自动查找以下位置的模型（按优先级）：
1. `../supervised_learning/checkpoints/checkpoint_latest.pth`（监督学习输出）
2. `checkpoints/initial.pth`（手动放置）

你也可以从 ModelScope 下载预训练模型放入 `checkpoints/` 目录。

### 3. 运行训练

```bash
cd reinforcement_learning
python run_loop.py
```

训练流程：
1. 加载最新模型
2. 生成自我对弈数据
3. 训练神经网络
4. 门控评估（新模型 vs 旧模型）
5. 胜率超过阈值则更新最佳模型

## TensorRT 编译

为了实现高效的自我对弈，我们使用 TensorRT 加速推理。训练循环会自动处理编译，也可以手动执行：

```bash
# 导出 ONNX
python pipeline/export_onnx.py <模型.pth> <输出.onnx>

# 构建 TensorRT 引擎
python pipeline/build_engine.py <输入.onnx> <输出.engine>
```

> **注意**: `.engine` 文件是硬件绑定的，不同 GPU 之间不能混用。

## 配置参数

在 `config.py` 中可调整：
- `SIMULATIONS`: MCTS 模拟次数（越多越强，但速度越慢）
- `NUM_WORKERS`: 并行工作进程数
- `BATCH_SIZE`: 训练批大小
- `GPUS`: 使用的 GPU 编号
