# 性能基准说明

性能比较必须固定模型、MCTS 模拟次数、batch 和 CPU 线程数。完整棋局的长度会
随落子变化，不能单独用“每局多少秒”判断算子优化是否有效。

## 端到端 MCTS

```bash
python benchmarks/benchmark_mcts.py \
  --engine reinforcement_learning/checkpoints/current_model.engine \
  --simulations 12000 --batch-size 64 --threads 5 \
  --processes 2 --gpus 6,7 --warmup-simulations 512 --repeats 5
```

主要观察 `median_search_seconds` 和 `steady_simulations_per_second`。
`startup_included_wall_seconds` 包含 Python、CUDA、TensorRT 初始化，只用于衡量
短任务启动成本。启用新版 C++ 引擎后还会输出 `deduplication_ratio`，它表示
原始叶子请求中有多少次复用了相同棋盘的网络结果。

比较两组 MCTS 参数的棋力和速度时，使用两个独立进程进行配对换色：

```bash
python benchmarks/compare_mcts_configs.py \
  --engine reinforcement_learning/checkpoints/current_model.engine \
  --gpu 6 --games 20 \
  --threads-a 5 --threads-b 24 \
  --batch-size-a 64 --batch-size-b 64 \
  --black-simulations 400 --white-simulations 1200
```

## TensorRT batch 曲线

```bash
CUDA_VISIBLE_DEVICES=6 python benchmarks/benchmark_tensorrt_engine.py \
  --engine reinforcement_learning/checkpoints/current_model.engine \
  --batches 16,32,64
```

窄 profile 引擎默认只支持到构建时的 `NEBULA_MCTS_BATCH_SIZE`。测试更大 batch
前，需要设置 `NEBULA_TRT_MAX_BATCH_SIZE` 重新构建宽 profile 引擎。

## 实验性相对位置注意力

```bash
CUDA_VISIBLE_DEVICES=6 python benchmarks/benchmark_relative_attention.py \
  --batch-size 64 --rounds 200
```

这个脚本只测单算子。即使单算子更快，也必须重新运行端到端 MCTS 基准；自定义
插件可能破坏 TensorRT 原有的 QKV、布局和 MHA 跨层融合。

## BF16 双卡训练

```bash
python benchmarks/benchmark_training.py \
  --gpus 6,7 --batch-size 386 --warmup 2 --rounds 5
```

脚本只执行随机数据上的前向、反向、梯度裁剪和 fused AdamW，不读取或修改训练
数据与 checkpoint。它用于判断多卡计算扩展，不代表真实数据加载吞吐。
