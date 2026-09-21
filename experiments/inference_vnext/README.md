# Inference-First A800 Models

这里承载面向 A800 自对弈吞吐设计的候选网络、训练、TensorRT 导出和数值验证。所有
训练产物默认写到仓库外的 `Nebula-Zero-Connect6-training/inference_vnext`，不会覆盖生产
checkpoint；仓库只保存代码、配置与可复现实验结论。

当前候选：

| 名称 | 结构 |
| --- | --- |
| `bottleneck_c192` | C192、12 个窄瓶颈卷积块、周期性全局池化 |
| `bottleneck_c256` | C256、12 个窄瓶颈卷积块、周期性全局池化 |
| `dual_scale_c192` | 19×19 局部卷积 + 5×5 全局注意力 |
| `sparse_stone_c192_k64` | 最多 64 个棋子 token + 361 个策略 query |
| `sparse_stone_c192_k96` | 最多 96 个棋子 token + 361 个策略 query |

另有接近 Strong 容量的扩大规格，可通过 `--help` 查看完整名称。加入 `--pair` 会在
同一次前向中导出第二子条件因子和全部 `q(s,a1)`；可用 `--pair-rank` 筛选低秩宽度。

## 训练

Strong 先学习最新 replay 的 MCTS 策略、胜负价值，以及 V2 教师的策略与价值：

```bash
CUDA_VISIBLE_DEVICES=0,7 conda run -n connect6 torchrun \
  --standalone --nproc_per_node=2 \
  experiments/inference_vnext/train.py \
  --architecture dual_scale_c320_d16 \
  --task exact \
  --recent-generations 128 \
  --epochs 48 \
  --batch-size 384 \
  --distill-policy-weight 0.5 \
  --distill-value-weight 0.7 \
  --cpu-affinity '0-29;62-91'
```

Pair 阶段从最佳 Strong checkpoint 初始化，联合训练第一子、条件第二子和
`q(s,a1)`：

```bash
CUDA_VISIBLE_DEVICES=0,7 conda run -n connect6 torchrun \
  --standalone --nproc_per_node=2 \
  experiments/inference_vnext/train.py \
  --architecture dual_scale_c320_d16 \
  --task pair \
  --init-checkpoint /path/to/strong/best.pth \
  --recent-generations 128 \
  --epochs 24 \
  --batch-size 384 \
  --distill-policy-weight 0.5 \
  --distill-value-weight 0.7 \
  --cpu-affinity '0-29;62-91'
```

训练入口支持原子 checkpoint、`--resume` 断点续训、DDP、D4 增强和训练/验证棋局
去重。Fast 模型使用 `sparse_stone_c192_k64`，并通过 `--maximum-stones 64` 限定其
服务区间。

隔离生成的新 replay 可重复传入 `--data-root` 与历史数据合并。原始自对弈 CSV 先用
`prepare_replay.py` 做确定性训练/验证切分，不需要复制生产 replay。

```bash
python experiments/inference_vnext/prepare_replay.py \
  --input /path/to/selfplay.csv \
  --output-dir /path/to/bootstrap_replay \
  --generation 273 \
  --validation-games 400

python experiments/inference_vnext/train.py \
  --architecture dual_scale_c320_d16 \
  --task pair \
  --data-root /path/to/production_replay \
  --data-root /path/to/bootstrap_replay \
  --init-checkpoint /path/to/pair/best.pth
```

长时间自对弈可用 `NEBULA_SELFPLAY_PROGRESS_INTERVAL` 控制日志频率，并用
`NEBULA_SELFPLAY_CPU_AFFINITY='0-29;62-91'` 将各 worker 固定到对应 NUMA 节点。

## 导出与基准

未训练结构的筛选产物统一写到 `/tmp`：

```bash
CUDA_VISIBLE_DEVICES=4 conda run -n connect6 python \
  experiments/inference_vnext/export_onnx.py \
  --architecture dual_scale_c320_d16 \
  --checkpoint /path/to/best.pth \
  --output /tmp/dual_scale_c320_d16.onnx

CUDA_VISIBLE_DEVICES=4 \
NEBULA_BUILD_GPU=0 \
NEBULA_MCTS_BATCH_SIZE=64 \
NEBULA_TRT_MAX_BATCH_SIZE=64 \
NEBULA_TRT_TIMING_CACHE=/tmp/dual_scale_c320_d16.timing.cache \
conda run -n connect6 python reinforcement_learning/pipeline/build_engine.py \
  /tmp/dual_scale_c320_d16.onnx /tmp/dual_scale_c320_d16.engine

CUDA_VISIBLE_DEVICES=4 conda run -n connect6 python \
  benchmarks/benchmark_tensorrt_engine.py \
  --engine /tmp/dual_scale_c320_d16.engine \
  --batches 1,2,4,8,16,32,64 --copy-outputs
```

随机权重测试只回答“部署图是否值得训练”，不能用于评价棋力。

首轮 A800 数据与候选决策见 [RESULTS.md](RESULTS.md)。
