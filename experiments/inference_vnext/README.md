# Inference-First A800 Screening

这个目录只用于隔离验证推理结构，不会读取或覆盖生产 checkpoint。第一轮使用固定
随机权重、相同原始棋盘编码、合法点 mask、规则修正与策略/价值输出，先比较实际
TensorRT 延迟，再决定是否投入蒸馏和自对弈训练。

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

所有大产物写到 `/tmp`：

```bash
CUDA_VISIBLE_DEVICES=4 conda run -n connect6 python \
  experiments/inference_vnext/export_onnx.py \
  --architecture bottleneck_c192 \
  --output /tmp/bottleneck_c192.onnx

CUDA_VISIBLE_DEVICES=4 \
NEBULA_BUILD_GPU=0 \
NEBULA_MCTS_BATCH_SIZE=64 \
NEBULA_TRT_MAX_BATCH_SIZE=64 \
NEBULA_TRT_TIMING_CACHE=/tmp/bottleneck_c192.timing.cache \
conda run -n connect6 python reinforcement_learning/pipeline/build_engine.py \
  /tmp/bottleneck_c192.onnx /tmp/bottleneck_c192.engine

CUDA_VISIBLE_DEVICES=4 conda run -n connect6 python \
  benchmarks/benchmark_tensorrt_engine.py \
  --engine /tmp/bottleneck_c192.engine \
  --batches 1,2,4,8,16,32,64 --copy-outputs
```

随机权重测试只回答“部署图是否值得训练”，不能用于评价棋力。

首轮 A800 数据与候选决策见 [RESULTS.md](RESULTS.md)。
