# A800 Inference Screening Results

首轮只筛选部署图，不评价随机权重棋力。所有延迟均在同一台
NVIDIA A800-SXM4-80GB、TensorRT 11.1、FP16、最大 batch 64 下测量；输入为原始
19×19 `int32` 棋盘，延迟包含全部输出异步复制到页锁定主存。

## Exact heads

| Architecture | Parameters | Batch 1 | Batch 64 | Throughput | vs current |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current V2 | 11.8M | 1.469 ms | 5.780 ms | 11,072 pos/s | 1.00× |
| `bottleneck_c192` | 1.67M | 0.852 ms | 1.227 ms | 52,173 pos/s | 4.71× |
| `bottleneck_c256` | 2.96M | 1.030 ms | 1.498 ms | 42,728 pos/s | 3.86× |
| `bottleneck_c320_d16` | 6.10M | 1.161 ms | 2.746 ms | 23,309 pos/s | 2.11× |
| `dual_scale_c192` | 2.56M | 0.839 ms | 1.104 ms | 57,984 pos/s | 5.24× |
| `dual_scale_c256_d14` | 5.92M | 1.311 ms | 1.664 ms | 38,474 pos/s | 3.47× |
| `dual_scale_c320_d16` | 9.90M | 1.144 ms | 2.661 ms | 24,055 pos/s | 2.17× |
| `sparse_stone_c192_k64` | 1.61M | 0.445 ms | 0.745 ms | 85,888 pos/s | 7.76× |
| `sparse_stone_c192_k96` | 1.61M | 0.475 ms | 0.838 ms | 76,392 pos/s | 6.90× |
| `sparse_stone_c256_k96_d6` | 3.88M | 0.640 ms | 1.189 ms | 53,835 pos/s | 4.86× |
| `sparse_stone_c320_k96_d8` | 7.66M | 0.929 ms | 1.901 ms | 33,665 pos/s | 3.04× |

## Native pair heads

一次前向同时输出第一子策略、价值、低秩第二子因子与全部 `q(s,a1)`。

| Architecture | Rank | Batch 64 | Throughput | vs current pair |
| --- | ---: | ---: | ---: | ---: |
| Current V2 pair | 16 | 6.331 ms | 10,110 pos/s | 1.00× |
| `bottleneck_c256` | 16 | 1.887 ms | 33,908 pos/s | 3.35× |
| `dual_scale_c192` | 16 | 1.366 ms | 46,863 pos/s | 4.64× |
| `dual_scale_c256_d14` | 16 | 1.989 ms | 32,179 pos/s | 3.18× |
| `sparse_stone_c192_k64` | 16 | 1.089 ms | 58,772 pos/s | 5.81× |
| `sparse_stone_c192_k96` | 16 | 1.179 ms | 54,285 pos/s | 5.37× |
| `sparse_stone_c256_k96_d6` | 16 | 1.586 ms | 40,360 pos/s | 3.99× |

对 `dual_scale_c256_d14` 单独筛选 rank 4/8/16 后，batch 64 分别为
1.923/1.932/1.989 ms。rank 4 相对 rank 16 只快 3.3%，因此首轮训练保留 rank 16，
后续以条件策略误差和门禁棋力决定是否降秩。

## Correctness and interpretation

- PyTorch 与 TensorRT 已在 0、9、65、97、180 子棋盘上对齐。稀疏 C256 的 policy
  最大绝对误差为 `8.1e-6`，value 为 `5.2e-4`；pair 因子最大为 `2.9e-3`。
- 稀疏模型超过 `K` 个棋子后会截断信息。它只能作为 `stone_count <= K` 的 Fast
  模型，超出后必须切换到 dense Strong 模型，不能独立接管整局。
- 随机网络会改变策略分布、搜索树形状和批内去重率。因此本轮端到端 MCTS 运行只作为
  ABI/稳定性 smoke test，其 simulations/s 不能与训练好的 V2 公平比较。
- ONNX、TensorRT engine、timing cache 和原始日志均位于
  `/tmp/nebula-inference-vnext`，不进入 Git。

## Decision

1. 首个 Strong 训练候选选择 `dual_scale_c256_d14`：约 592 万参数，exact/pair
   分别达到当前吞吐的 3.47×/3.18×，且没有信息截断。
2. 首个 Fast/开局草稿候选选择 `sparse_stone_c192_k64`：exact/pair 分别达到
   7.76×/5.81×，调度器必须在第 65 子前切换到 Strong。
3. 暂不训练 C320 与深层瓶颈版本；它们位于较差的容量—延迟曲线上。
4. 完成教师蒸馏后，使用相同搜索分布重测 MCTS、自对弈 games/s 和固定时间棋力。

## A800 infrastructure tuning

对 `dual_scale_c256_d14` 重新执行 TensorRT level-5 tactic 搜索，比较辅助流上限
0/1/2/4 与自动选择。构建阶段使用 3 次 tactic timing，基准仍包含全部 D2H 输出。

| Engine | Original | Tuned | Latency reduction | Setting |
| --- | ---: | ---: | ---: | --- |
| Exact, batch 64 | 1.671 ms | 1.539 ms | 7.9% | aux streams 0 |
| Pair, batch 32 | 1.562 ms | 1.325 ms | 15.2% | aux streams 1 |
| Pair, batch 64 | 1.948 ms | 1.793 ms | 8.0% | aux streams 1 |

自动辅助流即使增加到 3 次 timing，batch 64 仍为 1.679 ms；收益主要来自限制辅助流，
不是单纯延长构建。生产 Pair 候选应使用：

```bash
NEBULA_TRT_AUX_STREAMS=1 \
NEBULA_TRT_TIMING_ITERATIONS=3 \
python reinforcement_learning/pipeline/build_engine.py model.onnx model.engine
```

CPU 筛选显示单卡 24–32 个 MCTS 线程最合理，48/56 线程反而下降。GPU 0–3 属于
NUMA 0，GPU 4–7 属于 NUMA 1；多 worker 应按 GPU 分割本地 CPU 集合。跨 NUMA 对当前
大模型只有约 0–3% 影响，但快网进入 CPU 受限区后仍应避免线程迁移和同 socket 过量超卖。

## Trained-model gate

训练使用最近 128 代去重 replay、DDP、D4 增强与 V2 策略/价值蒸馏。正式门禁均为
100 局配对开局、完整换色、每步 1200 simulations。

| Candidate | Batch 64 | Gate vs incumbent | Decision |
| --- | ---: | ---: | --- |
| `dual_scale_c256_d14` Pair | 1.836 ms | 40–60 | reject |
| `dual_scale_c320_d16` Exact | 2.586 ms | 44–56 | reject |
| `dual_scale_c320_d16` Pair | 2.932 ms | 50–50 | bootstrap only |
| `sparse_stone_c192_k64` Exact | 0.733 ms | 2–8 smoke | provisional only |

C320 Pair 与 incumbent 的配对胜率为 `50.0% ± 4.0%`，尚未证明更强，因此不替换生产
模型；但相对当前 Pair 的 6.331 ms，纯推理吞吐提高约 2.16 倍。它被选为下一轮隔离
自对弈 teacher，待新数据精修并重新通过置信门禁后再考虑晋升。

### Bootstrap follow-up

C320 Pair 在两张 A800 上生成 13,200 局随机开局数据，耗时约 18.5 分钟，合计
11.9 games/s；相同配置的 incumbent 速度约为 6.3 games/s。合并历史数据四卡精修后，
固定 1200 simulations 门禁为 49–51，没有增益。

近似等墙钟门禁给候选 2400/2500 simulations、incumbent 1200 simulations。两组独立
开局分别为 53–47 和 91–109，合计 144–156。结论仍是“约等强、约 2 倍更快”，不能
声称更强，也不晋升生产；继续迭代需要提高搜索教师质量或改变训练分布。
