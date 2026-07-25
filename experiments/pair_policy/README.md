# 原生双落子策略实验

本实验让一次主干推理同时给出第一子策略、局面价值、第二子条件策略因子和
`q(s, a1)`。MCTS 在第一子后的低访问节点先使用轻量条件头，高价值白棋节点再
调用完整网络校正。它不裁剪合法动作；所有动作仍保存在树中，并沿用渐进展开。

当前仍是候选路径，不会自动替换 `checkpoints/current_model.engine` 或生产
`core/libmcts.so`。

## 当前候选结构

- 主干：现有 V2 主干联合微调，不单纯扩大参数量；
- 第二子策略：rank-16、隐藏维 128 的非线性投影；
- 几何信息：`37 x 37 x 16` 相对方向/距离门控和相对位置偏置；
- 条件价值：一次向量化输出 361 个 `q(s, a1)`；
- 推理图：单个 TensorRT 图输出主策略/价值和 FP16 条件因子；
- CPU：FP16 因子使用 AVX2/F16C 点积，并复用第一子的解码结果；
- 搜索：黑棋第一子后的第二子节点直接使用条件头，白棋节点访问达到 2 次时
  用完整网络同步校正。

保留的本地候选权重位于 `output/joint_aggressive/`。模型文件受 `.gitignore`
保护，不会误提交；TensorRT 引擎应在目标 GPU 服务器上重新构建。

## 棋力结果

gen25 的 3,000 个局面上，第一版联合候选达到：

- 第一子 CE `2.197`，top1 `62.97%`；
- 第二子 CE `2.173`，top1 `60.27%`；
- 第二子目标 top20 recall `99.47%`；
- 主价值 MAE `0.585`，条件价值 MAE `0.572`。

使用真实生产预算（黑 400、白 1200）对当前 incumbent 配对换色：两批共 60 局
为 `34:26`。额外三轮训练虽然把离线第二子 CE 降到 `2.138`、价值 MAE 降到
`0.574`，但同开局门禁为 `8:12`，上一版为 `10:10`，因此没有替换当前候选。
这也说明本实验只按真实 MCTS 门禁选模型，不按离线损失选模型。

32 与 64 CPU 线程在固定搜索预算下配对 40 局为 `19:21`，没有观察到高线程数
导致棋力下降。

## 2026-07-11 吞吐结果

本机两张 A800、每卡一个 worker、每卡 32 线程，固定黑 400/白 1200，生成
192 局：

| 路径 | 每卡并发棋局 | 对局/秒 | 搜索叶子/秒 |
| --- | ---: | ---: | ---: |
| 当前完整网络基线 | 12 | 5.203 | 124,407 |
| 双落子候选 | 12 | 6.115 | 132,803 |
| 双落子候选 | 24 | 7.619 | 156,268 |

相对当前完整网络，候选最终配置的对局吞吐提高约 `46.4%`，按搜索叶子归一化
仍提高约 `25.6%`，完整神经网络评估数约减少一半。每卡并发从 12 提高到 24
还能改善尾批和网络 batch 填充；双卡继续提高到每卡 48/56 线程会争抢本机
CPU/NUMA 资源，反而变慢。

单卡可使用 64 线程：96 局从 `3.599` 提高到 `4.558 局/s`。双卡应保持每卡
32 线程，给两个 worker 和系统留出调度空间。

## 复现实验

先在目标服务器编译候选 MCTS。Linux 默认 `-march=native`，会启用本机的
AVX2/F16C；若动态库需要搬到不同 CPU，可设置 `NEBULA_MCTS_NATIVE=0`。

```bash
python reinforcement_learning/core/compile_mcts.py \
  --output /tmp/libmcts_pair.so

CUDA_VISIBLE_DEVICES=6 python experiments/pair_policy/export_pair_onnx.py \
  experiments/pair_policy/output/joint_aggressive/main.pth \
  experiments/pair_policy/output/joint_aggressive/pair_heads.pt \
  /tmp/nebula_pair.onnx

CUDA_VISIBLE_DEVICES=6 python reinforcement_learning/pipeline/build_engine.py \
  /tmp/nebula_pair.onnx /tmp/nebula_pair.engine
```

双卡候选生成配置：

```bash
NEBULA_MCTS_LIBRARY=/tmp/libmcts_pair.so \
NEBULA_PAIR_HEADS=experiments/pair_policy/output/joint_aggressive/pair_heads.pt \
NEBULA_PAIR_REFRESH_VISITS_BLACK=1000000000 \
NEBULA_PAIR_REFRESH_VISITS_WHITE=2 \
NEBULA_PAIR_DEFER_REFRESH=0 \
NEBULA_SELFPLAY_GPUS=6,7 \
NEBULA_NUM_WORKERS=2 \
NEBULA_MCTS_THREADS=32 \
NEBULA_MCTS_CONCURRENT_GAMES=24 \
python reinforcement_learning/pipeline/generate.py \
  --engine /tmp/nebula_pair.engine --total 192 --out /tmp/pair_test.csv
```

上述命令只用于研究候选。正式接入主循环前还需在 H20 上重新构建、复测吞吐，
并完成更大规模门禁。
