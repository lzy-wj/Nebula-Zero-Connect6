# Pair Policy

该实验用一次主干推理同时预测第一子策略、局面价值、第二子条件因子和
`q(s, a1)`，减少第二子阶段的完整网络调用。

## 结构

| 组件 | 实现 |
| --- | --- |
| 主干 | C6TransNet V2，联合微调 |
| 第二子策略 | rank-16 非线性条件头 |
| 几何信息 | `37 × 37 × 16` 相对位置门控与偏置 |
| 条件价值 | 向量化输出 361 个 `q(s, a1)` |
| 推理 | 单个 TensorRT 图输出主策略、价值与条件因子 |
| CPU | AVX2/F16C 条件因子点积 |

所有合法动作仍保留在树中；条件头只改变第二子节点的估值路径，不执行永久 top-k
裁剪。该目录与经典强化学习循环隔离。

## 训练闭环

```text
pair self-play + exact anchors
            ↓
        replay buffer
            ↓
  joint trunk/head training
            ↓
   pair + exact TRT engines
            ↓
 color-swapped paired gate
```

运行前显式提供固定的主干与条件头种子：

```bash
NEBULA_PAIR_INITIAL_MAIN=/path/to/main.pth \
NEBULA_PAIR_INITIAL_HEADS=/path/to/pair_heads.pt \
NEBULA_SWANLAB_PROJECT=pair-experiment \
python experiments/pair_policy/run_loop.py --swanlab-mode disabled
```

持续运行：

```bash
python experiments/pair_policy/run_forever.py --swanlab-mode online
```

## 状态与产物

每个 run 独立保存在 `runs/<project>/`：

```text
data/          自我对弈、锚点和门禁棋谱
replay/        训练集、验证集与统计
checkpoints/   current 四件套和逐代候选
runtime/       MCTS 动态库、临时 ONNX、TRT cache
logs/          loop state、progress、gate 和 generation summary
```

`loop_state.json` 中的 phase 表示下一待执行阶段。恢复时会验证前序产物；停在
`evaluation` 不会重新训练候选。门禁 JSON 与棋谱均完整时可直接复用。

## 晋升规则

评估使用相同开局换色对局，并以开局对为统计单位。默认规则同时要求：

- 至少 50 组换色开局；
- 候选得分不低于 incumbent；
- 候选更强的近似概率不低于 90%；
- 候选执白胜率不低于 20%；
- 整局黑胜率位于 35%–65%。

阈值由 `NEBULA_PAIR_GATING_*` 环境变量控制。离线损失用于诊断，不用于替代实战门禁。

## 单独构建

```bash
python reinforcement_learning/core/compile_mcts.py --output /tmp/libmcts_pair.so
python experiments/pair_policy/export_pair_onnx.py \
  /path/to/main.pth /path/to/pair_heads.pt /tmp/pair.onnx
python reinforcement_learning/pipeline/build_engine.py \
  /tmp/pair.onnx /tmp/pair.engine
```

历史实验结果保存在 [RESULTS.md](RESULTS.md)。
