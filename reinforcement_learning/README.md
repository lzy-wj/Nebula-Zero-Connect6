# 强化学习系统

本目录实现生产训练闭环：模型导出、TensorRT 构建、自我对弈、replay 训练、
配对评估与候选晋升。

```text
checkpoint → ONNX/TRT → self-play → replay → train → gate → promote
```

## 运行约束

- 权重可以跨同结构设备迁移；TensorRT engine 必须在目标 GPU 上重建。
- 修改 C++ MCTS 后必须重新运行 `python core/compile_mcts.py`；旧 ABI 会被明确拒绝。
- `MCTS_BATCH_SIZE` 必须落在 engine profile 内；修改后应重建 engine。
- 同一训练谱系只运行一个 supervisor。远端节点只生成棋谱，不维护第二套状态。
- 训练数据、状态和候选文件按代保存；不要只替换单个 current 文件。

## 准备

```bash
python core/compile_mcts.py
python pipeline/export_onnx.py checkpoints/best.pth checkpoints/model.onnx
python pipeline/build_engine.py checkpoints/model.onnx checkpoints/model.engine
```

初始权重默认按以下顺序查找：

1. `../supervised_learning/checkpoints/checkpoint_latest.pth`
2. `checkpoints/initial.pth`

## 运行

只生成棋谱：

```bash
python pipeline/generate.py \
  --engine checkpoints/model.engine \
  --out data/raw/selfplay.csv \
  --total 300
```

单代与持续训练：

```bash
python run_loop.py --swanlab-mode disabled
python run_forever.py --swanlab-mode online
```

常用配置：

```bash
NEBULA_SELFPLAY_GPUS=0,1 \
NEBULA_NUM_WORKERS=2 \
NEBULA_MCTS_THREADS=32 \
NEBULA_MCTS_BATCH_SIZE=64 \
NEBULA_SIMULATIONS_BLACK=400 \
NEBULA_SIMULATIONS_WHITE=1200 \
python run_forever.py --swanlab-mode disabled
```

远端自我对弈：

```bash
NEBULA_SELFPLAY_BACKEND=remote \
NEBULA_REMOTE_SELFPLAY_HOST=user@host \
NEBULA_REMOTE_PROJECT_DIR=/path/to/Nebula-Zero-Connect6 \
NEBULA_REMOTE_PYTHON=python \
python run_forever.py --swanlab-mode online
```

远端执行器增量同步源码和权重，生成完成后原子取回棋谱。训练、门禁和实验记录仍由
本地 supervisor 维护。

## 晋升与门禁

incumbent 以不可变 bundle 保存，`checkpoints/incumbent.json` 同时指向权重与 engine，
并校验两者 SHA-256。晋升只原子切换该 manifest；`best.pth` 和
`current_model.engine` 是兼容别名，不是恢复依据。

生产门禁默认使用 200 局、100 组同开局换色对局。候选须满足最低得分、优势概率、
白方胜率与整局先手胜率约束；相关阈值均可通过 `NEBULA_GATING_*` 调整。

## 推理路径

默认路径使用融合棋盘编码、TensorRT FP16、CUDA Graph、页锁定复用缓冲，以及
batch 内等价叶子去重。实验开关必须通过固定搜索预算的端到端基准验证；单算子延迟
不足以代表完整 MCTS 吞吐或棋力。

```bash
python ../benchmarks/benchmark_mcts.py \
  --engine checkpoints/model.engine \
  --simulations 12000 \
  --batch-size 64 \
  --repeats 5
```

更多基准入口见 [benchmarks/README.md](../benchmarks/README.md)。

## 恢复

训练状态记录当前代和下一待执行阶段。阶段产物完整时，重启会从相应阶段继续；
不完整的棋谱、replay 或门禁结果会被拒绝。长期任务应先用一代小规模配置完成闭环
验收，再扩大数据量和搜索预算。
