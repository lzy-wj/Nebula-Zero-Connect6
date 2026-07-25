# NebulaNet V3 架构实验

这个目录与 H20 上的生产 AlphaZero loop 完全隔离，用本地 A800 6/7 对历史
自对弈数据训练新的候选网络。候选只有经过相同搜索预算的正式门控后，才会考虑
接入生产链路。

## 修复的主要问题

- 删除 RL 和 MCTS 从未训练、从未消费的第二策略头，保持逐子搜索语义一致。
- 输入从 17 个平面收敛到 5 个真实平面：己方、对方、空位、黑方身份、第二子阶段。
- 使用大核 ConvNeXt 块替代带 BatchNorm 的空洞卷积堆叠，避免运行统计漂移和
  空洞卷积栅格效应。
- 用二维 RoPE 代替稠密相对位置 bias，使 SDPA 能选择 Flash/高效内核。
- policy 使用标准软目标交叉熵，不再对 MCTS 分布错误套用 focal loss。
- 训练时屏蔽已占位置，避免 softmax 容量浪费在非法点上。
- value 使用 Smooth L1，并通过显著性池化保留局部杀棋信号。
- 按代数温和提高新数据采样概率，轻度补偿白胜和和棋，不再复制全部白胜数据。
- 最近 796–806 代作为固定验证集，指标不再只看训练集。

## 双卡训练

```bash
cd .
CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 \
  experiments/nebula_v3/train.py \
  --data-root /path/to/selfplay_data \
  --run-name network_v3_pretrain_a800 \
  --epochs 12 --batch-size 256 --workers 8
```

当前服务器建议直接使用封装好的启动脚本；它会隔离 `~/.local` 中不兼容的
CUDA 13 PyTorch：

```bash
torchrun ... experiments/nebula_v3/train.py  # see flags in train.py
```

`--batch-size` 是每张卡的 batch，因此默认全局 batch 为 512。A800 实测单卡
batch 256 峰值训练显存约 12.55GB，SDPA 明确进入 FlashAttention 前后向内核。
训练输出保存在
`experiments/nebula_v3/output/network_v3_pretrain_a800/`，SwanLab 中是独立实验，
不会写入任何 `gen_N` 或 `AlphaZero_Training_Loop`。

## 2026-07-11 实验结论

本轮所有候选都经过固定最近代验证集、TensorRT batch=64 基准，以及黑方每步
400 次、白方每步 1200 次搜索的成对换色门控。生产
`reinforcement_learning/checkpoints/best.pth` 和 `current_model.engine` 始终没有被
覆盖。

- 从零训练的 `NebulaNetV3` 价值误差更低，但策略弱于 V2，且 TensorRT
  batch=64 只有 8,548 positions/s，低于 V2 的 11,099，因此停止该架构。
- `FastC6NetV4` 把 V2 的 BatchNorm 精确折入卷积、输入收敛到 5 个真实平面，
  删除无效第二策略头，并裁掉一个全局块。它达到约 12,516 positions/s
  （相对 V2 +13.35%），但两轮 40 局门控分别为 15:25、17:23，未通过。
- 保守桥接版恢复被裁剪层的注意力、只省去该层 FFN。离线 policy loss
  1.8965（V2 为 1.9094），value MAE 0.4894（V2 为 0.6767），TensorRT
  batch=64 约快 3.69%；但 40 局门控仍为 17:23，最高 Top-1 checkpoint
  为 15:25，同样未通过。

因此当前生产模型继续使用 V2。Fast V4 的代码、checkpoint、ONNX 和独立引擎
均保留，适合作为后续“候选自身在线自对弈再训练”的起点；在新的正式门控通过前，
不应接入 H20 主循环。

主要产物：

- `output/fast_v4_distill_a800/`：整层裁剪版及第一阶段蒸馏。
- `output/fast_v4_distill_phase2_a800/`：整层裁剪版排序精修。
- `output/fast_v4_bridge_distill_a800/`：注意力桥接版、最终候选与部署引擎。
