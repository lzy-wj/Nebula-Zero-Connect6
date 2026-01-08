# 监督学习模块

本模块用于在人类对局数据上预训练神经网络，为强化学习提供初始模型。

## 数据准备

训练数据位于 `data/processed/connect6_cleaned.csv`，已经过清洗和格式化。

## 训练

```bash
cd supervised_learning
python train.py
```

训练完成后，模型保存在 `output/checkpoints/` 目录。

## 配置

主要参数在 `train.py` 中设置：
- `EPOCHS`: 训练轮数
- `BATCH_SIZE`: 批大小
- `LEARNING_RATE`: 学习率

## 输出

训练完成后，将生成：
- `output/checkpoints/checkpoint_latest.pth` - 最新模型
- `output/checkpoints/checkpoint_epoch_N.pth` - 各轮次检查点

将 `checkpoint_latest.pth` 复制到强化学习模块的 `checkpoints/initial.pth` 作为 RL 训练的起点。
