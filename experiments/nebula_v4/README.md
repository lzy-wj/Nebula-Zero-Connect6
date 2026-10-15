# Nebula-Zero V4 CPU 预实验

这里先验证 V4 的结构、联合 Pair 学习和整局 Top-1 MoE。离线监督结果只能用于排除
错误架构，不能代替目标部署环境的吞吐和实战门控。训练入口默认使用 CPU；传入
`--device cuda --dtype bf16` 可在单张 GPU 上运行相同对照。

Dense 示例：

```bash
python -m experiments.nebula_v4.train_cpu \
  --data-dir /path/to/replay/gen-N-attempt-00 \
  --output-dir /path/to/output/dense64 \
  --variant dense --summary-tokens 64
```

MoE 示例：

```bash
python -m experiments.nebula_v4.train_cpu \
  --data-dir /path/to/replay/gen-N-attempt-00 \
  --output-dir /path/to/output/moe64 \
  --variant moe --summary-tokens 64 --num-experts 4
```

训练入口会严格审计 Pair replay，并记录初始/训练后的 policy loss、value loss、速度和
各专家接收的完整局面数。训练产物必须放在仓库外。
