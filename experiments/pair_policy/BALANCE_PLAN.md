# Nebular-zero-two 平衡修复分支

状态：代码与离线数据检查已完成，训练尚未启动。

## 基线与隔离

- 旧 run `Nebular-zero-two` 保持只读，当前正式模型仍为 gen271。
- 新 run 名为 `Nebular-zero-two-balance`，初始化读取旧 run 的
  `current_main.pth` 和 `current_pair_heads.pt`。
- 新 run 拥有独立的 data、replay、checkpoints、runtime 和 logs，不覆盖旧四件套。

## 首轮平衡配置

- GPU 只使用物理 `0,1,2,3`；四 worker、每 worker 56 个 MCTS CPU 线程；
- MCTS batch 64；pair 并发棋局 24；exact 并发棋局 12；
- 黑棋 400、白棋 1200 次模拟，关闭动态提前停止；
- 黑白开局温度统一为 0.8；
- 首代 50% 棋局注入 5 颗合法中心区域随机开局，以获得两类开局的可靠对照；
  后续比例由控制器调整；注入着法不作为策略标签；
- 白胜棋局训练损失权重初始为 2.0，之后按 replay 白胜占比自动计算；
- 固定 200 局、尽量黑白各半的验证集；
- 门禁额外要求配对棋局全局黑胜率位于 35%～65%。

以上平衡参数是首轮诊断起点，不视为最终最优值。正式启动前应先用少量纯推理
对照确认随机开局比例和同温设置确实降低黑胜偏置。

## 自动平衡控制器

- 分别统计普通空盘棋局与注入开局棋局的黑胜率；
- 黑胜率 EMA 位于 47%～53% 时保持参数；
- 超出死区时，通过两种数据源胜率插值计算下一代随机开局比例；
- 每代最多改变 10 个百分点，范围限制在 0%～90%；
- 任一来源少于 20 局，或两种开局没有可测差异时保持参数；
- 白胜损失权重根据 replay 白胜率反算，使白胜样本的加权贡献接近 40%，
  权重限制在 1.0～3.0；
- 搜索预算、线程、batch 和黑白温度不参与自动调节。

状态保存在新 run 的 `logs/balance_controller.json`，只有一代完整通过收尾后才
写入下一代参数，中断重启不会提前漂移。

## SwanLab

- Outer：`AlphaZero_Training_Overview_v2`，只展示门禁、样本数量、数据质量、
  固定验证集模型质量和少量运行健康指标；
- Gen：step 0 为数据/buffer/训练前验证，step 1..N 为逐 epoch 指标，最后一步
  为门禁与是否晋升。

## 启动入口

通过环境变量启动独立 run（示例，按本机资源改写）：

```bash
export NEBULA_SWANLAB_PROJECT=Nebular-zero-two-balance
export NEBULA_AUTO_BALANCE=1
# export NEBULA_PAIR_INITIAL_MAIN=.../current_main.pth
# export NEBULA_PAIR_INITIAL_HEADS=.../current_pair_heads.pt
# export NEBULA_SELFPLAY_GPUS=0,1,2,3
python experiments/pair_policy/run_forever.py --swanlab-mode online
```

启动前检查 GPU 占用与是否已有重复训练进程。
