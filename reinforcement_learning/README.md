# Nebula Zero 强化学习与高速自对弈

这个目录包含 C6TransNet、C++ MCTS、自我对弈、门控评估和训练主循环。
当前默认推理链路针对“搜索次数不变、整局更快”进行了优化。

## 环境与编译

推荐使用项目专用环境，避免用户目录中的同名 CUDA 包污染依赖：

```bash
conda activate connect6
pip check
cd reinforcement_learning/core
python compile_mcts.py
```

Linux 需要 `g++` 和 OpenMP，Windows 需要 Visual Studio x64 编译环境。

## 准备模型

系统按以下顺序查找初始权重：

1. `../supervised_learning/checkpoints/checkpoint_latest.pth`
2. `checkpoints/initial.pth`

TensorRT 引擎与 GPU 架构、TensorRT 版本绑定。换到 A800、RTX 5090 或其他
GPU 后必须在目标机器重新导出和构建，不能复制旧 `.engine` 文件直接使用。

```bash
cd reinforcement_learning
python pipeline/export_onnx.py checkpoints/best.pth checkpoints/current_model.onnx
python pipeline/build_engine.py checkpoints/current_model.onnx checkpoints/current_model.engine
```

默认构建的动态范围为 `1..MCTS_BATCH_SIZE`，不会再为用不到的 batch 1024
预留大量激活显存。确实需要更大范围时可以显式设置：

```bash
NEBULA_TRT_MAX_BATCH_SIZE=256 python pipeline/build_engine.py \
  checkpoints/current_model.onnx checkpoints/current_model.engine
```

## 默认高速推理链路

- `NEBULA_TRT_FUSED_SELFPLAY=1`：输入原始整数棋盘，在一个 TensorRT 图中完成
  特征编码、神经网络、softmax 和规则价值修正。
- `NEBULA_CUDA_GRAPH=1`：满批时捕获 H2D、TensorRT 和 D2H；不足一个 batch
  的动态批量也使用 1–64 对应的精确 CUDA Graph，不做填充计算。
- 使用页锁定复用缓冲区和独立 CUDA 流，避免逐批分配与默认流隐式同步。
- 同一个 MCTS batch 内，相同叶子以及不同落子顺序形成的相同棋盘只进行一次
  网络评估；结果复用给原来的每一次模拟，模拟次数和回传次数完全不变。
- C++ MCTS 精确执行请求的模拟次数，不再向下取整；OpenMP 有效位数组不再
  使用存在数据竞争风险的 `vector<bool>`。
- `NEBULA_TRT_CUSTOM_ATTENTION=0`：实验性二维相对位置 Triton AOT 插件默认
  关闭。它的单算子更快，但当前会切断 TensorRT 的跨层融合，整网反而较慢。

所有开关都可以回退，方便在新 GPU 上做同模型 A/B 测试：

```bash
NEBULA_CUDA_GRAPH=0 python ../benchmarks/benchmark_mcts.py \
  --engine checkpoints/current_model.engine
```

## 不同 GPU 的调优方法

当前机器的默认值是两个 worker、物理卡 6/7 各一个、每个 worker 使用 32 个
MCTS CPU 线程和 batch 64。卡 0–5 不会被训练、自对弈或 TensorRT 构建使用。
worker 数、MCTS batch 和设备均支持环境变量，不需要修改源码：

```bash
NEBULA_SELFPLAY_GPUS=6,7 \
NEBULA_NUM_WORKERS=2 \
NEBULA_MCTS_BATCH_SIZE=64 \
python pipeline/generate.py --engine checkpoints/current_model.engine --total 20
```

改变 `NEBULA_MCTS_BATCH_SIZE` 后必须用相同值重新构建 TensorRT 引擎。建议在
每种目标 GPU 上测试 batch 32、64、128，并比较端到端 MCTS 吞吐，不要只看
神经网络单次延迟。在当前 A800 上，batch 128 虽然更快，但唯一评估局面明显
减少，并在 40 局配对换色测试中以 16:24 负于 batch 64，因此保持 batch 64。
同一张卡上增加 worker 没有提高总吞吐，因此每张卡只运行一个。两张卡之间采用
共享动态任务计数器，先完成棋局的卡立即领取下一局，避免固定平分后等待长局。

## H20 远端自对弈

训练、门控和 SwanLab 的 `AlphaZero_Training_Loop` 仍由本机唯一维护，H20
服务器只作为本代棋谱执行器，不会创建第二套训练 loop。启用方式：

```bash
NEBULA_SELFPLAY_BACKEND=remote \
NEBULA_REMOTE_SELFPLAY_HOST=user@remote-host \
NEBULA_REMOTE_SELFPLAY_GPUS=0,1,2,3 \
NEBULA_REMOTE_NUM_WORKERS=4 \
NEBULA_REMOTE_MCTS_THREADS=46 \
NEBULA_REMOTE_CPUSET=0-183 \
python run_forever.py --swanlab-mode online
```

远端执行器会按代增量同步源码；C++ MCTS 源码变化时才重编译，`best.pth`
变化时才在 H20 上重建专用 TensorRT 引擎。棋谱先保存在远端，再原子取回本机；
网络中断后会比较两侧已完成局数续跑。训练和 SwanLab 始终等待棋谱完整取回后才
进入下一阶段。SSH 使用短时复用连接，减少逐个校验步骤的握手开销。

H20 四卡实测的固定搜索吞吐如下：

| 配置 | CPU 范围 | 固定搜索吞吐 |
| --- | --- | ---: |
| 4 worker × 46 线程 | 0–183 | 115,414 simulations/s |
| 8 worker × 23 线程 | 0–183 | 108,310 simulations/s |
| 12 worker × 15 线程 | 0–183 | 98,892 simulations/s |

所以仍采用每卡一个 worker。`4×46` 的黑 400 / 白 1200 完整自对弈实测为
`4.88 局/s`，300 局约 61–65 秒；与 24 线程做 40 局配对换色为 20:20，
搜索吞吐快约 9.3%。CPU 0–183 使用 184/192 个物理核，给其他任务保留 8 个
物理核；四张卡上的原有进程不会被停止或修改。

## 性能基准

固定模拟次数的端到端基准：

```bash
python ../benchmarks/benchmark_mcts.py \
  --engine checkpoints/current_model.engine \
  --simulations 12000 --batch-size 64 --threads 5 \
  --processes 2 --gpus 6,7 --repeats 5
```

纯 TensorRT batch 曲线：

```bash
CUDA_VISIBLE_DEVICES=6 python ../benchmarks/benchmark_tensorrt_engine.py \
  --engine checkpoints/current_model.engine \
  --batches 16,32,64
```

在本次 A800 验收中，固定 12,000 次模拟的中位耗时从去重前约 `1.15s` 降至
约 `0.43s`，提升约 2.7 倍；完整自对弈通常能省去 80% 以上的重复网络评估。
动态 profile 收窄后，单执行上下文显存从约 `1849MiB` 降至约 `136MiB`。
黑棋 400、白棋 1200 的单卡 50 局样本耗时约 59 秒；卡 6+7 动态调度后约
28 秒。因此每代 300 局约 2.8 分钟、500 局约 4.7 分钟，实际时间会随棋局
长度变化。固定搜索基准中，双卡吞吐约为单卡的 2.03 倍。

默认严格执行完整搜索预算，不启用提前熔断：

```bash
NEBULA_SIMULATIONS_BLACK=400 \
NEBULA_SIMULATIONS_WHITE=1200 \
NEBULA_DYNAMIC_EARLY_STOP=0 \
python pipeline/generate.py --engine checkpoints/current_model.engine --total 300
```

TensorRT 默认使用 tactic 优化等级 5。首次构建约 75 秒，之后会复用
`checkpoints/tensorrt_timing.cache`，同结构模型的实测重建时间约 17 秒。

## 自我对弈与训练

只生成自我对弈数据：

```bash
cd reinforcement_learning
python pipeline/generate.py \
  --engine checkpoints/current_model.engine \
  --out data/raw/gen_data.csv --total 300
```

长期运行完整迭代：

```bash
python run_forever.py --swanlab-mode online
```

监督器每代启动一个新的主循环进程；因此 `config.py` 的修改会在下一代自动生效。
SwanLab 的 `AlphaZero_Training_Loop` 长期维护跨代 incumbent、buffer、耗时和门控
趋势；同一 Project 下的 `gen_N` 只展示本代信息：step 0 记录本代自我对弈与
数据分布，step 1..N 记录逐 epoch 训练指标，最后一步记录本代配对换色评估和
门控结果。两层实验由不同进程维护，互不覆盖。

主循环会依次执行模型导出、引擎构建、自我对弈、训练、配对换色评估和门控。
候选模型只有通过门控才会替换当前最佳模型；循环状态和关键文件均采用原子写入，
中断后可以从已完成阶段继续。

训练精度默认是 BF16，并让卡 6、7 同时可见：

```bash
NEBULA_TRAINING_GPUS=6,7 NEBULA_TRAIN_PRECISION=bf16 python run_forever.py
```

也可以设置为 `fp16` 或 `fp32`。实测全局 batch 386 时，BF16 训练从单卡约
`2322 samples/s` 提升到双卡约 `3834 samples/s`。TensorRT 推理精度与训练
精度分开控制：A800 上 BF16 推理在 batch 64 更慢且数值误差更大，因此自对弈
继续使用 FP16；这不会改变 BF16 训练设置。
