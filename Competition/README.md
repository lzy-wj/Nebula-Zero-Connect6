# Nebula Zero - 六子棋对弈与分析台

这是一个轻量级 Web 对弈服务，用于六子棋 AI 的在线对战、搜索观察和人工评估。

页面目前支持：

- 手动调用 AI，或开启“AI 自动应手”并选择玩家执黑/执白；
- 调整 MCTS 模拟次数、落子温度、GPU Batch 和 CPU 线程数；
- 查看真实搜索进度、耗时、局面胜率和前五个候选落点；
- 着法列表、按对弈回合悔棋；
- JSON 棋谱导入与导出；
- 桌面端与移动端自适应棋盘。

## 目录结构

- web/：Flask 后端与原生 HTML、CSS、Canvas、JavaScript 页面；
- core/：网页兼容入口；实际复用 reinforcement_learning/core 的已验证 MCTS/TensorRT 包装器；
- ai/：可中断思考的 AI Agent；
- utils/：棋谱记录工具。

## 快速运行

1. 安装仓库根目录依赖，并确保 reinforcement_learning/core/libmcts.so 已经编译。
2. 进入 Competition/web 目录。
3. 执行 python app.py 启动服务。
4. 访问 http://127.0.0.1:5000。

如果已经构建 `nebular_zero_two_gen_*_pair.engine`，也可以在仓库根目录直接执行：

```bash
./Competition/run_web_latest.sh
```

脚本默认使用物理 6 号 GPU，并自动选择最新的 Nebular-zero-two 引擎。

如果 MCTS 动态库暂时不可用，服务仍会展示棋盘和模型列表，并在页面内给出具体错误；AI 落子会保持禁用，不会让整个 Web 服务直接崩溃。

## 模型加载

服务默认读取 reinforcement_learning/checkpoints/current_model.engine。
请确保模型与当前 TensorRT、CUDA 环境兼容。

双落子增强引擎会自动读取同名的 `.pair_heads.pt` sidecar。例如：

```text
nebular_zero_two_gen_0216_pair.engine
nebular_zero_two_gen_0216_pair.pair_heads.pt
```

网页模型列表会用“`双落子增强`”标记这类完整模型包。

为避免网页读取任意服务器文件，模型选择接口只接受
reinforcement_learning/checkpoints/ 目录中的 .engine 文件。

## 棋谱格式

页面导出 UTF-8 JSON，核心字段包含 format、version、board_size、moves 和
search_settings。moves 中每手包含 move、color、row、col 与 coord。

导入时只要求存在 moves 数组；每手可以使用导出的对象格式，也可以写成
[row, col]。后端会重新校验占位、棋盘边界和终局状态，并同步重建 MCTS 树。
