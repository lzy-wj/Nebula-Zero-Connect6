# Nebula Zero - 在线对战平台 (Competition Web Arena)

这是一个轻量级的 Web 服务器，用于支持六子棋 AI 的在线对战和评估。前端界面允许用户直接在浏览器中与 AI 下棋。

## 目录结构
- `web/`: Flask 后端与前端页面代码。
- `web/models/`: (可选) 存放用于网页端加载的轻量级模型。

## 快速运行

1. 进入 web 目录：
   ```bash
   cd web
   ```

2. 启动服务器：
   ```bash
   python app.py
   ```

3. 访问本地地址 (通常是 `http://127.0.0.1:5000`) 即可开始对弈。

## 关于模型加载
`app.py` 中会指定加载的引擎路径。默认情况下，它可能会读取 `../reinforcement_learning/checkpoints/current_model.engine`。请确保路径配置正确且存在有效的 TensorRT 引擎文件。
