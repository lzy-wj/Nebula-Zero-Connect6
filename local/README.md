# Nebula Zero - 本地客户端 (Local Client)

本目录包含用于与 Nebula Zero AI 对弈的本地 GUI 客户端。界面使用 PyQt5 构建，底层核心 MCTS 计算由高性能 C++ 引擎驱动，神经网络推理使用 TensorRT 加速。

## 依赖要求 (Prerequisites)

**系统环境**:
- Python 3.8+
- NVIDIA GPU (带 CUDA 支持)
- C++ 编译器 (Windows: MSVC, Linux: GCC/Clang)

**Python 依赖**:
```bash
pip install PyQt5 numpy torch tensorrt
```

**TensorRT 安装**:
TensorRT 需要从 [NVIDIA 官网](https://developer.nvidia.com/tensorrt) 下载与你的 CUDA 版本匹配的安装包。安装后确保 `python -c "import tensorrt"` 不报错。

---

## 第一步：编译 C++ MCTS 引擎

核心 MCTS 引擎是用 C++ 编写的，你需要将其编译为动态链接库才能运行 Python 界面。

我们提供了一个跨平台编译脚本 `mcts/compile_dll.py`，支持 Windows 和 Linux。

### Windows
1. 打开 **"x64 Native Tools Command Prompt for VS 2022"** (或类似工具)。
2. 运行：
   ```cmd
   cd local/mcts
   python compile_dll.py
   ```

### Linux
确保已安装 `g++`，然后运行：
```bash
cd local/mcts
python compile_dll.py
```

脚本会自动检测操作系统，编译并将库文件复制到正确位置。

---

## 第二步：部署神经网络模型

如果你在服务器上训练了模型，想要在本地机器上运行，**不能直接拷贝 .engine 文件**（因为 TensorRT 引擎是硬件绑定的）。

你需要下载训练好的 `.pth` 权重文件，并在本地重新编译。

### 编译与部署指令

1. **准备模型**: 
   将下载好的 `.pth` 文件（例如 `best.pth`）放入 `local/tools/` 目录。

2. **一键编译与部署**:
   进入 `local/tools` 目录，运行以下指令：

   ```bash
   cd local/tools
   
   # 设置模型文件名 (不带 .pth 后缀)
   MODEL_NAME="best" 
   
   # 导出 ONNX
   python export_onnx.py ${MODEL_NAME}.pth model.onnx
   
   # 编译 TensorRT 引擎
   python build_engine.py model.onnx final.engine
   
   # 部署到游戏目录 (覆盖旧引擎)
   mv final.engine ../engine/current_model.engine
   
   # 清理中间文件
   rm model.onnx
   
   echo "部署完成！"
   ```

   > **Windows 用户**: 请依次执行上述命令，将 `mv` 替换为 `move`，`rm` 替换为 `del`。

---

## 第三步：启动游戏

完成上述两步后，运行以下命令启动界面：

```bash
cd local
python main.py
```

---

## 常见问题

- **找不到 DLL/SO**: 请检查 `local/core/` 目录下是否存在 `mcts.dll` (Windows) 或 `libmcts.so` (Linux)。
- **TensorRT 报错**: 请确保 TensorRT 版本与 CUDA 匹配，且 `tensorrt` Python 包已正确安装。
- **模型加载失败**: 确认 `local/engine/current_model.engine` 文件存在且是在当前机器上编译的。
