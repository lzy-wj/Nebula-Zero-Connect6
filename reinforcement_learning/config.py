"""
Connect6 AlphaZero 强化学习配置文件

所有可调参数都在这里，方便修改和调试。
"""
import os

# ==================================
# 路径配置
# ==================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 代码根目录
DATA_DIR = os.path.join(BASE_DIR, 'data')              # 数据目录
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')           # 原始对局数据（每代生成）
BUFFER_DIR = os.path.join(DATA_DIR, 'buffer')          # 回放缓冲区
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints') # 模型检查点
LOG_DIR = os.path.join(BASE_DIR, 'logs')               # 日志和图表

# ==================================
# 模型和引擎路径
# ==================================
# 初始模型：使用监督学习训练好的模型作为起点
# 请根据你的实际路径修改，或将模型放到 checkpoints/initial.pth
INITIAL_MODEL_PTH = os.path.join(CHECKPOINT_DIR, 'initial.pth')
INITIAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'initial_model.engine')  # 编译后的 TensorRT 引擎
CURRENT_ENGINE_PATH = os.path.join(CHECKPOINT_DIR, 'current_model.engine') # 当前使用的引擎
CURRENT_MODEL_PTH = os.path.join(CHECKPOINT_DIR, 'best.pth')               # 当前最佳 PyTorch 模型

# ==================================
# 自我对弈参数
# ==================================
NUM_WORKERS = 4           # 并行 worker 数量（每个 worker 独立运行对局）
MCTS_THREADS = 5          # 每个 worker 的 CPU 线程数
MCTS_BATCH_SIZE = 64      # GPU 批量推理大小（与 threads 配合填满 GPU）
GPUS = "0"                # 使用的 GPU ID（多卡用逗号分隔，如 "0,1"）
SIMULATIONS = 2000        # MCTS 模拟次数（越大越强，但越慢）
SIMULATIONS_BLACK = 200   # 黑棋模拟次数（可与白棋不同以调整平衡）
SIMULATIONS_WHITE = 1200   # 白棋模拟次数
GAMES_PER_LOOP = 300      # 每代生成的对局数
BATCH_SIZE = 512          # 数据加载批量大小

# ==================================
# 训练参数
# ==================================
TRAINING_GPU = "0"        # 训练使用的 GPU
BUFFER_SIZE = 50000       # 回放缓冲区大小（保持高周转以刷新旧数据）
TRAIN_EPOCHS = 3          # 每代训练轮数（动态调整）
BATCH_SIZE_TRAIN = 386    # 训练批量大小
LEARNING_RATE = 1e-5      # 学习率（微调模式，使用较低值）
MIN_LEARNING_RATE = 1e-6  # 最小学习率（余弦退火终点）

# ==================================
# MCTS 温度参数
# ==================================
# 温度控制探索程度：高温 = 更随机，低温 = 更确定
TEMP_OPENING_BLACK = 0.9  # 黑棋开局温度（前 N 步轻微随机）
TEMP_OPENING_WHITE = 0.2  # 白棋开局温度（保持对称）
TEMP_FINAL = 0.0          # 之后完全确定性选择
OPENING_MOVES = 12         # 使用开局温度的步数

# ==================================
# 动态模拟参数
# ==================================
DYNAMIC_CHECK_INTERVAL = 400  # 动态检查间隔（每 N 次模拟检查一次）
DYNAMIC_FUSE_RATIO = 10.0     # 熔断比例（Top1 访问次数 > Top2 * 此值时提前结束）

# ==================================
# 非对称自我对弈参数
# ==================================
ASYMMETRIC_SELFPLAY_RATIO = 0.0   # 使用旧模型作为陪练的比例（0 = 纯自我对弈）
OPPONENT_MODEL_GENERATION_GAP = 100  # 陪练模型与当前代数的差距
OPPONENT_ENGINE_PATH = os.path.join(CHECKPOINT_DIR, 'opponent_model.engine')

# ==================================
# 评估参数
# ==================================
EVAL_GAMES = 30           # 评估时每个对手的对局数
EVAL_SIMULATIONS = 1200   # 评估时的 MCTS 模拟次数

# 静态基准模型（用于跨代比较）
# 这些模型应放在 CHECKPOINT_DIR 目录下
STATIC_BENCHMARKS = [
    'model_gen_10.pth',
    'model_gen_100.pth', 
    'model_gen_200.pth',
]

# 相对代差评估（与当前代相差 N 代的模型对战）
EVAL_GENERATION_OFFSETS = [10, 20, 50]

# ==================================
# 门控阈值
# ==================================
GATING_MIN_WIN_RATE = 0.5       # 最低整体胜率（低于此值不更新主模型）
GATING_MIN_WHITE_WIN_RATE = 0.1  # 最低白棋胜率（防止黑棋偏向）

# ==================================
# 热启动参数
# ==================================
HOT_START_MIN_BUFFER = 2000     # 缓冲区最小样本数（低于此值跳过训练）


def ensure_dirs():
    """创建必要的目录"""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(BUFFER_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
