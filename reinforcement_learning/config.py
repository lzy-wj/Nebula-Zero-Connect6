"""
Connect6 AlphaZero 强化学习配置文件

所有可调参数都在这里，方便修改和调试。
"""
import os

# ==================================
# 可复现性与运行时配置
# ==================================
SEED = int(os.environ.get('NEBULA_SEED', '2026'))

# 训练精度：Ampere/Blackwell 推荐 bf16，也可选 fp16 或 fp32。
TRAIN_PRECISION = os.environ.get('NEBULA_TRAIN_PRECISION', 'bf16').lower()
ALLOW_TF32 = True
USE_FUSED_ADAMW = True
TORCH_COMPILE = False  # 短代训练需先测量编译开销，再决定是否开启。
TRT_PRECISION = os.environ.get('NEBULA_TRT_PRECISION', 'fp16').lower()
TRT_FUSED_SELFPLAY = os.environ.get('NEBULA_TRT_FUSED_SELFPLAY', '1') == '1'
# 使用 Connect6 专用的二维相对位置注意力 AOT 插件。设为 0 可随时回退
# 到 TensorRT 自带的 MHA 融合路径，便于做同模型 A/B 对比。
TRT_CUSTOM_ATTENTION = os.environ.get('NEBULA_TRT_CUSTOM_ATTENTION', '0') == '1'
DATALOADER_WORKERS = 4
DATALOADER_PREFETCH_FACTOR = 2
TRAIN_LOG_INTERVAL = 10
SWANLAB_MODE = os.environ.get('NEBULA_SWANLAB_MODE')  # online/offline/local/disabled
# SwanLab 最外层项目与长期强化学习实验名称，可用环境变量临时覆盖。
SWANLAB_PROJECT = os.environ.get('NEBULA_SWANLAB_PROJECT', 'Nebula-zero-one')
SWANLAB_RUN_PREFIX = os.environ.get('NEBULA_SWANLAB_RUN_PREFIX', 'gen')
SWANLAB_LOOP_RUN_NAME = os.environ.get(
    'NEBULA_SWANLAB_LOOP_RUN_NAME',
    'AlphaZero_Training_Loop',
)
# 云端短暂抖动时不能让代际监督器永久卡在 finish；本地 swanlog 仍会保留。
SWANLAB_FINISH_TIMEOUT = int(os.environ.get('NEBULA_SWANLAB_FINISH_TIMEOUT', '30'))

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
# 初始模型：优先查找监督学习的输出，否则使用 checkpoints/initial.pth
_SL_MODEL_PATH = os.path.join(BASE_DIR, '..', 'supervised_learning', 'checkpoints', 'checkpoint_latest.pth')
_DEFAULT_INITIAL_PATH = os.path.join(CHECKPOINT_DIR, 'initial.pth')

# 自动选择初始模型路径
if os.path.exists(_SL_MODEL_PATH):
    INITIAL_MODEL_PTH = os.path.abspath(_SL_MODEL_PATH)
else:
    INITIAL_MODEL_PTH = _DEFAULT_INITIAL_PATH

INITIAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'initial_model.engine')  # 编译后的 TensorRT 引擎
CURRENT_ENGINE_PATH = os.path.join(CHECKPOINT_DIR, 'current_model.engine') # 当前使用的引擎
CURRENT_MODEL_PTH = os.path.join(CHECKPOINT_DIR, 'best.pth')               # 当前最佳 PyTorch 模型

# ==================================
# 自我对弈参数
# ==================================
# 每卡仍只创建一个 worker 和一个 TensorRT context；worker 内并发推进多盘棋，
# 让不同搜索树共同填充一个网络 batch，避免多个 context 争用同一张卡。
NUM_WORKERS = int(os.environ.get('NEBULA_NUM_WORKERS', '2'))
MCTS_THREADS = int(os.environ.get('NEBULA_MCTS_THREADS', '32'))
MCTS_BATCH_SIZE = int(os.environ.get('NEBULA_MCTS_BATCH_SIZE', '64'))
MCTS_CONCURRENT_GAMES = int(os.environ.get('NEBULA_MCTS_CONCURRENT_GAMES', '12'))
MCTS_EVAL_CACHE_SIZE = int(os.environ.get('NEBULA_MCTS_EVAL_CACHE_SIZE', '32768'))
MCTS_CPUCT = float(os.environ.get('NEBULA_MCTS_CPUCT', '1.5'))
MCTS_WIDENING_BASE = int(os.environ.get('NEBULA_MCTS_WIDENING_BASE', '20'))
MCTS_WIDENING_SCALE = float(os.environ.get('NEBULA_MCTS_WIDENING_SCALE', '0.25'))
MCTS_DETERMINISTIC_SELECTION = (
    os.environ.get('NEBULA_MCTS_DETERMINISTIC_SELECTION', '0') == '1'
)
GPUS = os.environ.get('NEBULA_SELFPLAY_GPUS', '6,7')
SIMULATIONS = int(os.environ.get('NEBULA_SIMULATIONS', '1200'))
SIMULATIONS_BLACK = int(os.environ.get('NEBULA_SIMULATIONS_BLACK', '400'))
SIMULATIONS_WHITE = int(os.environ.get('NEBULA_SIMULATIONS_WHITE', '1200'))
GAMES_PER_LOOP = int(os.environ.get('NEBULA_GAMES_PER_LOOP', '300'))
BATCH_SIZE = 512          # 数据加载批量大小

# 自对弈可以留在本机，也可以交给独立的 GPU 服务器执行。训练、门控和
# AlphaZero_Training_Loop 始终由本地主循环唯一维护，远端只生成本代棋谱。
SELFPLAY_BACKEND = os.environ.get('NEBULA_SELFPLAY_BACKEND', 'local').lower()
REMOTE_SELFPLAY_HOST = os.environ.get('NEBULA_REMOTE_SELFPLAY_HOST', '')
REMOTE_PROJECT_DIR = os.environ.get('NEBULA_REMOTE_PROJECT_DIR', '')
REMOTE_PYTHON = os.environ.get('NEBULA_REMOTE_PYTHON', 'python')
# H20 实测最优生产点：四卡各一个 worker，46 线程时固定搜索吞吐约
# 115k simulations/s；40 局换色对照与 24 线程打平，同时单次搜索快约 9%。
REMOTE_SELFPLAY_GPUS = os.environ.get('NEBULA_REMOTE_SELFPLAY_GPUS', '0,1,2,3')
REMOTE_NUM_WORKERS = int(os.environ.get('NEBULA_REMOTE_NUM_WORKERS', '4'))
REMOTE_MCTS_THREADS = int(os.environ.get('NEBULA_REMOTE_MCTS_THREADS', '46'))
REMOTE_MCTS_BATCH_SIZE = int(os.environ.get('NEBULA_REMOTE_MCTS_BATCH_SIZE', '64'))
REMOTE_MCTS_CONCURRENT_GAMES = int(
    os.environ.get('NEBULA_REMOTE_MCTS_CONCURRENT_GAMES', str(MCTS_CONCURRENT_GAMES))
)
REMOTE_CPUSET = os.environ.get('NEBULA_REMOTE_CPUSET', '0-183')
REMOTE_BUILD_CPUSET = os.environ.get('NEBULA_REMOTE_BUILD_CPUSET', '0-7')
REMOTE_BUILD_GPU = os.environ.get('NEBULA_REMOTE_BUILD_GPU', '0')
REMOTE_SYNC_CODE = os.environ.get('NEBULA_REMOTE_SYNC_CODE', '1') == '1'
REMOTE_SSH_CONTROL_PATH = os.environ.get(
    'NEBULA_REMOTE_SSH_CONTROL_PATH',
    '/tmp/nebula-zero-ssh-%C',
)

# ==================================
# 训练参数
# ==================================
TRAINING_GPU = os.environ.get('NEBULA_TRAINING_GPU', '6')
TRAINING_GPUS = os.environ.get('NEBULA_TRAINING_GPUS', '6,7')
BUFFER_SIZE = 50000       # 回放缓冲区大小（保持高周转以刷新旧数据）
TRAIN_EPOCHS = 3          # 每代训练轮数（动态调整）
BATCH_SIZE_TRAIN = 386    # 训练批量大小
LEARNING_RATE = 1e-5      # 学习率（微调模式，使用较低值）
MIN_LEARNING_RATE = 1e-6  # 最小学习率（余弦退火终点）

# ==================================
# MCTS 温度参数
# ==================================
# 温度控制探索程度：高温 = 更随机，低温 = 更确定
TEMP_OPENING_BLACK = float(os.environ.get('NEBULA_TEMP_OPENING_BLACK', '0.9'))
TEMP_OPENING_WHITE = float(os.environ.get('NEBULA_TEMP_OPENING_WHITE', '0.2'))
TEMP_FINAL = float(os.environ.get('NEBULA_TEMP_FINAL', '0.0'))
OPENING_MOVES = int(os.environ.get('NEBULA_OPENING_MOVES', '12'))

# 可选的合法随机开局课程。被注入的着法不作为策略监督目标，只负责把部分
# 自对弈带到不同且合法的中前盘分布；0.0 保持历史行为不变。
FORCED_OPENING_RATIO = float(os.environ.get('NEBULA_FORCED_OPENING_RATIO', '0.0'))
FORCED_OPENING_STONES = int(os.environ.get('NEBULA_FORCED_OPENING_STONES', '5'))
FORCED_OPENING_RADIUS = int(os.environ.get('NEBULA_FORCED_OPENING_RADIUS', '4'))

# ==================================
# 动态模拟参数
# ==================================
DYNAMIC_CHECK_INTERVAL = 400  # 动态检查间隔（每 N 次模拟检查一次）
DYNAMIC_FUSE_RATIO = 10.0     # 熔断比例（Top1 访问次数 > Top2 * 此值时提前结束）
# 精确搜索深度默认不启用提前熔断；需要做速度/棋力实验时再显式开启。
DYNAMIC_EARLY_STOP = os.environ.get('NEBULA_DYNAMIC_EARLY_STOP', '0') == '1'

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
EVAL_SEED = 2026          # 固定开局套件，保证跨代可比较
EVAL_OPENING_STONES = 5   # 每组换色对局共享的中心区域随机开局

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
