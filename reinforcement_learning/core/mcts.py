import ctypes
import numpy as np
import os
import time
import torch
import tensorrt as trt
import platform

# Use pycuda for buffer management? No, let's use torch pointers for zero-copy if possible, 
# but simple numpy/pycuda is safer for now. Let's stick to torch-tensorrt interop via pointers if we can,
# or just use PyCUDA. To avoid installing pycuda, we can use torch tensors as buffers.

# We need CUDA runtime to manage memory if we don't use torch tensors.
# Actually, we can use torch tensors on GPU and pass their data_ptr() to TensorRT.
# This is the most efficient way (Zero Copy).

# 加载 C++ 动态库
if platform.system() == "Windows":
    lib_name = "mcts.dll"
else:
    lib_name = "libmcts.so"

default_lib_path = os.path.join(os.path.dirname(__file__), lib_name)
# 候选 MCTS 可以通过环境变量单独加载，验证通过前不覆盖生产动态库。
lib_path = os.environ.get('NEBULA_MCTS_LIBRARY', default_lib_path)
mcts_lib = ctypes.CDLL(lib_path)

# 定义 C++ 函数签名
mcts_lib.init_game.argtypes = []
mcts_lib.init_game.restype = None

mcts_lib.play_move.argtypes = [ctypes.c_int]
mcts_lib.play_move.restype = None

mcts_lib.run_mcts_simulations.argtypes = [ctypes.c_int]
mcts_lib.run_mcts_simulations.restype = None

mcts_lib.get_best_move.argtypes = [ctypes.c_float]
mcts_lib.get_best_move.restype = ctypes.c_int

mcts_lib.get_root_value.argtypes = []
mcts_lib.get_root_value.restype = ctypes.c_float

for stats_name in (
    'get_total_leaf_requests',
    'get_total_search_batches',
    'get_total_batch_unique_positions',
    'get_total_unique_evaluations',
    'get_total_second_stone_evaluations',
    'get_total_eval_batches',
    'get_total_cache_hits',
    'get_total_cache_misses',
    'get_eval_cache_size',
    'get_total_pair_provisional_evaluations',
    'get_total_pair_exact_refreshes',
    'get_total_pair_eager_exact_evaluations',
):
    if hasattr(mcts_lib, stats_name):
        stats_function = getattr(mcts_lib, stats_name)
        stats_function.argtypes = []
        stats_function.restype = ctypes.c_longlong

mcts_lib.get_policy.argtypes = [ctypes.POINTER(ctypes.c_float)]
mcts_lib.get_policy.restype = None

mcts_lib.print_top_moves.argtypes = []
mcts_lib.print_top_moves.restype = None

mcts_lib.set_random_seed.argtypes = [ctypes.c_int]
mcts_lib.set_random_seed.restype = None

# Safe binding for new params function (backward compatibility)
if hasattr(mcts_lib, 'set_mcts_params'):
    mcts_lib.set_mcts_params.argtypes = [ctypes.c_int, ctypes.c_int]
    mcts_lib.set_mcts_params.restype = None

if hasattr(mcts_lib, 'set_mcts_search_params'):
    mcts_lib.set_mcts_search_params.argtypes = [
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_float,
    ]
    mcts_lib.set_mcts_search_params.restype = None

if hasattr(mcts_lib, 'set_mcts_selection_mode'):
    mcts_lib.set_mcts_selection_mode.argtypes = [ctypes.c_int]
    mcts_lib.set_mcts_selection_mode.restype = None

if hasattr(mcts_lib, 'set_eval_cache_capacity'):
    mcts_lib.set_eval_cache_capacity.argtypes = [ctypes.c_longlong]
    mcts_lib.set_eval_cache_capacity.restype = None
    mcts_lib.clear_eval_cache.argtypes = []
    mcts_lib.clear_eval_cache.restype = None

if hasattr(mcts_lib, 'reset_mcts_statistics'):
    mcts_lib.reset_mcts_statistics.argtypes = []
    mcts_lib.reset_mcts_statistics.restype = None

# 多棋局上下文是可选 ABI，旧动态库仍可照常运行单棋局代码。
_MULTI_CONTEXT_FUNCTIONS = (
    'create_mcts_context',
    'destroy_mcts_context',
    'reset_mcts_context',
    'set_mcts_context_seed',
    'play_move_context',
    'run_mcts_simulations_context',
    'run_mcts_simulations_multi',
    'get_best_move_context',
    'get_root_value_context',
    'get_policy_context',
)
HAS_MULTI_CONTEXT = all(hasattr(mcts_lib, name) for name in _MULTI_CONTEXT_FUNCTIONS)

if HAS_MULTI_CONTEXT:
    mcts_lib.create_mcts_context.argtypes = [ctypes.c_int]
    mcts_lib.create_mcts_context.restype = ctypes.c_void_p
    mcts_lib.destroy_mcts_context.argtypes = [ctypes.c_void_p]
    mcts_lib.destroy_mcts_context.restype = None
    mcts_lib.reset_mcts_context.argtypes = [ctypes.c_void_p]
    mcts_lib.reset_mcts_context.restype = None
    mcts_lib.set_mcts_context_seed.argtypes = [ctypes.c_void_p, ctypes.c_int]
    mcts_lib.set_mcts_context_seed.restype = None
    mcts_lib.play_move_context.argtypes = [ctypes.c_void_p, ctypes.c_int]
    mcts_lib.play_move_context.restype = None
    mcts_lib.run_mcts_simulations_context.argtypes = [ctypes.c_void_p, ctypes.c_int]
    mcts_lib.run_mcts_simulations_context.restype = None
    mcts_lib.run_mcts_simulations_multi.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    mcts_lib.run_mcts_simulations_multi.restype = None
    mcts_lib.get_best_move_context.argtypes = [ctypes.c_void_p, ctypes.c_float]
    mcts_lib.get_best_move_context.restype = ctypes.c_int
    mcts_lib.get_root_value_context.argtypes = [ctypes.c_void_p]
    mcts_lib.get_root_value_context.restype = ctypes.c_float
    mcts_lib.get_policy_context.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
    ]
    mcts_lib.get_policy_context.restype = None
    if hasattr(mcts_lib, 'get_root_action_count_context'):
        mcts_lib.get_root_action_count_context.argtypes = [ctypes.c_void_p]
        mcts_lib.get_root_action_count_context.restype = ctypes.c_int
    if hasattr(mcts_lib, 'get_root_active_action_count_context'):
        mcts_lib.get_root_active_action_count_context.argtypes = [ctypes.c_void_p]
        mcts_lib.get_root_active_action_count_context.restype = ctypes.c_int
    if hasattr(mcts_lib, 'print_top_moves_context'):
        mcts_lib.print_top_moves_context.argtypes = [ctypes.c_void_p]
        mcts_lib.print_top_moves_context.restype = None
    if hasattr(mcts_lib, 'get_second_stone_visit_counts_context'):
        mcts_lib.get_second_stone_visit_counts_context.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_longlong),
        ]
        mcts_lib.get_second_stone_visit_counts_context.restype = None

# 回调函数类型: (batch_size, boards_ptr, policies_ptr, values_ptr)
# batch_size: int
# boards_ptr: int* (flattened batch)
# policies_ptr: float* (flattened batch)
# values_ptr: float*
CALLBACK_FUNC_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float))

PAIR_CALLBACK_FUNC_TYPE = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_uint16),
    ctypes.POINTER(ctypes.c_uint16),
    ctypes.POINTER(ctypes.c_uint16),
)

mcts_lib.set_eval_callback.argtypes = [CALLBACK_FUNC_TYPE]
mcts_lib.set_eval_callback.restype = None
if hasattr(mcts_lib, 'set_pair_eval_callback'):
    mcts_lib.set_pair_eval_callback.argtypes = [PAIR_CALLBACK_FUNC_TYPE]
    mcts_lib.set_pair_eval_callback.restype = None
if hasattr(mcts_lib, 'set_pair_policy_params'):
    mcts_lib.set_pair_policy_params.argtypes = [
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
    ]
    mcts_lib.set_pair_policy_params.restype = None
if hasattr(mcts_lib, 'set_pair_refresh_mode'):
    mcts_lib.set_pair_refresh_mode.argtypes = [ctypes.c_int]
    mcts_lib.set_pair_refresh_mode.restype = None
if hasattr(mcts_lib, 'set_pair_value_scale'):
    mcts_lib.set_pair_value_scale.argtypes = [ctypes.c_float]
    mcts_lib.set_pair_value_scale.restype = None
if hasattr(mcts_lib, 'set_pair_relative_gate'):
    mcts_lib.set_pair_relative_gate.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
    ]
    mcts_lib.set_pair_relative_gate.restype = None
if hasattr(mcts_lib, 'set_pair_refresh_visits_by_player'):
    mcts_lib.set_pair_refresh_visits_by_player.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
    ]
    mcts_lib.set_pair_refresh_visits_by_player.restype = None

class MCTSEngine:
    def __init__(self, engine_path, device='cuda', pair_heads_path=None):
        self.device = device
        self.engine_path = engine_path
        
        # 1. Initialize TensorRT
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(self.engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"无法反序列化 TensorRT 引擎: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"无法创建 TensorRT 执行上下文: {self.engine_path}")
        
        # 2. Allocate Buffers (using Torch for GPU memory management)
        # Input: (32, 17, 19, 19)
        self.max_batch_size = 32
        
        # Inspect bindings to find input name and index
        num_io = self.engine.num_io_tensors
        input_name = None
        move1_name = None
        raw_board_name = None
        
        for i in range(num_io):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                if 'board' in name:
                    raw_board_name = name
                    input_name = name
                elif 'input' in name:
                    input_name = name
                elif 'move1' in name or 'idx' in name:
                    move1_name = name
        
        if input_name is None:
            raise RuntimeError("No input tensor found in TensorRT engine!")

        self.input_name = input_name
        self.move1_name = move1_name
        self.fused_selfplay = raw_board_name is not None
        min_shape, _, max_shape = self.engine.get_tensor_profile_shape(input_name, 0)
        self.profile_min_batch = int(min_shape[0])
        self.profile_max_batch = int(max_shape[0])
        self.max_batch_size = max(
            self.profile_min_batch,
            min(self.max_batch_size, self.profile_max_batch),
        )

        # Set input shape explicitly (Required for dynamic shape engines)
        if self.fused_selfplay:
            self._set_input_shape_checked(input_name, (self.max_batch_size, 19, 19))
        else:
            self._set_input_shape_checked(input_name, (self.max_batch_size, 17, 19, 19))
        if move1_name:
            self._set_input_shape_checked(move1_name, (self.max_batch_size,))
        self.execution_batch_size = self.max_batch_size
        output_names = {
            self.engine.get_tensor_name(index)
            for index in range(num_io)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index))
            == trt.TensorIOMode.OUTPUT
        }
        pair_outputs = {'pair_candidate', 'pair_first', 'pair_value'}
        self.pair_enabled = pair_outputs.issubset(output_names)
        self.pair_heads_path = pair_heads_path or os.environ.get('NEBULA_PAIR_HEADS')
        if self.pair_enabled:
            if not self.fused_selfplay:
                raise RuntimeError('成对落子增强输出目前只支持融合棋盘输入引擎')
            required_pair_abi = (
                hasattr(mcts_lib, 'set_pair_eval_callback')
                and hasattr(mcts_lib, 'set_pair_policy_params')
            )
            if not required_pair_abi:
                raise RuntimeError('当前 MCTS 动态库不支持成对落子增强回调')
            if not self.pair_heads_path:
                raise ValueError(
                    '增强 TensorRT 引擎需要 pair_heads_path，或设置 NEBULA_PAIR_HEADS'
                )
            pair_shape = tuple(self.context.get_tensor_shape('pair_candidate'))
            self.pair_rank = int(pair_shape[-1])
            if self.pair_rank != 16:
                raise ValueError(f'候选 MCTS 目前固定支持 pair rank=16，实际为 {self.pair_rank}')
        else:
            self.pair_rank = 0
        
        # Bindings
        # We need to map binding indices to pointers
        # For execute_v2, it expects a list of pointers in order of binding indices
        
        # Note: In newer TRT, bindings are deprecated for set_tensor_address, 
        # but execute_v2 still takes list of pointers.
        # We need to ensure the order matches the engine's binding order.
        
        input_shape = (
            (self.max_batch_size, 19, 19)
            if self.fused_selfplay
            else (self.max_batch_size, 17, 19, 19)
        )
        input_dtype = torch.int32 if self.fused_selfplay else torch.float32
        self.input_tensor = torch.zeros(input_shape, dtype=input_dtype, device=self.device)
        self.board_staging_tensor = torch.empty(
            (self.max_batch_size, 19, 19), dtype=torch.int32, device=self.device
        )
        self.board_host_tensor = torch.empty(
            (self.max_batch_size, 19, 19), dtype=torch.int32, pin_memory=True
        )
        self.move1_tensor = torch.full((self.max_batch_size,), 361, dtype=torch.long, device=self.device) # 361 is standard SOS/Padding

        self.policy1_tensor = torch.zeros((self.max_batch_size, 361), dtype=torch.float32, device=self.device)
        self.policy2_tensor = torch.zeros((self.max_batch_size, 361), dtype=torch.float32, device=self.device)
        self.value_tensor = torch.zeros((self.max_batch_size, 1), dtype=torch.float32, device=self.device)
        self.policy_host_tensor = torch.empty(
            (self.max_batch_size, 361), dtype=torch.float32, pin_memory=True
        )
        self.value_host_tensor = torch.empty(
            (self.max_batch_size,), dtype=torch.float32, pin_memory=True
        )
        if self.pair_enabled:
            for name in ('pair_candidate', 'pair_first', 'pair_value'):
                if self.engine.get_tensor_dtype(name) != trt.float16:
                    raise TypeError(f'{name} 必须是 FP16 输出')
            factor_shape = (self.max_batch_size, 361, self.pair_rank)
            self.pair_candidate_tensor = torch.empty(
                factor_shape,
                dtype=torch.float16,
                device=self.device,
            )
            self.pair_first_tensor = torch.empty(
                factor_shape,
                dtype=torch.float16,
                device=self.device,
            )
            self.pair_value_tensor = torch.empty(
                (self.max_batch_size, 361),
                dtype=torch.float16,
                device=self.device,
            )
            self.pair_candidate_host_tensor = torch.empty(
                factor_shape,
                dtype=torch.float16,
                pin_memory=True,
            )
            self.pair_first_host_tensor = torch.empty(
                factor_shape,
                dtype=torch.float16,
                pin_memory=True,
            )
            self.pair_value_host_tensor = torch.empty(
                (self.max_batch_size, 361),
                dtype=torch.float16,
                pin_memory=True,
            )
        else:
            self.pair_candidate_tensor = None
            self.pair_first_tensor = None
            self.pair_value_tensor = None
            self.pair_candidate_host_tensor = None
            self.pair_first_host_tensor = None
            self.pair_value_host_tensor = None
        
        # Map pointers by name
        self.context.set_tensor_address(input_name, int(self.input_tensor.data_ptr()))
        if move1_name:
            self.context.set_tensor_address(move1_name, int(self.move1_tensor.data_ptr()))

        # Find output names
        for i in range(num_io):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                if name == 'pair_candidate':
                    self.context.set_tensor_address(
                        name,
                        int(self.pair_candidate_tensor.data_ptr()),
                    )
                elif name == 'pair_first':
                    self.context.set_tensor_address(
                        name,
                        int(self.pair_first_tensor.data_ptr()),
                    )
                elif name == 'pair_value':
                    self.context.set_tensor_address(
                        name,
                        int(self.pair_value_tensor.data_ptr()),
                    )
                elif 'policy1' in name:
                    self.context.set_tensor_address(name, int(self.policy1_tensor.data_ptr()))
                elif 'policy2' in name:
                    self.context.set_tensor_address(name, int(self.policy2_tensor.data_ptr()))
                elif 'value' in name:
                    self.context.set_tensor_address(name, int(self.value_tensor.data_ptr()))
        
        # 初始化 C++ 引擎
        mcts_lib.init_game()
        
        # 融合引擎已包含规则修正；旧引擎仍在 Python 侧执行。
        self.rule_helper = None if self.fused_selfplay else RuleHelper(self.device)

        # 可选的细粒度性能统计，默认关闭，避免基准外运行产生额外事件开销。
        self.profile_enabled = os.environ.get('NEBULA_PROFILE_MCTS', '0') == '1'
        self.profile_interval = int(os.environ.get('NEBULA_PROFILE_INTERVAL', '100'))
        self.profile_calls = 0
        self.profile_positions = 0
        self.profile_cpu_seconds = 0.0
        self.profile_gpu_ms = {
            'input': 0.0,
            'tensorrt': 0.0,
            'rules': 0.0,
            'postprocess': 0.0,
            'd2h': 0.0,
        }
        if self.profile_enabled:
            self.profile_events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]

        # TensorRT 在 CUDA 默认流上会额外插入同步。自对弈的回调本来就是
        # 串行等待结果，因此使用独立非默认流既保持语义，又能去掉这部分
        # 隐式同步，并为后续 CUDA Graph 捕获提供稳定的流和地址。
        self.inference_stream = torch.cuda.Stream(device=self.device)
        self.cuda_graph_enabled = (
            self.fused_selfplay
            and os.environ.get('NEBULA_CUDA_GRAPH', '1') == '1'
        )
        self.cuda_graphs = {}
        self.cuda_graph_buckets = []
        
        # 注册回调
        self.c_callback = CALLBACK_FUNC_TYPE(self._eval_callback)
        self.c_pair_callback = (
            PAIR_CALLBACK_FUNC_TYPE(self._eval_pair_callback)
            if self.pair_enabled
            else None
        )
        self.null_pair_callback = PAIR_CALLBACK_FUNC_TYPE()
        if self.pair_enabled:
            saved_heads = torch.load(
                self.pair_heads_path,
                map_location='cpu',
                weights_only=True,
            )
            pair_state = saved_heads['pair_head']
            self.pair_base_scale = float(pair_state['base_scale'])
            self.pair_relative_bias = np.ascontiguousarray(
                pair_state['relative_bias'].float().numpy(),
                dtype=np.float32,
            )
            self.pair_refresh_visits = int(
                os.environ.get('NEBULA_PAIR_REFRESH_VISITS', '2')
            )
            self.pair_black_refresh_visits = int(
                os.environ.get(
                    'NEBULA_PAIR_REFRESH_VISITS_BLACK',
                    str(self.pair_refresh_visits),
                )
            )
            self.pair_white_refresh_visits = int(
                os.environ.get(
                    'NEBULA_PAIR_REFRESH_VISITS_WHITE',
                    str(self.pair_refresh_visits),
                )
            )
            relative_gate = pair_state.get('relative_gate')
            if relative_gate is None:
                self.pair_relative_gate = None
            else:
                self.pair_relative_gate = np.ascontiguousarray(
                    relative_gate.float().numpy(),
                    dtype=np.float32,
                )
            self.pair_deferred_refresh = (
                os.environ.get('NEBULA_PAIR_DEFER_REFRESH', '1') == '1'
            )
            self.pair_value_scale = float(
                os.environ.get('NEBULA_PAIR_VALUE_SCALE', '1.0')
            )
        self._activate_callbacks()
        if hasattr(mcts_lib, 'set_eval_cache_capacity'):
            cache_capacity = int(os.environ.get('NEBULA_EVAL_CACHE_CAPACITY', '32768'))
            mcts_lib.set_eval_cache_capacity(max(cache_capacity, 0))

    def _activate_callbacks(self):
        """多个 Python 引擎共享动态库时，搜索前显式切换完整回调集合。"""

        mcts_lib.set_eval_callback(self.c_callback)
        if hasattr(mcts_lib, 'set_pair_eval_callback'):
            callback = (
                self.c_pair_callback
                if self.pair_enabled
                else self.null_pair_callback
            )
            mcts_lib.set_pair_eval_callback(callback)
        if self.pair_enabled:
            # 两个候选模型共用同一个 C++ 动态库时，方向门控和相对偏置也
            # 必须随回调一起切换，否则后创建的模型会污染前一个模型的门禁。
            mcts_lib.set_pair_policy_params(
                self.pair_base_scale,
                self.pair_relative_bias.ctypes.data_as(
                    ctypes.POINTER(ctypes.c_float)
                ),
                self.pair_relative_bias.size,
                max(self.pair_refresh_visits, 1),
            )
            mcts_lib.set_pair_refresh_visits_by_player(
                max(self.pair_black_refresh_visits, 1),
                max(self.pair_white_refresh_visits, 1),
            )
            if self.pair_relative_gate is None:
                mcts_lib.set_pair_relative_gate(None, 0, self.pair_rank)
            else:
                mcts_lib.set_pair_relative_gate(
                    self.pair_relative_gate.ctypes.data_as(
                        ctypes.POINTER(ctypes.c_float)
                    ),
                    self.pair_relative_gate.size,
                    self.pair_rank,
                )
            mcts_lib.set_pair_refresh_mode(
                1 if self.pair_deferred_refresh else 0
            )
            mcts_lib.set_pair_value_scale(self.pair_value_scale)

    def _set_input_shape_checked(self, name, shape):
        """设置动态形状，并在运行配置超过引擎 profile 时尽早报错。"""

        if not self.context.set_input_shape(name, shape):
            raise ValueError(
                f"TensorRT profile 不支持输入 {name}={shape}；"
                "请用相同的 NEBULA_MCTS_BATCH_SIZE 重新构建引擎"
            )

    def _validate_batch_size(self, batch_size):
        if not self.profile_min_batch <= batch_size <= self.profile_max_batch:
            raise ValueError(
                f"MCTS batch={batch_size} 超出引擎 profile "
                f"[{self.profile_min_batch}, {self.profile_max_batch}]；"
                "请用相同的 NEBULA_MCTS_BATCH_SIZE 重新构建引擎"
            )

    def _resize_buffers(self, batch_size):
        """在搜索开始前扩容，避免第一次回调临时重新分配显存。"""
        self._validate_batch_size(batch_size)
        if batch_size <= self.input_tensor.size(0):
            return

        self.max_batch_size = batch_size
        input_shape = (
            (batch_size, 19, 19)
            if self.fused_selfplay
            else (batch_size, 17, 19, 19)
        )
        input_dtype = torch.int32 if self.fused_selfplay else torch.float32
        self.input_tensor = torch.zeros(input_shape, dtype=input_dtype, device=self.device)
        self.board_staging_tensor = torch.empty(
            (batch_size, 19, 19), dtype=torch.int32, device=self.device
        )
        self.board_host_tensor = torch.empty(
            (batch_size, 19, 19), dtype=torch.int32, pin_memory=True
        )
        self.move1_tensor = torch.full(
            (batch_size,), 361, dtype=torch.long, device=self.device
        )
        self.policy1_tensor = torch.zeros(
            (batch_size, 361), dtype=torch.float32, device=self.device
        )
        self.policy2_tensor = torch.zeros(
            (batch_size, 361), dtype=torch.float32, device=self.device
        )
        self.value_tensor = torch.zeros(
            (batch_size, 1), dtype=torch.float32, device=self.device
        )
        self.policy_host_tensor = torch.empty(
            (batch_size, 361), dtype=torch.float32, pin_memory=True
        )
        self.value_host_tensor = torch.empty(
            (batch_size,), dtype=torch.float32, pin_memory=True
        )
        if self.pair_enabled:
            factor_shape = (batch_size, 361, self.pair_rank)
            self.pair_candidate_tensor = torch.empty(
                factor_shape, dtype=torch.float16, device=self.device
            )
            self.pair_first_tensor = torch.empty(
                factor_shape, dtype=torch.float16, device=self.device
            )
            self.pair_value_tensor = torch.empty(
                (batch_size, 361), dtype=torch.float16, device=self.device
            )
            self.pair_candidate_host_tensor = torch.empty(
                factor_shape, dtype=torch.float16, pin_memory=True
            )
            self.pair_first_host_tensor = torch.empty(
                factor_shape, dtype=torch.float16, pin_memory=True
            )
            self.pair_value_host_tensor = torch.empty(
                (batch_size, 361), dtype=torch.float16, pin_memory=True
            )

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                if 'board' in name:
                    self._set_input_shape_checked(name, (batch_size, 19, 19))
                    self.context.set_tensor_address(name, int(self.input_tensor.data_ptr()))
                elif 'input' in name:
                    self._set_input_shape_checked(name, (batch_size, 17, 19, 19))
                    self.context.set_tensor_address(name, int(self.input_tensor.data_ptr()))
                elif 'move1' in name or 'idx' in name:
                    self._set_input_shape_checked(name, (batch_size,))
                    self.context.set_tensor_address(name, int(self.move1_tensor.data_ptr()))
            elif name == 'pair_candidate':
                self.context.set_tensor_address(
                    name, int(self.pair_candidate_tensor.data_ptr())
                )
            elif name == 'pair_first':
                self.context.set_tensor_address(
                    name, int(self.pair_first_tensor.data_ptr())
                )
            elif name == 'pair_value':
                self.context.set_tensor_address(
                    name, int(self.pair_value_tensor.data_ptr())
                )
            elif 'policy1' in name:
                self.context.set_tensor_address(name, int(self.policy1_tensor.data_ptr()))
            elif 'policy2' in name:
                self.context.set_tensor_address(name, int(self.policy2_tensor.data_ptr()))
            elif 'value' in name:
                self.context.set_tensor_address(name, int(self.value_tensor.data_ptr()))
        self.execution_batch_size = batch_size

    def _set_execution_batch_size(self, batch_size):
        """动态 profile 按真实叶子数执行，避免尾批仍计算整块容量。"""
        self._validate_batch_size(batch_size)
        if batch_size == self.execution_batch_size:
            return
        input_shape = (
            (batch_size, 19, 19)
            if self.fused_selfplay
            else (batch_size, 17, 19, 19)
        )
        self._set_input_shape_checked(self.input_name, input_shape)
        if self.move1_name:
            self._set_input_shape_checked(self.move1_name, (batch_size,))
        self.execution_batch_size = batch_size

    def _copy_results_to_cpp(
        self,
        probabilities,
        values,
        batch_size,
        policies_ptr,
        values_ptr,
        cpu_started_at=None,
        profile_events=None,
        pair_ptrs=None,
    ):
        """复用页锁定缓冲区，一次同步后把结果交还给 C++。"""
        self.policy_host_tensor[:batch_size].copy_(probabilities, non_blocking=True)
        self.value_host_tensor[:batch_size].copy_(values.flatten(), non_blocking=True)
        if pair_ptrs is not None:
            self.pair_candidate_host_tensor[:batch_size].copy_(
                self.pair_candidate_tensor[:batch_size],
                non_blocking=True,
            )
            self.pair_first_host_tensor[:batch_size].copy_(
                self.pair_first_tensor[:batch_size],
                non_blocking=True,
            )
            self.pair_value_host_tensor[:batch_size].copy_(
                self.pair_value_tensor[:batch_size],
                non_blocking=True,
            )

        if profile_events is not None:
            profile_copy = profile_events[-1]
            profile_copy.record()
            profile_copy.synchronize()
        else:
            torch.cuda.current_stream().synchronize()

        self._write_host_results_to_cpp(batch_size, policies_ptr, values_ptr)
        if pair_ptrs is not None:
            self._write_pair_results_to_cpp(batch_size, *pair_ptrs)

        if profile_events is None:
            return

        self.profile_calls += 1
        self.profile_positions += batch_size
        self.profile_cpu_seconds += time.perf_counter() - cpu_started_at
        profile_start, profile_input, profile_trt, profile_rules, profile_post, profile_copy = profile_events
        event_pairs = [
            ('input', profile_start, profile_input),
            ('tensorrt', profile_input, profile_trt),
            ('rules', profile_trt, profile_rules),
            ('postprocess', profile_rules, profile_post),
            ('d2h', profile_post, profile_copy),
        ]
        for name, start_event, end_event in event_pairs:
            self.profile_gpu_ms[name] += start_event.elapsed_time(end_event)

        if self.profile_calls % self.profile_interval == 0:
            total_gpu_ms = sum(self.profile_gpu_ms.values())
            positions_per_second = self.profile_positions / max(self.profile_cpu_seconds, 1e-9)
            breakdown = ', '.join(
                f"{name}={duration / max(total_gpu_ms, 1e-9):.1%}"
                for name, duration in self.profile_gpu_ms.items()
            )
            print(
                f"[MCTS Profile] {self.profile_calls} batches | "
                f"{positions_per_second:.0f} positions/s | {breakdown}"
            )

    def _write_host_results_to_cpp(self, batch_size, policies_ptr, values_ptr):
        """把已经落入页锁定内存的结果复制到 C++ 回调缓冲区。"""

        policy_view = self.policy_host_tensor[:batch_size].numpy()
        value_view = self.value_host_tensor[:batch_size].numpy()
        ctypes.memmove(policies_ptr, policy_view.ctypes.data, policy_view.nbytes)
        ctypes.memmove(values_ptr, value_view.ctypes.data, value_view.nbytes)

    def _write_pair_results_to_cpp(
        self,
        batch_size,
        candidate_ptr,
        first_ptr,
        pair_values_ptr,
    ):
        """FP16 位模式直接交给 C++，避免 Python 逐元素转换。"""

        tensors_and_pointers = (
            (self.pair_candidate_host_tensor[:batch_size], candidate_ptr),
            (self.pair_first_host_tensor[:batch_size], first_ptr),
            (self.pair_value_host_tensor[:batch_size], pair_values_ptr),
        )
        for tensor, pointer in tensors_and_pointers:
            view = tensor.numpy()
            ctypes.memmove(pointer, view.ctypes.data, view.nbytes)

    def _capture_cuda_graph(self, batch_size):
        """为每个动态 batch 捕获图，避免填充计算和 batch 相关数值变化。"""

        if not self.cuda_graph_enabled:
            return
        if self.cuda_graph_buckets and self.cuda_graph_buckets[-1] == batch_size:
            return

        stream = self.inference_stream
        buckets = list(range(1, batch_size + 1))
        self.cuda_graphs = {}

        for capture_size in buckets:
            self._set_execution_batch_size(capture_size)

            def graph_work(size=capture_size):
                self.input_tensor[:size].copy_(
                    self.board_host_tensor[:size],
                    non_blocking=True,
                )
                success = self.context.execute_async_v3(
                    stream_handle=stream.cuda_stream
                )
                if not success:
                    raise RuntimeError('TensorRT CUDA Graph 预热执行失败')
                self.policy_host_tensor[:size].copy_(
                    self.policy1_tensor[:size],
                    non_blocking=True,
                )
                self.value_host_tensor[:size].copy_(
                    self.value_tensor[:size].flatten(),
                    non_blocking=True,
                )
                if self.pair_enabled:
                    self.pair_candidate_host_tensor[:size].copy_(
                        self.pair_candidate_tensor[:size],
                        non_blocking=True,
                    )
                    self.pair_first_host_tensor[:size].copy_(
                        self.pair_first_tensor[:size],
                        non_blocking=True,
                    )
                    self.pair_value_host_tensor[:size].copy_(
                        self.pair_value_tensor[:size],
                        non_blocking=True,
                    )

            # 每种动态形状先执行一次，避免惰性初始化发生在 capture 中。
            with torch.cuda.stream(stream):
                graph_work()
            stream.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                graph_work()
            stream.synchronize()
            self.cuda_graphs[capture_size] = graph

        self.cuda_graph_buckets = buckets

    def _cuda_graph_bucket(self, batch_size):
        if not self.cuda_graph_enabled or self.profile_enabled:
            return None
        for bucket in self.cuda_graph_buckets:
            if batch_size <= bucket:
                return bucket
        return None
        
    def _eval_callback(self, batch_size, boards_ptr, policies_ptr, values_ptr):
        """把整次 GPU 回调固定到非默认流，避免 TensorRT 的默认流同步。"""

        with torch.cuda.stream(self.inference_stream):
            self._eval_callback_on_stream(
                batch_size,
                boards_ptr,
                policies_ptr,
                values_ptr,
            )

    def _eval_pair_callback(
        self,
        batch_size,
        boards_ptr,
        policies_ptr,
        values_ptr,
        candidate_ptr,
        first_ptr,
        pair_values_ptr,
    ):
        """增强引擎回调：主输出与成对落子辅助输出只同步一次。"""

        with torch.cuda.stream(self.inference_stream):
            self._eval_callback_on_stream(
                batch_size,
                boards_ptr,
                policies_ptr,
                values_ptr,
                pair_ptrs=(candidate_ptr, first_ptr, pair_values_ptr),
            )

    def _eval_callback_on_stream(
        self,
        batch_size,
        boards_ptr,
        policies_ptr,
        values_ptr,
        pair_ptrs=None,
    ):
        """
        Batch Evaluation Callback using TensorRT.
        Vectorized optimization to remove Python loops and leverage GPU.
        """
        cpu_started_at = time.perf_counter() if self.profile_enabled else None
        if self.profile_enabled:
            profile_start, profile_input, profile_trt, profile_rules, profile_post, profile_copy = self.profile_events
            profile_start.record()

        # Dynamic Resizing if batch_size exceeds current max
        if batch_size > self.input_tensor.size(0):
            self._resize_buffers(batch_size)
        graph_bucket = self._cuda_graph_bucket(batch_size)
        if graph_bucket is None:
            self._set_execution_batch_size(batch_size)

        total_elements = batch_size * 361
        
        # 先拷到可复用页锁定内存，再异步传入 GPU；避免每批创建新张量。
        boards_array = np.ctypeslib.as_array(boards_ptr, shape=(total_elements,))
        board_host_view = self.board_host_tensor[:batch_size].numpy()
        np.copyto(board_host_view.reshape(-1), boards_array)

        if graph_bucket is not None:
            self.cuda_graphs[graph_bucket].replay()
            self.inference_stream.synchronize()
            self._write_host_results_to_cpp(batch_size, policies_ptr, values_ptr)
            if pair_ptrs is not None:
                self._write_pair_results_to_cpp(batch_size, *pair_ptrs)
            return

        board_target = self.input_tensor if self.fused_selfplay else self.board_staging_tensor
        board_target[:batch_size].copy_(self.board_host_tensor[:batch_size], non_blocking=True)

        if self.fused_selfplay:
            if self.profile_enabled:
                profile_input.record()
            if not self.context.execute_async_v3(
                stream_handle=torch.cuda.current_stream().cuda_stream
            ):
                raise RuntimeError('TensorRT 融合自对弈推理失败')
            if self.profile_enabled:
                # 规则修正和 softmax 已经在 TensorRT 图内，两个阶段只记录边界。
                profile_trt.record()
                profile_rules.record()
                profile_post.record()
            self._copy_results_to_cpp(
                self.policy1_tensor[:batch_size],
                self.value_tensor[:batch_size],
                batch_size,
                policies_ptr,
                values_ptr,
                cpu_started_at,
                self.profile_events if self.profile_enabled else None,
                pair_ptrs=pair_ptrs,
            )
            return

        boards_gpu = self.board_staging_tensor[:batch_size]
        
        # CRITICAL FIX: Normalize Board Values
        # Ensure White stones are represented as -1, not 2
        # Some C++ engines use 2 for White, while our logic expects -1
        boards_gpu.masked_fill_(boards_gpu == 2, -1)
        
        # 2. Vectorized Feature Engineering
        self.input_tensor.zero_() # Reset tensor
        
        # Count stones per board: (B,)
        n_stones = (boards_gpu != 0).sum(dim=(1, 2)).int()
        
        # Determine Current Player & Stones Remaining (Connect6 Logic)
        # n=0: Black(1), Rem=1
        # n>0, n%2!=0 (Odd: 1,3,5): Rem=2
        # n>0, n%2==0 (Even: 2,4,6): Rem=1
        # Player: ((n+1)//2) % 2 == 0 ? Black(1) : White(-1)
        # n=0 -> (1)//2=0 -> 0%2=0 -> B
        # n=1 -> (2)//2=1 -> 1%2=1 -> W
        # n=2 -> (3)//2=1 -> 1%2=1 -> W
        # n=3 -> (4)//2=2 -> 2%2=0 -> B
        
        stones_rem = torch.where((n_stones > 0) & (n_stones % 2 != 0), 
                                 torch.tensor(2, device=self.device), 
                                 torch.tensor(1, device=self.device))
        
        # 0 is Even, so ((0+1)//2)%2 = 0 -> Black. Correct.
        rank = (n_stones + 1) // 2
        is_white = (rank % 2 == 1)
        curr_player = torch.where(is_white, 
                                  torch.tensor(-1.0, device=self.device), 
                                  torch.tensor(1.0, device=self.device))
        
        # Construct Feature Planes (Vectorized)
        # Plane 0: Self stones
        # Plane 1: Opponent stones
        # Plane 16: Color (1.0 for Black, 0.0 for White)
        
        # Expand dims for broadcasting: (B, 1, 19, 19)
        boards_gpu_expanded = boards_gpu.unsqueeze(1)
        curr_player_expanded = curr_player.view(batch_size, 1, 1, 1)
        
        self.input_tensor[:batch_size, 0:1] = (boards_gpu_expanded == curr_player_expanded).float()
        self.input_tensor[:batch_size, 1:2] = (boards_gpu_expanded == -curr_player_expanded).float()
        
        # Color plane
        # Black (1) -> 1.0, White (-1) -> 0.0
        color_plane = (curr_player == 1.0).float().view(batch_size, 1, 1, 1)
        self.input_tensor[:batch_size, 16:17] = color_plane

        if self.profile_enabled:
            profile_input.record()

        # 3. Inference (TensorRT V3 API)
        if not self.context.execute_async_v3(
            stream_handle=torch.cuda.current_stream().cuda_stream
        ):
            raise RuntimeError('TensorRT 推理失败')

        if self.profile_enabled:
            profile_trt.record()
        
        # 4. Rule-based Reward Shaping (Parallel with TRT Inference)
        # Use slices to avoid copies
        self_stones = self.input_tensor[:batch_size, 0:1]
        opp_stones = self.input_tensor[:batch_size, 1:2]
        
        self_threats = self.rule_helper.detect(self_stones, exclusion_tensor=opp_stones)
        opp_threats = self.rule_helper.detect(opp_stones, exclusion_tensor=self_stones)

        if self.profile_enabled:
            profile_rules.record()
        
        # 5. Process Outputs (Vectorized on GPU)
        # Logits -> Probs
        logits = self.policy1_tensor[:batch_size]
        probs = torch.softmax(logits, dim=1) # Keep on GPU
        
        values = self.value_tensor[:batch_size] # Keep on GPU (B, 1)
        
        # Helper to get batch-wise max (B,) bool tensor
        def get_trigger(threat_dict, key):
            # threat_dict[key] is (B, 1, H, W) -> max over H,W -> (B, 1) -> flatten
            return threat_dict[key].view(batch_size, -1).max(dim=1)[0] > 0.5

        my_win = get_trigger(self_threats, 'win') # (B,)
        my_c5  = get_trigger(self_threats, 'c5')
        my_c4  = get_trigger(self_threats, 'c4')
        
        opp_win = get_trigger(opp_threats, 'win')
        opp_c5  = get_trigger(opp_threats, 'c5')
        opp_c4  = get_trigger(opp_threats, 'c4')
        
        # Flatten values for indexing (B,)
        values_flat = values.flatten()
        
        # Apply Logic using Masks (Priority Order handled by overwriting)
        # All operations are on GPU
        
        # Priority 4: Opponent Threats (Penalties)
        # These are soft hints, easily overwritten by wins/losses
        
        # opp_c4 -> min(v, -0.2)
        values_flat = torch.where(opp_c4, torch.min(values_flat, torch.tensor(-0.2, device=self.device, dtype=torch.float32)), values_flat)
        
        # opp_c5 -> min(v, -0.25)
        values_flat = torch.where(opp_c5, torch.min(values_flat, torch.tensor(-0.25, device=self.device, dtype=torch.float32)), values_flat)
        
        # Priority 3: My Potential Win (If I move, I win)
        # (my_c5) -> 1.0 (Need 1 stone, I have >=1)
        # (my_c4 AND rem >= 2) -> 1.0 (Need 2 stones, I have 2)
        mask_can_win = my_c5 | (my_c4 & (stones_rem >= 2))
        values_flat = torch.where(mask_can_win, torch.tensor(1.0, device=self.device, dtype=torch.float32), values_flat)
        
        # Priority 2: Opponent Actual Win (Game Over - I Lose)
        # If opponent has 6 stones, I lost. It doesn't matter if I "could" have won.
        values_flat = torch.where(opp_win, torch.tensor(-1.0, device=self.device, dtype=torch.float32), values_flat)
        
        # Priority 1: My Actual Win (Game Over - I Won)
        # Rare state: I already have 6 stones? (Shouldn't happen for curr_player usually, but strictly implies 1.0)
        values_flat = torch.where(my_win, torch.tensor(1.0, device=self.device, dtype=torch.float32), values_flat)

        if self.profile_enabled:
            profile_post.record()

        self._copy_results_to_cpp(
            probs,
            values_flat,
            batch_size,
            policies_ptr,
            values_ptr,
            cpu_started_at,
            self.profile_events if self.profile_enabled else None,
        )


    def reset(self):
        mcts_lib.init_game()

    def set_random_seed(self, seed):
        mcts_lib.set_random_seed(seed)
        
    def set_params(self, batch_size=32, num_threads=4):
        """
        Set MCTS execution parameters. Safe to call even if engine doesn't support it.
        :param batch_size: Number of evaluations to batch for GPU.
        :param num_threads: Number of CPU threads for tree search.
        """
        self._resize_buffers(batch_size)
        self._capture_cuda_graph(batch_size)
        if hasattr(mcts_lib, 'set_mcts_params'):
            mcts_lib.set_mcts_params(batch_size, num_threads)
        else:
            print("Warning: set_params ignored (C++ engine too old)")

    def set_search_params(self, cpuct=1.5, widening_base=20, widening_scale=0.25):
        """设置 PUCT 与渐进动作激活；动作始终完整保存在树中，不做永久裁剪。"""

        if not hasattr(mcts_lib, 'set_mcts_search_params'):
            return False
        mcts_lib.set_mcts_search_params(
            float(cpuct),
            int(widening_base),
            float(widening_scale),
        )
        return True

    def set_deterministic_selection(self, enabled):
        """门控可顺序选择保证复现；生产自对弈默认保留并行探索吞吐。"""

        if not hasattr(mcts_lib, 'set_mcts_selection_mode'):
            return False
        mcts_lib.set_mcts_selection_mode(1 if enabled else 0)
        return True

    def update_state(self, move):
        mcts_lib.play_move(move)

    def run_simulations(self, simulations):
        """Run MCTS simulations without returning a move (for dynamic search)"""
        self._activate_callbacks()
        mcts_lib.run_mcts_simulations(simulations)

    @property
    def supports_multi_context(self):
        """当前动态库是否支持一张卡共享多盘棋。"""

        return HAS_MULTI_CONTEXT

    def create_game_context(self, seed=42):
        """创建一盘独立搜索树；所有上下文仍共享本对象的 TensorRT context。"""

        if not HAS_MULTI_CONTEXT:
            raise RuntimeError(
                "当前 MCTS 动态库不支持多棋局上下文；"
                "请加载新版 libmcts.so 或设置 NEBULA_MCTS_LIBRARY"
            )
        return MCTSGameContext(self, seed=seed)

    def run_simulations_multi(self, contexts, simulations):
        """把多盘棋的搜索请求合并成同一个 GPU 推理批次。"""

        contexts = list(contexts)
        if np.isscalar(simulations):
            budgets = [int(simulations)] * len(contexts)
        else:
            budgets = [int(value) for value in simulations]
        if len(contexts) != len(budgets):
            raise ValueError("contexts 与 simulations 的长度必须一致")
        if not contexts:
            return
        if any(context.owner is not self for context in contexts):
            raise ValueError("不能混用来自不同 TensorRT 引擎的 MCTS 上下文")
        if any(context.closed for context in contexts):
            raise RuntimeError("不能搜索已经关闭的 MCTS 上下文")
        if any(value < 0 for value in budgets):
            raise ValueError("模拟次数不能为负数")

        # 多个 MCTSEngine 仍共享同一个 C++ 动态库；调用前明确激活本模型。
        self._activate_callbacks()
        handle_array = (ctypes.c_void_p * len(contexts))(
            *(context.handle for context in contexts)
        )
        budget_array = (ctypes.c_int * len(budgets))(*budgets)
        mcts_lib.run_mcts_simulations_multi(
            handle_array,
            budget_array,
            len(contexts),
        )
        
    def get_mcts_move(self, simulations=1000, temperature=0.0):
        # Support existing calls but allow 0 simulations if run_simulations was called manually
        if simulations > 0:
            self._activate_callbacks()
            mcts_lib.run_mcts_simulations(simulations)
        return mcts_lib.get_best_move(temperature)
        
    def get_win_rate(self):
        val = mcts_lib.get_root_value()
        return float(val)

    def get_search_stats(self):
        """返回本局叶子去重统计；旧版动态库则返回空字典。"""

        required = (
            'get_total_leaf_requests',
            'get_total_unique_evaluations',
            'get_total_eval_batches',
        )
        if not all(hasattr(mcts_lib, name) for name in required):
            return {}
        leaf_requests = int(mcts_lib.get_total_leaf_requests())
        unique_evaluations = int(mcts_lib.get_total_unique_evaluations())
        stats = {
            'leaf_requests': leaf_requests,
            'unique_evaluations': unique_evaluations,
            'eval_batches': int(mcts_lib.get_total_eval_batches()),
            'deduplication_ratio': 1.0 - unique_evaluations / max(leaf_requests, 1),
        }
        if hasattr(mcts_lib, 'get_total_second_stone_evaluations'):
            second_stone_evaluations = int(
                mcts_lib.get_total_second_stone_evaluations()
            )
            stats['second_stone_evaluations'] = second_stone_evaluations
            stats['second_stone_evaluation_ratio'] = (
                second_stone_evaluations / max(unique_evaluations, 1)
            )
        if hasattr(mcts_lib, 'get_total_batch_unique_positions'):
            batch_unique = int(mcts_lib.get_total_batch_unique_positions())
            stats['batch_unique_positions'] = batch_unique
            stats['within_batch_deduplication_ratio'] = (
                1.0 - batch_unique / max(leaf_requests, 1)
            )
            if hasattr(mcts_lib, 'get_total_search_batches'):
                search_batches = int(mcts_lib.get_total_search_batches())
                stats['search_batches'] = search_batches
                stats['average_unique_search_batch'] = (
                    batch_unique / max(search_batches, 1)
                )
        if hasattr(mcts_lib, 'get_total_cache_hits'):
            cache_hits = int(mcts_lib.get_total_cache_hits())
            cache_misses = int(mcts_lib.get_total_cache_misses())
            stats.update({
                'cache_hits': cache_hits,
                'cache_misses': cache_misses,
                'cache_hit_rate': cache_hits / max(cache_hits + cache_misses, 1),
                'cache_size': int(mcts_lib.get_eval_cache_size()),
            })
        if hasattr(mcts_lib, 'get_total_pair_provisional_evaluations'):
            provisional = int(
                mcts_lib.get_total_pair_provisional_evaluations()
            )
            refreshes = int(mcts_lib.get_total_pair_exact_refreshes())
            eager_exact = int(
                mcts_lib.get_total_pair_eager_exact_evaluations()
            )
            stats.update({
                'pair_provisional_evaluations': provisional,
                'pair_exact_refreshes': refreshes,
                'pair_eager_exact_evaluations': eager_exact,
                'pair_saved_full_evaluations': provisional - refreshes,
                'pair_provisional_ratio': provisional / max(leaf_requests, 1),
            })
        stats['average_eval_batch'] = unique_evaluations / max(stats['eval_batches'], 1)
        stats['eval_batch_fill_ratio'] = (
            stats['average_eval_batch'] / max(self.max_batch_size, 1)
        )
        return stats

    def reset_search_stats(self):
        """只重置计数器，保留搜索树和跨批评估缓存。"""

        if hasattr(mcts_lib, 'reset_mcts_statistics'):
            mcts_lib.reset_mcts_statistics()

    def clear_eval_cache(self):
        """模型或推理后处理发生变化时手动清空评估缓存。"""

        if hasattr(mcts_lib, 'clear_eval_cache'):
            mcts_lib.clear_eval_cache()

    def set_eval_cache_capacity(self, capacity):
        """设置每张卡共享的局面缓存容量；设为 0 可做无缓存基准。"""

        if not hasattr(mcts_lib, 'set_eval_cache_capacity'):
            raise RuntimeError("当前 MCTS 动态库不支持跨批评估缓存")
        mcts_lib.set_eval_cache_capacity(max(int(capacity), 0))
        
    def get_policy(self):
        policy = np.zeros(361, dtype=np.float32)
        mcts_lib.get_policy(policy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return policy
    
    def print_top_debug(self):
        mcts_lib.print_top_moves()


class MCTSGameContext:
    """一盘棋的 C++ 棋盘与搜索树句柄，不拥有额外的 TensorRT context。"""

    def __init__(self, owner, seed=42):
        self.owner = owner
        raw_handle = mcts_lib.create_mcts_context(int(seed))
        if not raw_handle:
            raise MemoryError("创建 MCTS 棋局上下文失败")
        self.handle = raw_handle
        self.closed = False

    def _require_open(self):
        if self.closed:
            raise RuntimeError("MCTS 棋局上下文已经关闭")

    def reset(self):
        self._require_open()
        mcts_lib.reset_mcts_context(self.handle)

    def set_random_seed(self, seed):
        self._require_open()
        mcts_lib.set_mcts_context_seed(self.handle, int(seed))

    def update_state(self, move):
        self._require_open()
        mcts_lib.play_move_context(self.handle, int(move))

    def run_simulations(self, simulations):
        self.owner.run_simulations_multi([self], [simulations])

    def get_mcts_move(self, simulations=0, temperature=0.0):
        if simulations > 0:
            self.run_simulations(simulations)
        self._require_open()
        return int(mcts_lib.get_best_move_context(self.handle, float(temperature)))

    def get_win_rate(self):
        self._require_open()
        return float(mcts_lib.get_root_value_context(self.handle))

    def get_policy(self):
        self._require_open()
        policy = np.zeros(361, dtype=np.float32)
        mcts_lib.get_policy_context(
            self.handle,
            policy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return policy

    def get_root_action_count(self):
        self._require_open()
        if not hasattr(mcts_lib, 'get_root_action_count_context'):
            return None
        return int(mcts_lib.get_root_action_count_context(self.handle))

    def get_root_active_action_count(self):
        self._require_open()
        if not hasattr(mcts_lib, 'get_root_active_action_count_context'):
            return None
        return int(mcts_lib.get_root_active_action_count_context(self.handle))

    def print_top_debug(self):
        self._require_open()
        if hasattr(mcts_lib, 'print_top_moves_context'):
            mcts_lib.print_top_moves_context(self.handle)

    def get_second_stone_visit_counts(self, thresholds=(1, 2, 4, 8, 16, 32)):
        """返回第二颗子阶段中访问次数不低于各阈值的累计节点数。"""

        self._require_open()
        if not hasattr(mcts_lib, 'get_second_stone_visit_counts_context'):
            return {}
        thresholds = [max(1, int(value)) for value in thresholds]
        threshold_array = (ctypes.c_int * len(thresholds))(*thresholds)
        output_array = (ctypes.c_longlong * len(thresholds))()
        mcts_lib.get_second_stone_visit_counts_context(
            self.handle,
            threshold_array,
            len(thresholds),
            output_array,
        )
        return {
            threshold: int(output_array[index])
            for index, threshold in enumerate(thresholds)
        }

    def close(self):
        if self.closed:
            return
        mcts_lib.destroy_mcts_context(self.handle)
        self.handle = None
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            # 解释器退出阶段动态库对象可能已经被回收。
            pass


class RuleHelper:
    """
    GPU-based Threat Detector.
    Detects 4, 5, 6 in a row using Convolution.
    """
    def __init__(self, device):
        self.device = device
        self.kernels = self._build_kernels()
        
    def _build_kernels(self):
        import torch.nn.functional as F
        # 4 directions: Horizontal, Vertical, Diagonal, Anti-Diagonal
        # We use simple counting kernels.
        # Shape: (Out, 1, K, K)
        
        # We need to detect length 4, 5, 6.
        # To avoid many separate convs, we can use one set of 6x6 kernels
        # and threshold the output value.
        # Output value 4 => Connect 4 (or jumped 4)
        # Output value 5 => Connect 5
        # Output value 6 => Connect 6
        
        kernels = []
        
        # Horizontal: [1,1,1,1,1,1] (6x6 padded with 0s vertically)
        k_h = torch.zeros((1, 1, 6, 6), device=self.device)
        k_h[0, 0, 0, :] = 1 # Top row
        kernels.append(k_h)
        
        # Vertical
        k_v = torch.zeros((1, 1, 6, 6), device=self.device)
        k_v[0, 0, :, 0] = 1 # Left col
        kernels.append(k_v)
        
        # Diagonal
        k_d = torch.eye(6, device=self.device).reshape(1, 1, 6, 6)
        kernels.append(k_d)
        
        # Anti-Diagonal
        k_ad = torch.rot90(torch.eye(6, device=self.device), 1, [0, 1]).reshape(1, 1, 6, 6)
        kernels.append(k_ad)
        
        return torch.cat(kernels, dim=0) # (4, 1, 6, 6)
        
    def detect(self, board_tensor, exclusion_tensor=None):
        """
        board_tensor: (B, 1, 19, 19)
        exclusion_tensor: (B, 1, 19, 19)
        """
        import torch.nn.functional as F
        
        # Use padding to check boundary conditions robustly
        # We pad inputs so that the 6x6 kernel can slide partially off-board.
        # Board: Pad with 0 (Simulates empty off-board, but since it's "Self", 0 means no stone)
        # Exclusion: Pad with 1 (Simulates Boundary as Blocker)
        pad_size = 5
        
        # Check self counts
        # (B, 1, 19+10, 19+10)
        padded_board = F.pad(board_tensor, (pad_size, pad_size, pad_size, pad_size), value=0)
        out = F.conv2d(padded_board, self.kernels, padding=0)
        
        if exclusion_tensor is not None:
             # Check blockers with Boundary = 1
             padded_excl = F.pad(exclusion_tensor, (pad_size, pad_size, pad_size, pad_size), value=1)
             blocked = F.conv2d(padded_excl, self.kernels, padding=0)
             
             valid_mask = (blocked == 0).float()
             out = out * valid_mask
             
        threats = {}
        # Result map is larger (B, 4, 19+5, 19+5), but we only care about max value existence
        threats['win'] = (out >= 6.0).float()
        threats['c5'] = (out == 5.0).float()
        threats['c4'] = (out == 4.0).float()
        
        return threats
