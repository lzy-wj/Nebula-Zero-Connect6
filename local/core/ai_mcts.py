import ctypes
import numpy as np
import os
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

lib_path = os.path.join(os.path.dirname(__file__), lib_name)
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

if hasattr(mcts_lib, 'set_pruning_k'):
    mcts_lib.set_pruning_k.argtypes = [ctypes.c_int]
    mcts_lib.set_pruning_k.restype = None

# reexpand_root: 重新展开根节点，补充被剪枝的候选
if hasattr(mcts_lib, 'reexpand_root'):
    mcts_lib.reexpand_root.argtypes = [ctypes.POINTER(ctypes.c_float)]
    mcts_lib.reexpand_root.restype = None

# ============================================================
# Multi-Instance API Bindings (new _ex functions)
# ============================================================

# create_instance: 创建新的 MCTS 实例
if hasattr(mcts_lib, 'create_instance'):
    mcts_lib.create_instance.argtypes = []
    mcts_lib.create_instance.restype = ctypes.c_void_p

# destroy_instance: 销毁 MCTS 实例
if hasattr(mcts_lib, 'destroy_instance'):
    mcts_lib.destroy_instance.argtypes = [ctypes.c_void_p]
    mcts_lib.destroy_instance.restype = None

# init_game_ex: 初始化实例的游戏状态
if hasattr(mcts_lib, 'init_game_ex'):
    mcts_lib.init_game_ex.argtypes = [ctypes.c_void_p]
    mcts_lib.init_game_ex.restype = None

# set_random_seed_ex
if hasattr(mcts_lib, 'set_random_seed_ex'):
    mcts_lib.set_random_seed_ex.argtypes = [ctypes.c_void_p, ctypes.c_int]
    mcts_lib.set_random_seed_ex.restype = None

# set_mcts_params_ex
if hasattr(mcts_lib, 'set_mcts_params_ex'):
    mcts_lib.set_mcts_params_ex.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    mcts_lib.set_mcts_params_ex.restype = None

# set_pruning_k_ex
if hasattr(mcts_lib, 'set_pruning_k_ex'):
    mcts_lib.set_pruning_k_ex.argtypes = [ctypes.c_void_p, ctypes.c_int]
    mcts_lib.set_pruning_k_ex.restype = None

# play_move_ex
if hasattr(mcts_lib, 'play_move_ex'):
    mcts_lib.play_move_ex.argtypes = [ctypes.c_void_p, ctypes.c_int]
    mcts_lib.play_move_ex.restype = None

# run_mcts_simulations_ex
if hasattr(mcts_lib, 'run_mcts_simulations_ex'):
    mcts_lib.run_mcts_simulations_ex.argtypes = [ctypes.c_void_p, ctypes.c_int]
    mcts_lib.run_mcts_simulations_ex.restype = None

# get_best_move_ex
if hasattr(mcts_lib, 'get_best_move_ex'):
    mcts_lib.get_best_move_ex.argtypes = [ctypes.c_void_p, ctypes.c_float]
    mcts_lib.get_best_move_ex.restype = ctypes.c_int

# get_root_value_ex
if hasattr(mcts_lib, 'get_root_value_ex'):
    mcts_lib.get_root_value_ex.argtypes = [ctypes.c_void_p]
    mcts_lib.get_root_value_ex.restype = ctypes.c_float

# get_policy_ex
if hasattr(mcts_lib, 'get_policy_ex'):
    mcts_lib.get_policy_ex.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    mcts_lib.get_policy_ex.restype = None

# get_visit_count_ex: 获取根节点访问次数
if hasattr(mcts_lib, 'get_visit_count_ex'):
    mcts_lib.get_visit_count_ex.argtypes = [ctypes.c_void_p]
    mcts_lib.get_visit_count_ex.restype = ctypes.c_int

# reexpand_root_ex
if hasattr(mcts_lib, 'reexpand_root_ex'):
    mcts_lib.reexpand_root_ex.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    mcts_lib.reexpand_root_ex.restype = None

# clone_instance: 深拷贝实例的搜索树
if hasattr(mcts_lib, 'clone_instance'):
    mcts_lib.clone_instance.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    mcts_lib.clone_instance.restype = None

# copy_instance_to_default: 复制实例树到主引擎
if hasattr(mcts_lib, 'copy_instance_to_default'):
    mcts_lib.copy_instance_to_default.argtypes = [ctypes.c_void_p]
    mcts_lib.copy_instance_to_default.restype = None

# 回调函数类型: (batch_size, boards_ptr, policies_ptr, values_ptr)
# batch_size: int
# boards_ptr: int* (flattened batch)
# policies_ptr: float* (flattened batch)
# values_ptr: float*
CALLBACK_FUNC_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float))

mcts_lib.set_eval_callback.argtypes = [CALLBACK_FUNC_TYPE]
mcts_lib.set_eval_callback.restype = None

# --- Python Board Logic for Rule-based Override ---
class PyConnect6Board:
    def __init__(self):
        self.board = [0] * 361
        self.current_player = 1 # Black
        self.stones_in_turn = 0
        self.total_stones = 0
        
    def reset(self):
        self.board = [0] * 361
        self.current_player = 1
        self.stones_in_turn = 0
        self.total_stones = 0
        
    def make_move(self, move):
        if self.board[move] != 0: return
        self.board[move] = self.current_player
        self.stones_in_turn += 1
        self.total_stones += 1
        
        if self.total_stones == 1:
            self.current_player = -self.current_player
            self.stones_in_turn = 0
        elif self.stones_in_turn >= 2:
            self.current_player = -self.current_player
            self.stones_in_turn = 0
            
    def check_potential_threat(self, move_idx, color):
        # Check if placing 'color' at 'move_idx' creates 5 or more
        r, c = divmod(move_idx, 19)
        dr = [0, 1, 1, 1]
        dc = [1, 0, 1, -1]
        
        for i in range(4):
            count = 1
            # Forward
            for k in range(1, 6):
                nr, nc = r + dr[i]*k, c + dc[i]*k
                if not (0 <= nr < 19 and 0 <= nc < 19) or self.board[nr*19 + nc] != color:
                    break
                count += 1
            # Backward
            for k in range(1, 6):
                nr, nc = r - dr[i]*k, c - dc[i]*k
                if not (0 <= nr < 19 and 0 <= nc < 19) or self.board[nr*19 + nc] != color:
                    break
                count += 1
            
            if count >= 5: return True
        return False

    def check_winning_move(self, move_idx, color):
        # Check if placing 'color' at 'move_idx' creates 6 or more
        r, c = divmod(move_idx, 19)
        dr = [0, 1, 1, 1]
        dc = [1, 0, 1, -1]
        
        for i in range(4):
            count = 1
            # Forward
            for k in range(1, 6):
                nr, nc = r + dr[i]*k, c + dc[i]*k
                if not (0 <= nr < 19 and 0 <= nc < 19) or self.board[nr*19 + nc] != color:
                    break
                count += 1
            # Backward
            for k in range(1, 6):
                nr, nc = r - dr[i]*k, c - dc[i]*k
                if not (0 <= nr < 19 and 0 <= nc < 19) or self.board[nr*19 + nc] != color:
                    break
                count += 1
            
            if count >= 6: return True
        return False

    def get_winning_moves(self):
        moves = []
        for i in range(361):
            if self.board[i] == 0:
                if self.check_winning_move(i, self.current_player):
                    moves.append(i)
        return moves

    def get_forced_moves(self):
        moves = []
        opponent = -self.current_player
        for i in range(361):
            if self.board[i] == 0:
                if self.check_potential_threat(i, opponent):
                    moves.append(i)
        return moves

    def get_winning_pairs(self):
        """
        检测两子必杀：找到所有能用1-2子完成六连的组合
        
        检测模式:
        - 5颗棋子 + 1空位 = 单子必杀（返回 (空位, None)）
        - 4颗棋子 + 2空位 = 两子必杀（返回 (空位1, 空位2)）
        
        Returns:
            list of (move1, move2): 必杀着法对列表，move2 可能为 None
        """
        color = self.current_player
        winning_pairs = []
        
        # 检查每个方向上的必杀机会
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # 横、竖、斜、反斜
        
        for start_idx in range(361):
            r, c = divmod(start_idx, 19)
            
            for dr, dc in directions:
                # 检查连续 6 格的窗口
                window = []
                for k in range(6):
                    nr, nc = r + dr * k, c + dc * k
                    if 0 <= nr < 19 and 0 <= nc < 19:
                        window.append((nr * 19 + nc, self.board[nr * 19 + nc]))
                    else:
                        break
                
                if len(window) < 6:
                    continue
                
                # 统计这个窗口内的情况
                our_stones = sum(1 for _, v in window if v == color)
                empty_cells = [(idx, v) for idx, v in window if v == 0]
                opponent_stones = sum(1 for _, v in window if v == -color)
                
                # 如果窗口内有对手棋子，无法成六连
                if opponent_stones > 0:
                    continue
                
                # 5颗棋子 + 1空位 = 单子必杀（优先）
                if our_stones == 5 and len(empty_cells) == 1:
                    move1 = empty_cells[0][0]
                    pair = (move1, None)
                    if pair not in winning_pairs:
                        winning_pairs.insert(0, pair)  # 插入到最前面，优先处理
                
                # 4颗棋子 + 2空位 = 两子必杀
                elif our_stones == 4 and len(empty_cells) == 2:
                    move1 = empty_cells[0][0]
                    move2 = empty_cells[1][0]
                    if (move1, move2) not in winning_pairs and (move2, move1) not in winning_pairs:
                        winning_pairs.append((move1, move2))
        
        return winning_pairs

    def get_best_winning_pair(self):
        """
        获取最佳必杀着法对
        
        Returns:
            (move1, move2) 或 None
        """
        pairs = self.get_winning_pairs()
        if pairs:
            # 返回第一个找到的必杀组合
            return pairs[0]
        return None

class MCTSEngine:
    def __init__(self, engine_path, device='cuda'):
        self.device = device
        self.engine_path = engine_path
        
        # Python Board for Rule-based Override
        self.py_board = PyConnect6Board()
        
        # 1. Initialize TensorRT
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(self.engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        
        # 2. Allocate Buffers (using Torch for GPU memory management)
        # Input: (32, 17, 19, 19)
        self.max_batch_size = 32
        
        # Inspect bindings to find input name and index
        num_io = self.engine.num_io_tensors
        input_name = None
        for i in range(num_io):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                input_name = name
                print(f"Found Input Tensor: {name}")
                break
        
        if input_name is None:
            raise RuntimeError("No input tensor found in TensorRT engine!")

        # Set input shape explicitly (Required for dynamic shape engines)
        self.context.set_input_shape(input_name, (self.max_batch_size, 17, 19, 19))
        
        # Bindings
        # We need to map binding indices to pointers
        # For execute_v2, it expects a list of pointers in order of binding indices
        
        # Note: In newer TRT, bindings are deprecated for set_tensor_address, 
        # but execute_v2 still takes list of pointers.
        # We need to ensure the order matches the engine's binding order.
        
        self.input_tensor = torch.zeros((self.max_batch_size, 17, 19, 19), dtype=torch.float32, device=self.device)
        # New input: move1_idx for autoregressive head
        self.move1_tensor = torch.full((self.max_batch_size,), 361, dtype=torch.long, device=self.device)
        
        self.policy1_tensor = torch.zeros((self.max_batch_size, 361), dtype=torch.float32, device=self.device)
        self.policy2_tensor = torch.zeros((self.max_batch_size, 361), dtype=torch.float32, device=self.device)
        self.value_tensor = torch.zeros((self.max_batch_size, 1), dtype=torch.float32, device=self.device)
        
        # Map pointers by name
        # Input 1: Board
        self.context.set_tensor_address(input_name, int(self.input_tensor.data_ptr()))
        
        # Input 2: Move1 Index (Find name if dynamic)
        # We assume export_onnx.py named it 'move1_idx'
        for i in range(num_io):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if 'move1' in name or 'idx' in name:
                     self.context.set_input_shape(name, (self.max_batch_size,))
                     self.context.set_tensor_address(name, int(self.move1_tensor.data_ptr()))

        # Find output names
        for i in range(num_io):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                if 'policy1' in name:
                    self.context.set_tensor_address(name, int(self.policy1_tensor.data_ptr()))
                elif 'policy2' in name:
                    self.context.set_tensor_address(name, int(self.policy2_tensor.data_ptr()))
                elif 'value' in name:
                    self.context.set_tensor_address(name, int(self.value_tensor.data_ptr()))
        
        # 初始化 C++ 引擎
        mcts_lib.init_game()
        
        # 注册回调
        self.c_callback = CALLBACK_FUNC_TYPE(self._eval_callback)
        mcts_lib.set_eval_callback(self.c_callback)
        
        # === TensorRT 预热 ===
        # 移除手动预热，改为在 interface 层通过真实调用来预热
        print("🔥 [TensorRT] Ready.")
        
    def _eval_callback(self, batch_size, boards_ptr, policies_ptr, values_ptr):
        """
        Batch Evaluation Callback using TensorRT.
        Vectorized optimization to remove Python loops and leverage GPU.
        """
        # Dynamic Resizing if batch_size exceeds current max
        if batch_size > self.input_tensor.size(0):
            # print(f"[MCTS] Resizing tensors: {self.input_tensor.size(0)} -> {batch_size}")
            self.max_batch_size = batch_size
            
            # Re-allocate Tensors
            self.input_tensor = torch.zeros((self.max_batch_size, 17, 19, 19), dtype=torch.float32, device=self.device)
            self.move1_tensor = torch.full((self.max_batch_size,), 361, dtype=torch.long, device=self.device)
            
            self.policy1_tensor = torch.zeros((self.max_batch_size, 361), dtype=torch.float32, device=self.device)
            self.policy2_tensor = torch.zeros((self.max_batch_size, 361), dtype=torch.float32, device=self.device)
            self.value_tensor = torch.zeros((self.max_batch_size, 1), dtype=torch.float32, device=self.device)
            
            # Re-bind addresses to TensorRT Context
            num_io = self.engine.num_io_tensors
            for i in range(num_io):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                
                if mode == trt.TensorIOMode.INPUT:
                    if 'input' in name:
                        self.context.set_input_shape(name, (self.max_batch_size, 17, 19, 19))
                        self.context.set_tensor_address(name, int(self.input_tensor.data_ptr()))
                    elif 'move1' in name or 'idx' in name:
                        self.context.set_input_shape(name, (self.max_batch_size,))
                        self.context.set_tensor_address(name, int(self.move1_tensor.data_ptr()))
                else:
                    if 'policy1' in name:
                        self.context.set_tensor_address(name, int(self.policy1_tensor.data_ptr()))
                    elif 'policy2' in name:
                        self.context.set_tensor_address(name, int(self.policy2_tensor.data_ptr()))
                    elif 'value' in name:
                        self.context.set_tensor_address(name, int(self.value_tensor.data_ptr()))


        total_elements = batch_size * 361
        
        # 1. Read Batch Boards from C++ (CPU -> GPU)
        # Use a view to avoid copy on CPU side, then direct copy to GPU
        boards_array = np.ctypeslib.as_array(boards_ptr, shape=(total_elements,))
            
        boards_reshaped = boards_array.reshape(batch_size, 19, 19)
        
        # Move to GPU immediately for vectorized processing
        boards_gpu = torch.from_numpy(boards_reshaped).to(self.device, dtype=torch.float32) # (B, 19, 19)
        
        # CRITICAL FIX: Normalize Board Values
        # Ensure White stones are represented as -1, not 2
        # Some C++ engines use 2 for White, while our logic expects -1
        boards_gpu = torch.where(boards_gpu == 2, torch.tensor(-1.0, device=self.device), boards_gpu)
        
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

        # 3. Inference (TensorRT V3 API)
        self.context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        
        # Wait for TRT completion
        torch.cuda.current_stream().synchronize()
        
        # 4. Process Outputs (Vectorized on GPU)
        # Logits -> Probs
        logits = self.policy1_tensor[:batch_size]
        probs = torch.softmax(logits, dim=1) # Keep on GPU
        
        # 6. Copy back to CPU for C++
        values = self.value_tensor[:batch_size] # Keep on GPU (B, 1)
        values_flat = values.flatten()
        # Ensure float32 and contiguous
        probs_cpu = probs.cpu().numpy().astype(np.float32)
        values_cpu = values_flat.cpu().numpy().astype(np.float32)
        
        ctypes.memmove(policies_ptr, probs_cpu.ctypes.data, probs_cpu.nbytes)
        ctypes.memmove(values_ptr, values_cpu.ctypes.data, values_cpu.nbytes)


    def reset(self):
        mcts_lib.init_game()
        self.py_board.reset()
        self._callback_count = 0  # 重置回调计数
        
    def set_random_seed(self, seed):
        mcts_lib.set_random_seed(seed)
        
    def set_params(self, batch_size=32, num_threads=4):
        """
        Set MCTS execution parameters. Safe to call even if engine doesn't support it.
        :param batch_size: Number of evaluations to batch for GPU.
        :param num_threads: Number of CPU threads for tree search.
        """
        if hasattr(mcts_lib, 'set_mcts_params'):
            mcts_lib.set_mcts_params(batch_size, num_threads)
        else:
            print("Warning: set_params ignored (C++ engine too old)")

    def set_pruning_k(self, k):
        """
        Set the pruning parameter K (Top-K children to consider).
        0 means no pruning.
        """
        if hasattr(mcts_lib, 'set_pruning_k'):
            mcts_lib.set_pruning_k(k)

    def update_state(self, move):
        mcts_lib.play_move(move)
        self.py_board.make_move(move)

    def sync_state_from_moves(self, moves):
        """
        从着法历史重新同步 C++ 引擎状态。
        用于 Ponder 未命中时恢复正确状态。
        
        :param moves: 着法列表 [move_idx, ...]
        """
        # 重置 C++ 引擎和 py_board
        mcts_lib.init_game()
        self.py_board.reset()
        
        # 重放所有着法
        for move in moves:
            mcts_lib.play_move(move)
            self.py_board.make_move(move)

    def get_board_state(self):
        """
        获取当前棋盘状态的元组表示，用于状态校验。
        
        :return: tuple(board_tuple, current_player, stones_in_turn)
        """
        return (
            tuple(self.py_board.board),
            self.py_board.current_player,
            self.py_board.stones_in_turn
        )
    
    def verify_state(self, ui_board_flat, ui_current_player):
        """
        校验 AI 内部状态与 UI 状态是否一致。
        
        :param ui_board_flat: UI 棋盘的扁平化列表 (361 元素)
        :param ui_current_player: UI 当前玩家 (1=黑, -1=白)
        :return: (is_match, mismatch_info)
        """
        ai_board = self.py_board.board
        ai_player = self.py_board.current_player
        
        # 检查棋盘
        board_match = True
        mismatch_positions = []
        for i in range(361):
            if ai_board[i] != ui_board_flat[i]:
                board_match = False
                r, c = divmod(i, 19)
                mismatch_positions.append((r, c, ai_board[i], ui_board_flat[i]))
        
        # 检查当前玩家
        player_match = (ai_player == ui_current_player)
        
        is_match = board_match and player_match
        
        mismatch_info = None
        if not is_match:
            mismatch_info = {
                'board_match': board_match,
                'player_match': player_match,
                'ai_player': ai_player,
                'ui_player': ui_current_player,
                'mismatch_positions': mismatch_positions[:5]  # 只显示前5个
            }
        
        return is_match, mismatch_info
        
    def get_mcts_move(self, simulations=1000, temperature=0.0):
        # Run simulations
        mcts_lib.run_mcts_simulations(simulations)
        # Get best move with temperature
        return mcts_lib.get_best_move(float(temperature))
    
    def run_simulations(self, simulations=1000):
        """仅运行 MCTS 模拟，不获取着法。用于 Ponder 预测时避免副作用。"""
        mcts_lib.run_mcts_simulations(simulations)
        
    def get_win_rate(self, ai_color=None):
        """
        获取 AI 的胜率，范围 [0, 1]。
        
        :param ai_color: AI 执棋颜色 (1=黑, -1=白)。如果为 None，返回当前玩家的胜率。
        :return: 胜率 [0, 1]
        """
        # q_value 是黑方视角: +1=黑胜, -1=白胜
        q = mcts_lib.get_root_value()
        
        # 如果指定了 AI 颜色，调整视角
        if ai_color is not None and ai_color == -1:  # AI 是白方
            q = -q
        
        # 从 [-1, 1] 映射到 [0, 1]
        win_rate = (q + 1.0) / 2.0
        
        # 确保在 [0, 1] 范围内
        return max(0.0, min(1.0, float(win_rate)))

    def get_root_value(self):
        """
        获取原始 Q 值（黑方视角），范围 [-1, 1]。
        用于 MCTSStrategy 内部逻辑。
        """
        return float(mcts_lib.get_root_value())

    def get_best_move(self):
        """Get best move without running simulations."""
        return mcts_lib.get_best_move(0.0)
        
    def get_policy(self):
        policy = np.zeros(361, dtype=np.float32)
        mcts_lib.get_policy(policy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return policy
    
    def print_top_debug(self):
        mcts_lib.print_top_moves()

    def reexpand_root(self):
        """
        重新展开根节点，补充被剪枝丢弃的候选着法。
        用于同回合两子复用时，确保第二子有完整的候选集。
        
        会调用神经网络获取当前局面的策略，然后：
        1. 更新已有子节点的 prior
        2. 添加之前被剪掉的候选
        """
        if not hasattr(mcts_lib, 'reexpand_root'):
            print("Warning: reexpand_root not available in C++ engine")
            return
        
        # 获取当前局面的神经网络策略
        # 需要单独调用一次 NN 评估
        policy = self._get_current_policy()
        if policy is None:
            return
        
        # 调用 C++ 的 reexpand_root
        policy_ptr = policy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        mcts_lib.reexpand_root(policy_ptr)

    def _get_current_policy(self):
        """
        获取当前局面的神经网络策略输出。
        """
        # 构建当前局面的特征
        board = np.array(self.py_board.board, dtype=np.int32).reshape(19, 19)
        
        # 计算当前玩家
        n_stones = np.sum(board != 0)
        rank = (n_stones + 1) // 2
        if rank % 2 == 1:
            curr_player = -1  # White
        else:
            curr_player = 1   # Black
        
        # 构建输入特征
        self.input_tensor.zero_()
        board_tensor = torch.from_numpy(board).to(self.device, dtype=torch.float32)
        
        if curr_player == 1:  # Black
            self.input_tensor[0, 0] = (board_tensor == 1).float()
            self.input_tensor[0, 1] = (board_tensor == -1).float()
            self.input_tensor[0, 16] = 1.0
        else:  # White
            self.input_tensor[0, 0] = (board_tensor == -1).float()
            self.input_tensor[0, 1] = (board_tensor == 1).float()
            self.input_tensor[0, 16] = 0.0
        
        # 运行推理
        self.context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        torch.cuda.current_stream().synchronize()
        
        # 获取策略输出
        logits = self.policy1_tensor[0]
        probs = torch.softmax(logits, dim=0).cpu().numpy().astype(np.float32)
        
        return probs






# ============================================================
# Multi-Instance MCTS Wrapper Classes
# ============================================================

class MCTSInstanceHandle:
    """
    Wrapper for a single MCTS instance (C++ tree).
    Provides Python-friendly interface to control one independent MCTS tree.
    """
    def __init__(self, handle, batch_size=32, num_threads=4):
        self.handle = handle
        self.move_history = []
        self.batch_size = batch_size
        self.num_threads = num_threads
        
        # Initialize params
        if hasattr(mcts_lib, 'set_mcts_params_ex'):
            mcts_lib.set_mcts_params_ex(self.handle, batch_size, num_threads)
    
    def reset(self):
        """Reset this instance to initial game state."""
        if hasattr(mcts_lib, 'init_game_ex'):
            mcts_lib.init_game_ex(self.handle)
        self.move_history = []
    
    def set_random_seed(self, seed):
        """Set random seed for this instance."""
        if hasattr(mcts_lib, 'set_random_seed_ex'):
            mcts_lib.set_random_seed_ex(self.handle, seed)
    
    def set_pruning_k(self, k):
        """Set pruning K value for this instance."""
        if hasattr(mcts_lib, 'set_pruning_k_ex'):
            mcts_lib.set_pruning_k_ex(self.handle, k)
    
    def play_move(self, move):
        """Play a move and update tree (switches to child subtree if exists)."""
        if hasattr(mcts_lib, 'play_move_ex'):
            mcts_lib.play_move_ex(self.handle, move)
        self.move_history.append(move)
    
    def run_simulations(self, num_simulations):
        """Run MCTS simulations on this instance."""
        if hasattr(mcts_lib, 'run_mcts_simulations_ex'):
            mcts_lib.run_mcts_simulations_ex(self.handle, num_simulations)
    
    def get_best_move(self, temperature=0.0):
        """Get best move from this instance."""
        if hasattr(mcts_lib, 'get_best_move_ex'):
            return mcts_lib.get_best_move_ex(self.handle, float(temperature))
        return -1
    
    def get_root_value(self):
        """Get Q-value of root node (black perspective)."""
        if hasattr(mcts_lib, 'get_root_value_ex'):
            return float(mcts_lib.get_root_value_ex(self.handle))
        return 0.0
    
    def get_visit_count(self):
        """Get total visit count of root node."""
        if hasattr(mcts_lib, 'get_visit_count_ex'):
            return mcts_lib.get_visit_count_ex(self.handle)
        return 0
    
    def get_policy(self):
        """Get policy distribution from root node."""
        policy = np.zeros(361, dtype=np.float32)
        if hasattr(mcts_lib, 'get_policy_ex'):
            mcts_lib.get_policy_ex(self.handle, policy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return policy
    
    def sync_from_moves(self, moves):
        """Reset and replay moves to reach a specific state."""
        self.reset()
        for move in moves:
            self.play_move(move)
    
    def get_last_move(self):
        """Get the last move played on this instance."""
        if self.move_history:
            return self.move_history[-1]
        return -1
    
    def clone_from(self, other):
        """
        Deep copy another instance's MCTS tree into this instance.
        This allows inheriting search results from a hit instance.
        """
        if hasattr(mcts_lib, 'clone_instance'):
            mcts_lib.clone_instance(other.handle, self.handle)
            # Also copy move history
            self.move_history = other.move_history.copy()
    
    def copy_to_default(self):
        """
        Copy this instance's MCTS tree to the default (main) engine.
        This enables true tree reuse when a ponder hit occurs.
        """
        if hasattr(mcts_lib, 'copy_instance_to_default'):
            mcts_lib.copy_instance_to_default(self.handle)
            return True
        return False


class MultiMCTSEngine:
    """
    Manager for multiple MCTS instances.
    Used for parallel pondering on multiple opponent move predictions.
    """
    def __init__(self, num_instances=3, batch_size=32, num_threads=4):
        self.num_instances = num_instances
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.instances = []
        
        # Check if multi-instance is supported
        if not hasattr(mcts_lib, 'create_instance'):
            print("⚠️ [MultiMCTS] Multi-instance not supported by current DLL")
            self.supported = False
            return
        
        self.supported = True
        
        # Create instances
        for i in range(num_instances):
            handle = mcts_lib.create_instance()
            if handle:
                inst = MCTSInstanceHandle(handle, batch_size, num_threads)
                self.instances.append(inst)
                print(f"🔧 [MultiMCTS] Created instance {i}: handle={handle}")
            else:
                print(f"❌ [MultiMCTS] Failed to create instance {i}")
        
        print(f"✅ [MultiMCTS] Initialized {len(self.instances)} instances")
    
    def is_supported(self):
        """Check if multi-instance is supported."""
        return self.supported and len(self.instances) > 0
    
    def get_instance(self, idx):
        """Get instance by index."""
        if 0 <= idx < len(self.instances):
            return self.instances[idx]
        return None
    
    def reset_all(self):
        """Reset all instances to initial state."""
        for inst in self.instances:
            inst.reset()
            
    def set_pruning_k(self, k):
        """Set pruning k for all instances."""
        for inst in self.instances:
            inst.set_pruning_k(k)
    
    def find_matching_instance(self, move):
        """
        Find instance whose last move matches the given move.
        Returns (index, instance) or (-1, None) if not found.
        """
        for i, inst in enumerate(self.instances):
            if inst.get_last_move() == move:
                return i, inst
        return -1, None
    
    def clone_all_from(self, source_idx):
        """
        Clone source instance to all other instances.
        Used when first move hits to inherit search tree.
        Returns the source instance.
        """
        if source_idx < 0 or source_idx >= len(self.instances):
            return None
        
        source = self.instances[source_idx]
        for i, inst in enumerate(self.instances):
            if i != source_idx:
                inst.clone_from(source)
        
        return source
    
    def cleanup(self):
        """Destroy all instances."""
        if not hasattr(mcts_lib, 'destroy_instance'):
            return
        
        for i, inst in enumerate(self.instances):
            if inst.handle:
                mcts_lib.destroy_instance(inst.handle)
                print(f"🗑️ [MultiMCTS] Destroyed instance {i}")
        
        self.instances = []
    
    def __del__(self):
        """Destructor - cleanup instances."""
        try:
            self.cleanup()
        except:
            pass
