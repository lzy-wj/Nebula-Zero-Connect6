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

# 回调函数类型: (batch_size, boards_ptr, policies_ptr, values_ptr)
# batch_size: int
# boards_ptr: int* (flattened batch)
# policies_ptr: float* (flattened batch)
# values_ptr: float*
CALLBACK_FUNC_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float))

mcts_lib.set_eval_callback.argtypes = [CALLBACK_FUNC_TYPE]
mcts_lib.set_eval_callback.restype = None

class MCTSEngine:
    def __init__(self, engine_path, device='cuda'):
        self.device = device
        self.engine_path = engine_path
        
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
        move1_name = None
        
        for i in range(num_io):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                if 'input' in name:
                    input_name = name
                elif 'move1' in name or 'idx' in name:
                    move1_name = name
        
        if input_name is None:
            raise RuntimeError("No input tensor found in TensorRT engine!")

        # Set input shape explicitly (Required for dynamic shape engines)
        self.context.set_input_shape(input_name, (self.max_batch_size, 17, 19, 19))
        if move1_name:
            self.context.set_input_shape(move1_name, (self.max_batch_size,))
        
        # Bindings
        # We need to map binding indices to pointers
        # For execute_v2, it expects a list of pointers in order of binding indices
        
        # Note: In newer TRT, bindings are deprecated for set_tensor_address, 
        # but execute_v2 still takes list of pointers.
        # We need to ensure the order matches the engine's binding order.
        
        self.input_tensor = torch.zeros((self.max_batch_size, 17, 19, 19), dtype=torch.float32, device=self.device)
        self.move1_tensor = torch.full((self.max_batch_size,), 361, dtype=torch.long, device=self.device) # 361 is standard SOS/Padding

        self.policy1_tensor = torch.zeros((self.max_batch_size, 361), dtype=torch.float32, device=self.device)
        self.policy2_tensor = torch.zeros((self.max_batch_size, 361), dtype=torch.float32, device=self.device)
        self.value_tensor = torch.zeros((self.max_batch_size, 1), dtype=torch.float32, device=self.device)
        
        # Map pointers by name
        self.context.set_tensor_address(input_name, int(self.input_tensor.data_ptr()))
        if move1_name:
            self.context.set_tensor_address(move1_name, int(self.move1_tensor.data_ptr()))

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
        
        # 初始化规则检测器 (用于辅助 Value 评估)
        self.rule_helper = RuleHelper(self.device)
        
        # 注册回调
        self.c_callback = CALLBACK_FUNC_TYPE(self._eval_callback)
        mcts_lib.set_eval_callback(self.c_callback)
        
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
        
        # 4. Rule-based Reward Shaping (Parallel with TRT Inference)
        # Use slices to avoid copies
        self_stones = self.input_tensor[:batch_size, 0:1]
        opp_stones = self.input_tensor[:batch_size, 1:2]
        
        self_threats = self.rule_helper.detect(self_stones, exclusion_tensor=opp_stones)
        opp_threats = self.rule_helper.detect(opp_stones, exclusion_tensor=self_stones)
        
        # Wait for TRT completion
        torch.cuda.current_stream().synchronize()
        
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
        
        # 6. Copy back to CPU for C++
        # Ensure float32 and contiguous
        probs_cpu = probs.cpu().numpy().astype(np.float32)
        values_cpu = values_flat.cpu().numpy().astype(np.float32)
        
        ctypes.memmove(policies_ptr, probs_cpu.ctypes.data, probs_cpu.nbytes)
        ctypes.memmove(values_ptr, values_cpu.ctypes.data, values_cpu.nbytes)


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
        if hasattr(mcts_lib, 'set_mcts_params'):
            mcts_lib.set_mcts_params(batch_size, num_threads)
        else:
            print("Warning: set_params ignored (C++ engine too old)")

    def update_state(self, move):
        mcts_lib.play_move(move)

    def run_simulations(self, simulations):
        """Run MCTS simulations without returning a move (for dynamic search)"""
        mcts_lib.run_mcts_simulations(simulations)
        
    def get_mcts_move(self, simulations=1000, temperature=0.0):
        # Support existing calls but allow 0 simulations if run_simulations was called manually
        if simulations > 0:
            mcts_lib.run_mcts_simulations(simulations)
        return mcts_lib.get_best_move(temperature)
        
    def get_win_rate(self):
        val = mcts_lib.get_root_value()
        return float(val)
        
    def get_policy(self):
        policy = np.zeros(361, dtype=np.float32)
        mcts_lib.get_policy(policy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return policy
    
    def print_top_debug(self):
        mcts_lib.print_top_moves()


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
