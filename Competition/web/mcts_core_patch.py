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
        self.policy1_tensor = torch.zeros((self.max_batch_size, 361), dtype=torch.float32, device=self.device)
        self.policy2_tensor = torch.zeros((self.max_batch_size, 361), dtype=torch.float32, device=self.device)
        self.value_tensor = torch.zeros((self.max_batch_size, 1), dtype=torch.float32, device=self.device)
        
        # Map pointers by name
        self.context.set_tensor_address(input_name, int(self.input_tensor.data_ptr()))
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
        """
        total_elements = batch_size * 361
        
        # 1. Read Batch Boards from C++
        boards_array = np.ctypeslib.as_array(boards_ptr, shape=(total_elements,))
        boards_reshaped = boards_array.reshape(batch_size, 19, 19)
        
        # 2. Preprocess Features directly into GPU Input Tensor
        # Optimization: Reset tensor
        self.input_tensor.zero_()
        
        # Store stones_remaining for each batch item
        stones_rem_batch = np.zeros(batch_size, dtype=np.int32)
        
        for b in range(batch_size):
            board_2d = boards_reshaped[b]
            n_stones = np.sum(board_2d != 0)
            
            # Connect6 Turn Logic:
            # Move 0: 1 stone (Black)
            # Subsequent: 2 stones
            # If n_stones is Even (0, 2, 4...): Next player has 1 stone to play? 
            # Wait:
            # 0 stones -> Black plays 1. (stones_rem=1)
            # 1 stone -> White plays 2. (stones_rem=2)
            # 2 stones -> White has played 1/2? No.
            # Let's trace:
            # Start: 0. Black turn. 1 to play.
            # After Black: 1. White turn. 2 to play.
            # White plays 1: 2. White turn. 1 to play.
            # White plays 2: 3. Black turn. 2 to play.
            # Black plays 1: 4. Black turn. 1 to play.
            # So: 
            # n=0: 1
            # n Odd (1, 3): 2
            # n Even >0 (2, 4): 1
            
            if n_stones == 0:
                stones_rem = 1
                curr_player = 1 # Black
            else:
                if n_stones % 2 != 0:
                    stones_rem = 2
                    # n=1 (B), Next White (-1)
                    # n=3 (BWW), Next Black (1)
                    # (1+1)//2 = 1 (odd) -> White
                    # (3+1)//2 = 2 (even) -> Black
                    rank = (n_stones + 1) // 2
                    curr_player = -1 if rank % 2 == 1 else 1
                else:
                    stones_rem = 1
                    # n=2 (BW), White still playing.
                    # n=4 (BWWB), Black still playing.
                    # (2)//2 = 1 -> White
                    # (4)//2 = 2 -> Black
                    rank = n_stones // 2
                    curr_player = -1 if rank % 2 == 1 else 1
            
            stones_rem_batch[b] = stones_rem

            if curr_player == 1: # Black
                self.input_tensor[b, 0] = torch.from_numpy(board_2d == 1).float()
                self.input_tensor[b, 1] = torch.from_numpy(board_2d == -1).float()
                self.input_tensor[b, 16] = 1.0
            else: # White
                self.input_tensor[b, 0] = torch.from_numpy(board_2d == -1).float()
                self.input_tensor[b, 1] = torch.from_numpy(board_2d == 1).float()
                self.input_tensor[b, 16] = 0.0

        # 3. Inference (TensorRT V3 API)
        self.context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        
        # 4. Rule-based Reward Shaping (Parallel with TRT Inference)
        # Check for threats for BOTH Self and Opponent
        self_stones = self.input_tensor[:batch_size, 0:1, :, :]
        opp_stones = self.input_tensor[:batch_size, 1:2, :, :]
        
        self_threats = self.rule_helper.detect(self_stones, exclusion_tensor=opp_stones)
        opp_threats = self.rule_helper.detect(opp_stones, exclusion_tensor=self_stones)
        
        # Wait for completion (Synchronize)
        torch.cuda.current_stream().synchronize()
        
        # 5. Process Outputs
        logits = self.policy1_tensor[:batch_size]
        probs = torch.softmax(logits, dim=1).cpu().numpy() # (B, 361)
        
        values = self.value_tensor[:batch_size].cpu().numpy().flatten() # (B,)
        
        # Helper to get batch-wise max (did any part of the board trigger?)
        def get_trigger(threat_dict, key):
            # threat_dict[key] is (B, 1, H, W) -> max over H,W -> (B, 1) -> flatten
            return threat_dict[key].view(batch_size, -1).max(dim=1)[0].cpu().numpy() > 0.5

        my_win = get_trigger(self_threats, 'win')
        my_c5  = get_trigger(self_threats, 'c5')
        my_c4  = get_trigger(self_threats, 'c4')
        
        opp_win = get_trigger(opp_threats, 'win')
        opp_c5  = get_trigger(opp_threats, 'c5')
        opp_c4  = get_trigger(opp_threats, 'c4')
        
        for i in range(batch_size):
            rem = stones_rem_batch[i]
            
            # Priority 1: I win
            # my_c5: Need 1. Have >=1 (rem is 1 or 2). Win.
            # my_c4: Need 2. Only win if rem >= 2.
            if my_win[i] or my_c5[i]:
                values[i] = 1.0
                continue
            
            if my_c4[i] and rem >= 2:
                values[i] = 1.0
                continue
                
            # Priority 2: Opponent wins (Game Over / Checkmate)
            # opp_win: Opponent connected 6. I lost.
            if opp_win[i]:
                values[i] = -1.0
                continue
            
            # Priority 3: Opponent Threats
            # opp_c5: Opponent has 5. I MUST block. (Defendable) -> Penalty -0.25
            # opp_c4: Opponent has 4. I should block. -> Penalty -0.2
            
            # Note: If I have 1 stone left, and opp has c5, I block it.
            # If opp has c4, I block it? Opp needs 2 to win.
            # If I have 1 stone, I block 1. Opp has 1 hole left? No c4 becomes c5 blocked.
            
            if opp_c5[i]:
                values[i] = min(values[i], -0.25)
            elif opp_c4[i]:
                values[i] = min(values[i], -0.2)
            
            # Bonus: If I have my_c4 but rem=1 (Mid-turn)
            # Previously we gave a bonus here (0.25), but it led to mindless attacks.
            # Now we give NO bonus. Let MCTS figure out if it's actually good.
            # if my_c4[i] and rem == 1:
            #     values[i] = max(values[i], 0.25)
        ctypes.memmove(policies_ptr, probs.ctypes.data, probs.nbytes)
        ctypes.memmove(values_ptr, values.ctypes.data, values.nbytes)


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
        
    def get_mcts_move(self, simulations=1000, temperature=0.0):
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
        exclusion_tensor: (B, 1, 19, 19) - Stones that block the line
        Returns dict of masks for 'win' (6), 'c5' (5), 'c4' (4)
        """
        import torch.nn.functional as F
        
        # Check self counts
        out = F.conv2d(board_tensor, self.kernels, padding=0)
        
        # Check exclusion counts (if provided)
        if exclusion_tensor is not None:
            blocked = F.conv2d(exclusion_tensor, self.kernels, padding=0)
            # We only care if blocked == 0
            # Create a mask: 1 if not blocked, 0 if blocked
            valid_mask = (blocked == 0).float()
            # Apply mask
            out = out * valid_mask
        
        # out shape: (B, 4, 14, 14)
        
        threats = {}
        # Check max value in the 6x6 window
        # If max >= 6 -> Win
        threats['win'] = (out >= 6.0).float()
        # If max == 5 -> C5
        threats['c5'] = (out == 5.0).float()
        # If max == 4 -> C4
        threats['c4'] = (out == 4.0).float()
        
        return threats

