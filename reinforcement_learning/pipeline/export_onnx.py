import os
import sys

# ONNX 导出会执行一次模型前向，默认固定到预留的物理卡 6。
os.environ.setdefault(
    'CUDA_VISIBLE_DEVICES',
    os.environ.get('NEBULA_BUILD_GPU', '6'),
)

import torch
import torch.onnx
import torch.nn.functional as F

# Add path to phrase4 root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from core.model import C6TransNet


class PrecisionExportWrapper(torch.nn.Module):
    """保持 FP32 I/O，在图内部按 TensorRT 精度执行模型。"""

    def __init__(self, model, compute_dtype):
        super().__init__()
        self.model = model
        self.compute_dtype = compute_dtype

    def forward(self, inputs, move1_idx):
        policy1, policy2, value = self.model(inputs.to(self.compute_dtype), move1_idx)
        return policy1.float(), policy2.float(), value.float()


class FusedSelfPlayExportWrapper(torch.nn.Module):
    """把棋盘编码、网络、规则修正和 softmax 合成一个 TensorRT 图。"""

    def __init__(self, model, compute_dtype):
        super().__init__()
        self.model = model
        self.compute_dtype = compute_dtype

        kernels = torch.zeros((4, 1, 6, 6), dtype=torch.float32)
        kernels[0, 0, 0, :] = 1
        kernels[1, 0, :, 0] = 1
        kernels[2, 0] = torch.eye(6)
        kernels[3, 0] = torch.rot90(torch.eye(6), 1, [0, 1])
        self.register_buffer('threat_kernels', kernels)

    def _detect_threats(self, stones, blockers):
        padded_stones = F.pad(stones, (5, 5, 5, 5), value=0)
        counts = F.conv2d(padded_stones, self.threat_kernels)
        padded_blockers = F.pad(blockers, (5, 5, 5, 5), value=1)
        blocked = F.conv2d(padded_blockers, self.threat_kernels)
        counts = counts * (blocked == 0).float()
        return (
            counts.ge(6).float().flatten(1).amax(dim=1),
            counts.eq(5).float().flatten(1).amax(dim=1),
            counts.eq(4).float().flatten(1).amax(dim=1),
        )

    def forward(self, board):
        # 兼容白棋编码为 -1 或 2 的旧 C++ 棋盘。
        board = torch.where(board == 2, -torch.ones_like(board), board)
        stone_count = board.ne(0).sum(dim=(1, 2))
        rank = (stone_count + 1) // 2
        current_player = torch.where(rank.remainder(2).eq(1), -1, 1)
        player_view = current_player.view(-1, 1, 1)

        self_stones = board.eq(player_view).float().unsqueeze(1)
        opponent_stones = board.eq(-player_view).float().unsqueeze(1)
        empty_planes = torch.zeros(
            (board.shape[0], 14, 19, 19),
            dtype=torch.float32,
            device=board.device,
        )
        color_plane = current_player.eq(1).float().view(-1, 1, 1, 1).expand(-1, 1, 19, 19)
        features = torch.cat((self_stones, opponent_stones, empty_planes, color_plane), dim=1)

        policy_logits, _, value = self.model(features.to(self.compute_dtype))
        policy = torch.softmax(policy_logits.float(), dim=1)
        value = value.float().flatten()

        my_win, my_c5, my_c4 = self._detect_threats(self_stones, opponent_stones)
        opp_win, opp_c5, opp_c4 = self._detect_threats(opponent_stones, self_stones)
        stones_remaining = torch.where(
            stone_count.gt(0) & stone_count.remainder(2).eq(1),
            2,
            1,
        )

        value = torch.where(opp_c4.bool(), value.clamp(max=-0.2), value)
        value = torch.where(opp_c5.bool(), value.clamp(max=-0.25), value)
        can_win = my_c5.bool() | (my_c4.bool() & stones_remaining.ge(2))
        value = torch.where(can_win, torch.ones_like(value), value)
        value = torch.where(opp_win.bool(), -torch.ones_like(value), value)
        value = torch.where(my_win.bool(), torch.ones_like(value), value)
        return policy, value.unsqueeze(1)


def export_to_onnx(model_path, output_path, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    print(f"Loading model from {model_path}...")
    model = C6TransNet(input_planes=17).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # Remove module. prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    precision_to_dtype = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
    }
    compute_dtype = precision_to_dtype.get(config.TRT_PRECISION, torch.float16)
    model = model.to(dtype=compute_dtype)
    if config.TRT_CUSTOM_ATTENTION:
        if compute_dtype != torch.float16:
            raise ValueError('二维相对位置注意力 AOT 插件目前要求 TRT_PRECISION=fp16')
        for module in model.modules():
            if hasattr(module, 'use_trt_attention_plugin'):
                module.use_trt_attention_plugin = True
        print('启用 Nebula 二维相对位置注意力 AOT 节点')
    if config.TRT_FUSED_SELFPLAY:
        export_model = FusedSelfPlayExportWrapper(model, compute_dtype).to(device).eval()
    else:
        export_model = PrecisionExportWrapper(model, compute_dtype).to(device).eval()
    print(f"TensorRT graph compute precision: {config.TRT_PRECISION}")
    
    # 2. Create Dummy Input
    # Shape: [Batch, 17, 19, 19]
    if config.TRT_FUSED_SELFPLAY:
        dummy_args = (torch.zeros(batch_size, 19, 19, dtype=torch.int32, device=device),)
        input_names = ['board']
        output_names = ['policy1', 'value']
        dynamic_axes = {
            'board': {0: 'batch_size'},
            'policy1': {0: 'batch_size'},
            'value': {0: 'batch_size'},
        }
    else:
        dummy_input = torch.randn(batch_size, 17, 19, 19, device=device)
        dummy_move1 = torch.zeros(batch_size, dtype=torch.long, device=device)
        dummy_args = (dummy_input, dummy_move1)
        input_names = ['input', 'move1_idx']
        output_names = ['policy1', 'policy2', 'value']
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'move1_idx': {0: 'batch_size'},
            'policy1': {0: 'batch_size'},
            'policy2': {0: 'batch_size'},
            'value': {0: 'batch_size'},
        }
    
    # Export
    print(f"Exporting to {output_path} with batch_size={batch_size}...")
    
    # Exporting with opset 11 or higher is recommended
    torch.onnx.export(
        export_model,
        dummy_args,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        # TensorRT 当前链路使用稳定的 TorchScript 导出器；显式关闭 dynamo
        # 可以避免额外依赖 onnxscript，并保持原有动态轴语义。
        dynamo=False,
        dynamic_axes=dynamic_axes,
        custom_opsets={'nebula': 1} if config.TRT_CUSTOM_ATTENTION else None,
    )
    print("Export success!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python export_onnx.py <ckpt_path> <onnx_path>")
        sys.exit(1)
        
    ckpt_path = sys.argv[1]
    onnx_path = sys.argv[2]
    
    # Use config batch size to match runtime environment
    target_batch = config.MCTS_BATCH_SIZE
    
    export_to_onnx(ckpt_path, onnx_path, batch_size=target_batch)
