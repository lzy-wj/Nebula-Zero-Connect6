import torch
import torch.onnx
import sys
import os

# Local import adjustment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from core.model import C6TransNet
except ImportError:
    from model import C6TransNet # Local fallback

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
    
    # 2. Create Dummy Input
    # Shape: [Batch, 17, 19, 19]
    dummy_input = torch.randn(batch_size, 17, 19, 19, device=device)
    
    # Create Dummy Move1 Input for Autoregressive Head
    # Shape: [Batch]
    # Use index 361 (last embedding) as "padding/none"
    dummy_move1 = torch.full((batch_size,), 361, dtype=torch.long, device=device)
    
    # Export
    print(f"Exporting to {output_path} with batch_size={batch_size}...")
    
    # Exporting with opset 11 or higher is recommended
    torch.onnx.export(
        model, 
        (dummy_input, dummy_move1), # Args: Tuple of all inputs
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input', 'move1_idx'],
        output_names=['policy1', 'policy2', 'value'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'move1_idx': {0: 'batch_size'},
            'policy1': {0: 'batch_size'},
            'policy2': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        }
    )
    print("Export success!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python export_onnx.py <ckpt_path> <onnx_path>")
        sys.exit(1)
        
    ckpt_path = sys.argv[1]
    onnx_path = sys.argv[2]
    
    # We fix batch size to 32 for optimization, but dynamic axes allows flexibility
    export_to_onnx(ckpt_path, onnx_path, batch_size=32)
