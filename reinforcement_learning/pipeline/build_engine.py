import torch
# Force CUDA initialization
if torch.cuda.is_available():
    _ = torch.tensor([1.0]).cuda()
import tensorrt as trt
import os
import sys

# Add path to root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def build_engine(onnx_path, engine_path):
    # Read batch size from config
    target_batch = config.MCTS_BATCH_SIZE
    max_batch_limit = 1024  # Increased for RTX 5090
    
    print(f"Building Engine | Target Batch: {target_batch} | Max Limit: {max_batch_limit}")

    # 1. Setup Logger
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    
    # 2. Create Builder, Network, and Config
    builder = trt.Builder(TRT_LOGGER)
    
    # EXPLICIT_BATCH flag is required for ONNX models with dynamic axes
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config_trt = builder.create_builder_config()
    
    # 3. Parser
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    print(f"Parsing ONNX model from {onnx_path}...")
    # Use parse_from_file instead of parse(read()) so it can find external data files
    if not parser.parse_from_file(onnx_path):
        print('ERROR: Failed to parse the ONNX file.')
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        return False
            
    print("ONNX parsed successfully.")
    
    # 4. Optimization Profile (for dynamic batch size)
    # Even if we want fixed batch size, it's good practice to define profile
    profile = builder.create_optimization_profile()
    
    # Input name 'input' from export script
    # Shape: (Batch, 17, 19, 19)
    # We need to set shapes for ALL inputs
    
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        name = tensor.name
        print(f"Configuring profile for input: {name}")
        
        if 'input' in name:
            profile.set_shape(name, (1, 17, 19, 19), (target_batch, 17, 19, 19), (max_batch_limit, 17, 19, 19))
        elif 'move1' in name or 'idx' in name:
            profile.set_shape(name, (1,), (target_batch,), (max_batch_limit,))
            
    config_trt.add_optimization_profile(profile)
    
    # 5. FP16 Mode
    if builder.platform_has_fast_fp16:
        print("Enabling FP16 mode...")
        config_trt.set_flag(trt.BuilderFlag.FP16)
    else:
        print("FP16 not supported, using FP32.")
        
    # Workspace size
    # set_memory_pool_limit is the new API
    config_trt.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4096 * 1024 * 1024) # 4GB
    
    # 6. Build Engine
    print("Building TensorRT engine... This may take a few minutes.")
    serialized_engine = builder.build_serialized_network(network, config_trt)
    
    if serialized_engine is None:
        print("Failed to build engine.")
        return False
        
    print(f"Saving engine to {engine_path}...")
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
        
    print("Engine built and saved successfully!")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python build_engine.py <onnx_path> <engine_path>")
        sys.exit(1)
        
    onnx_p = sys.argv[1]
    engine_p = sys.argv[2]
    
    try:
        import tensorrt
        print(f"TensorRT Version: {tensorrt.__version__}")
        build_engine(onnx_p, engine_p)
    except ImportError:
        print("TensorRT python library not found. Please install it.")
