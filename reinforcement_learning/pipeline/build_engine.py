import os
import sys

# TensorRT tactic 测量必须在目标卡上完成；默认只使用预留的物理卡 6。
os.environ.setdefault(
    'CUDA_VISIBLE_DEVICES',
    os.environ.get('NEBULA_BUILD_GPU', '6'),
)

import torch
# Force CUDA initialization
if torch.cuda.is_available():
    _ = torch.tensor([1.0]).cuda()
import tensorrt as trt

# Add path to root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def build_engine(onnx_path, engine_path):
    # Read batch size from config
    target_batch = config.MCTS_BATCH_SIZE
    # profile 的 max 会直接决定执行上下文的显存上界。自对弈只会产生
    # 1..MCTS_BATCH_SIZE 的批量，默认不再为 1024 预留近 2GB 激活内存；
    # 做跨对局集中批处理时可通过环境变量显式放大后重新构建。
    max_batch_limit = int(
        os.environ.get('NEBULA_TRT_MAX_BATCH_SIZE', str(target_batch))
    )
    if max_batch_limit < target_batch:
        raise ValueError('NEBULA_TRT_MAX_BATCH_SIZE 不能小于 MCTS_BATCH_SIZE')
    
    print(f"Building Engine | Target Batch: {target_batch} | Max Limit: {max_batch_limit}")

    # 1. Setup Logger
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    
    # 2. Create Builder, Network, and Config
    builder = trt.Builder(TRT_LOGGER)
    
    # TensorRT 10 及更早版本需要显式 batch 标志；TensorRT 11 已移除该
    # 枚举并默认使用显式 batch，因此按版本能力选择。
    explicit_batch = getattr(trt.NetworkDefinitionCreationFlag, 'EXPLICIT_BATCH', None)
    network_flags = 0 if explicit_batch is None else 1 << int(explicit_batch)
    if config.TRT_CUSTOM_ATTENTION:
        # 导入模块会向 TensorRT 注册插件创建器；AOT PTX 会嵌入最终引擎，
        # 所以部署和运行阶段不需要 Python 回调。
        from core import trt_relative_attention  # noqa: F401

        prefer_aot = getattr(
            trt.NetworkDefinitionCreationFlag,
            'PREFER_AOT_PYTHON_PLUGINS',
            None,
        )
        if prefer_aot is None:
            raise RuntimeError('当前 TensorRT 版本不支持 Python AOT 插件')
        network_flags |= 1 << int(prefer_aot)
    network = builder.create_network(network_flags)
    config_trt = builder.create_builder_config()
    optimization_level = int(os.environ.get('NEBULA_TRT_OPT_LEVEL', '5'))
    if not 0 <= optimization_level <= 5:
        raise ValueError('NEBULA_TRT_OPT_LEVEL 必须位于 0..5')
    config_trt.builder_optimization_level = optimization_level
    config_trt.avg_timing_iterations = int(
        os.environ.get('NEBULA_TRT_TIMING_ITERATIONS', '1')
    )
    aux_streams = os.environ.get('NEBULA_TRT_AUX_STREAMS')
    if aux_streams is not None:
        config_trt.max_aux_streams = int(aux_streams)
    print(
        f"TensorRT tactic 等级: {config_trt.builder_optimization_level} | "
        f"计时轮数: {config_trt.avg_timing_iterations} | "
        f"辅助流上限: {config_trt.max_aux_streams}"
    )

    # 各代模型只改变权重，层形状保持一致。持久化 tactic 计时缓存可以让
    # 等级 5 的搜索成本只在首次构建支付，后续代直接复用已测得的内核。
    timing_cache_path = os.environ.get(
        'NEBULA_TRT_TIMING_CACHE',
        os.path.join(config.CHECKPOINT_DIR, 'tensorrt_timing.cache'),
    )
    cache_data = b''
    if os.path.exists(timing_cache_path):
        with open(timing_cache_path, 'rb') as cache_file:
            cache_data = cache_file.read()
        print(f"加载 TensorRT timing cache: {timing_cache_path}")
    timing_cache = config_trt.create_timing_cache(cache_data)
    if not config_trt.set_timing_cache(timing_cache, ignore_mismatch=False):
        raise RuntimeError('TensorRT timing cache 与当前 GPU/版本不兼容')
    
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
        
        if 'board' in name:
            profile.set_shape(name, (1, 19, 19), (target_batch, 19, 19), (max_batch_limit, 19, 19))
        elif 'input' in name:
            profile.set_shape(name, (1, 17, 19, 19), (target_batch, 17, 19, 19), (max_batch_limit, 17, 19, 19))
        elif 'move1' in name or 'idx' in name:
            profile.set_shape(name, (1,), (target_batch,), (max_batch_limit,))
            
    config_trt.add_optimization_profile(profile)
    
    # 推理精度与训练精度分开配置。通常 TensorRT FP16 的兼容性和吞吐
    # 更稳定；如需验证 BF16，可通过 NEBULA_TRT_PRECISION=bf16 开启。
    requested_precision = config.TRT_PRECISION
    if (
        requested_precision == 'bf16'
        and hasattr(trt.BuilderFlag, 'BF16')
        and getattr(builder, 'platform_has_fast_bf16', True)
    ):
        config_trt.set_flag(trt.BuilderFlag.BF16)
        print("Enabling BF16 mode...")
    elif (
        requested_precision == 'fp16'
        and hasattr(trt.BuilderFlag, 'FP16')
        and getattr(builder, 'platform_has_fast_fp16', True)
    ):
        config_trt.set_flag(trt.BuilderFlag.FP16)
        print("Enabling FP16 mode...")
    elif requested_precision in {'fp16', 'bf16'}:
        # TensorRT 11 使用强类型网络，精度已由 ONNX 图中的 Cast 和权重
        # 类型确定，不再提供 FP16/BF16 BuilderFlag。
        print(f"Using strongly typed {requested_precision.upper()} ONNX graph...")
    else:
        print("Using FP32 mode.")
        
    # Workspace size
    # set_memory_pool_limit is the new API
    config_trt.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4096 * 1024 * 1024) # 4GB
    
    # 6. Build Engine
    print("Building TensorRT engine... This may take a few minutes.")
    serialized_engine = builder.build_serialized_network(network, config_trt)
    
    if serialized_engine is None:
        print("Failed to build engine.")
        return False

    updated_cache = config_trt.get_timing_cache()
    if updated_cache is not None:
        cache_dir = os.path.dirname(timing_cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        cache_tmp_path = timing_cache_path + '.tmp'
        with open(cache_tmp_path, 'wb') as cache_file:
            cache_file.write(bytes(updated_cache.serialize()))
            cache_file.flush()
            os.fsync(cache_file.fileno())
        os.replace(cache_tmp_path, timing_cache_path)
        print(f"保存 TensorRT timing cache: {timing_cache_path}")
        
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
        success = build_engine(onnx_p, engine_p)
        sys.exit(0 if success else 1)
    except ImportError:
        print("TensorRT python library not found. Please install it.")
        sys.exit(1)
    except Exception as e:
        print(f"TensorRT engine build failed: {e}")
        sys.exit(1)
