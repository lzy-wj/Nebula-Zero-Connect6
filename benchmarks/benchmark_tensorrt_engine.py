"""测量 TensorRT 引擎在不同 batch 下的纯推理吞吐与上下文显存。"""

import argparse
import json
import statistics

import tensorrt as trt
import torch


TRT_TO_TORCH = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.bfloat16: torch.bfloat16,
    trt.int32: torch.int32,
    trt.int64: torch.int64,
    trt.bool: torch.bool,
}


def input_shape(name, batch_size):
    if 'board' in name:
        return (batch_size, 19, 19)
    if 'move1' in name or 'idx' in name:
        return (batch_size,)
    return (batch_size, 17, 19, 19)


def benchmark_batch(
    engine,
    batch_size,
    warmup,
    iterations,
    samples,
    copy_outputs=False,
):
    context = engine.create_execution_context()
    tensors = {}
    host_outputs = {}

    # 必须先设置全部动态输入，之后才能查询动态输出的实际形状。
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
            continue
        shape = input_shape(name, batch_size)
        if not context.set_input_shape(name, shape):
            raise ValueError(f'引擎 profile 不支持 {name}={shape}')

    unresolved = context.infer_shapes()
    if unresolved:
        raise RuntimeError(f'仍有无法推断形状的张量: {unresolved}')

    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        dtype = TRT_TO_TORCH[engine.get_tensor_dtype(name)]
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            shape = input_shape(name, batch_size)
        else:
            shape = tuple(context.get_tensor_shape(name))
        tensor = torch.zeros(shape, dtype=dtype, device='cuda')
        tensors[name] = tensor
        context.set_tensor_address(name, tensor.data_ptr())
        if copy_outputs and engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            host_outputs[name] = torch.empty(shape, dtype=dtype, pin_memory=True)

    def execute(stream):
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('TensorRT 基准执行失败')
        for name, host_tensor in host_outputs.items():
            host_tensor.copy_(tensors[name], non_blocking=True)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            execute(stream)
    stream.synchronize()

    latency_samples = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record(stream)
            for _ in range(iterations):
                execute(stream)
            end.record(stream)
        end.synchronize()
        latency_samples.append(start.elapsed_time(end) / iterations)

    median_ms = statistics.median(latency_samples)
    return {
        'batch_size': batch_size,
        'latency_ms': round(median_ms, 4),
        'positions_per_second': round(batch_size * 1000.0 / median_ms),
        'copy_outputs': copy_outputs,
        'samples_ms': [round(value, 4) for value in latency_samples],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', required=True)
    parser.add_argument('--batches', default='16,32,64,128,256')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument(
        '--copy-outputs',
        action='store_true',
        help='把全部输出异步拷到页锁定主存，并把 D2H 计入延迟',
    )
    args = parser.parse_args()

    logger = trt.Logger(trt.Logger.ERROR)
    with open(args.engine, 'rb') as file, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(file.read())
    if engine is None:
        raise RuntimeError(f'无法反序列化引擎: {args.engine}')

    batches = [int(value) for value in args.batches.split(',') if value.strip()]
    result = {
        'engine': args.engine,
        'context_memory_mib': round(engine.device_memory_size_v2 / 2**20, 2),
        'batches': [
            benchmark_batch(
                engine,
                batch,
                args.warmup,
                args.iterations,
                args.samples,
                args.copy_outputs,
            )
            for batch in batches
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
