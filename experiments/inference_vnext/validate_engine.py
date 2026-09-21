"""Compare a random-weight candidate with its exported TensorRT engine."""

import argparse
import json
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="Validate PyTorch/TensorRT numerical parity",
    )
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--board-seed", type=int, default=99)
    parser.add_argument("--stone-counts", default="0,9,65,97,180")
    parser.add_argument("--pair", action="store_true")
    parser.add_argument("--pair-rank", type=int, default=16)
    args = parser.parse_args()

    import tensorrt as trt
    import torch

    from experiments.inference_vnext.model import build_architecture
    from experiments.inference_vnext.wrapper import FusedPairSelfPlayWrapper
    from experiments.nebula_v3.export_onnx import FusedV3SelfPlayWrapper

    trt_to_torch = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.bfloat16: torch.bfloat16,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.bool: torch.bool,
    }
    counts = [
        int(value)
        for value in args.stone_counts.split(",")
        if value.strip()
    ]
    if not counts or any(count < 0 or count > 361 for count in counts):
        raise ValueError("stone counts must be integers in [0, 361]")

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    model = build_architecture(args.architecture)
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device=device, dtype=torch.float16).eval()
    if args.pair:
        wrapper = FusedPairSelfPlayWrapper(
            model,
            compute_dtype=torch.float16,
            pair_rank=args.pair_rank,
        ).to(device=device).eval()
        if checkpoint is not None:
            if "pair_state_dict" not in checkpoint:
                raise ValueError("pair validation requires pair_state_dict")
            wrapper.pair_heads.load_state_dict(checkpoint["pair_state_dict"])
        wrapper.pair_heads.to(dtype=torch.float16)
        output_names = [
            "policy1",
            "value",
            "pair_candidate",
            "pair_first",
            "pair_value",
        ]
    else:
        wrapper = FusedV3SelfPlayWrapper(
            model,
            compute_dtype=torch.float16,
        ).to(device=device).eval()
        output_names = ["policy1", "value"]

    boards = torch.zeros(
        (len(counts), 19, 19),
        dtype=torch.int32,
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(args.board_seed)
    for batch_index, count in enumerate(counts):
        indices = torch.randperm(
            361,
            generator=generator,
            device=device,
        )[:count]
        flat_board = boards[batch_index].flatten()
        flat_board[indices[0::2]] = 1
        flat_board[indices[1::2]] = 2

    with torch.inference_mode():
        expected_outputs = wrapper(boards)

    logger = trt.Logger(trt.Logger.ERROR)
    with open(args.engine, "rb") as engine_file, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(engine_file.read())
    if engine is None:
        raise RuntimeError(f"cannot deserialize engine: {args.engine}")
    context = engine.create_execution_context()
    if not context.set_input_shape("board", tuple(boards.shape)):
        raise ValueError(f"engine does not support batch {len(counts)}")
    unresolved = context.infer_shapes()
    if unresolved:
        raise RuntimeError(f"unresolved TensorRT shapes: {unresolved}")

    tensors = {"board": boards}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if name == "board":
            continue
        tensors[name] = torch.empty(
            tuple(context.get_tensor_shape(name)),
            dtype=trt_to_torch[engine.get_tensor_dtype(name)],
            device=device,
        )
    for name, tensor in tensors.items():
        context.set_tensor_address(name, tensor.data_ptr())

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
    stream.synchronize()

    metrics = {}
    for name, expected in zip(output_names, expected_outputs):
        error = (tensors[name].float() - expected.float()).abs()
        metrics[name] = {
            "max_abs": float(error.max()),
            "mean_abs": float(error.mean()),
        }
    print(json.dumps({
        "architecture": args.architecture,
        "engine": os.path.abspath(args.engine),
        "pair": args.pair,
        "pair_rank": args.pair_rank if args.pair else None,
        "stone_counts": counts,
        "outputs": metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
