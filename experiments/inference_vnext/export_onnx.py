"""Export inference-first checkpoints or random candidates to ONNX."""

import argparse
import json
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from experiments.inference_vnext.model import ARCHITECTURES, build_architecture
from experiments.inference_vnext.wrapper import FusedPairSelfPlayWrapper
from experiments.nebula_v3.export_onnx import FusedV3SelfPlayWrapper


def main():
    parser = argparse.ArgumentParser(
        description="Export an inference-first checkpoint or random candidate to ONNX",
    )
    parser.add_argument("--architecture", choices=sorted(ARCHITECTURES), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="optional vNext checkpoint containing model and pair states",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--pair",
        action="store_true",
        help="also export rank-16 pair factors and conditional values",
    )
    parser.add_argument(
        "--pair-rank",
        type=int,
        choices=(4, 8, 16, 32),
        default=16,
        help="low-rank width used by --pair (default: 16)",
    )
    args = parser.parse_args()

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
        checkpoint_architecture = checkpoint.get("architecture")
        if checkpoint_architecture and checkpoint_architecture != args.architecture:
            raise ValueError(
                f"checkpoint architecture {checkpoint_architecture!r} does not match "
                f"{args.architecture!r}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
    parameter_count = model.parameter_count
    model = model.to(device=device, dtype=torch.float16).eval()
    if args.pair:
        wrapper = FusedPairSelfPlayWrapper(
            model,
            compute_dtype=torch.float16,
            pair_rank=args.pair_rank,
        ).to(device=device).eval()
        if checkpoint is not None:
            if "pair_state_dict" not in checkpoint:
                raise ValueError("pair export requires pair_state_dict in checkpoint")
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
        ).to(device).eval()
        output_names = ["policy1", "value"]
    dummy_board = torch.zeros(
        (args.batch_size, 19, 19),
        dtype=torch.int32,
        device=device,
    )

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy_board,),
        args.output,
        opset_version=18,
        do_constant_folding=True,
        input_names=["board"],
        output_names=output_names,
        dynamic_axes={
            "board": {0: "batch_size"},
            **{name: {0: "batch_size"} for name in output_names},
        },
        dynamo=False,
    )
    print(json.dumps({
        "architecture": args.architecture,
        "parameter_count": parameter_count,
        "onnx": os.path.abspath(args.output),
        "seed": args.seed,
        "pair": args.pair,
        "pair_rank": args.pair_rank if args.pair else None,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
