"""Measure single-GPU and DDP training-step throughput for V4 candidates."""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from experiments.nebula_v4.model import BOARD_POINTS, CANDIDATE_CONFIGS, build_model


class TrainingGraph(nn.Module):
    """Keep Pair scoring inside DDP's forward graph."""

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, features: torch.Tensor):
        unary, pair_embeddings, value, occupied, auxiliary = self.core(
            features, return_aux=True
        )
        scores = self.core.pair_scores(unary, pair_embeddings, occupied).flatten(1)
        return scores, value, auxiliary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("dense", "moe"), required=True)
    parser.add_argument("--global-batch", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    arguments = parser.parse_args()

    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    if arguments.global_batch % world_size:
        raise ValueError("global batch must be divisible by world size")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(20261013 + rank)
    model = TrainingGraph(
        build_model({**CANDIDATE_CONFIGS[arguments.variant], "summary_tokens": 64})
    ).to(device, memory_format=torch.channels_last)
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=arguments.variant == "moe",
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, fused=True)
    local_batch = arguments.global_batch // world_size
    features = torch.zeros((local_batch, 5, 19, 19), device=device).contiguous(
        memory_format=torch.channels_last
    )
    features[:, 2].fill_(1.0)
    features[:, 3].fill_(1.0)
    target = torch.full((local_batch,), 1, dtype=torch.long, device=device)
    value_target = torch.empty(local_batch, device=device).uniform_(-1.0, 1.0)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores, value, auxiliary = model(features)
            loss = F.cross_entropy(scores, target) + F.mse_loss(value, value_target)
            if "balance_loss" in auxiliary:
                loss = loss + 0.1 * auxiliary["balance_loss"]
        loss.backward()
        optimizer.step()

    for _ in range(arguments.warmup):
        step()
    torch.cuda.synchronize()
    if distributed:
        dist.barrier()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(arguments.steps):
        step()
    torch.cuda.synchronize()
    if distributed:
        dist.barrier()
    seconds = time.perf_counter() - started
    if rank == 0:
        print(
            json.dumps(
                {
                    "variant": arguments.variant,
                    "world_size": world_size,
                    "global_batch": arguments.global_batch,
                    "steps": arguments.steps,
                    "seconds": seconds,
                    "samples_per_second": arguments.global_batch * arguments.steps / seconds,
                    "peak_memory_gib_per_rank": torch.cuda.max_memory_allocated()
                    / 1024**3,
                }
            ),
            flush=True,
        )
    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
