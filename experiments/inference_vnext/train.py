"""DDP distillation and native-pair training for inference-first models."""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.inference_vnext.dataset import (
    ExactPositionDataset,
    PairPositionDataset,
    load_replay_records,
    seed_data_worker,
)
from experiments.inference_vnext.model import ARCHITECTURES, build_architecture
from experiments.inference_vnext.wrapper import NativePairHeads
from experiments.nebula_v3.train import (
    DistributedShardSampler,
    WeightedDistributedSampler,
    adapt_v2_features,
    cosine_warmup_lambda,
    load_v2_teacher,
)


def default_paths():
    train_root = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
    production = os.path.join(train_root, "Nebula-Zero-Connect6")
    return {
        "data": os.path.join(
            production,
            "experiments",
            "pair_policy",
            "runs",
            "Nebular-zero-two",
            "replay",
        ),
        "teacher": os.path.join(
            production,
            "reinforcement_learning",
            "checkpoints",
            "best.pth",
        ),
        "output": os.path.join(
            train_root,
            "Nebula-Zero-Connect6-training",
            "inference_vnext",
        ),
    }


def parse_args():
    paths = default_paths()
    parser = argparse.ArgumentParser(
        description="Train inference-first Connect6 exact or pair models",
    )
    parser.add_argument("--architecture", choices=sorted(ARCHITECTURES), required=True)
    parser.add_argument("--task", choices=("exact", "pair"), default="exact")
    parser.add_argument("--data-root", default=paths["data"])
    parser.add_argument("--teacher-checkpoint", default=paths["teacher"])
    parser.add_argument("--output-dir", default=paths["output"])
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--recent-generations", type=int, default=128)
    parser.add_argument("--max-train-games", type=int, default=0)
    parser.add_argument("--max-validation-games", type=int, default=0)
    parser.add_argument("--maximum-stones", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=96, help="batch per GPU")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--samples-per-epoch", type=int, default=0)
    parser.add_argument("--validation-positions", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--trunk-learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--value-loss-weight", type=float, default=0.5)
    parser.add_argument("--second-policy-weight", type=float, default=1.0)
    parser.add_argument("--conditional-value-weight", type=float, default=0.5)
    parser.add_argument("--distill-policy-weight", type=float, default=0.3)
    parser.add_argument("--pair-rank", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--cpu-affinity",
        default="",
        help="semicolon-separated CPU sets, one per local rank",
    )
    return parser.parse_args()


def parse_cpu_set(value):
    cpus = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    return cpus


def apply_cpu_affinity(specification, local_rank):
    if not specification or not hasattr(os, "sched_setaffinity"):
        return None
    entries = specification.split(";")
    if local_rank >= len(entries):
        raise ValueError("--cpu-affinity has fewer entries than local ranks")
    cpus = parse_cpu_set(entries[local_rank])
    if not cpus:
        raise ValueError(f"empty CPU set for local rank {local_rank}")
    os.sched_setaffinity(0, cpus)
    return sorted(cpus)


def distributed_context():
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def seed_everything(seed, rank):
    effective = seed + rank
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    torch.cuda.manual_seed_all(effective)


def strip_module_prefix(state):
    return {key.removeprefix("module."): value for key, value in state.items()}


def checkpoint_model_state(checkpoint):
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    return strip_module_prefix(state)


def atomic_torch_save(payload, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json_dump(payload, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


class TrainingModel(nn.Module):
    def __init__(self, architecture, task, pair_rank):
        super().__init__()
        self.architecture = architecture
        self.task = task
        self.backbone = build_architecture(architecture)
        self.pair_heads = (
            NativePairHeads(self.backbone.feature_dim, pair_rank=pair_rank)
            if task == "pair"
            else None
        )

    def forward(self, features, first_moves=None):
        policy_logits, _, value, board_features = self.backbone(
            features,
            return_features=True,
        )
        if self.pair_heads is None:
            return policy_logits, value
        if first_moves is None:
            raise ValueError("pair training requires first_moves")
        candidate, first, pair_values, _ = self.pair_heads.forward_all(
            board_features,
            value,
        )
        second_logits, conditional_value = self.pair_heads.conditional_outputs(
            policy_logits,
            candidate,
            first,
            pair_values,
            first_moves,
        )
        return policy_logits, value, second_logits, conditional_value


def masked_policy_metrics(logits, target, occupied):
    logits = logits.float()
    illegal_mass = (
        torch.softmax(logits, dim=1) * occupied.float()
    ).sum(dim=1)
    masked = logits.masked_fill(occupied, -10_000.0)
    log_probability = torch.log_softmax(masked, dim=1)
    loss = -(target.float() * log_probability).sum(dim=1)
    target_move = target.argmax(dim=1)
    return {
        "loss": loss,
        "top1": masked.argmax(dim=1).eq(target_move).float(),
        "top5": masked.topk(5, dim=1).indices.eq(
            target_move[:, None]
        ).any(dim=1).float(),
        "illegal_mass": illegal_mass,
        "log_probability": log_probability,
    }


def reduce_metrics(metric_sums, device, world_size):
    names = sorted(metric_sums)
    values = torch.tensor(
        [metric_sums[name] for name in names],
        dtype=torch.float64,
        device=device,
    )
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    reduced = {name: float(value) for name, value in zip(names, values.cpu())}
    count = max(1.0, reduced.pop("count"))
    return {name: value / count for name, value in reduced.items()}


def empty_metric_sums(task):
    names = [
        "loss",
        "first_policy_loss",
        "first_top1",
        "first_top5",
        "illegal_mass",
        "value_loss",
        "value_mae",
        "distill_policy_loss",
    ]
    if task == "pair":
        names.extend([
            "second_policy_loss",
            "second_top1",
            "second_top5",
            "conditional_value_loss",
            "conditional_value_mae",
        ])
    return {"count": 0.0, **{name: 0.0 for name in names}}


def run_epoch(
    model,
    loader,
    task,
    device,
    world_size,
    args,
    teacher=None,
    optimizer=None,
    scheduler=None,
    global_step=0,
    rank=0,
):
    training = optimizer is not None
    model.train(training)
    if teacher is not None:
        teacher.eval()
    sums = empty_metric_sums(task)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    for batch_index, batch in enumerate(loader, start=1):
        features = batch[0].to(
            device,
            dtype=torch.float32,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
        first_target = batch[1].to(device, non_blocking=True)
        if task == "exact":
            first_moves = None
            second_target = None
            value_target = batch[2].to(device, non_blocking=True)
        else:
            first_moves = batch[2].to(device, non_blocking=True)
            second_target = batch[3].to(device, non_blocking=True)
            value_target = batch[4].to(device, non_blocking=True)

        occupied = (features[:, 0] + features[:, 1]).gt(0.5).flatten(1)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            outputs = model(features, first_moves)
            policy_logits = outputs[0]
            value_prediction = outputs[1].float().flatten()
            first_metrics = masked_policy_metrics(
                policy_logits,
                first_target,
                occupied,
            )
            first_policy_loss = first_metrics["loss"].mean()
            distill_per_sample = torch.zeros_like(first_metrics["loss"])
            if training and teacher is not None and args.distill_policy_weight > 0:
                with torch.no_grad():
                    teacher_logits, _, _ = teacher(adapt_v2_features(features))
                    teacher_probability = torch.softmax(
                        teacher_logits.float().masked_fill(occupied, -10_000.0),
                        dim=1,
                    )
                distill_per_sample = -(
                    teacher_probability * first_metrics["log_probability"]
                ).sum(dim=1)
            distill_loss = distill_per_sample.mean()
            optimized_first_loss = (
                (1.0 - args.distill_policy_weight) * first_policy_loss
                + args.distill_policy_weight * distill_loss
                if teacher is not None and args.distill_policy_weight > 0
                else first_policy_loss
            )
            value_loss_per_sample = F.smooth_l1_loss(
                value_prediction,
                value_target.float(),
                beta=0.5,
                reduction="none",
            )
            value_loss = value_loss_per_sample.mean()
            loss = optimized_first_loss + args.value_loss_weight * value_loss

            second_metrics = None
            conditional_loss_per_sample = None
            conditional_value = None
            if task == "pair":
                second_logits = outputs[2]
                conditional_value = outputs[3].float().flatten()
                occupied_second = occupied.clone()
                occupied_second.scatter_(1, first_moves[:, None], True)
                second_metrics = masked_policy_metrics(
                    second_logits,
                    second_target,
                    occupied_second,
                )
                conditional_loss_per_sample = F.smooth_l1_loss(
                    conditional_value,
                    value_target.float(),
                    beta=0.5,
                    reduction="none",
                )
                loss = (
                    loss
                    + args.second_policy_weight * second_metrics["loss"].mean()
                    + args.conditional_value_weight
                    * conditional_loss_per_sample.mean()
                )

        if training:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
        else:
            grad_norm = torch.zeros((), device=device)

        batch_size = features.shape[0]
        sums["count"] += batch_size
        sums["loss"] += float(loss.detach()) * batch_size
        sums["first_policy_loss"] += float(first_metrics["loss"].detach().sum())
        sums["first_top1"] += float(first_metrics["top1"].detach().sum())
        sums["first_top5"] += float(first_metrics["top5"].detach().sum())
        sums["illegal_mass"] += float(
            first_metrics["illegal_mass"].detach().sum()
        )
        sums["value_loss"] += float(value_loss_per_sample.detach().sum())
        sums["value_mae"] += float(
            (value_prediction - value_target.float()).detach().abs().sum()
        )
        sums["distill_policy_loss"] += float(distill_per_sample.detach().sum())
        if task == "pair":
            sums["second_policy_loss"] += float(
                second_metrics["loss"].detach().sum()
            )
            sums["second_top1"] += float(second_metrics["top1"].detach().sum())
            sums["second_top5"] += float(second_metrics["top5"].detach().sum())
            sums["conditional_value_loss"] += float(
                conditional_loss_per_sample.detach().sum()
            )
            sums["conditional_value_mae"] += float(
                (conditional_value - value_target.float()).detach().abs().sum()
            )

        if training and rank == 0 and batch_index % args.log_interval == 0:
            elapsed = time.perf_counter() - started
            throughput = sums["count"] * world_size / max(elapsed, 1e-9)
            print(
                f"step={global_step} loss={float(loss.detach()):.4f} "
                f"grad={float(grad_norm.detach()):.3f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} "
                f"throughput={throughput:.0f} samples/s",
                flush=True,
            )

    metrics = reduce_metrics(sums, device, world_size)
    elapsed = time.perf_counter() - started
    metrics["seconds"] = elapsed
    metrics["samples_per_second"] = sums["count"] * world_size / max(elapsed, 1e-9)
    metrics["rank0_peak_memory_gib"] = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    )
    return metrics, global_step


def save_checkpoint(raw_model, optimizer, scheduler, args, epoch, global_step, best, metrics, path):
    payload = {
        "architecture": args.architecture,
        "task": args.task,
        "pair_rank": args.pair_rank if args.task == "pair" else None,
        "maximum_stones": args.maximum_stones,
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_loss": best,
        "metrics": metrics,
        "model_state_dict": raw_model.backbone.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
    }
    if raw_model.pair_heads is not None:
        payload["pair_state_dict"] = raw_model.pair_heads.state_dict()
    atomic_torch_save(payload, path)


def save_pair_metadata(raw_model, args, output_dir):
    if raw_model.pair_heads is None:
        return
    heads = raw_model.pair_heads
    atomic_torch_save({
        "pair_head": {
            "base_scale": heads.base_scale.detach().cpu(),
            "relative_bias": heads.relative_bias.detach().cpu(),
        },
        "args": {
            "rank": args.pair_rank,
            "tied_factors": False,
            "relative_gating": False,
            "native_vnext": True,
        },
    }, os.path.join(output_dir, "pair_heads.pt"))


def main():
    args = parse_args()
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if not 0.0 <= args.distill_policy_weight <= 1.0:
        raise ValueError("--distill-policy-weight must be in [0, 1]")
    if args.recent_generations < 0:
        raise ValueError("--recent-generations must not be negative")
    if args.maximum_stones <= 0:
        args.maximum_stones = 64 if "sparse_stone" in args.architecture else 360
    run_name = args.run_name or f"{args.architecture}_{args.task}"
    args.output_dir = os.path.join(args.output_dir, run_name)

    local_rank_hint = int(os.environ.get("LOCAL_RANK", "0"))
    affinity = apply_cpu_affinity(args.cpu_affinity, local_rank_hint)
    rank, local_rank, world_size = distributed_context()
    seed_everything(args.seed, rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    training_records, validation_records, data_stats = load_replay_records(
        args.data_root,
        recent_generations=args.recent_generations,
        max_train_games=args.max_train_games,
        max_validation_games=args.max_validation_games,
    )
    dataset_type = ExactPositionDataset if args.task == "exact" else PairPositionDataset
    training_dataset = dataset_type(
        training_records,
        training=True,
        positions_per_game=1,
        maximum_stones=args.maximum_stones,
    )
    validation_dataset = dataset_type(
        validation_records,
        training=False,
        positions_per_game=args.validation_positions,
        maximum_stones=args.maximum_stones,
    )
    global_samples = args.samples_per_epoch or len(training_dataset)
    training_sampler = WeightedDistributedSampler(
        training_dataset.record_weights(),
        global_samples,
        rank,
        world_size,
        args.seed,
    )
    validation_sampler = DistributedShardSampler(
        len(validation_dataset),
        rank,
        world_size,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "worker_init_fn": seed_data_worker,
    }
    if args.workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 3})
    training_loader = DataLoader(
        training_dataset,
        sampler=training_sampler,
        drop_last=True,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        sampler=validation_sampler,
        drop_last=False,
        **loader_kwargs,
    )

    model = TrainingModel(args.architecture, args.task, args.pair_rank)
    if args.init_checkpoint:
        initial = torch.load(
            args.init_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        model.backbone.load_state_dict(checkpoint_model_state(initial))
        if model.pair_heads is not None and "pair_state_dict" in initial:
            model.pair_heads.load_state_dict(initial["pair_state_dict"])
    model = model.to(device, memory_format=torch.channels_last)

    teacher = None
    if args.distill_policy_weight > 0:
        teacher = load_v2_teacher(args.teacher_checkpoint, device=device)

    if args.task == "pair":
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": args.trunk_learning_rate},
                {"params": model.pair_heads.parameters(), "lr": args.head_learning_rate},
            ],
            weight_decay=args.weight_decay,
            fused=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            fused=True,
        )
    total_steps = max(1, len(training_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup_lambda(
            step,
            total_steps,
            warmup_steps,
            args.min_lr_ratio,
        ),
    )

    start_epoch = 0
    global_step = 0
    best_validation_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("architecture") != args.architecture:
            raise ValueError("resume checkpoint architecture mismatch")
        model.backbone.load_state_dict(checkpoint["model_state_dict"])
        if model.pair_heads is not None:
            model.pair_heads.load_state_dict(checkpoint["pair_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_validation_loss = float(checkpoint.get("best_validation_loss", float("inf")))

    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(json.dumps({
            "run": run_name,
            "architecture": args.architecture,
            "task": args.task,
            "parameters": sum(parameter.numel() for parameter in raw_model.parameters()),
            "world_size": world_size,
            "global_batch": args.batch_size * world_size,
            "training_positions": len(training_dataset),
            "validation_positions": len(validation_dataset),
            "maximum_stones": args.maximum_stones,
            "data": data_stats,
            "affinity": affinity,
        }, ensure_ascii=False), flush=True)

    history_path = os.path.join(args.output_dir, "metrics_history.json")
    history = {"epochs": []}
    if args.resume and os.path.exists(history_path):
        with open(history_path, encoding="utf-8") as source:
            history = json.load(source)

    try:
        for epoch in range(start_epoch, args.epochs):
            training_sampler.set_epoch(epoch)
            train_metrics, global_step = run_epoch(
                model,
                training_loader,
                args.task,
                device,
                world_size,
                args,
                teacher=teacher,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=global_step,
                rank=rank,
            )
            with torch.no_grad():
                validation_metrics, _ = run_epoch(
                    model,
                    validation_loader,
                    args.task,
                    device,
                    world_size,
                    args,
                    teacher=None,
                    optimizer=None,
                    scheduler=None,
                    global_step=global_step,
                    rank=rank,
                )
            if rank == 0:
                epoch_record = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train": train_metrics,
                    "validation": validation_metrics,
                }
                history["epochs"].append(epoch_record)
                atomic_json_dump(history, history_path)
                improved = validation_metrics["loss"] < best_validation_loss
                if improved:
                    best_validation_loss = validation_metrics["loss"]
                save_checkpoint(
                    raw_model,
                    optimizer,
                    scheduler,
                    args,
                    epoch,
                    global_step,
                    best_validation_loss,
                    epoch_record,
                    os.path.join(args.output_dir, "last.pth"),
                )
                if improved:
                    save_checkpoint(
                        raw_model,
                        optimizer,
                        scheduler,
                        args,
                        epoch,
                        global_step,
                        best_validation_loss,
                        epoch_record,
                        os.path.join(args.output_dir, "best.pth"),
                    )
                    save_pair_metadata(raw_model, args, args.output_dir)
                print(json.dumps(epoch_record, ensure_ascii=False), flush=True)
            if world_size > 1:
                dist.barrier(device_ids=[local_rank])
    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
