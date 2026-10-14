"""Small, strict CPU supervision experiment for Nebula V4 candidates."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from experiments.nebula_v4.model import (
    BOARD_POINTS,
    CANDIDATE_CONFIGS,
    active_parameter_count,
    build_model,
    parameter_count,
)
from experiments.nebula_v4.mmap_dataset import MMapTurnDataset, collate_mmap_turns
from experiments.pair_native.pair_replay import PASS_MOVE
from experiments.pair_native.train import PairTurnDataset, collate_turns, load_data


def limited_rows(rows, maximum: int, seed: int):
    if maximum <= 0 or len(rows) <= maximum:
        return rows
    generator = np.random.default_rng(seed)
    indices = generator.choice(len(rows), size=maximum, replace=False)
    return [rows[int(index)] for index in sorted(indices)]


def hierarchical_auxiliary_loss(
    model,
    unary,
    pair_embeddings,
    occupied,
    target_a,
    target_b,
    target_probability,
):
    """Teach a first-stone marginal and bidirectional conditional second stone."""
    probability = target_probability.float()
    valid = probability.gt(0)
    singleton = target_b.eq(PASS_MOVE) & valid
    pair = target_b.ge(0) & valid
    safe_a = target_a.clamp_min(0)
    safe_b = target_b.clamp_min(0)

    first_target = torch.zeros_like(unary, dtype=torch.float32)
    first_target.scatter_add_(
        1,
        safe_a,
        probability * (singleton.float() + 0.5 * pair.float()),
    )
    first_target.scatter_add_(1, safe_b, probability * 0.5 * pair.float())
    first_logits = unary.float().masked_fill(occupied, -1.0e9)
    first_loss = -(
        first_target * torch.log_softmax(first_logits, dim=1)
    ).sum(dim=1).mean()

    target_index = probability.argmax(dim=1, keepdim=True)
    top_a = safe_a.gather(1, target_index).squeeze(1)
    top_b_raw = target_b.gather(1, target_index).squeeze(1)
    top_pair = top_b_raw.ge(0)
    top_b = torch.where(top_pair, top_b_raw, top_a)
    first_moves = torch.stack((top_a, top_b), dim=1)
    rows = model.pair_row_scores(
        unary,
        pair_embeddings,
        occupied,
        first_moves,
    ).float()
    row_log_probability = torch.log_softmax(rows, dim=2)
    a_to_b = row_log_probability[:, 0].gather(1, top_b[:, None]).squeeze(1)
    b_to_a = row_log_probability[:, 1].gather(1, top_a[:, None]).squeeze(1)
    conditional = -0.5 * (a_to_b + b_to_a)
    conditional_loss = (
        conditional[top_pair].mean()
        if top_pair.any()
        else first_loss.new_zeros(())
    )
    return first_loss + conditional_loss, first_loss, conditional_loss


def compute_loss(
    model,
    batch,
    device: torch.device,
    dtype: torch.dtype,
    value_weight: float,
    router_balance_weight: float,
    router_z_weight: float,
    hierarchical_aux_weight: float = 0.0,
):
    non_blocking = device.type == "cuda"
    features = batch["features"].to(
        device,
        non_blocking=non_blocking,
        memory_format=torch.channels_last,
    )
    target_a = batch["policy_a"].to(device, non_blocking=non_blocking)
    target_b = batch["policy_b"].to(device, non_blocking=non_blocking)
    target_probability = batch["policy_probability"].to(
        device,
        non_blocking=non_blocking,
    )
    value_target = batch["value"].to(device, non_blocking=non_blocking)
    chosen_a = batch["chosen_a"].to(device, non_blocking=non_blocking)
    chosen_b = batch["chosen_b"].to(device, non_blocking=non_blocking)

    with torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=device.type == "cuda",
    ):
        unary, pair_embeddings, value, occupied, pair_scores, auxiliary = model(
            features,
            return_aux=True,
            return_pair_scores=True,
        )
    unary = unary.float().masked_fill(occupied, -1.0e9)
    pair_scores = pair_scores.float().flatten(1)
    unary_log_probability = torch.log_softmax(unary, dim=1)
    pair_log_probability = torch.log_softmax(pair_scores, dim=1)
    singleton = target_b.eq(PASS_MOVE)
    valid = target_probability.gt(0)
    safe_a = target_a.clamp_min(0)
    safe_b = target_b.clamp_min(0)
    pair_ids = safe_a * BOARD_POINTS + safe_b
    pair_terms = pair_log_probability.gather(1, pair_ids)
    singleton_terms = unary_log_probability.gather(1, safe_a)
    log_terms = torch.where(singleton, singleton_terms, pair_terms)
    policy_loss = -(target_probability * log_terms * valid).sum(dim=1).mean()
    value_loss = F.mse_loss(value.float(), value_target)
    balance_loss = auxiliary.get("balance_loss", policy_loss.new_zeros(()))
    router_z_loss = auxiliary.get("router_z_loss", policy_loss.new_zeros(()))
    loss = (
        policy_loss
        + value_weight * value_loss
        + router_balance_weight * balance_loss
        + router_z_weight * router_z_loss
    )
    hierarchical_loss = policy_loss.new_zeros(())
    hierarchical_first_loss = policy_loss.new_zeros(())
    hierarchical_conditional_loss = policy_loss.new_zeros(())
    if hierarchical_aux_weight > 0.0:
        core_model = (
            model.module if isinstance(model, DistributedDataParallel) else model
        )
        (
            hierarchical_loss,
            hierarchical_first_loss,
            hierarchical_conditional_loss,
        ) = hierarchical_auxiliary_loss(
            core_model,
            unary,
            pair_embeddings,
            occupied,
            target_a,
            target_b,
            target_probability,
        )
        loss = loss + hierarchical_aux_weight * hierarchical_loss

    target_index = target_probability.argmax(dim=1, keepdim=True)
    target_top_a = target_a.gather(1, target_index).squeeze(1)
    target_top_b = target_b.gather(1, target_index).squeeze(1)
    pair_top5 = pair_scores.topk(k=5, dim=1).indices
    pair_prediction = pair_top5[:, 0]
    predicted_a = pair_prediction.div(BOARD_POINTS, rounding_mode="floor")
    predicted_b = pair_prediction % BOARD_POINTS
    unary_top5 = unary.topk(k=5, dim=1).indices
    unary_prediction = unary_top5[:, 0]
    target_singleton = target_top_b.eq(PASS_MOVE)
    predicted_a = torch.where(target_singleton, unary_prediction, predicted_a)
    predicted_b = torch.where(
        target_singleton,
        torch.full_like(predicted_b, PASS_MOVE),
        predicted_b,
    )
    pair_top5_a = pair_top5.div(BOARD_POINTS, rounding_mode="floor")
    pair_top5_b = pair_top5 % BOARD_POINTS
    pair_top5_hit = (
        pair_top5_a.eq(target_top_a[:, None])
        & pair_top5_b.eq(target_top_b[:, None])
    ).any(dim=1)
    unary_top5_hit = unary_top5.eq(target_top_a[:, None]).any(dim=1)
    target_top5 = torch.where(target_singleton, unary_top5_hit, pair_top5_hit)
    support_mass = (log_terms.exp() * valid).sum(dim=1)
    chosen_top1 = predicted_a.eq(chosen_a) & predicted_b.eq(chosen_b)
    decisive = value_target.abs().gt(0.25)
    value_sign_correct = (
        torch.sign(value.float()).eq(torch.sign(value_target)) & decisive
    )
    metrics = {
        "loss": loss.detach(),
        "supervised_loss": (policy_loss + value_weight * value_loss).detach(),
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "policy_top1": (
            predicted_a.eq(target_top_a) & predicted_b.eq(target_top_b)
        ).float().mean().detach(),
        "policy_top5": target_top5.float().mean().detach(),
        "chosen_action_top1": chosen_top1.float().mean().detach(),
        "policy_support_mass": support_mass.mean().detach(),
        "value_mae": F.l1_loss(value.float(), value_target).detach(),
        "value_decisive_fraction": decisive.float().mean().detach(),
        "value_decisive_correct_fraction": value_sign_correct.float().mean().detach(),
        "router_balance_loss": balance_loss.detach(),
        "router_z_loss": router_z_loss.detach(),
        "hierarchical_loss": hierarchical_loss.detach(),
        "hierarchical_first_loss": hierarchical_first_loss.detach(),
        "hierarchical_conditional_loss": hierarchical_conditional_loss.detach(),
        "router_entropy": (
            -(auxiliary["probabilities"] * auxiliary["probabilities"].clamp_min(1e-9).log())
            .sum(dim=1)
            .mean()
            .detach()
            if "probabilities" in auxiliary
            else policy_loss.new_zeros(())
        ),
        "router_confidence": (
            auxiliary["probabilities"].amax(dim=1).mean().detach()
            if "probabilities" in auxiliary
            else policy_loss.new_zeros(())
        ),
    }
    assignments = auxiliary.get("assignments")
    return loss, metrics, assignments


def run_epoch(
    model,
    loader,
    device: torch.device,
    dtype: torch.dtype,
    value_weight: float,
    router_balance_weight: float,
    router_z_weight: float,
    hierarchical_aux_weight: float = 0.0,
    optimizer=None,
    scheduler=None,
    metric_log=None,
    epoch: int = 0,
    log_interval: int = 0,
):
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    core_model = model.module if isinstance(model, DistributedDataParallel) else model
    training = optimizer is not None
    model.train(training)
    totals = {}
    expert_counts_tensor = None
    samples = 0
    gradient_norm_total = None
    gradient_norm_maximum = None
    optimizer_steps = 0
    learning_rate_start = optimizer.param_groups[0]["lr"] if training else 0.0
    window_totals = {}
    window_samples = 0
    window_expert_counts = None
    window_gradient_norm = None
    window_steps = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for batch in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss, metrics, assignments = compute_loss(
                model,
                batch,
                device,
                dtype,
                value_weight,
                router_balance_weight,
                router_z_weight,
                hierarchical_aux_weight,
            )
        if training:
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach()
            gradient_norm_total = (
                gradient_norm
                if gradient_norm_total is None
                else gradient_norm_total + gradient_norm
            )
            gradient_norm_maximum = (
                gradient_norm
                if gradient_norm_maximum is None
                else torch.maximum(gradient_norm_maximum, gradient_norm)
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer_steps += 1
        batch_size = batch["features"].shape[0]
        samples += batch_size
        for name, value in metrics.items():
            weighted = value.detach() * batch_size
            totals[name] = weighted if name not in totals else totals[name] + weighted
            window_totals[name] = (
                weighted
                if name not in window_totals
                else window_totals[name] + weighted
            )
        window_samples += batch_size
        if assignments is not None:
            counts = torch.bincount(
                assignments,
                minlength=core_model.moe.num_experts,
            )
            expert_counts_tensor = (
                counts
                if expert_counts_tensor is None
                else expert_counts_tensor + counts
            )
            window_expert_counts = (
                counts
                if window_expert_counts is None
                else window_expert_counts + counts
            )
        if training:
            window_gradient_norm = (
                gradient_norm
                if window_gradient_norm is None
                else window_gradient_norm + gradient_norm
            )
            window_steps += 1
        if (
            training
            and log_interval > 0
            and optimizer_steps % log_interval == 0
        ):
            logged_samples = window_samples
            logged_totals = window_totals
            logged_expert_counts = window_expert_counts
            if distributed:
                sample_tensor = torch.tensor(
                    window_samples, device=device, dtype=torch.long
                )
                dist.all_reduce(sample_tensor, op=dist.ReduceOp.SUM)
                logged_samples = int(sample_tensor)
                logged_totals = dict(window_totals)
                for value in logged_totals.values():
                    dist.all_reduce(value, op=dist.ReduceOp.SUM)
                if logged_expert_counts is not None:
                    logged_expert_counts = logged_expert_counts.clone()
                    dist.all_reduce(logged_expert_counts, op=dist.ReduceOp.SUM)
            event = {
                "epoch": epoch,
                "optimizer_step": scheduler.last_epoch if scheduler is not None else optimizer_steps,
                "samples": logged_samples,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm_mean": float(window_gradient_norm) / max(window_steps, 1),
                **{
                    name: float(value) / max(logged_samples, 1)
                    for name, value in logged_totals.items()
                },
            }
            if logged_expert_counts is not None:
                counts = logged_expert_counts.cpu().tolist()
                total = max(sum(counts), 1)
                event["expert_counts"] = counts
                event["expert_shares"] = [value / total for value in counts]
            if rank == 0:
                metric_log.write(json.dumps(event, ensure_ascii=False) + "\n")
                metric_log.flush()
            window_totals = {}
            window_samples = 0
            window_expert_counts = None
            window_gradient_norm = None
            window_steps = 0
    if training and window_samples:
        logged_samples = window_samples
        logged_totals = window_totals
        logged_expert_counts = window_expert_counts
        if distributed:
            sample_tensor = torch.tensor(window_samples, device=device, dtype=torch.long)
            dist.all_reduce(sample_tensor, op=dist.ReduceOp.SUM)
            logged_samples = int(sample_tensor)
            logged_totals = dict(window_totals)
            for value in logged_totals.values():
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
            if logged_expert_counts is not None:
                logged_expert_counts = logged_expert_counts.clone()
                dist.all_reduce(logged_expert_counts, op=dist.ReduceOp.SUM)
        event = {
            "epoch": epoch,
            "optimizer_step": scheduler.last_epoch if scheduler is not None else optimizer_steps,
            "samples": logged_samples,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "gradient_norm_mean": float(window_gradient_norm) / max(window_steps, 1),
            **{
                name: float(value) / max(logged_samples, 1)
                for name, value in logged_totals.items()
            },
        }
        if logged_expert_counts is not None:
            counts = logged_expert_counts.cpu().tolist()
            total = max(sum(counts), 1)
            event["expert_counts"] = counts
            event["expert_shares"] = [value / total for value in counts]
        if rank == 0:
            metric_log.write(json.dumps(event, ensure_ascii=False) + "\n")
            metric_log.flush()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    if distributed:
        sample_tensor = torch.tensor(samples, device=device, dtype=torch.long)
        dist.all_reduce(sample_tensor, op=dist.ReduceOp.SUM)
        samples = int(sample_tensor)
        for name in totals:
            dist.all_reduce(totals[name], op=dist.ReduceOp.SUM)
        seconds_tensor = torch.tensor(seconds, device=device)
        dist.all_reduce(seconds_tensor, op=dist.ReduceOp.MAX)
        seconds = float(seconds_tensor)
        if expert_counts_tensor is not None:
            dist.all_reduce(expert_counts_tensor, op=dist.ReduceOp.SUM)
    expert_counts = (
        {
            index: int(count)
            for index, count in enumerate(expert_counts_tensor.cpu().tolist())
        }
        if expert_counts_tensor is not None
        else {}
    )
    result = {
        **{name: float(value) / max(samples, 1) for name, value in totals.items()},
        "samples": samples,
        "seconds": seconds,
        "samples_per_second": samples / max(seconds, 1.0e-9),
        "expert_counts": dict(sorted(expert_counts.items())),
    }
    result["value_decisive_sign_accuracy"] = (
        result.pop("value_decisive_correct_fraction")
        / max(result["value_decisive_fraction"], 1.0e-12)
    )
    if expert_counts:
        loads = np.asarray(
            [expert_counts[index] for index in range(core_model.moe.num_experts)],
            dtype=np.float64,
        )
        shares = loads / max(float(loads.sum()), 1.0)
        result["expert_shares"] = shares.tolist()
        result["expert_max_share"] = float(shares.max())
        result["expert_load_cv"] = float(loads.std() / max(loads.mean(), 1.0))
        nonzero = shares[shares > 0]
        result["expert_load_entropy"] = float(-(nonzero * np.log(nonzero)).sum())
    if training:
        result.update({
            "gradient_norm_mean": float(gradient_norm_total) / max(optimizer_steps, 1),
            "gradient_norm_max": float(gradient_norm_maximum),
            "learning_rate_start": learning_rate_start,
            "learning_rate_end": optimizer.param_groups[0]["lr"],
            "optimizer_steps": optimizer_steps,
        })
    return result


def atomic_save(payload, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", action="append", default=[])
    parser.add_argument("--validation-data-dir", action="append", default=[])
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=("dense", "moe"), default="dense")
    parser.add_argument("--summary-tokens", type=int, default=64)
    parser.add_argument("--feature-dim", type=int, default=0)
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--num-heads", type=int, default=0)
    parser.add_argument("--pair-rank", type=int, default=48)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--expert-expansion", type=int, default=0)
    parser.add_argument("--expert-depth", type=int, default=0)
    parser.add_argument("--shared-expansion", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--final-full-validation", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--router-balance-weight", type=float, default=0.01)
    parser.add_argument("--router-z-weight", type=float, default=1e-4)
    parser.add_argument("--hierarchical-aux-weight", type=float, default=0.0)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--replay-audit-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    arguments = parser.parse_args()
    distributed = "RANK" in os.environ
    if distributed:
        if arguments.device != "cuda":
            raise ValueError("DDP mode requires --device cuda")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        rank = 0
        world_size = 1
    preset = CANDIDATE_CONFIGS[arguments.variant]
    feature_dim = arguments.feature_dim or preset["feature_dim"]
    depth = arguments.depth or preset["depth"]
    num_heads = arguments.num_heads or preset["num_heads"]
    expert_expansion = arguments.expert_expansion or preset["expert_expansion"]
    expert_depth = arguments.expert_depth or preset["expert_depth"]
    if feature_dim % num_heads:
        raise ValueError("feature-dim must be divisible by num-heads")
    if arguments.torch_threads < 1:
        raise ValueError("torch-threads must be positive")
    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not 0.0 <= arguments.warmup_ratio < 1.0:
        raise ValueError("warmup-ratio must be in [0, 1)")
    if not 0.0 <= arguments.minimum_learning_rate <= arguments.learning_rate:
        raise ValueError("minimum-learning-rate must be in [0, learning-rate]")
    if arguments.log_interval < 0:
        raise ValueError("log-interval must be non-negative")
    if arguments.hierarchical_aux_weight < 0.0:
        raise ValueError("hierarchical-aux-weight must be non-negative")
    if bool(arguments.cache_dir) == bool(arguments.data_dir):
        raise ValueError("provide exactly one of --cache-dir or --data-dir")
    if arguments.batch_size % world_size:
        raise ValueError("batch-size must be divisible by DDP world size")

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.set_num_threads(arguments.torch_threads)
    torch.set_num_interop_threads(1)

    cache_metadata = None
    if arguments.cache_dir:
        cache_dir = Path(arguments.cache_dir).expanduser().resolve()
        cache_metadata = json.loads(
            (cache_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if not cache_metadata.get("strict_audit"):
            raise ValueError("mmap cache is not marked strictly audited")
        if int(cache_metadata["seed"]) != arguments.seed:
            raise ValueError("mmap cache seed does not match training seed")
        training_probe = MMapTurnDataset(cache_dir, "train", training=True)
        training_selection = None
        if 0 < arguments.train_samples < len(training_probe):
            generator = np.random.default_rng(arguments.seed)
            training_selection = np.sort(
                generator.choice(
                    len(training_probe),
                    size=arguments.train_samples,
                    replace=False,
                )
            )
        training_dataset = MMapTurnDataset(
            cache_dir,
            "train",
            training=True,
            indices=training_selection,
        )
        full_validation_dataset = MMapTurnDataset(
            cache_dir, "validation", training=False
        )
        validation_indices = np.load(
            cache_dir / "validation_indices.npy", mmap_mode="r"
        )
        if 0 < arguments.validation_samples < len(validation_indices):
            generator = np.random.default_rng(arguments.seed + 1)
            selected = np.sort(
                generator.choice(
                    len(validation_indices),
                    size=arguments.validation_samples,
                    replace=False,
                )
            )
            validation_indices = validation_indices[selected]
        validation_dataset = MMapTurnDataset(
            cache_dir,
            "validation",
            training=False,
            indices=validation_indices,
        )
        collate_function = collate_mmap_turns
        games_count = int(cache_metadata.get("source_games", 0))
        audited_shards = int(cache_metadata["source_shards"])
    else:
        data_dirs = [Path(value).expanduser().resolve() for value in arguments.data_dir]
        games, _, training_rows, validation_rows, audits = load_data(
            data_dirs,
            arguments.validation_fraction,
            single_pass_audit_load=True,
            replay_audit_workers=arguments.replay_audit_workers,
        )
        if arguments.validation_data_dir:
            # Generation-level holdout: keep every row from the training directories
            # in training and every row from the validation directories in validation.
            training_rows = training_rows + validation_rows
            validation_dirs = [
                Path(value).expanduser().resolve()
                for value in arguments.validation_data_dir
            ]
            validation_games, _, validation_train, validation_rows, validation_audits = load_data(
                validation_dirs,
                arguments.validation_fraction,
                single_pass_audit_load=True,
                replay_audit_workers=arguments.replay_audit_workers,
            )
            overlap = set(games).intersection(validation_games)
            if overlap:
                raise ValueError(f"training/validation game leakage: {len(overlap)} games")
            games.update(validation_games)
            validation_rows = validation_train + validation_rows
            audits.extend(validation_audits)
        full_validation_rows = validation_rows
        training_rows = limited_rows(
            training_rows,
            arguments.train_samples,
            arguments.seed,
        )
        validation_rows = limited_rows(
            validation_rows,
            arguments.validation_samples,
            arguments.seed + 1,
        )
        training_dataset = PairTurnDataset(training_rows, games, training=True)
        validation_dataset = PairTurnDataset(validation_rows, games, training=False)
        full_validation_dataset = PairTurnDataset(
            full_validation_rows, games, training=False
        )
        collate_function = collate_turns
        games_count = len(games)
        audited_shards = len(audits)
    loader_options = {
        "batch_size": arguments.batch_size // world_size,
        "num_workers": arguments.workers,
        "collate_fn": collate_function,
        "pin_memory": arguments.device == "cuda",
    }
    if arguments.workers:
        loader_options.update({"persistent_workers": True, "prefetch_factor": 2})
    training_sampler = (
        DistributedSampler(
            training_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=arguments.seed,
        )
        if distributed
        else None
    )
    validation_sampler = (
        DistributedSampler(
            validation_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        if distributed
        else None
    )
    training_loader = DataLoader(
        training_dataset,
        shuffle=training_sampler is None,
        sampler=training_sampler,
        generator=torch.Generator().manual_seed(arguments.seed),
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        sampler=validation_sampler,
        **loader_options,
    )
    full_validation_loader = None
    if arguments.final_full_validation and len(validation_dataset) < len(full_validation_dataset):
        full_validation_sampler = (
            DistributedSampler(
                full_validation_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
            )
            if distributed
            else None
        )
        full_validation_loader = DataLoader(
            full_validation_dataset,
            shuffle=False,
            sampler=full_validation_sampler,
            **loader_options,
        )

    model_config = {
        "variant": arguments.variant,
        "feature_dim": feature_dim,
        "summary_tokens": arguments.summary_tokens,
        "depth": depth,
        "num_heads": num_heads,
        "pair_rank": arguments.pair_rank,
        "num_experts": arguments.num_experts,
        "expert_expansion": expert_expansion,
        "expert_depth": expert_depth,
        "shared_expansion": arguments.shared_expansion,
    }
    device = (
        torch.device("cuda", local_rank)
        if distributed
        else torch.device(arguments.device)
    )
    dtype = torch.float16 if arguments.dtype == "fp16" else torch.bfloat16
    model = build_model(model_config).to(device, memory_format=torch.channels_last)
    if arguments.init_checkpoint:
        initial_checkpoint = torch.load(
            Path(arguments.init_checkpoint).expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        if initial_checkpoint["model_config"] != model_config:
            raise ValueError("initial checkpoint model config does not match")
        model.load_state_dict(initial_checkpoint["model_state_dict"])
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=arguments.variant == "moe",
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        fused=device.type == "cuda",
    )
    total_steps = arguments.epochs * len(training_loader)
    warmup_steps = (
        max(1, round(total_steps * arguments.warmup_ratio))
        if arguments.warmup_ratio > 0.0
        else 0
    )
    minimum_ratio = arguments.minimum_learning_rate / arguments.learning_rate

    def learning_rate_scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1.0e-8)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)
    output_dir = Path(arguments.output_dir).expanduser().resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    core_model = model.module if isinstance(model, DistributedDataParallel) else model
    history = {
        "purpose": "structural supervision pilot; not a strength result",
        "config": vars(arguments),
        "model_config": model_config,
        "parameters": parameter_count(core_model),
        "active_parameters": active_parameter_count(core_model),
        "games": games_count,
        "training_turns": len(training_dataset),
        "validation_turns": len(validation_dataset),
        "full_validation_turns": len(full_validation_dataset),
        "audited_shards": audited_shards,
        "cache_metadata": cache_metadata,
        "device": str(device),
        "world_size": world_size,
        "gradient_checkpointing": False,
        "total_optimizer_steps": total_steps,
        "warmup_steps": warmup_steps,
        "epochs": [],
    }
    with torch.no_grad():
        history["initial_validation"] = run_epoch(
            model,
            validation_loader,
            device,
            dtype,
            arguments.value_weight,
            arguments.router_balance_weight,
            arguments.router_z_weight,
            arguments.hierarchical_aux_weight,
        )
    best_loss = math.inf
    metrics_path = output_dir / "metrics.jsonl"
    metric_log = metrics_path.open("w", encoding="utf-8") if rank == 0 else None
    try:
        for epoch in range(arguments.epochs):
            if training_sampler is not None:
                training_sampler.set_epoch(epoch)
            training = run_epoch(
                model,
                training_loader,
                device,
                dtype,
                arguments.value_weight,
                arguments.router_balance_weight,
                arguments.router_z_weight,
                arguments.hierarchical_aux_weight,
                optimizer,
                scheduler,
                metric_log,
                epoch + 1,
                arguments.log_interval,
            )
            with torch.no_grad():
                validation = run_epoch(
                    model,
                    validation_loader,
                    device,
                    dtype,
                    arguments.value_weight,
                    arguments.router_balance_weight,
                    arguments.router_z_weight,
                    arguments.hierarchical_aux_weight,
                )
            record = {"epoch": epoch + 1, "training": training, "validation": validation}
            history["epochs"].append(record)
            payload = {
                "schema_version": 1,
                "model_config": model_config,
                "model_state_dict": core_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "metrics": record,
            }
            if rank == 0:
                atomic_save(payload, output_dir / "last.pth")
                atomic_save(payload, output_dir / f"epoch-{epoch + 1}.pth")
                if validation["supervised_loss"] < best_loss:
                    best_loss = validation["supervised_loss"]
                    atomic_save(payload, output_dir / "best.pth")
                (output_dir / "history.json").write_text(
                    json.dumps(history, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        if metric_log is not None:
            metric_log.close()
    if full_validation_loader is not None:
        with torch.no_grad():
            history["final_full_validation"] = run_epoch(
                model,
                full_validation_loader,
                device,
                dtype,
                arguments.value_weight,
                arguments.router_balance_weight,
                arguments.router_z_weight,
                arguments.hierarchical_aux_weight,
            )
        if rank == 0:
            (output_dir / "history.json").write_text(
                json.dumps(history, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    if rank == 0:
        print(json.dumps(history, ensure_ascii=False, indent=2))
    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
