"""Train the native unordered-Pair model from strictly audited self-play."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from experiments.pair_native.model import load_checkpoint, parameter_count
from experiments.pair_native.pair_replay import (
    PASS_MOVE,
    audit_shard,
    load_and_audit_shard,
    load_shard,
    unpack_points,
)


BOARD_SIZE = 19
BOARD_POINTS = BOARD_SIZE * BOARD_SIZE
LENGTH_BUCKET_NAMES = ("short", "medium", "long", "very_long")
WINNER_BUCKET_NAMES = ("black", "white", "draw")


def parse_cpu_affinity(specification: str) -> list[int]:
    cpus = set()
    for item in specification.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = (int(value) for value in item.split("-", 1))
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(item))
    return sorted(cpus)


def validation_game(game_id: str, fraction: float) -> bool:
    digest = hashlib.blake2b(game_id.encode(), digest_size=8).digest()
    value = int.from_bytes(digest, "little") / 2**64
    return value < fraction


def game_length_bucket(stones: int) -> int:
    if stones <= 40:
        return 0
    if stones <= 80:
        return 1
    if stones <= 140:
        return 2
    return 3


def parse_distribution(specification: str, expected: int, name: str) -> list[float]:
    values = [float(value) for value in specification.split(",") if value.strip()]
    if len(values) != expected or any(value < 0.0 for value in values):
        raise ValueError(f"{name} must contain {expected} non-negative values")
    total = sum(values)
    if total <= 0.0:
        raise ValueError(f"{name} must have positive total mass")
    return [value / total for value in values]


def transform_index(index: int, transform: int) -> int:
    if index == PASS_MOVE:
        return PASS_MOVE
    row, column = divmod(int(index), BOARD_SIZE)
    if transform >= 4:
        column = BOARD_SIZE - 1 - column
    for _ in range(transform % 4):
        row, column = BOARD_SIZE - 1 - column, row
    return row * BOARD_SIZE + column


def transform_board(board: np.ndarray, transform: int) -> np.ndarray:
    result = np.fliplr(board) if transform >= 4 else board
    if transform % 4:
        result = np.rot90(result, transform % 4)
    return np.ascontiguousarray(result)


class PairTurnDataset(Dataset):
    def __init__(
        self,
        rows,
        winners,
        training: bool,
        value_target: str = "outcome",
        value_discount: float = 1.0,
    ):
        self.rows = rows
        self.winners = winners
        self.training = bool(training)
        self.value_target = value_target
        self.value_discount = float(value_discount)
        self.last_turn = {}
        for row in rows:
            game_id = row["game_id"]
            self.last_turn[game_id] = max(
                self.last_turn.get(game_id, -1),
                int(row["turn_index"]),
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        black = unpack_points(row["board_black"])
        white = unpack_points(row["board_white"])
        board = black.astype(np.int8) - white.astype(np.int8)
        transform = random.randrange(8) if self.training else 0
        board = transform_board(board, transform)
        player = int(row["player"])
        own = board == player
        opponent = board == -player
        features = np.stack(
            (
                own,
                opponent,
                board == 0,
                np.full_like(board, player == 1, dtype=bool),
                np.zeros_like(board, dtype=bool),
            ),
            axis=0,
        ).astype(np.float32)
        policy_a = [transform_index(value, transform) for value in row["policy_a"]]
        policy_b = [transform_index(value, transform) for value in row["policy_b"]]
        for action_index, second in enumerate(policy_b):
            if second != PASS_MOVE and policy_a[action_index] > second:
                policy_a[action_index], policy_b[action_index] = (
                    second,
                    policy_a[action_index],
                )
        chosen_a = transform_index(row["chosen_a"], transform)
        chosen_b = transform_index(row["chosen_b"], transform)
        if chosen_b != PASS_MOVE and chosen_a > chosen_b:
            chosen_a, chosen_b = chosen_b, chosen_a
        remaining_turns = self.last_turn[row["game_id"]] - int(row["turn_index"])
        if self.value_target == "search":
            value = float(row["root_value"])
        else:
            value = float(self.winners[row["game_id"]] * player)
            value *= self.value_discount**remaining_turns
        return {
            "features": torch.from_numpy(features),
            "policy_a": torch.tensor(policy_a, dtype=torch.long),
            "policy_b": torch.tensor(policy_b, dtype=torch.long),
            "policy_probability": torch.tensor(
                row["policy_probability"],
                dtype=torch.float32,
            ),
            "chosen_a": chosen_a,
            "chosen_b": chosen_b,
            "value": value,
            "remaining_turns": remaining_turns,
            "stones_before": int(row["stones_before"]),
        }


def collate_turns(samples):
    maximum = max(len(sample["policy_a"]) for sample in samples)
    batch = len(samples)
    policy_a = torch.full((batch, maximum), -1, dtype=torch.long)
    policy_b = torch.full((batch, maximum), -2, dtype=torch.long)
    probability = torch.zeros((batch, maximum), dtype=torch.float32)
    for index, sample in enumerate(samples):
        length = len(sample["policy_a"])
        policy_a[index, :length] = sample["policy_a"]
        policy_b[index, :length] = sample["policy_b"]
        probability[index, :length] = sample["policy_probability"]
    return {
        "features": torch.stack([sample["features"] for sample in samples]),
        "policy_a": policy_a,
        "policy_b": policy_b,
        "policy_probability": probability,
        "chosen_a": torch.tensor([sample["chosen_a"] for sample in samples]),
        "chosen_b": torch.tensor([sample["chosen_b"] for sample in samples]),
        "value": torch.tensor([sample["value"] for sample in samples]),
        "remaining_turns": torch.tensor([sample["remaining_turns"] for sample in samples]),
        "stones_before": torch.tensor([sample["stones_before"] for sample in samples]),
    }


def load_data(
    data_dirs: Path | list[Path],
    validation_fraction: float,
    single_pass_audit_load: bool = False,
    replay_audit_workers: int = 1,
):
    games = {}
    game_lengths = {}
    turns = []
    audits = []
    if isinstance(data_dirs, Path):
        data_dirs = [data_dirs]
    manifest_groups = []
    for data_dir in data_dirs:
        manifests = sorted(data_dir.glob("*.manifest.json"))
        if not manifests:
            raise ValueError(f"no Pair replay manifests under {data_dir}")
        manifest_groups.append((data_dir, manifests))
    if replay_audit_workers < 1:
        raise ValueError("replay_audit_workers must be positive")
    parallel_audits = {}
    if replay_audit_workers > 1:
        all_manifests = [
            manifest
            for _, manifests in manifest_groups
            for manifest in manifests
        ]
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(replay_audit_workers, len(all_manifests)),
            mp_context=mp.get_context("spawn"),
        ) as executor:
            reports = list(executor.map(audit_shard, all_manifests))
        parallel_audits = dict(zip(all_manifests, reports))
    for data_dir, manifests in manifest_groups:
        directory_provenance = {
            "generations": set(),
            "model_sha256": set(),
            "engine_sha256": set(),
            "search_config_sha256": set(),
        }
        for manifest in manifests:
            if replay_audit_workers > 1:
                audit = parallel_audits[manifest]
                _, shard_games, shard_turns = load_shard(manifest)
            elif single_pass_audit_load:
                audit, _, shard_games, shard_turns = load_and_audit_shard(manifest)
            else:
                audit = audit_shard(manifest)
                _, shard_games, shard_turns = load_shard(manifest)
            if not audit["valid"]:
                raise ValueError(f"invalid shard {manifest}: {audit['errors']}")
            audits.append(audit)
            for name, values in audit["provenance"].items():
                directory_provenance[name].update(values)
            for game in shard_games:
                if game["game_id"] in games:
                    raise ValueError(f"duplicate game id {game['game_id']}")
                games[game["game_id"]] = int(game["winner"])
                game_lengths[game["game_id"]] = len(game["moves"])
            turns.extend(shard_turns)
        mixed = {
            name: sorted(values)
            for name, values in directory_provenance.items()
            if len(values) != 1
        }
        if mixed:
            raise ValueError(f"mixed replay provenance under {data_dir}: {mixed}")
    if len(games) < 2:
        raise ValueError("training requires at least two audited games")
    validation_ids = {
        game_id for game_id in games if validation_game(game_id, validation_fraction)
    }
    if not validation_ids or len(validation_ids) == len(games):
        ordered = sorted(games)
        count = max(1, min(len(ordered) - 1, round(len(ordered) * validation_fraction)))
        validation_ids = set(ordered[:count])
    training_rows = [row for row in turns if row["game_id"] not in validation_ids]
    validation_rows = [row for row in turns if row["game_id"] in validation_ids]
    if not training_rows or not validation_rows:
        raise ValueError("game-level split produced an empty train or validation set")
    return games, game_lengths, training_rows, validation_rows, audits


def phase_weights(rows):
    bins = [0 if row["stones_before"] < 32 else 1 if row["stones_before"] <= 64 else 2 for row in rows]
    counts = Counter(bins)
    weights = torch.tensor([1.0 / counts[value] for value in bins], dtype=torch.double)
    return weights / weights.mean(), counts


def _supported_targets(categories: np.ndarray, targets: list[float]) -> np.ndarray:
    result = np.asarray(targets, dtype=np.float64).copy()
    present = np.bincount(categories, minlength=len(result)) > 0
    result[~present] = 0.0
    if result.sum() <= 0.0:
        result[present] = 1.0
    return result / result.sum()


def _weighted_marginal(weights: np.ndarray, categories: np.ndarray, size: int) -> list[float]:
    masses = np.bincount(categories, weights=weights, minlength=size).astype(np.float64)
    masses /= max(float(masses.sum()), 1.0e-12)
    return masses.tolist()


def joint_replay_weights(
    rows,
    winners: dict[str, int],
    game_lengths: dict[str, int],
    length_targets: list[float],
    winner_targets: list[float],
    maximum_weight_ratio: float = 4.0,
    iterations: int = 30,
):
    """Rake replay turns toward phase, game-length, and outcome targets.

    The calculation is repeated for every learner launch, so it adapts to the
    current replay window. Structural-zero categories are removed and targets
    are renormalized instead of manufacturing nonexistent examples.
    """
    if not rows:
        raise ValueError("cannot balance an empty replay")
    if maximum_weight_ratio < 1.0 or iterations < 1:
        raise ValueError("invalid replay balancing limit or iteration count")
    phases = np.asarray([
        0 if row["stones_before"] < 32 else 1 if row["stones_before"] <= 64 else 2
        for row in rows
    ], dtype=np.int8)
    lengths = np.asarray([
        game_length_bucket(game_lengths[row["game_id"]])
        for row in rows
    ], dtype=np.int8)
    winner_index = {1: 0, -1: 1, 0: 2}
    outcomes = np.asarray([
        winner_index[winners[row["game_id"]]]
        for row in rows
    ], dtype=np.int8)
    dimensions = (
        (phases, _supported_targets(phases, [1.0 / 3.0] * 3)),
        (lengths, _supported_targets(lengths, length_targets)),
        (outcomes, _supported_targets(outcomes, winner_targets)),
    )
    weights = np.ones(len(rows), dtype=np.float64)
    # ``maximum_weight_ratio`` is the literal largest/smallest row-weight
    # ratio, not a symmetric multiplier around one (which would square it).
    upper = math.sqrt(maximum_weight_ratio)
    lower = 1.0 / upper
    for _ in range(iterations):
        for categories, targets in dimensions:
            total = float(weights.sum())
            for category, target in enumerate(targets):
                mask = categories == category
                current = float(weights[mask].sum()) / max(total, 1.0e-12)
                if current > 0.0 and target > 0.0:
                    weights[mask] *= target / current
            weights /= weights.mean()
            np.clip(weights, lower, upper, out=weights)
    weights /= weights.mean()
    joint_counts = Counter(
        (LENGTH_BUCKET_NAMES[length], WINNER_BUCKET_NAMES[outcome])
        for length, outcome in zip(lengths, outcomes)
    )
    stats = {
        "mode": "joint",
        "phase_source": _weighted_marginal(np.ones_like(weights), phases, 3),
        "phase_sampled": _weighted_marginal(weights, phases, 3),
        "phase_target": [1.0 / 3.0] * 3,
        "length_source": _weighted_marginal(np.ones_like(weights), lengths, 4),
        "length_sampled": _weighted_marginal(weights, lengths, 4),
        "length_target": list(length_targets),
        "winner_source": _weighted_marginal(np.ones_like(weights), outcomes, 3),
        "winner_sampled": _weighted_marginal(weights, outcomes, 3),
        "winner_target": list(winner_targets),
        "joint_counts": {
            f"{length}:{winner}": joint_counts[(length, winner)]
            for length in LENGTH_BUCKET_NAMES
            for winner in WINNER_BUCKET_NAMES
        },
        "minimum_weight": float(weights.min()),
        "maximum_weight": float(weights.max()),
        "maximum_to_minimum_weight_ratio": float(weights.max() / weights.min()),
        "effective_sample_size": float(weights.sum() ** 2 / np.square(weights).sum()),
        "rows": len(rows),
    }
    return torch.from_numpy(weights).to(dtype=torch.double), stats


def compute_loss(model, batch, device, value_weight, decisive_value_weight):
    features = batch["features"].to(
        device,
        non_blocking=True,
        memory_format=torch.channels_last,
    )
    target_a = batch["policy_a"].to(device, non_blocking=True)
    target_b = batch["policy_b"].to(device, non_blocking=True)
    target_probability = batch["policy_probability"].to(device, non_blocking=True)
    value_target = batch["value"].to(device, non_blocking=True)
    chosen_a = batch["chosen_a"].to(device, non_blocking=True)
    chosen_b = batch["chosen_b"].to(device, non_blocking=True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        unary, pair_embeddings, value, occupied = model(features)
        pair_scores = model.pair_scores(unary, pair_embeddings, occupied)
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
    policy_per_sample = -(target_probability * log_terms * valid).sum(dim=1)
    policy_loss = policy_per_sample.mean()
    value_importance = 1.0 + (decisive_value_weight - 1.0) * value_target.abs()
    value_loss = (
        value_importance * (value.float() - value_target).square()
    ).sum() / value_importance.sum().clamp_min(1.0)
    loss = policy_loss + value_weight * value_loss

    pair_prediction = pair_scores.argmax(dim=1)
    predicted_a = pair_prediction.div(BOARD_POINTS, rounding_mode="floor")
    predicted_b = pair_prediction % BOARD_POINTS
    unary_prediction = unary.argmax(dim=1)
    predicted_a = torch.where(chosen_b.eq(PASS_MOVE), unary_prediction, predicted_a)
    predicted_b = torch.where(
        chosen_b.eq(PASS_MOVE),
        torch.full_like(predicted_b, PASS_MOVE),
        predicted_b,
    )
    chosen_top1 = predicted_a.eq(chosen_a) & predicted_b.eq(chosen_b)
    target_top1_index = target_probability.argmax(dim=1, keepdim=True)
    target_top1_a = target_a.gather(1, target_top1_index).squeeze(1)
    target_top1_b = target_b.gather(1, target_top1_index).squeeze(1)
    target_top1 = predicted_a.eq(target_top1_a) & predicted_b.eq(target_top1_b)
    decisive = value_target.abs().gt(0.25)
    value_sign = torch.sign(value.float()).eq(torch.sign(value_target)) & decisive
    return loss, {
        "loss": loss.detach(),
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "policy_top1": target_top1.float().mean().detach(),
        "chosen_action_top1": chosen_top1.float().mean().detach(),
        "value_mae": F.l1_loss(value.float(), value_target).detach(),
        "value_decisive_fraction": decisive.float().mean().detach(),
        "value_decisive_correct_fraction": value_sign.float().mean().detach(),
    }


def run_epoch(
    model,
    loader,
    device,
    value_weight,
    decisive_value_weight,
    optimizer=None,
):
    training = optimizer is not None
    model.train(training)
    totals = Counter()
    samples = 0
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()
    for batch in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss, metrics = compute_loss(
                model,
                batch,
                device,
                value_weight,
                decisive_value_weight,
            )
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        batch_size = batch["features"].shape[0]
        samples += batch_size
        for name, value in metrics.items():
            totals[name] += float(value) * batch_size
    finished.record()
    torch.cuda.synchronize(device)
    seconds = started.elapsed_time(finished) / 1000.0
    result = {
        **{name: value / max(samples, 1) for name, value in totals.items()},
        "samples": samples,
        "seconds": seconds,
        "samples_per_second": samples / max(seconds, 1.0e-9),
    }
    result["value_decisive_sign_accuracy"] = (
        result.pop("value_decisive_correct_fraction")
        / max(result["value_decisive_fraction"], 1.0e-12)
    )
    return result


def atomic_save(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", action="append", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--decisive-value-weight", type=float, default=1.0)
    parser.add_argument("--value-target", choices=("outcome", "search"), default="outcome")
    parser.add_argument("--value-discount", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help=(
            "maximum weighted training draws per epoch; 0 uses every training "
            "row once in expectation"
        ),
    )
    parser.add_argument(
        "--sampling-mode",
        choices=("phase", "joint"),
        default="phase",
        help="historical phase-only sampling or dynamic phase/length/outcome balancing",
    )
    parser.add_argument("--length-targets", default="0.20,0.35,0.30,0.15")
    parser.add_argument("--winner-targets", default="0.475,0.475,0.05")
    parser.add_argument("--sampler-max-weight-ratio", type=float, default=4.0)
    parser.add_argument("--sampler-raking-iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu-affinity", default="")
    parser.add_argument(
        "--single-pass-audit-load",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("NEBULA_TRAIN_SINGLE_PASS_AUDIT_LOAD", "0") == "1",
        help="reuse decoded rows from the strict audit instead of loading each shard twice",
    )
    parser.add_argument(
        "--replay-audit-workers",
        type=int,
        default=1,
        help="strict audit processes before the parent loads each verified shard",
    )
    arguments = parser.parse_args()
    if not 0.0 < arguments.value_discount <= 1.0:
        raise ValueError("value_discount must be in (0, 1]")
    if arguments.decisive_value_weight < 1.0:
        raise ValueError("decisive_value_weight must be at least 1")
    length_targets = parse_distribution(arguments.length_targets, 4, "length_targets")
    winner_targets = parse_distribution(arguments.winner_targets, 3, "winner_targets")
    if arguments.sampler_max_weight_ratio < 1.0:
        raise ValueError("sampler_max_weight_ratio must be at least 1")
    if arguments.sampler_raking_iterations < 1:
        raise ValueError("sampler_raking_iterations must be positive")
    if arguments.replay_audit_workers < 1:
        raise ValueError("replay_audit_workers must be positive")
    if arguments.max_train_samples < 0:
        raise ValueError("max_train_samples must be non-negative")
    affinity = parse_cpu_affinity(arguments.cpu_affinity)
    if affinity:
        os.sched_setaffinity(0, affinity)

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    device = torch.device("cuda:0")
    model, checkpoint = load_checkpoint(arguments.init_checkpoint)
    model = model.to(device, memory_format=torch.channels_last)
    games, game_lengths, training_rows, validation_rows, audits = load_data(
        [Path(value).expanduser().resolve() for value in arguments.data_dir],
        arguments.validation_fraction,
        single_pass_audit_load=arguments.single_pass_audit_load,
        replay_audit_workers=arguments.replay_audit_workers,
    )
    training_dataset = PairTurnDataset(
        training_rows,
        games,
        training=True,
        value_target=arguments.value_target,
        value_discount=arguments.value_discount,
    )
    validation_dataset = PairTurnDataset(
        validation_rows,
        games,
        training=False,
        value_target=arguments.value_target,
        value_discount=arguments.value_discount,
    )
    phase_counts = Counter(
        0 if row["stones_before"] < 32 else 1 if row["stones_before"] <= 64 else 2
        for row in training_rows
    )
    if arguments.sampling_mode == "joint":
        weights, sampling_stats = joint_replay_weights(
            training_rows,
            games,
            game_lengths,
            length_targets,
            winner_targets,
            maximum_weight_ratio=arguments.sampler_max_weight_ratio,
            iterations=arguments.sampler_raking_iterations,
        )
    else:
        weights, phase_counts = phase_weights(training_rows)
        sampling_stats = {
            "mode": "phase",
            "phase_counts": dict(phase_counts),
        }
    train_samples_per_epoch = (
        min(len(training_dataset), arguments.max_train_samples)
        if arguments.max_train_samples
        else len(training_dataset)
    )
    sampler = WeightedRandomSampler(
        weights,
        num_samples=train_samples_per_epoch,
        replacement=True,
        generator=torch.Generator().manual_seed(arguments.seed),
    )
    loader_options = {
        "batch_size": arguments.batch_size,
        "num_workers": arguments.workers,
        "pin_memory": True,
        "collate_fn": collate_turns,
    }
    if arguments.workers:
        loader_options.update({"persistent_workers": True, "prefetch_factor": 2})
    training_loader = DataLoader(training_dataset, sampler=sampler, **loader_options)
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_options,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        fused=True,
    )
    output_dir = Path(arguments.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history = {
        "config": vars(arguments),
        "parameters": parameter_count(model),
        "games": len(games),
        "training_turns": len(training_rows),
        "train_samples_per_epoch": train_samples_per_epoch,
        "validation_turns": len(validation_rows),
        "training_phase_counts": dict(phase_counts),
        "sampling": sampling_stats,
        "audited_shards": len(audits),
        "cpu_affinity": affinity,
        "epochs": [],
    }
    with torch.no_grad():
        initial_validation = run_epoch(
            model,
            validation_loader,
            device,
            arguments.value_weight,
            arguments.decisive_value_weight,
        )
    history["initial_validation"] = initial_validation
    best_loss = math.inf
    for epoch in range(arguments.epochs):
        training_metrics = run_epoch(
            model,
            training_loader,
            device,
            arguments.value_weight,
            arguments.decisive_value_weight,
            optimizer,
        )
        with torch.no_grad():
            validation_metrics = run_epoch(
                model,
                validation_loader,
                device,
                arguments.value_weight,
                arguments.decisive_value_weight,
            )
        record = {
            "epoch": epoch + 1,
            "training": training_metrics,
            "validation": validation_metrics,
        }
        history["epochs"].append(record)
        payload = {
            "schema_version": 1,
            "model_config": checkpoint["model_config"],
            "seed": arguments.seed,
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": record,
        }
        atomic_save(payload, output_dir / "last.pth")
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            atomic_save(payload, output_dir / "best.pth")
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(record), flush=True)
    print(json.dumps(history, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
