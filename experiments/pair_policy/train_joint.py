"""联合微调主干与方向条件头，让一次完整推理原生覆盖整回合两颗子。"""

import argparse
import csv
import json
import math
import os
import random
import sys

import numpy as np
import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RL_DIR = os.path.join(ROOT, "reinforcement_learning")
sys.path.insert(0, RL_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from core.model import C6TransNet
from model import PairPolicyHead, PairValueHead
from probe import make_features, parse_move, parse_policy, player_at


def load_joint_samples(paths, limit, seed):
    """提取回合起点、第一/第二子搜索策略和当前玩家最终胜负。"""

    samples = []
    for path in paths:
        with open(path, newline="") as source:
            for row in csv.DictReader(source):
                winner = {"black": 1, "white": -1, "draw": 0}[row["winner"]]
                moves = [parse_move(value) for value in row["moves"].split(",") if value]
                policies = row["policies"].split("|")
                board = np.zeros(361, dtype=np.int8)
                for move_index, move in enumerate(moves):
                    if (
                        move_index % 2 == 1
                        and move_index + 1 < len(moves)
                        and move_index + 1 < len(policies)
                    ):
                        first_policy = parse_policy(policies[move_index])
                        second_policy = parse_policy(policies[move_index + 1])
                        if first_policy.sum() > 0 and second_policy.sum() > 0:
                            player = player_at(move_index)
                            samples.append(
                                (
                                    board.copy(),
                                    player,
                                    move,
                                    first_policy,
                                    second_policy,
                                    winner * player,
                                    winner,
                                )
                            )
                    if board[move] != 0:
                        break
                    board[move] = player_at(move_index)

    random.Random(seed).shuffle(samples)
    if limit > 0:
        samples = samples[:limit]
    if not samples:
        raise RuntimeError("没有提取到联合成对训练样本")
    return samples


def sample_batch(samples, indices):
    selected = [samples[int(index)] for index in indices]
    return (
        torch.from_numpy(np.stack([item[0] for item in selected])),
        torch.tensor([item[1] for item in selected], dtype=torch.int64),
        torch.tensor([item[2] for item in selected], dtype=torch.int64),
        torch.from_numpy(np.stack([item[3] for item in selected])),
        torch.from_numpy(np.stack([item[4] for item in selected])),
        torch.tensor([item[5] for item in selected], dtype=torch.float32),
        torch.tensor([item[6] for item in selected], dtype=torch.int64),
    )


def atomic_json_dump(payload, path):
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def normalized_outcome_weights(game_winners, white_win_weight):
    weights = torch.where(
        game_winners.eq(-1),
        torch.full_like(game_winners, float(white_win_weight), dtype=torch.float32),
        torch.ones_like(game_winners, dtype=torch.float32),
    )
    return weights / weights.mean().clamp_min(1e-6)


def transform_spatial(values, rotations, reflect):
    values = torch.rot90(values, rotations, dims=(-2, -1))
    return torch.flip(values, dims=(-1,)) if reflect else values


def augment_batch(boards, first_moves, first_policy, second_policy, rng):
    """整批使用同一个D4变换，保持连续内存并降低数据准备开销。"""

    rotations = int(rng.integers(0, 4))
    reflect = bool(rng.integers(0, 2))
    boards_2d = transform_spatial(boards.view(-1, 19, 19), rotations, reflect)
    first_policy_2d = transform_spatial(
        first_policy.view(-1, 19, 19),
        rotations,
        reflect,
    )
    second_policy_2d = transform_spatial(
        second_policy.view(-1, 19, 19),
        rotations,
        reflect,
    )
    first_mask = torch.zeros(
        (first_moves.shape[0], 361),
        dtype=first_policy.dtype,
        device=first_policy.device,
    )
    first_mask.scatter_(1, first_moves.unsqueeze(1), 1.0)
    first_mask = transform_spatial(
        first_mask.view(-1, 19, 19),
        rotations,
        reflect,
    )
    transformed_first = first_mask.reshape(-1, 361).argmax(dim=1)
    return (
        boards_2d.reshape(-1, 361).contiguous(),
        transformed_first,
        first_policy_2d.reshape(-1, 361).contiguous(),
        second_policy_2d.reshape(-1, 361).contiguous(),
    )


def forward_joint(model, pair_policy, pair_value, inputs, first_moves):
    features_2d = model.relu(model.bn_in(model.conv_in(inputs)))
    features_2d = model.res_stack(features_2d)
    features = model.forward_transformer(features_2d.flatten(2).transpose(1, 2))
    first_logits = model.head_move1(features).squeeze(-1)
    value = model.value_head(features.mean(dim=1)).flatten()
    second_logits = pair_policy(features, first_moves, first_logits)
    conditional_value = pair_value(features, first_moves, value)
    return first_logits, second_logits, value, conditional_value


def mask_logits(logits, boards, first_moves=None):
    occupied = boards.to(device=logits.device).ne(0)
    if first_moves is not None:
        occupied.scatter_(1, first_moves.unsqueeze(1), True)
    return logits.masked_fill(occupied, -1e9)


@torch.no_grad()
def evaluate(model, pair_policy, pair_value, samples, batch_size, device):
    model.eval()
    pair_policy.eval()
    pair_value.eval()
    totals = {
        "first_ce": 0.0,
        "second_ce": 0.0,
        "first_top1": 0.0,
        "second_top1": 0.0,
        "second_recall_top20": 0.0,
        "second_mass_top20": 0.0,
        "value_mae": 0.0,
        "conditional_value_mae": 0.0,
    }
    count = 0
    for start in range(0, len(samples), batch_size):
        indices = np.arange(start, min(start + batch_size, len(samples)))
        boards, players, first_moves, first_target, second_target, value_target, _ = sample_batch(
            samples,
            indices,
        )
        inputs = make_features(boards, players, device)
        first_moves = first_moves.to(device)
        first_target = first_target.to(device)
        second_target = second_target.to(device)
        value_target = value_target.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            first_logits, second_logits, value, conditional_value = forward_joint(
                model,
                pair_policy,
                pair_value,
                inputs,
                first_moves,
            )
        first_logits = mask_logits(first_logits.float(), boards)
        second_logits = mask_logits(second_logits.float(), boards, first_moves)
        totals["first_ce"] += float(
            (-(first_target * first_logits.log_softmax(dim=1)).sum(dim=1)).sum()
        )
        totals["second_ce"] += float(
            (-(second_target * second_logits.log_softmax(dim=1)).sum(dim=1)).sum()
        )
        first_moves_target = first_target.argmax(dim=1)
        second_moves_target = second_target.argmax(dim=1)
        totals["first_top1"] += float(
            (first_logits.argmax(dim=1) == first_moves_target).sum()
        )
        totals["second_top1"] += float(
            (second_logits.argmax(dim=1) == second_moves_target).sum()
        )
        second_top20 = second_logits.topk(20, dim=1).indices
        totals["second_recall_top20"] += float(
            (second_top20 == second_moves_target.unsqueeze(1)).any(dim=1).sum()
        )
        totals["second_mass_top20"] += float(
            second_target.gather(1, second_top20).sum()
        )
        totals["value_mae"] += float((value.float() - value_target).abs().sum())
        totals["conditional_value_mae"] += float(
            (conditional_value.float() - value_target).abs().sum()
        )
        count += len(indices)
    model.train()
    pair_policy.train()
    pair_value.train()
    return {name: value / count for name, value in totals.items()}


def save_candidate(output_dir, model, pair_policy, pair_value, args, metrics):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "metrics": metrics},
        os.path.join(output_dir, "main.pth"),
    )
    torch.save(
        {
            "pair_head": pair_policy.state_dict(),
            "pair_value_head": pair_value.state_dict(),
            "args": {
                "rank": args.rank,
                "tied_factors": False,
                "projection_hidden": args.projection_hidden,
                "relative_gating": args.relative_gating,
                "joint_training": True,
                "white_win_weight": args.white_win_weight,
            },
            "metrics": metrics,
        },
        os.path.join(output_dir, "pair_heads.pt"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(RL_DIR, "checkpoints", "best.pth"))
    parser.add_argument("--pair-heads", required=True)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--validation", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-train", type=int, default=100000)
    parser.add_argument("--max-validation", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--trunk-learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=3e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--projection-hidden", type=int, default=128)
    parser.add_argument("--relative-gating", action="store_true")
    parser.add_argument("--white-win-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")

    train_samples = load_joint_samples(args.train, args.max_train, args.seed)
    validation_samples = load_joint_samples(
        args.validation,
        args.max_validation,
        args.seed + 1,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model = C6TransNet(input_planes=17).to(device)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})

    pair_policy = PairPolicyHead(
        rank=args.rank,
        projection_hidden=args.projection_hidden,
        relative_gating=args.relative_gating,
    ).to(device)
    pair_value = PairValueHead().to(device)
    saved_heads = torch.load(args.pair_heads, map_location=device, weights_only=True)
    pair_policy.load_state_dict(saved_heads["pair_head"])
    pair_value.load_state_dict(saved_heads["pair_value_head"])

    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": args.trunk_learning_rate},
            {
                "params": [*pair_policy.parameters(), *pair_value.parameters()],
                "lr": args.head_learning_rate,
            },
        ],
        weight_decay=1e-4,
    )
    best_score = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)
    initial_metrics = evaluate(
        model,
        pair_policy,
        pair_value,
        validation_samples,
        args.batch_size,
        device,
    )
    history_path = os.path.join(args.output_dir, "metrics_history.json")
    history = {"initial": initial_metrics, "epochs": []}
    atomic_json_dump(history, history_path)
    print(json.dumps({
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "white_win_weight": args.white_win_weight,
        "initial": initial_metrics,
    }, ensure_ascii=False))

    for epoch in range(1, args.epochs + 1):
        permutation = rng.permutation(len(train_samples))
        running = []
        for start in range(0, len(train_samples), args.batch_size):
            indices = permutation[start:start + args.batch_size]
            boards, players, first_moves, first_target, second_target, value_target, game_winners = sample_batch(
                train_samples,
                indices,
            )
            boards, first_moves, first_target, second_target = augment_batch(
                boards,
                first_moves,
                first_target,
                second_target,
                rng,
            )
            inputs = make_features(boards, players, device)
            first_moves = first_moves.to(device)
            first_target = first_target.to(device)
            second_target = second_target.to(device)
            value_target = value_target.to(device)
            game_winners = game_winners.to(device)
            sample_weights = normalized_outcome_weights(
                game_winners,
                args.white_win_weight,
            )

            with torch.autocast("cuda", dtype=torch.bfloat16):
                first_logits, second_logits, value, conditional_value = forward_joint(
                    model,
                    pair_policy,
                    pair_value,
                    inputs,
                    first_moves,
                )
            first_logits = mask_logits(first_logits.float(), boards)
            second_logits = mask_logits(second_logits.float(), boards, first_moves)
            first_losses = -(
                first_target * first_logits.log_softmax(dim=1)
            ).sum(dim=1)
            second_losses = -(
                second_target * second_logits.log_softmax(dim=1)
            ).sum(dim=1)
            value_losses = torch.nn.functional.mse_loss(
                value.float(), value_target, reduction="none"
            )
            conditional_value_losses = torch.nn.functional.mse_loss(
                conditional_value.float(),
                value_target,
                reduction="none",
            )
            first_loss = (first_losses * sample_weights).mean()
            second_loss = (second_losses * sample_weights).mean()
            value_loss = (value_losses * sample_weights).mean()
            conditional_value_loss = (
                conditional_value_losses * sample_weights
            ).mean()
            loss = first_loss + second_loss + 0.5 * (
                value_loss + conditional_value_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [
                    *model.parameters(),
                    *pair_policy.parameters(),
                    *pair_value.parameters(),
                ],
                1.0,
            )
            optimizer.step()
            running.append(
                (
                    float(first_loss.detach()),
                    float(second_loss.detach()),
                    float(value_loss.detach()),
                    float(conditional_value_loss.detach()),
                )
            )

        metrics = evaluate(
            model,
            pair_policy,
            pair_value,
            validation_samples,
            args.batch_size,
            device,
        )
        train_means = np.asarray(running).mean(axis=0)
        metrics.update({
            "epoch": epoch,
            "train_samples": len(train_samples),
            "validation_samples": len(validation_samples),
            "batches_per_epoch": math.ceil(len(train_samples) / args.batch_size),
            "optimizer_steps": math.ceil(len(train_samples) / args.batch_size) * epoch,
            "samples_seen": len(train_samples) * epoch,
            "white_win_weight": args.white_win_weight,
            "train_white_win_fraction": sum(
                int(sample[6] == -1) for sample in train_samples
            ) / len(train_samples),
            "train_first_ce": float(train_means[0]),
            "train_second_ce": float(train_means[1]),
            "train_value_mse": float(train_means[2]),
            "train_conditional_value_mse": float(train_means[3]),
        })
        print(json.dumps(metrics, ensure_ascii=False))
        history["epochs"].append(metrics)
        atomic_json_dump(history, history_path)
        score = metrics["first_ce"] + metrics["second_ce"]
        if score < best_score:
            best_score = score
            save_candidate(
                args.output_dir,
                model,
                pair_policy,
                pair_value,
                args,
                metrics,
            )


if __name__ == "__main__":
    main()
