"""冻结 V2 主干，比较旧 policy2 与低秩条件第二子头的可学习性。"""

import argparse
import copy
import csv
import json
import os
import random
import sys

import numpy as np
import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RL_DIR = os.path.join(ROOT, "reinforcement_learning")
sys.path.insert(0, RL_DIR)

from core.model import C6TransNet
from model import PairPolicyHead, PairValueHead


def parse_move(text):
    text = text.strip().lower()
    column = ord(text[0]) - ord("a")
    row = int(text[1:]) - 1
    if not (0 <= row < 19 and 0 <= column < 19):
        raise ValueError(f"非法坐标：{text}")
    return row * 19 + column


def parse_policy(text):
    policy = np.zeros(361, dtype=np.float32)
    for item in text.split(";"):
        if not item:
            continue
        index, probability = item.split(":", 1)
        policy[int(index)] = float(probability)
    total = float(policy.sum())
    return policy / total if total > 0 else policy


def player_at(move_index):
    if move_index == 0:
        return 1
    return -1 if ((move_index + 1) // 2) % 2 == 1 else 1


def load_pair_samples(paths, limit, seed):
    """提取每个双子回合开始状态及下一颗子的真实 MCTS 策略。"""

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
                        target = parse_policy(policies[move_index + 1])
                        if target.sum() > 0:
                            samples.append(
                                (
                                    board.copy(),
                                    player_at(move_index),
                                    move,
                                    target,
                                    moves[move_index + 1],
                                    winner * player_at(move_index),
                                )
                            )
                    if board[move] != 0:
                        break
                    board[move] = player_at(move_index)

    random.Random(seed).shuffle(samples)
    if limit > 0:
        samples = samples[:limit]
    if not samples:
        raise RuntimeError("没有提取到成对落子样本")
    return samples


def make_features(boards, players, device):
    boards = boards.to(device=device)
    players = players.to(device=device)
    features = torch.zeros(
        (boards.shape[0], 17, 19, 19),
        device=device,
        dtype=torch.float32,
    )
    board_2d = boards.view(-1, 19, 19)
    player_view = players.view(-1, 1, 1)
    features[:, 0] = board_2d.eq(player_view)
    features[:, 1] = board_2d.eq(-player_view)
    features[:, 16] = players.eq(1).float().view(-1, 1, 1)
    return features


def trunk_forward(model, inputs, amp_dtype):
    with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype):
        output = model.relu(model.bn_in(model.conv_in(inputs)))
        output = model.res_stack(output)
        features = model.forward_transformer(output.flatten(2).transpose(1, 2))
        policy1 = model.head_move1(features).squeeze(-1)
        value = model.value_head(features.mean(dim=1)).squeeze(1)
    return features.float(), policy1.float(), value.float()


def sample_batch(samples, indices):
    selected = [samples[int(index)] for index in indices]
    return (
        torch.from_numpy(np.stack([item[0] for item in selected])),
        torch.tensor([item[1] for item in selected], dtype=torch.int64),
        torch.tensor([item[2] for item in selected], dtype=torch.int64),
        torch.from_numpy(np.stack([item[3] for item in selected])),
        torch.tensor([item[4] for item in selected], dtype=torch.int64),
        torch.tensor([item[5] for item in selected], dtype=torch.float32),
    )


def mask_second_move(logits, boards, first_moves):
    occupied = boards.to(device=logits.device).ne(0)
    occupied.scatter_(1, first_moves.unsqueeze(1), True)
    return logits.masked_fill(occupied, -1e9)


def evaluate(
    model,
    legacy_embed,
    legacy_head,
    pair_head,
    pair_value_head,
    samples,
    batch_size,
    amp_dtype,
):
    totals = {
        "target_entropy": 0.0,
        "pair_after_policy_l1": 0.0,
        "pair_after_top1_agreement": 0.0,
        "pair_after_top20_overlap": 0.0,
        "parent_value_mae": 0.0,
        "pair_value_mae": 0.0,
        "after_value_mae": 0.0,
        "pair_after_value_mae": 0.0,
    }
    for name in ("legacy", "pair", "after"):
        totals[f"{name}_ce"] = 0.0
        totals[f"{name}_target_top1"] = 0.0
        totals[f"{name}_actual_top1"] = 0.0
        for top_k in (5, 20):
            totals[f"{name}_target_recall_top{top_k}"] = 0.0
            totals[f"{name}_actual_recall_top{top_k}"] = 0.0
            totals[f"{name}_target_mass_top{top_k}"] = 0.0
    count = 0
    for start in range(0, len(samples), batch_size):
        indices = np.arange(start, min(start + batch_size, len(samples)))
        boards, players, first_moves, targets, actual_moves, value_targets = sample_batch(
            samples,
            indices,
        )
        inputs = make_features(boards, players, next(model.parameters()).device)
        first_moves = first_moves.to(inputs.device)
        targets = targets.to(inputs.device)
        actual_moves = actual_moves.to(inputs.device)
        value_targets = value_targets.to(inputs.device)

        features, parent_policy, parent_value = trunk_forward(model, inputs, amp_dtype)
        with torch.no_grad():
            legacy_logits = legacy_head(features + legacy_embed(first_moves).unsqueeze(1))
            legacy_logits = legacy_logits.squeeze(-1)
            pair_logits = pair_head(features, first_moves, parent_policy)
            pair_value = pair_value_head(features, first_moves, parent_value)

            after_boards = boards.clone()
            after_boards.scatter_(1, first_moves.cpu().unsqueeze(1), players.unsqueeze(1).to(torch.int8))
            after_inputs = make_features(after_boards, players, inputs.device)
            _, after_logits, after_value = trunk_forward(model, after_inputs, amp_dtype)

            logits_by_name = {
                "legacy": mask_second_move(legacy_logits, boards, first_moves),
                "pair": mask_second_move(pair_logits, boards, first_moves),
                "after": after_logits.masked_fill(after_boards.to(inputs.device).ne(0), -1e9),
            }
            pair_probabilities = logits_by_name["pair"].softmax(dim=1)
            after_probabilities = logits_by_name["after"].softmax(dim=1)
            totals["pair_after_policy_l1"] += float(
                (pair_probabilities - after_probabilities).abs().sum()
            )
            totals["pair_after_top1_agreement"] += float(
                (
                    logits_by_name["pair"].argmax(dim=1)
                    == logits_by_name["after"].argmax(dim=1)
                ).sum()
            )
            pair_top20 = logits_by_name["pair"].topk(20, dim=1).indices
            after_top20 = logits_by_name["after"].topk(20, dim=1).indices
            totals["pair_after_top20_overlap"] += float(
                (
                    pair_top20.unsqueeze(2)
                    == after_top20.unsqueeze(1)
                ).any(dim=2).float().mean(dim=1).sum()
            )
            target_moves = targets.argmax(dim=1)
            totals["target_entropy"] += float(
                (-(targets * targets.clamp_min(1e-12).log()).sum(dim=1)).sum()
            )
            for name, logits in logits_by_name.items():
                totals[f"{name}_ce"] += float(
                    (-(targets * logits.log_softmax(dim=1)).sum(dim=1)).sum()
                )
                predictions = logits.argmax(dim=1)
                totals[f"{name}_target_top1"] += float((predictions == target_moves).sum())
                totals[f"{name}_actual_top1"] += float((predictions == actual_moves).sum())
                for top_k in (5, 20):
                    top_moves = logits.topk(top_k, dim=1).indices
                    totals[f"{name}_target_recall_top{top_k}"] += float(
                        (top_moves == target_moves.unsqueeze(1)).any(dim=1).sum()
                    )
                    totals[f"{name}_actual_recall_top{top_k}"] += float(
                        (top_moves == actual_moves.unsqueeze(1)).any(dim=1).sum()
                    )
                    totals[f"{name}_target_mass_top{top_k}"] += float(
                        targets.gather(1, top_moves).sum()
                    )
            totals["parent_value_mae"] += float(
                (parent_value - value_targets).abs().sum()
            )
            totals["pair_value_mae"] += float(
                (pair_value - value_targets).abs().sum()
            )
            totals["after_value_mae"] += float(
                (after_value - value_targets).abs().sum()
            )
            totals["pair_after_value_mae"] += float(
                (pair_value - after_value).abs().sum()
            )
        count += len(indices)
    return {name: value / count for name, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(RL_DIR, "checkpoints", "best.pth"))
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--validation", nargs="+", required=True)
    parser.add_argument("--max-train", type=int, default=12000)
    parser.add_argument("--max-validation", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument(
        "--tied-factors",
        action="store_true",
        help="两颗子共享低秩投影，验证无序棋子对假设",
    )
    parser.add_argument(
        "--projection-hidden",
        type=int,
        default=0,
        help="低秩投影前的非线性隐藏维度，0 表示保持线性投影",
    )
    parser.add_argument(
        "--relative-gating",
        action="store_true",
        help="按两颗子的相对方向和距离门控每个低秩通道",
    )
    parser.add_argument(
        "--distill-after-state",
        action="store_true",
        help="用落第一子后的完整网络 policy/value 作为教师目标",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default=None)
    parser.add_argument("--load-heads", default=None)
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="仅打印最后一轮指标，便于批量结构扫描",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16

    train_samples = load_pair_samples(args.train, args.max_train, args.seed)
    validation_samples = load_pair_samples(
        args.validation,
        args.max_validation,
        args.seed + 1,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model = C6TransNet(input_planes=17).to(device).eval()
    model.load_state_dict({key.removeprefix("module."): value for key, value in state_dict.items()})
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    legacy_embed = copy.deepcopy(model.move1_embed).to(device).train()
    legacy_head = copy.deepcopy(model.head_move2).to(device).train()
    pair_head = PairPolicyHead(
        rank=args.rank,
        tied_factors=args.tied_factors,
        projection_hidden=args.projection_hidden,
        relative_gating=args.relative_gating,
    ).to(device).train()
    # value 头使用独立随机流，避免 policy rank 改变前序随机数消耗后影响比较。
    torch.manual_seed(args.seed + 10_000)
    pair_value_head = PairValueHead().to(device).train()
    if args.load_heads:
        saved_heads = torch.load(args.load_heads, map_location=device, weights_only=True)
        legacy_embed.load_state_dict(saved_heads["legacy_embed"])
        legacy_head.load_state_dict(saved_heads["legacy_head"])
        pair_head.load_state_dict(saved_heads["pair_head"])
        pair_value_head.load_state_dict(saved_heads["pair_value_head"])
    optimizer = torch.optim.AdamW(
        [
            *legacy_embed.parameters(),
            *legacy_head.parameters(),
            *pair_head.parameters(),
            *pair_value_head.parameters(),
        ],
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    initial_metrics = {
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "epoch": 0,
        **evaluate(
            model,
            legacy_embed,
            legacy_head,
            pair_head,
            pair_value_head,
            validation_samples,
            args.batch_size,
            amp_dtype,
        ),
    }
    if not args.final_only or args.epochs == 0:
        print(json.dumps(initial_metrics, ensure_ascii=False))

    for epoch in range(1, args.epochs + 1):
        permutation = np.random.permutation(len(train_samples))
        losses = []
        for start in range(0, len(train_samples), args.batch_size):
            indices = permutation[start:start + args.batch_size]
            boards, players, first_moves, targets, _, value_targets = sample_batch(
                train_samples,
                indices,
            )
            inputs = make_features(boards, players, device)
            first_moves = first_moves.to(device)
            targets = targets.to(device)
            value_targets = value_targets.to(device)
            features, parent_policy, parent_value = trunk_forward(model, inputs, amp_dtype)

            legacy_logits = legacy_head(features + legacy_embed(first_moves).unsqueeze(1)).squeeze(-1)
            pair_logits = pair_head(features, first_moves, parent_policy)
            pair_value = pair_value_head(features, first_moves, parent_value)
            legacy_logits = mask_second_move(legacy_logits, boards, first_moves)
            pair_logits = mask_second_move(pair_logits, boards, first_moves)
            legacy_loss = -(targets * legacy_logits.log_softmax(dim=1)).sum(dim=1).mean()
            if args.distill_after_state:
                after_boards = boards.clone()
                after_boards.scatter_(
                    1,
                    first_moves.cpu().unsqueeze(1),
                    players.unsqueeze(1).to(torch.int8),
                )
                after_inputs = make_features(after_boards, players, device)
                _, teacher_logits, teacher_value = trunk_forward(
                    model,
                    after_inputs,
                    amp_dtype,
                )
                teacher_logits = teacher_logits.masked_fill(
                    after_boards.to(device).ne(0),
                    -1e9,
                )
                teacher_policy = teacher_logits.softmax(dim=1)
                pair_loss = -(
                    teacher_policy * pair_logits.log_softmax(dim=1)
                ).sum(dim=1).mean()
                value_loss = torch.nn.functional.mse_loss(
                    pair_value,
                    teacher_value,
                )
            else:
                pair_loss = -(
                    targets * pair_logits.log_softmax(dim=1)
                ).sum(dim=1).mean()
                value_loss = torch.nn.functional.mse_loss(
                    pair_value,
                    value_targets,
                )
            loss = legacy_loss + pair_loss + value_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(
                (
                    float(legacy_loss.detach()),
                    float(pair_loss.detach()),
                    float(value_loss.detach()),
                )
            )

        metrics = evaluate(
            model,
            legacy_embed,
            legacy_head,
            pair_head,
            pair_value_head,
            validation_samples,
            args.batch_size,
            amp_dtype,
        )
        metrics.update({
            "epoch": epoch,
            "train_legacy_ce": float(np.mean([item[0] for item in losses])),
            "train_pair_ce": float(np.mean([item[1] for item in losses])),
            "train_pair_value_mse": float(np.mean([item[2] for item in losses])),
        })
        if not args.final_only or epoch == args.epochs:
            print(json.dumps(metrics, ensure_ascii=False))

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        torch.save(
            {
                "legacy_embed": legacy_embed.state_dict(),
                "legacy_head": legacy_head.state_dict(),
                "pair_head": pair_head.state_dict(),
                "pair_value_head": pair_value_head.state_dict(),
                "args": vars(args),
            },
            args.output,
        )


if __name__ == "__main__":
    main()
