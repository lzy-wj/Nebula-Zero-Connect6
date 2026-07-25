"""为双落子联合训练构造按来源配比、可复现的逐代回放集。"""

import argparse
import csv
import json
import os
import random

from data_metrics import analyze_games


HEADER = ["moves", "winner", "policies", "bonuses"]


def read_games(paths):
    """读取并按整行内容去重，保留标准四列 CSV。"""

    games = []
    seen = set()
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8", errors="replace") as source:
            for row in csv.DictReader(source):
                game = tuple(row.get(name, "") for name in HEADER)
                if not game[0] or game in seen:
                    continue
                seen.add(game)
                games.append(game)
    return games


def sample_at_most(games, limit, rng):
    if limit <= 0 or len(games) <= limit:
        return list(games)
    return rng.sample(games, limit)


def stratified_validation_split(games, validation_ratio, rng):
    """Keep validation useful by balancing black/white outcomes when possible."""

    target = max(1, round(len(games) * validation_ratio))
    groups = {"black": [], "white": [], "draw": []}
    for game in games:
        groups.setdefault(game[1], []).append(game)
    for values in groups.values():
        rng.shuffle(values)

    validation = []
    per_color = target // 2
    for winner in ("black", "white"):
        take = min(per_color, len(groups[winner]))
        validation.extend(groups[winner][:take])
        groups[winner] = groups[winner][take:]

    remaining = [*groups["black"], *groups["white"], *groups["draw"]]
    rng.shuffle(remaining)
    validation.extend(remaining[: max(0, target - len(validation))])
    validation_set = set(validation)
    training = [game for game in games if game not in validation_set]
    rng.shuffle(validation)
    rng.shuffle(training)
    return training, validation


def atomic_write(path, games):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(HEADER)
        writer.writerows(games)
    os.replace(temporary, path)


def atomic_json_write(path, payload):
    """原子写入 buffer 清单，供 SwanLab 和断点恢复共同读取。"""

    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", nargs="+", required=True)
    parser.add_argument("--anchor", nargs="+", required=True)
    parser.add_argument("--legacy", nargs="*", default=[])
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--validation-output", required=True)
    parser.add_argument("--fixed-validation", default=None)
    parser.add_argument("--fixed-validation-games", type=int, default=0)
    parser.add_argument(
        "--stats-output",
        default=None,
        help="可选：保存本代 replay buffer 的来源、容量和配比",
    )
    parser.add_argument("--max-online-games", type=int, default=1200)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--legacy-ratio-to-anchor", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not 0.0 < args.validation_ratio < 1.0:
        raise ValueError("validation-ratio 必须位于 (0, 1)")

    rng = random.Random(args.seed)
    online_available = read_games(args.online)
    anchor_available = read_games(args.anchor)
    legacy_available = read_games(args.legacy)
    online = sample_at_most(
        online_available,
        args.max_online_games,
        rng,
    )
    anchors = list(anchor_available)
    legacy = list(legacy_available)
    if not online:
        raise RuntimeError("没有可用的优化版在线对局")
    if len(anchors) < 2:
        raise RuntimeError("精确锚点至少需要两局，才能拆分训练和验证")

    fixed_validation = False
    if args.fixed_validation:
        if os.path.exists(args.fixed_validation):
            validation = read_games([args.fixed_validation])
        else:
            pool = read_games([*args.anchor, *args.legacy])
            target = min(
                len(pool),
                max(2, args.fixed_validation_games),
            )
            _, validation = stratified_validation_split(
                pool,
                target / max(len(pool), 1),
                rng,
            )
            atomic_write(args.fixed_validation, validation)
        validation_set = set(validation)
        anchors = [game for game in anchors if game not in validation_set]
        legacy = [game for game in legacy if game not in validation_set]
        anchor_train = list(anchors)
        fixed_validation = True
    else:
        anchor_train, validation = stratified_validation_split(
            anchors,
            args.validation_ratio,
            rng,
        )
    legacy_count = round(len(anchor_train) * args.legacy_ratio_to_anchor)
    legacy_train = sample_at_most(legacy, legacy_count, rng)

    # 训练集以在线数据为主；精确锚点约束近似头，旧精确数据防止短代遗忘。
    train = [*online, *anchor_train, *legacy_train]
    rng.shuffle(train)
    atomic_write(args.train_output, train)
    atomic_write(args.validation_output, validation)

    train_quality = analyze_games(args.train_output)
    validation_quality = analyze_games(args.validation_output)

    train_count = len(train)
    stats = {
        "online_source_files": sum(os.path.exists(path) for path in args.online),
        "anchor_source_files": sum(os.path.exists(path) for path in args.anchor),
        "legacy_source_files": sum(os.path.exists(path) for path in args.legacy),
        "online_available_games": len(online_available),
        "online_games": len(online),
        "online_capacity_games": args.max_online_games,
        "anchor_available_games": len(anchor_available),
        "anchor_train_games": len(anchor_train),
        "legacy_available_games": len(legacy_available),
        "legacy_train_games": len(legacy_train),
        "train_games": train_count,
        "validation_games": len(validation),
        "online_fraction": len(online) / max(train_count, 1),
        "anchor_fraction": len(anchor_train) / max(train_count, 1),
        "legacy_fraction": len(legacy_train) / max(train_count, 1),
        "validation_ratio": args.validation_ratio,
        "fixed_validation": int(fixed_validation),
        "fixed_validation_games": len(validation) if fixed_validation else 0,
        "train_positions": train_quality["joint_samples"],
        "validation_positions": validation_quality["joint_samples"],
        "train_black_win_rate": train_quality["black_win_rate"],
        "train_white_win_rate": train_quality["white_win_rate"],
        "train_white_win_sample_fraction": train_quality[
            "white_win_sample_fraction"
        ],
        "validation_black_win_rate": validation_quality["black_win_rate"],
        "validation_white_win_rate": validation_quality["white_win_rate"],
        "validation_white_win_sample_fraction": validation_quality[
            "white_win_sample_fraction"
        ],
        "train_black_positive_target_rate": train_quality[
            "black_positive_target_rate"
        ],
        "train_white_positive_target_rate": train_quality[
            "white_positive_target_rate"
        ],
        "validation_black_positive_target_rate": validation_quality[
            "black_positive_target_rate"
        ],
        "validation_white_positive_target_rate": validation_quality[
            "white_positive_target_rate"
        ],
        "train_output": args.train_output,
        "validation_output": args.validation_output,
    }
    atomic_json_write(args.stats_output, stats)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
