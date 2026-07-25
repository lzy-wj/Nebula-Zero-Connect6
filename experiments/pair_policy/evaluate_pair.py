"""对两套主网络+双落子头进行隔离门禁，并原子保存结果。"""

import argparse
import json
import os
import sys
import time


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RL_DIR = os.path.join(ROOT, "reinforcement_learning")
sys.path.insert(0, RL_DIR)

from pipeline.evaluate import play_match


def atomic_json_dump(payload, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def atomic_csv_dump(lines, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8", newline="") as output:
        output.write("moves,winner,policies,bonuses\n")
        for line in lines:
            output.write(line + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-engine", required=True)
    parser.add_argument("--candidate-heads", required=True)
    parser.add_argument("--incumbent-engine", required=True)
    parser.add_argument("--incumbent-heads", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-output", default=None)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--simulations-black", type=int, default=400)
    parser.add_argument("--simulations-white", type=int, default=1200)
    parser.add_argument("--opening-stones", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    started_at = time.perf_counter()
    stats, data_lines = play_match(
        args.candidate_engine,
        args.incumbent_engine,
        games=args.games,
        simulations=max(args.simulations_black, args.simulations_white),
        simulations_black=args.simulations_black,
        simulations_white=args.simulations_white,
        gpu_id=args.gpu,
        seed=args.seed,
        opening_stones=args.opening_stones,
        engine1_pair_heads=args.candidate_heads,
        engine2_pair_heads=args.incumbent_heads,
    )
    if not stats:
        raise RuntimeError("双落子门禁没有返回有效结果")

    total = stats["wins"] + stats["losses"] + stats["draws"]
    if total != args.games:
        raise RuntimeError(f"门禁仅完成 {total}/{args.games} 局")
    result = {
        "games": total,
        "wins": stats["wins"],
        "losses": stats["losses"],
        "draws": stats["draws"],
        "score_rate": (stats["wins"] + 0.5 * stats["draws"]) / total,
        "black_win_rate": (
            stats["black_wins"] / stats["black_games"]
            if stats["black_games"] else 0.0
        ),
        "white_win_rate": (
            stats["white_wins"] / stats["white_games"]
            if stats["white_games"] else 0.0
        ),
        "game_black_win_rate": stats.get("game_black_wins", 0) / total,
        "game_white_win_rate": stats.get("game_white_wins", 0) / total,
        "average_steps": sum(
            stats["win_steps"] + stats["loss_steps"] + stats["draw_steps"]
        ) / total,
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    atomic_json_dump(result, args.output)
    if args.data_output:
        atomic_csv_dump(data_lines, args.data_output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
