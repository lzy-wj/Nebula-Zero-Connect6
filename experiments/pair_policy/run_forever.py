"""双落子优化版常驻监督器：每代使用独立子进程和独立 SwanLab 实验。"""

import argparse
import glob
import json
import os
import subprocess
import sys

import swanlab


EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.environ.get("NEBULA_SWANLAB_PROJECT", "Nebular-zero-two")
RUN_ROOT = os.path.abspath(
    os.environ.get(
        "NEBULA_PAIR_RUN_DIR",
        os.path.join(EXPERIMENT_DIR, "runs", PROJECT),
    )
)
LOG_DIR = os.path.join(RUN_ROOT, "logs")
SUMMARY_DIR = os.path.join(LOG_DIR, "generation_summaries")
REPLAY_DIR = os.path.join(RUN_ROOT, "replay")


def atomic_text_write(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as output:
        output.write(str(value))
    os.replace(temporary, path)


def load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as source:
            return json.load(source)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def count_csv_games(path):
    """只统计 CSV 对局数；历史回填时不把整份策略数据载入内存。"""

    try:
        with open(path, encoding="utf-8", errors="replace") as source:
            return max(0, sum(1 for _ in source) - 1)
    except OSError:
        return 0


def buffer_stats_for_generation(generation, summary):
    """读取新版清单；旧代没有清单时从 replay 文件恢复基础容量指标。"""

    stats_path = os.path.join(REPLAY_DIR, f"gen_{generation:04d}_stats.json")
    stats = load_json(stats_path, None)
    if not isinstance(stats, dict):
        stats = summary.get("buffer") if isinstance(summary, dict) else None
    if isinstance(stats, dict):
        return stats

    train_path = os.path.join(REPLAY_DIR, f"gen_{generation:04d}_train.csv")
    validation_path = os.path.join(
        REPLAY_DIR,
        f"gen_{generation:04d}_validation.csv",
    )
    return {
        "generation": generation,
        "train_games": count_csv_games(train_path),
        "validation_games": count_csv_games(validation_path),
        "train_size_mb": (
            os.path.getsize(train_path) / (1024 * 1024)
            if os.path.exists(train_path)
            else 0.0
        ),
        "validation_size_mb": (
            os.path.getsize(validation_path) / (1024 * 1024)
            if os.path.exists(validation_path)
            else 0.0
        ),
    }


def scalar_payload(prefix, values):
    return {
        f"{prefix}/{key}": value
        for key, value in values.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def sync_buffer_history(active):
    """使用独立游标回填 buffer 曲线，不重复上传已有的代际主指标。"""

    if not active:
        return
    marker_path = os.path.join(
        LOG_DIR,
        "swanlab",
        "outer_last_buffer_generation.txt",
    )
    try:
        with open(marker_path, encoding="utf-8") as source:
            last_generation = int(source.read().strip())
    except (FileNotFoundError, ValueError, OSError):
        last_generation = -1

    for path in sorted(glob.glob(os.path.join(SUMMARY_DIR, "gen_*.json"))):
        summary = load_json(path, None)
        if not isinstance(summary, dict):
            continue
        generation = int(summary["generation"])
        if generation <= last_generation:
            continue
        stats = buffer_stats_for_generation(generation, summary)
        swanlab.log(scalar_payload("buffer", stats), step=generation)
        atomic_text_write(marker_path, generation)
        last_generation = generation


def init_outer_swanlab(mode):
    if mode == "disabled":
        return False
    run_id_path = os.path.join(LOG_DIR, "swanlab", "outer_loop.txt")
    existing_id = None
    try:
        with open(run_id_path, encoding="utf-8") as source:
            existing_id = source.read().strip() or None
    except FileNotFoundError:
        pass
    kwargs = {
        "project": PROJECT,
        "name": os.environ.get(
            "NEBULA_SWANLAB_LOOP_RUN_NAME",
            "AlphaZero_Training_Loop",
        ),
        "description": "原生双落子优化版跨代训练总览",
        "config": {
            "role": "pair_generation_loop",
            "selfplay_gpus": os.environ.get("NEBULA_SELFPLAY_GPUS", "6,7"),
            "training_gpu": os.environ.get("NEBULA_TRAINING_GPU", "6"),
            "train_precision": "bf16",
        },
        "mode": mode,
    }
    if existing_id:
        kwargs.update({"id": existing_id, "resume": "allow"})
    try:
        run = swanlab.init(**kwargs)
        run_id = getattr(run, "id", None)
        if run_id:
            atomic_text_write(run_id_path, run_id)
        return True
    except Exception as error:
        print(f"外层 SwanLab 初始化失败，将继续训练: {error}", flush=True)
        return False


def sync_summaries(active):
    if not active:
        return
    marker_path = os.path.join(LOG_DIR, "swanlab", "outer_last_generation.txt")
    try:
        with open(marker_path, encoding="utf-8") as source:
            last_generation = int(source.read().strip())
    except (FileNotFoundError, ValueError, OSError):
        last_generation = -1

    summaries = []
    for path in sorted(glob.glob(os.path.join(SUMMARY_DIR, "gen_*.json"))):
        value = load_json(path, None)
        if isinstance(value, dict):
            summaries.append(value)

    for index, summary in enumerate(summaries):
        generation = int(summary["generation"])
        if generation <= last_generation:
            continue
        gate = summary.get("eval", {})
        train = summary.get("train", {})
        buffer = summary.get("buffer", {})
        quality = summary.get("data_quality", {})
        balance = summary.get("balance", {})
        online = quality.get("online", {})
        anchor = quality.get("anchor", {})
        recent = summaries[max(0, index - 19): index + 1]
        accept_rate_20 = sum(bool(item.get("accepted")) for item in recent) / len(recent)
        accepted_generation = int(summary.get("accepted_generation", -1))
        payload = {
            "generation": generation,
            "gate/pass": int(bool(summary.get("accepted"))),
            "gate/score_rate": gate.get("score_rate", 0.0),
            "gate/score_margin": gate.get("score_rate", 0.5) - 0.5,
            "gate/candidate_black_win_rate": gate.get("black_win_rate", 0.0),
            "gate/candidate_white_win_rate": gate.get("white_win_rate", 0.0),
            "gate/game_black_win_rate": gate.get("game_black_win_rate", 0.0),
            "gate/accept_rate_20": accept_rate_20,
            "gate/accepted_generation": accepted_generation,
            "gate/incumbent_age": generation - accepted_generation,
            "samples/online_games": online.get("games", summary.get("online_games", 0)),
            "samples/anchor_games": anchor.get("games", summary.get("anchor_games", 0)),
            "samples/replay_games": buffer.get("train_games", 0),
            "samples/train_positions": buffer.get("train_positions", 0),
            "samples/validation_games": buffer.get("validation_games", 0),
            "samples/validation_positions": buffer.get("validation_positions", 0),
            "samples/online_fraction": buffer.get("online_fraction", 0.0),
            "samples/actual_samples_seen": buffer.get("actual_samples_seen", 0),
            "quality/online_black_win_rate": online.get("black_win_rate", 0.0),
            "quality/anchor_black_win_rate": anchor.get("black_win_rate", 0.0),
            "quality/train_white_win_rate": buffer.get("train_white_win_rate", 0.0),
            "quality/validation_white_win_rate": buffer.get("validation_white_win_rate", 0.0),
            "quality/black_positive_target_rate": buffer.get(
                "train_black_positive_target_rate", 0.0
            ),
            "quality/white_positive_target_rate": buffer.get(
                "train_white_positive_target_rate", 0.0
            ),
            "quality/mean_stones": online.get("mean_stones", 0.0),
            "quality/p90_stones": online.get("p90_stones", 0.0),
            "quality/policy_entropy": online.get("policy_entropy", 0.0),
            "quality/effective_actions": online.get("effective_actions", 0.0),
            "quality/opening_unique_ratio": online.get("opening_unique_ratio", 0.0),
            "quality/injected_opening_rate": online.get("injected_opening_rate", 0.0),
            "quality/invalid_game_rate": online.get("invalid_game_rate", 0.0),
            "model/validation_first_ce": train.get("first_ce", 0.0),
            "model/validation_second_ce": train.get("second_ce", 0.0),
            "model/value_mae": train.get("value_mae", 0.0),
            "model/conditional_value_mae": train.get("conditional_value_mae", 0.0),
            "health/selfplay_games_per_second": summary.get(
                "selfplay_games_per_second", 0
            ),
            "health/generation_minutes": summary.get("phase_times", {}).get(
                "total_seconds", 0.0
            ) / 60.0,
            "balance/observed_black_win_rate": balance.get(
                "observed_black_win_rate", 0.0
            ),
            "balance/black_win_rate_ema": balance.get("black_win_rate_ema", 0.0),
            "balance/standard_black_win_rate": balance.get(
                "standard_black_win_rate", 0.0
            ),
            "balance/injected_black_win_rate": balance.get(
                "injected_black_win_rate", 0.0
            ),
            "balance/current_opening_ratio": balance.get(
                "current_opening_ratio", 0.0
            ),
            "balance/next_opening_ratio": balance.get("next_opening_ratio", 0.0),
            "balance/white_win_weight": balance.get("white_win_weight", 1.0),
            "balance/white_win_sample_fraction": balance.get(
                "white_win_sample_fraction", 0.0
            ),
            "balance/effective_white_contribution": balance.get(
                "effective_white_contribution", 0.0
            ),
        }
        swanlab.log(payload, step=generation)
        atomic_text_write(marker_path, generation)
        last_generation = generation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "offline", "local", "disabled"],
        default=os.environ.get("NEBULA_SWANLAB_MODE", "online"),
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=None,
        help="仅用于本地闭环验证；正式常驻时不设置",
    )
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    active = init_outer_swanlab(args.swanlab_mode)
    finish_state = "aborted"
    completed = 0
    loop_script = os.path.join(EXPERIMENT_DIR, "run_loop.py")
    try:
        sync_summaries(active)
        while args.max_generations is None or completed < args.max_generations:
            print("\n双落子监督器：启动下一代", flush=True)
            result = subprocess.run(
                [
                    sys.executable,
                    loop_script,
                    "--swanlab-mode",
                    args.swanlab_mode,
                ],
                check=False,
            )
            if result.returncode != 0:
                finish_state = "crashed"
                raise SystemExit(result.returncode)
            completed += 1
            sync_summaries(active)
        finish_state = "success"
    finally:
        if active:
            try:
                swanlab.finish(state=finish_state, async_log_timeout=30)
            except Exception as error:
                print(f"外层 SwanLab 收尾失败: {error}", flush=True)


if __name__ == "__main__":
    main()
