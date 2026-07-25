"""双落子优化版逐代训练循环；每次进程默认只完成一代。"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import swanlab

from balance_controller import (
    combine_opening_counts,
    update_opening_ratio,
    white_win_weight,
)
from data_metrics import analyze_games


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RL_DIR = os.path.join(ROOT, "reinforcement_learning")
PYTHON = sys.executable

PROJECT = os.environ.get("NEBULA_SWANLAB_PROJECT", "Nebular-zero-two")
RUN_ROOT = os.path.abspath(
    os.environ.get(
        "NEBULA_PAIR_RUN_DIR",
        os.path.join(EXPERIMENT_DIR, "runs", PROJECT),
    )
)
DATA_DIR = os.path.join(RUN_ROOT, "data")
REPLAY_DIR = os.path.join(RUN_ROOT, "replay")
CHECKPOINT_DIR = os.path.join(RUN_ROOT, "checkpoints")
LOG_DIR = os.path.join(RUN_ROOT, "logs")
RUNTIME_DIR = os.path.join(RUN_ROOT, "runtime")
SUMMARY_DIR = os.path.join(LOG_DIR, "generation_summaries")
STATE_PATH = os.path.join(LOG_DIR, "loop_state.json")
CONTROLLER_PATH = os.path.join(LOG_DIR, "balance_controller.json")

INITIAL_MAIN = os.path.abspath(
    os.environ.get(
        "NEBULA_PAIR_INITIAL_MAIN",
        os.path.join(EXPERIMENT_DIR, "output", "joint_aggressive", "main.pth"),
    )
)
INITIAL_HEADS = os.path.abspath(
    os.environ.get(
        "NEBULA_PAIR_INITIAL_HEADS",
        os.path.join(EXPERIMENT_DIR, "output", "joint_aggressive", "pair_heads.pt"),
    )
)

CURRENT_MAIN = os.path.join(CHECKPOINT_DIR, "current_main.pth")
CURRENT_HEADS = os.path.join(CHECKPOINT_DIR, "current_pair_heads.pt")
CURRENT_PAIR_ENGINE = os.path.join(CHECKPOINT_DIR, "current_pair.engine")
CURRENT_EXACT_ENGINE = os.path.join(CHECKPOINT_DIR, "current_exact.engine")
MCTS_LIBRARY = os.path.join(RUNTIME_DIR, "libmcts.so")


def env_int(name, default):
    return int(os.environ.get(name, str(default)))


def env_float(name, default):
    return float(os.environ.get(name, str(default)))


def ensure_directories():
    for path in (
        DATA_DIR,
        REPLAY_DIR,
        CHECKPOINT_DIR,
        LOG_DIR,
        RUNTIME_DIR,
        SUMMARY_DIR,
    ):
        os.makedirs(path, exist_ok=True)


def atomic_json_dump(payload, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as source:
            return json.load(source)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def atomic_copy(source, destination):
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    temporary = f"{destination}.tmp"
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def run_command(command, env_vars=None):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if env_vars:
        env.update({key: str(value) for key, value in env_vars.items()})
    print("Running:", " ".join(str(value) for value in command), flush=True)
    subprocess.run(command, check=True, env=env)


def count_games(path):
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8", errors="replace") as source:
        return max(0, sum(1 for _ in source) - 1)


def load_state():
    state = load_json(STATE_PATH, None)
    if not isinstance(state, dict):
        state = {
            "version": 1,
            "generation": 0,
            "accepted_generation": -1,
            "phase": "selfplay",
        }
    return state


def save_state(generation, accepted_generation, phase):
    atomic_json_dump(
        {
            "version": 1,
            "generation": generation,
            "accepted_generation": accepted_generation,
            "phase": phase,
            "updated_at": time.time(),
        },
        STATE_PATH,
    )


def compile_mcts_if_needed():
    source_paths = [
        os.path.join(RL_DIR, "core", name)
        for name in ("mcts_engine.cpp", "c6_logic.h", "compile_mcts.py")
    ]
    newest_source = max(os.path.getmtime(path) for path in source_paths)
    if os.path.exists(MCTS_LIBRARY) and os.path.getmtime(MCTS_LIBRARY) >= newest_source:
        return
    run_command([
        PYTHON,
        os.path.join(RL_DIR, "core", "compile_mcts.py"),
        "--output",
        MCTS_LIBRARY,
    ])


def build_bundle(main_path, heads_path, output_dir, tag, build_gpu):
    """为同一主网络同时构建近似双落子引擎和精确锚点引擎。"""

    os.makedirs(output_dir, exist_ok=True)
    pair_onnx = os.path.join(RUNTIME_DIR, f"{tag}_pair.onnx")
    exact_onnx = os.path.join(RUNTIME_DIR, f"{tag}_exact.onnx")
    pair_engine = os.path.join(output_dir, f"{tag}_pair.engine")
    exact_engine = os.path.join(output_dir, f"{tag}_exact.engine")
    build_env = {
        "CUDA_VISIBLE_DEVICES": build_gpu,
        "NEBULA_BUILD_GPU": build_gpu,
        "NEBULA_TRT_TIMING_CACHE": os.path.join(RUNTIME_DIR, "tensorrt_timing.cache"),
        "NEBULA_TRT_MAX_BATCH_SIZE": env_int("NEBULA_MCTS_BATCH_SIZE", 64),
    }
    run_command(
        [
            PYTHON,
            os.path.join(EXPERIMENT_DIR, "export_pair_onnx.py"),
            main_path,
            heads_path,
            pair_onnx,
        ],
        build_env,
    )
    run_command(
        [
            PYTHON,
            os.path.join(RL_DIR, "pipeline", "build_engine.py"),
            pair_onnx,
            pair_engine,
        ],
        build_env,
    )
    run_command(
        [
            PYTHON,
            os.path.join(RL_DIR, "pipeline", "export_onnx.py"),
            main_path,
            exact_onnx,
        ],
        build_env,
    )
    run_command(
        [
            PYTHON,
            os.path.join(RL_DIR, "pipeline", "build_engine.py"),
            exact_onnx,
            exact_engine,
        ],
        build_env,
    )
    for path in (pair_onnx, exact_onnx):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    return pair_engine, exact_engine


def initialize_bundle(build_gpu):
    for path in (INITIAL_MAIN, INITIAL_HEADS):
        if not os.path.exists(path):
            raise FileNotFoundError(f"初始双落子候选不存在: {path}")
    if not os.path.exists(CURRENT_MAIN):
        atomic_copy(INITIAL_MAIN, CURRENT_MAIN)
    if not os.path.exists(CURRENT_HEADS):
        atomic_copy(INITIAL_HEADS, CURRENT_HEADS)
    compile_mcts_if_needed()
    if not os.path.exists(CURRENT_PAIR_ENGINE) or not os.path.exists(CURRENT_EXACT_ENGINE):
        pair_engine, exact_engine = build_bundle(
            CURRENT_MAIN,
            CURRENT_HEADS,
            RUNTIME_DIR,
            "initial",
            build_gpu,
        )
        atomic_copy(pair_engine, CURRENT_PAIR_ENGINE)
        atomic_copy(exact_engine, CURRENT_EXACT_ENGINE)


def generation_environment(pair_heads=None, concurrent_games=24):
    values = {
        "NEBULA_MCTS_LIBRARY": MCTS_LIBRARY,
        "NEBULA_SELFPLAY_GPUS": os.environ.get("NEBULA_SELFPLAY_GPUS", "6,7"),
        "NEBULA_NUM_WORKERS": env_int("NEBULA_NUM_WORKERS", 2),
        "NEBULA_MCTS_THREADS": env_int("NEBULA_MCTS_THREADS", 32),
        "NEBULA_MCTS_BATCH_SIZE": env_int("NEBULA_MCTS_BATCH_SIZE", 64),
        "NEBULA_MCTS_CONCURRENT_GAMES": concurrent_games,
        "NEBULA_SIMULATIONS_BLACK": env_int("NEBULA_SIMULATIONS_BLACK", 400),
        "NEBULA_SIMULATIONS_WHITE": env_int("NEBULA_SIMULATIONS_WHITE", 1200),
        "NEBULA_DYNAMIC_EARLY_STOP": 0,
        "NEBULA_CUDA_GRAPH": 1,
    }
    if pair_heads:
        values.update({
            "NEBULA_PAIR_HEADS": pair_heads,
            "NEBULA_PAIR_REFRESH_VISITS_BLACK": env_int(
                "NEBULA_PAIR_REFRESH_VISITS_BLACK", 1_000_000_000
            ),
            "NEBULA_PAIR_REFRESH_VISITS_WHITE": env_int(
                "NEBULA_PAIR_REFRESH_VISITS_WHITE", 2
            ),
            "NEBULA_PAIR_DEFER_REFRESH": 0,
        })
    return values


def generate_games(output_path, target_games, engine, seed, pair_heads=None):
    existing = count_games(output_path)
    remaining = max(0, target_games - existing)
    if remaining == 0:
        return 0
    concurrent_games = env_int(
        "NEBULA_PAIR_CONCURRENT_GAMES" if pair_heads else "NEBULA_EXACT_CONCURRENT_GAMES",
        24 if pair_heads else 12,
    )
    run_command(
        [
            PYTHON,
            os.path.join(RL_DIR, "pipeline", "generate.py"),
            "--engine",
            engine,
            "--out",
            output_path,
            "--total",
            str(remaining),
            "--seed",
            str(seed),
        ],
        generation_environment(pair_heads, concurrent_games),
    )
    completed = count_games(output_path)
    if completed != target_games:
        raise RuntimeError(f"生成数量异常: {output_path} 为 {completed}/{target_games}")
    return remaining


def legacy_data_paths():
    configured = os.environ.get("NEBULA_PAIR_LEGACY_DATA", "")
    if configured:
        paths = []
        for pattern in configured.split(os.pathsep):
            paths.extend(glob.glob(pattern))
        return sorted(set(paths))
    return [
        os.path.join(RL_DIR, "data", "raw", f"gen_{generation}.csv")
        for generation in range(20, 26)
        if os.path.exists(os.path.join(RL_DIR, "data", "raw", f"gen_{generation}.csv"))
    ]


def build_replay(generation, seed):
    window = env_int("NEBULA_PAIR_REPLAY_GENERATIONS", 2)
    online_paths = [
        os.path.join(DATA_DIR, f"gen_{index:04d}_pair.csv")
        for index in range(max(0, generation - window + 1), generation + 1)
    ]
    anchor_paths = [
        os.path.join(DATA_DIR, f"gen_{index:04d}_anchor.csv")
        for index in range(max(0, generation - window + 1), generation + 1)
    ]
    train_path = os.path.join(REPLAY_DIR, f"gen_{generation:04d}_train.csv")
    validation_path = os.path.join(REPLAY_DIR, f"gen_{generation:04d}_validation.csv")
    stats_path = os.path.join(REPLAY_DIR, f"gen_{generation:04d}_stats.json")
    command = [
        PYTHON,
        os.path.join(EXPERIMENT_DIR, "build_replay.py"),
        "--online",
        *online_paths,
        "--anchor",
        *anchor_paths,
        "--train-output",
        train_path,
        "--validation-output",
        validation_path,
        "--stats-output",
        stats_path,
        "--max-online-games",
        str(env_int("NEBULA_PAIR_MAX_ONLINE_REPLAY", 1200)),
        "--validation-ratio",
        str(env_float("NEBULA_PAIR_VALIDATION_RATIO", 0.2)),
        "--seed",
        str(seed),
    ]
    legacy = legacy_data_paths()
    if legacy:
        command.extend(["--legacy", *legacy])
    fixed_validation_games = env_int("NEBULA_PAIR_FIXED_VALIDATION_GAMES", 0)
    if fixed_validation_games > 0:
        command.extend([
            "--fixed-validation",
            os.path.join(REPLAY_DIR, "fixed_validation.csv"),
            "--fixed-validation-games",
            str(fixed_validation_games),
        ])
    run_command(command)
    stats = load_json(stats_path, {})
    if not isinstance(stats, dict):
        stats = {}
    stats.update({
        "generation": generation,
        "replay_window_generations": window,
        "oldest_online_generation": max(0, generation - window + 1),
        "newest_online_generation": generation,
        "train_size_mb": os.path.getsize(train_path) / (1024 * 1024),
        "validation_size_mb": os.path.getsize(validation_path) / (1024 * 1024),
    })
    atomic_json_dump(stats, stats_path)
    return train_path, validation_path, stats


def load_candidate_metrics(path):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }


def init_generation_swanlab(generation, mode, config_payload):
    if mode == "disabled":
        return False
    run_id_path = os.path.join(LOG_DIR, "swanlab", f"gen_{generation:04d}.txt")
    os.makedirs(os.path.dirname(run_id_path), exist_ok=True)
    existing_id = None
    try:
        with open(run_id_path, encoding="utf-8") as source:
            existing_id = source.read().strip() or None
    except FileNotFoundError:
        pass
    kwargs = {
        "project": PROJECT,
        "name": f"gen_{generation:04d}",
        "description": "原生双落子 MCTS 的逐代在线训练、精确锚点与成套门禁",
        "config": config_payload,
        "mode": mode,
    }
    if existing_id:
        kwargs.update({"id": existing_id, "resume": "allow"})
    try:
        run = swanlab.init(**kwargs)
        run_id = getattr(run, "id", None)
        if run_id:
            temporary = f"{run_id_path}.tmp"
            with open(temporary, "w", encoding="utf-8") as output:
                output.write(run_id)
            os.replace(temporary, run_id_path)
        return True
    except Exception as error:
        print(f"本代 SwanLab 初始化失败，将继续本地执行: {error}", flush=True)
        return False


def swanlab_log(active, payload, step):
    if not active:
        return
    try:
        swanlab.log(payload, step=step)
    except Exception as error:
        print(f"SwanLab step={step} 记录失败: {error}", flush=True)


def finish_swanlab(active, state):
    if not active:
        return
    try:
        swanlab.finish(state=state, async_log_timeout=30)
    except Exception as error:
        print(f"SwanLab 收尾失败: {error}", flush=True)


def training_epoch_payload(metrics):
    validation_names = {
        "first_ce",
        "second_ce",
        "first_top1",
        "second_top1",
        "second_recall_top20",
        "second_mass_top20",
        "value_mae",
        "conditional_value_mae",
    }
    payload = {}
    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue
        if key.startswith("train_"):
            name = f"train/{key.removeprefix('train_')}"
        elif key in validation_names:
            name = f"validation/{key}"
        else:
            name = f"training/{key}"
        payload[name] = value
    return payload


def run_one_generation(mode):
    ensure_directories()
    state = load_state()
    generation = int(state["generation"])
    accepted_generation = int(state.get("accepted_generation", -1))
    auto_balance = os.environ.get("NEBULA_AUTO_BALANCE", "0") == "1"
    controller_state = load_json(CONTROLLER_PATH, {})
    if not isinstance(controller_state, dict):
        controller_state = {}
    opening_ratio = env_float("NEBULA_FORCED_OPENING_RATIO", 0.0)
    if (
        auto_balance
        and int(controller_state.get("next_generation", -1)) == generation
    ):
        opening_ratio = float(
            controller_state.get("next_opening_ratio", opening_ratio)
        )
    os.environ["NEBULA_FORCED_OPENING_RATIO"] = str(opening_ratio)
    training_gpu = os.environ.get("NEBULA_TRAINING_GPU", "6")
    online_games = env_int("NEBULA_PAIR_ONLINE_GAMES", 500)
    anchor_games = env_int("NEBULA_PAIR_ANCHOR_GAMES", 100)
    eval_games = env_int("NEBULA_PAIR_EVAL_GAMES", 200)
    generation_seed = env_int("NEBULA_SEED", 2026) + generation * 100_000
    config_payload = {
        "generation": generation,
        "online_games": online_games,
        "exact_anchor_games": anchor_games,
        "eval_games": eval_games,
        "selfplay_gpus": os.environ.get("NEBULA_SELFPLAY_GPUS", "6,7"),
        "mcts_threads_per_worker": env_int("NEBULA_MCTS_THREADS", 32),
        "pair_concurrent_games": env_int("NEBULA_PAIR_CONCURRENT_GAMES", 24),
        "black_simulations": env_int("NEBULA_SIMULATIONS_BLACK", 400),
        "white_simulations": env_int("NEBULA_SIMULATIONS_WHITE", 1200),
        "opening_temperature_black": env_float("NEBULA_TEMP_OPENING_BLACK", 0.9),
        "opening_temperature_white": env_float("NEBULA_TEMP_OPENING_WHITE", 0.2),
        "auto_balance": int(auto_balance),
        "forced_opening_ratio": opening_ratio,
        "forced_opening_stones": env_int("NEBULA_FORCED_OPENING_STONES", 5),
        "train_precision": "bf16",
        "trunk_learning_rate": env_float("NEBULA_PAIR_TRUNK_LR", 3e-6),
        "head_learning_rate": env_float("NEBULA_PAIR_HEAD_LR", 8e-5),
        "train_epochs": env_int("NEBULA_PAIR_TRAIN_EPOCHS", 2),
        "white_win_weight": env_float("NEBULA_PAIR_WHITE_WIN_WEIGHT", 1.0),
        "replay_window_generations": env_int("NEBULA_PAIR_REPLAY_GENERATIONS", 2),
        "max_online_replay_games": env_int("NEBULA_PAIR_MAX_ONLINE_REPLAY", 1200),
        "max_train_samples": env_int("NEBULA_PAIR_MAX_TRAIN_SAMPLES", 30_000),
        "max_validation_samples": env_int(
            "NEBULA_PAIR_MAX_VALIDATION_SAMPLES", 3_000
        ),
    }
    swan_active = init_generation_swanlab(generation, mode, config_payload)
    finish_state = "crashed"
    generation_started = time.perf_counter()
    phase_times = {}

    try:
        initialize_bundle(training_gpu)

        save_state(generation, accepted_generation, "selfplay")
        started = time.perf_counter()
        pair_path = os.path.join(DATA_DIR, f"gen_{generation:04d}_pair.csv")
        anchor_path = os.path.join(DATA_DIR, f"gen_{generation:04d}_anchor.csv")
        new_pair = generate_games(
            pair_path,
            online_games,
            CURRENT_PAIR_ENGINE,
            generation_seed,
            pair_heads=CURRENT_HEADS,
        )
        new_anchor = generate_games(
            anchor_path,
            anchor_games,
            CURRENT_EXACT_ENGINE,
            generation_seed + 50_000,
        )
        phase_times["selfplay_seconds"] = time.perf_counter() - started
        total_new = new_pair + new_anchor
        generation_speed = total_new / max(phase_times["selfplay_seconds"], 1e-9)
        pair_quality = analyze_games(pair_path)
        anchor_quality = analyze_games(anchor_path)
        opening_counts = combine_opening_counts(pair_quality, anchor_quality)
        opening_control = update_opening_ratio(
            opening_ratio,
            opening_counts,
            previous_ema=controller_state.get("black_win_rate_ema"),
            target_black_rate=env_float("NEBULA_BALANCE_TARGET_BLACK_RATE", 0.50),
            deadband=env_float("NEBULA_BALANCE_DEADBAND", 0.03),
            ema_alpha=env_float("NEBULA_BALANCE_EMA_ALPHA", 0.35),
            max_step=env_float("NEBULA_BALANCE_MAX_OPENING_STEP", 0.10),
            minimum_ratio=env_float("NEBULA_BALANCE_MIN_OPENING_RATIO", 0.0),
            maximum_ratio=env_float("NEBULA_BALANCE_MAX_OPENING_RATIO", 0.90),
        )
        if not auto_balance:
            opening_control["next_opening_ratio"] = opening_ratio
            opening_control["reason"] = "automatic_balance_disabled"
        swanlab_log(swan_active, {
            "generation": generation,
            **{f"data/online/{key}": value for key, value in pair_quality.items()},
            **{f"data/anchor/{key}": value for key, value in anchor_quality.items()},
            "performance/selfplay_games_per_second": generation_speed,
            "timing/selfplay_seconds": phase_times["selfplay_seconds"],
            **{
                f"balance/{key}": value
                for key, value in opening_control.items()
                if isinstance(value, (int, float))
            },
        }, step=0)

        save_state(generation, accepted_generation, "training")
        started = time.perf_counter()
        train_path, validation_path, replay_stats = build_replay(
            generation,
            generation_seed + 70_000,
        )
        if auto_balance:
            config_payload["white_win_weight"] = white_win_weight(
                replay_stats.get(
                    "train_white_win_sample_fraction",
                    replay_stats.get("train_white_win_rate", 0.0),
                ),
                target_fraction=env_float(
                    "NEBULA_BALANCE_TARGET_WHITE_CONTRIBUTION", 0.40
                ),
                minimum=env_float("NEBULA_BALANCE_MIN_WHITE_WEIGHT", 1.0),
                maximum=env_float("NEBULA_BALANCE_MAX_WHITE_WEIGHT", 3.0),
            )
        swanlab_log(swan_active, {
            "generation": generation,
            **{
                f"buffer/{key}": value
                for key, value in replay_stats.items()
                if isinstance(value, (int, float))
            },
            "balance/white_win_weight": config_payload["white_win_weight"],
        }, step=0)
        candidate_dir = os.path.join(CHECKPOINT_DIR, f"gen_{generation:04d}")
        os.makedirs(candidate_dir, exist_ok=True)
        run_command(
            [
                PYTHON,
                os.path.join(EXPERIMENT_DIR, "train_joint.py"),
                "--checkpoint",
                CURRENT_MAIN,
                "--pair-heads",
                CURRENT_HEADS,
                "--train",
                train_path,
                "--validation",
                validation_path,
                "--output-dir",
                candidate_dir,
                "--max-train",
                str(env_int("NEBULA_PAIR_MAX_TRAIN_SAMPLES", 30_000)),
                "--max-validation",
                str(env_int("NEBULA_PAIR_MAX_VALIDATION_SAMPLES", 3_000)),
                "--batch-size",
                str(env_int("NEBULA_PAIR_TRAIN_BATCH_SIZE", 96)),
                "--epochs",
                str(config_payload["train_epochs"]),
                "--trunk-learning-rate",
                str(config_payload["trunk_learning_rate"]),
                "--head-learning-rate",
                str(config_payload["head_learning_rate"]),
                "--rank",
                "16",
                "--projection-hidden",
                "128",
                "--relative-gating",
                "--white-win-weight",
                str(config_payload["white_win_weight"]),
                "--seed",
                str(generation_seed + 80_000),
            ],
            {"CUDA_VISIBLE_DEVICES": training_gpu},
        )
        candidate_main = os.path.join(candidate_dir, "main.pth")
        candidate_heads = os.path.join(candidate_dir, "pair_heads.pt")
        if not os.path.exists(candidate_main) or not os.path.exists(candidate_heads):
            raise RuntimeError("联合训练没有生成完整候选包")
        train_metrics = load_candidate_metrics(candidate_main)
        training_history = load_json(
            os.path.join(candidate_dir, "metrics_history.json"),
            {},
        )
        if not isinstance(training_history, dict):
            training_history = {}
        replay_stats.update({
            "train_samples_used": int(train_metrics.get("train_samples", 0)),
            "validation_samples_used": int(
                train_metrics.get("validation_samples", 0)
            ),
            "batches_per_epoch": int(train_metrics.get("batches_per_epoch", 0)),
            "selected_epoch": int(train_metrics.get("epoch", 0)),
            "selected_optimizer_steps": int(train_metrics.get("optimizer_steps", 0)),
            "selected_samples_seen": int(train_metrics.get("samples_seen", 0)),
            "executed_epochs": int(config_payload["train_epochs"]),
            "actual_optimizer_steps": int(
                train_metrics.get("batches_per_epoch", 0)
                * config_payload["train_epochs"]
            ),
            "actual_samples_seen": int(
                train_metrics.get("train_samples", 0)
                * config_payload["train_epochs"]
            ),
        })
        atomic_json_dump(
            replay_stats,
            os.path.join(REPLAY_DIR, f"gen_{generation:04d}_stats.json"),
        )
        phase_times["training_seconds"] = time.perf_counter() - started
        initial_metrics = training_history.get("initial", {})
        if isinstance(initial_metrics, dict):
            swanlab_log(swan_active, {
                "generation": generation,
                **{
                    f"validation/initial_{key}": value
                    for key, value in initial_metrics.items()
                    if isinstance(value, (int, float))
                },
            }, step=0)
        for epoch_metrics in training_history.get("epochs", []):
            if not isinstance(epoch_metrics, dict):
                continue
            epoch = int(epoch_metrics.get("epoch", 0))
            swanlab_log(swan_active, {
                "generation": generation,
                **training_epoch_payload(epoch_metrics),
                "timing/training_seconds": phase_times["training_seconds"],
            }, step=epoch)

        save_state(generation, accepted_generation, "engine")
        started = time.perf_counter()
        candidate_pair_engine, candidate_exact_engine = build_bundle(
            candidate_main,
            candidate_heads,
            candidate_dir,
            f"gen_{generation:04d}",
            training_gpu,
        )
        phase_times["engine_seconds"] = time.perf_counter() - started

        save_state(generation, accepted_generation, "evaluation")
        started = time.perf_counter()
        gate_path = os.path.join(LOG_DIR, f"gen_{generation:04d}_gate.json")
        gate_data_path = os.path.join(DATA_DIR, f"gen_{generation:04d}_gate.csv")
        eval_env = generation_environment(CURRENT_HEADS, 1)
        run_command(
            [
                PYTHON,
                os.path.join(EXPERIMENT_DIR, "evaluate_pair.py"),
                "--candidate-engine",
                candidate_pair_engine,
                "--candidate-heads",
                candidate_heads,
                "--incumbent-engine",
                CURRENT_PAIR_ENGINE,
                "--incumbent-heads",
                CURRENT_HEADS,
                "--output",
                gate_path,
                "--data-output",
                gate_data_path,
                "--games",
                str(eval_games),
                "--gpu",
                training_gpu,
                "--simulations-black",
                str(env_int("NEBULA_SIMULATIONS_BLACK", 400)),
                "--simulations-white",
                str(env_int("NEBULA_SIMULATIONS_WHITE", 1200)),
                "--seed",
                str(generation_seed + 90_000),
            ],
            eval_env,
        )
        gate = load_json(gate_path, None)
        if not isinstance(gate, dict):
            raise RuntimeError("门禁结果不存在")
        phase_times["evaluation_seconds"] = time.perf_counter() - started
        passed = (
            float(gate["score_rate"]) >= env_float("NEBULA_PAIR_GATING_SCORE", 0.5)
            and float(gate["white_win_rate"])
            >= env_float("NEBULA_PAIR_GATING_WHITE", 0.2)
            and float(gate.get("game_black_win_rate", 0.5))
            >= env_float("NEBULA_PAIR_GATING_GAME_BLACK_MIN", 0.0)
            and float(gate.get("game_black_win_rate", 0.5))
            <= env_float("NEBULA_PAIR_GATING_GAME_BLACK_MAX", 1.0)
        )
        if passed:
            atomic_copy(candidate_main, CURRENT_MAIN)
            atomic_copy(candidate_heads, CURRENT_HEADS)
            atomic_copy(candidate_pair_engine, CURRENT_PAIR_ENGINE)
            atomic_copy(candidate_exact_engine, CURRENT_EXACT_ENGINE)
            accepted_generation = generation
            print(f">>> 第 {generation} 代成套门禁通过，已原子晋升", flush=True)
        else:
            print(
                f">>> 第 {generation} 代门禁未通过：score={gate['score_rate']:.2%}, "
                f"white={gate['white_win_rate']:.2%}；保留 incumbent",
                flush=True,
            )

        phase_times["total_seconds"] = time.perf_counter() - generation_started
        swanlab_log(swan_active, {
            "generation": generation,
            "eval/gating_passed": int(passed),
            **{f"eval/{key}": value for key, value in gate.items()},
            **{f"timing/{key}": value for key, value in phase_times.items()},
        }, step=config_payload["train_epochs"] + 1)
        summary = {
            "version": 1,
            "generation": generation,
            "accepted": passed,
            "accepted_generation": accepted_generation,
            "online_games": count_games(pair_path),
            "anchor_games": count_games(anchor_path),
            "selfplay_games_per_second": generation_speed,
            "buffer": replay_stats,
            "data_quality": {
                "online": pair_quality,
                "anchor": anchor_quality,
            },
            "train": train_metrics,
            "training_history": training_history,
            "balance": {
                **opening_control,
                "white_win_weight": config_payload["white_win_weight"],
                "white_win_sample_fraction": replay_stats.get(
                    "train_white_win_sample_fraction", 0.0
                ),
                "effective_white_contribution": (
                    replay_stats.get("train_white_win_sample_fraction", 0.0)
                    * config_payload["white_win_weight"]
                    / max(
                        replay_stats.get("train_white_win_sample_fraction", 0.0)
                        * config_payload["white_win_weight"]
                        + 1.0
                        - replay_stats.get("train_white_win_sample_fraction", 0.0),
                        1e-9,
                    )
                ),
                "target_white_contribution": env_float(
                    "NEBULA_BALANCE_TARGET_WHITE_CONTRIBUTION", 0.40
                ),
            },
            "eval": gate,
            "phase_times": phase_times,
        }
        atomic_json_dump(
            summary,
            os.path.join(SUMMARY_DIR, f"gen_{generation:04d}.json"),
        )
        if auto_balance:
            atomic_json_dump(
                {
                    "version": 1,
                    "generation": generation,
                    "next_generation": generation + 1,
                    **opening_control,
                    "white_win_weight": config_payload["white_win_weight"],
                },
                CONTROLLER_PATH,
            )
        save_state(generation + 1, accepted_generation, "selfplay")
        finish_state = "success"
        print(
            f"第 {generation} 代完成，总耗时 {phase_times['total_seconds']:.1f}s",
            flush=True,
        )
        return summary
    finally:
        finish_swanlab(swan_active, finish_state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "offline", "local", "disabled"],
        default=os.environ.get("NEBULA_SWANLAB_MODE", "online"),
    )
    args = parser.parse_args()
    run_one_generation(args.swanlab_mode)


if __name__ == "__main__":
    main()
