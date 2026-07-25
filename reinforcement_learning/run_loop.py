import argparse
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time

import swanlab

import config


PYTHON = sys.executable


def signal_handler(sig, frame):
    """把退出信号交给主循环的 finally，确保状态和日志能正常收尾。"""
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def run_command(cmd, env_vars=None):
    """运行子进程；列表参数不经过 shell，避免路径和转义问题。"""
    display_cmd = ' '.join(str(part) for part in cmd)
    print(f"Running: {display_cmd}")
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    if env_vars:
        env.update({key: str(value) for key, value in env_vars.items()})
    subprocess.run(cmd, check=True, env=env)


def atomic_json_dump(payload, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temp_path = f"{path}.tmp"
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    os.replace(temp_path, path)


def load_json(path, default):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def atomic_copy(source, destination):
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    temp_path = f"{destination}.tmp"
    shutil.copy2(source, temp_path)
    os.replace(temp_path, destination)


def count_csv_rows(path):
    if not os.path.exists(path):
        return 0
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return max(0, sum(1 for _ in f) - 1)


def manage_buffer(new_data_path):
    """幂等合并回放数据，重启后重复导入同一文件也不会复制样本。"""
    buffer_path = os.path.join(config.BUFFER_DIR, 'replay_buffer.csv')
    if not os.path.exists(new_data_path):
        raise FileNotFoundError(f"回放数据不存在: {new_data_path}")

    with open(new_data_path, 'r', encoding='utf-8', errors='replace') as f:
        incoming_lines = f.readlines()
    if not incoming_lines:
        raise ValueError(f"回放数据为空: {new_data_path}")

    header = incoming_lines[0].rstrip('\n')
    incoming = [line.rstrip('\n') for line in incoming_lines[1:] if line.strip()]
    existing = []
    if os.path.exists(buffer_path):
        with open(buffer_path, 'r', encoding='utf-8', errors='replace') as f:
            buffer_lines = f.readlines()
        if buffer_lines:
            existing = [line.rstrip('\n') for line in buffer_lines[1:] if line.strip()]

    seen = set(existing)
    added = []
    for line in incoming:
        if line not in seen:
            seen.add(line)
            added.append(line)

    combined = existing + added
    if len(combined) > config.BUFFER_SIZE:
        print(f"回放池从 {len(combined)} 局裁剪到 {config.BUFFER_SIZE} 局")
        combined = combined[-config.BUFFER_SIZE:]

    temp_path = f"{buffer_path}.tmp"
    with open(temp_path, 'w', encoding='utf-8', newline='') as f:
        f.write(header + '\n')
        for line in combined:
            f.write(line + '\n')
    os.replace(temp_path, buffer_path)

    result = {
        'added': len(added),
        'duplicates': len(incoming) - len(added),
        'size': len(combined),
    }
    print(
        f"回放池新增 {result['added']} 局，跳过 {result['duplicates']} 局重复数据，"
        f"当前共 {result['size']} 局"
    )
    return result


def update_history_plot(history_file, plot_file):
    # 回放池与状态恢复等无界面逻辑不应强制依赖 matplotlib。
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    history = load_json(history_file, [])
    if not history:
        return

    generations = [entry['generation'] for entry in history]
    scores = [entry.get('incumbent_score_rate', entry.get('avg_win_rate', 0)) for entry in history]
    white_win_rates = [entry.get('white_win_rate', entry.get('avg_white_win_rate', 0)) for entry in history]

    plt.figure(figsize=(10, 6))
    plt.plot(generations, scores, marker='o', label='Score vs incumbent')
    plt.plot(generations, white_win_rates, marker='x', linestyle='--', label='White win rate')
    plt.title('Gating Trend')
    plt.xlabel('Generation')
    plt.ylabel('Rate')
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file, dpi=120)
    plt.close()


def analyze_generation_data(data_path, generation):
    """分析对局分布并生成轻量图表。"""
    import pandas as pd

    if not os.path.exists(data_path):
        return {}
    try:
        df = pd.read_csv(data_path, on_bad_lines='skip')
    except Exception as e:
        print(f"读取对局数据失败: {e}")
        return {}
    if df.empty or 'winner' not in df.columns:
        return {}

    df['winner'] = df['winner'].astype(str).str.lower()
    total_games = len(df)
    black_wins = int(df['winner'].str.contains('black', na=False).sum())
    white_wins = int(df['winner'].str.contains('white', na=False).sum())
    draws = total_games - black_wins - white_wins

    def count_moves(moves_str):
        if not isinstance(moves_str, str) or moves_str == 'nan':
            return 0
        return moves_str.count(',') + 1

    def get_opening(moves_str, n_moves=5):
        if not isinstance(moves_str, str) or moves_str == 'nan':
            return ''
        return ','.join(moves_str.split(',')[:n_moves])

    moves_column = 'moves' if 'moves' in df.columns else df.columns[0]
    df['length'] = df[moves_column].apply(count_moves)
    df['opening'] = df[moves_column].apply(get_opening)
    average_length = float(df['length'].mean())
    unique_openings = int(df['opening'].nunique())

    stats = {
        'total_games': total_games,
        'black_wins': black_wins,
        'white_wins': white_wins,
        'draws': draws,
        'black_win_rate': black_wins / total_games,
        'white_win_rate': white_wins / total_games,
        'avg_game_length': average_length,
        'opening_diversity': unique_openings / total_games,
        'unique_openings': unique_openings,
    }

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        labels = ['Black', 'Draw', 'White']
        counts = [black_wins, draws, white_wins]
        bars = axes[0].bar(labels, counts, color=['black', 'gray', 'white'], edgecolor='black')
        axes[0].set_title(f'Gen {generation} Win Distribution')
        axes[0].set_ylabel('Games')
        for bar, count in zip(bars, counts):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f'{count}\n({count / total_games:.1%})',
                ha='center',
                va='bottom',
            )

        axes[1].hist(df['length'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        axes[1].axvline(average_length, color='red', linestyle='--', label=f'Avg: {average_length:.1f}')
        axes[1].set_title(f'Gen {generation} Game Length')
        axes[1].set_xlabel('Moves')
        axes[1].set_ylabel('Games')
        axes[1].legend()
        plt.tight_layout()

        chart_path = os.path.join(config.LOG_DIR, f'gen_{generation}_analysis.png')
        plt.savefig(chart_path, dpi=100)
        plt.close(fig)
        stats['chart_path'] = chart_path
    except Exception as e:
        print(f"生成数据图表失败: {e}")

    return stats


def parse_generation(path, prefix):
    name = os.path.basename(path)
    try:
        return int(name.replace(prefix, '').split('.')[0])
    except (TypeError, ValueError):
        return None


def discover_generation():
    """在没有状态文件时，从原始数据和候选权重推断下一代。"""
    raw_files = glob.glob(os.path.join(config.RAW_DATA_DIR, 'gen_*.csv'))
    raw_generations = [parse_generation(path, 'gen_') for path in raw_files]
    raw_generations = [value for value in raw_generations if value is not None]

    checkpoint_files = glob.glob(os.path.join(config.CHECKPOINT_DIR, 'model_gen_*.pth'))
    checkpoint_generations = [parse_generation(path, 'model_gen_') for path in checkpoint_files]
    checkpoint_generations = [value for value in checkpoint_generations if value is not None]

    candidates = [0]
    if raw_generations:
        last_raw = max(raw_generations)
        last_path = os.path.join(config.RAW_DATA_DIR, f'gen_{last_raw}.csv')
        candidates.append(last_raw if count_csv_rows(last_path) < config.GAMES_PER_LOOP else last_raw + 1)
    if checkpoint_generations:
        candidates.append(max(checkpoint_generations) + 1)
    return max(candidates)


def load_loop_state(state_path):
    state = load_json(state_path, None)
    if not isinstance(state, dict) or 'generation' not in state:
        state = {
            'generation': discover_generation(),
            'phase': 'selfplay',
            'current_epochs': config.TRAIN_EPOCHS,
            'accepted_generation': None,
        }
    return state


def save_loop_state(state_path, generation, phase, current_epochs, accepted_generation):
    state = {
        'version': 2,
        'generation': generation,
        'phase': phase,
        'current_epochs': current_epochs,
        'accepted_generation': accepted_generation,
        'updated_at': time.time(),
    }
    atomic_json_dump(state, state_path)


def save_generation_summary(generation, payload):
    """保存本代标量摘要，供最外层 loop 实验汇总跨代趋势。"""
    summary_dir = os.path.join(config.LOG_DIR, 'generation_summaries')
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, f'gen_{generation}.json')
    atomic_json_dump(payload, summary_path)


def selfplay_runtime_config():
    """返回实际执行自对弈的设备参数，供日志和 SwanLab 保持一致。"""
    if config.SELFPLAY_BACKEND == 'remote':
        return {
            'backend': 'remote',
            'host': config.REMOTE_SELFPLAY_HOST,
            'gpus': config.REMOTE_SELFPLAY_GPUS,
            'workers': config.REMOTE_NUM_WORKERS,
            'mcts_threads': config.REMOTE_MCTS_THREADS,
            'mcts_batch_size': config.REMOTE_MCTS_BATCH_SIZE,
            'cpu_set': config.REMOTE_CPUSET,
        }
    if config.SELFPLAY_BACKEND != 'local':
        raise ValueError(
            f"未知 SELFPLAY_BACKEND={config.SELFPLAY_BACKEND!r}，只支持 local/remote"
        )
    return {
        'backend': 'local',
        'host': '',
        'gpus': config.GPUS,
        'workers': config.NUM_WORKERS,
        'mcts_threads': config.MCTS_THREADS,
        'mcts_batch_size': config.MCTS_BATCH_SIZE,
        'cpu_set': '',
    }


def initialize_swanlab(generation, current_epochs, accepted_generation, mode=None):
    """初始化或恢复当前代实验；失败时不阻塞本地训练。"""
    # 每个 Project、每一代单独保存续传 ID，中断恢复时不会创建重复实验。
    project_slug = ''.join(
        char if char.isalnum() or char in '-_' else '_'
        for char in config.SWANLAB_PROJECT
    )
    run_id_dir = os.path.join(config.LOG_DIR, 'swanlab_runs', project_slug)
    os.makedirs(run_id_dir, exist_ok=True)
    run_id_path = os.path.join(run_id_dir, f'gen_{generation}.txt')
    existing_run_id = None
    try:
        with open(run_id_path, 'r', encoding='utf-8') as f:
            existing_run_id = f.read().strip() or None
    except FileNotFoundError:
        pass

    selfplay_runtime = selfplay_runtime_config()
    init_kwargs = {
        'reinit': True,
        'project': config.SWANLAB_PROJECT,
        'name': f'{config.SWANLAB_RUN_PREFIX}_{generation}',
        'description': f'Nebula Zero 第 {generation} 代：自我对弈、训练与门控',
        'config': {
            'generation': generation,
            'accepted_generation': accepted_generation,
            'seed': config.SEED,
            'train_precision': config.TRAIN_PRECISION,
            'train_epochs': current_epochs,
            'train_batch_size': config.BATCH_SIZE_TRAIN,
            'learning_rate': config.LEARNING_RATE,
            'buffer_size': config.BUFFER_SIZE,
            'selfplay_games': config.GAMES_PER_LOOP,
            'selfplay_backend': selfplay_runtime['backend'],
            'selfplay_host': selfplay_runtime['host'],
            'selfplay_gpus': selfplay_runtime['gpus'],
            'selfplay_workers': selfplay_runtime['workers'],
            'mcts_threads': selfplay_runtime['mcts_threads'],
            'mcts_batch_size': selfplay_runtime['mcts_batch_size'],
            'selfplay_cpu_set': selfplay_runtime['cpu_set'],
            'simulations_black': config.SIMULATIONS_BLACK,
            'simulations_white': config.SIMULATIONS_WHITE,
            'eval_games': config.EVAL_GAMES,
            'eval_simulations': config.EVAL_SIMULATIONS,
        },
    }
    if mode:
        init_kwargs['mode'] = mode
    if existing_run_id:
        init_kwargs.update({'id': existing_run_id, 'resume': 'allow'})

    try:
        run = swanlab.init(**init_kwargs)
        run_id = getattr(run, 'id', None)
        if run_id and not existing_run_id:
            temp_path = f"{run_id_path}.tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(run_id)
            os.replace(temp_path, run_id_path)
        return True
    except Exception as e:
        print(f"SwanLab 初始化失败，将继续本地训练: {e}")
        return False


def swanlab_log(active, payload, step):
    if not active:
        return
    try:
        swanlab.log(payload, step=step)
    except Exception as e:
        print(f"SwanLab 记录失败: {e}")


def finish_swanlab(active, state):
    """结束当前代实验；收尾异常不影响本地断点。"""
    if not active:
        return
    try:
        swanlab.finish(
            state=state,
            async_log_timeout=config.SWANLAB_FINISH_TIMEOUT,
        )
    except Exception as e:
        print(f"SwanLab 收尾失败: {e}")


def build_engine_from_checkpoint(checkpoint_path, engine_path, gpu_id, tag):
    onnx_path = os.path.join(config.CHECKPOINT_DIR, f'{tag}.onnx')
    export_script = os.path.join(config.BASE_DIR, 'pipeline', 'export_onnx.py')
    build_script = os.path.join(config.BASE_DIR, 'pipeline', 'build_engine.py')
    env = {'CUDA_VISIBLE_DEVICES': gpu_id}
    try:
        run_command([PYTHON, export_script, checkpoint_path, onnx_path], env_vars=env)
        run_command([PYTHON, build_script, onnx_path, engine_path], env_vars=env)
        if not os.path.exists(engine_path):
            raise RuntimeError(f"TensorRT 引擎未生成: {engine_path}")
    finally:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)


def ensure_initial_assets(train_gpu):
    """确保 incumbent 的 PyTorch 权重和 TensorRT 引擎同时存在。"""
    if not os.path.exists(config.CURRENT_MODEL_PTH):
        if not os.path.exists(config.INITIAL_MODEL_PTH):
            raise FileNotFoundError(f"找不到初始权重: {config.INITIAL_MODEL_PTH}")
        atomic_copy(config.INITIAL_MODEL_PTH, config.CURRENT_MODEL_PTH)
        print(f"初始化 best.pth: {config.INITIAL_MODEL_PTH}")

    if os.path.exists(config.CURRENT_ENGINE_PATH):
        return
    if os.path.exists(config.INITIAL_MODEL_PATH):
        atomic_copy(config.INITIAL_MODEL_PATH, config.CURRENT_ENGINE_PATH)
        return

    print("首次运行：从 best.pth 构建 TensorRT 引擎")
    build_engine_from_checkpoint(
        config.CURRENT_MODEL_PTH,
        config.CURRENT_ENGINE_PATH,
        train_gpu,
        'initial',
    )
    atomic_copy(config.CURRENT_ENGINE_PATH, config.INITIAL_MODEL_PATH)


def prepare_opponent_engine(generation, train_gpu):
    if config.ASYMMETRIC_SELFPLAY_RATIO <= 0:
        return None
    target_generation = generation - config.OPPONENT_MODEL_GENERATION_GAP
    if target_generation <= 0:
        return None

    accepted_pth = os.path.join(config.CHECKPOINT_DIR, f'accepted_gen_{target_generation}.pth')
    legacy_pth = os.path.join(config.CHECKPOINT_DIR, f'model_gen_{target_generation}.pth')
    opponent_pth = accepted_pth if os.path.exists(accepted_pth) else legacy_pth
    if not os.path.exists(opponent_pth):
        return None

    engine_path = os.path.join(config.CHECKPOINT_DIR, f'opponent_gen_{target_generation}.engine')
    if not os.path.exists(engine_path):
        build_engine_from_checkpoint(opponent_pth, engine_path, train_gpu, f'opponent_{target_generation}')
    return engine_path


def flatten_eval_results(results):
    payload = {}
    for opponent_name, stats in results.items():
        for metric_name, value in stats.items():
            if isinstance(value, (int, float)):
                payload[f'eval/{opponent_name}/{metric_name}'] = value
    return payload


def run_training_loop(max_generations=None, swanlab_mode=None):
    config.ensure_dirs()
    state_path = os.path.join(config.LOG_DIR, 'loop_state.json')
    history_path = os.path.join(config.LOG_DIR, 'eval_history.json')
    if not os.path.exists(history_path):
        atomic_json_dump([], history_path)

    train_gpu = str(config.TRAINING_GPU)
    training_gpus = str(config.TRAINING_GPUS)
    ensure_initial_assets(train_gpu)
    state = load_loop_state(state_path)
    generation = int(state['generation'])
    current_epochs = int(state.get('current_epochs', config.TRAIN_EPOCHS))
    accepted_generation = state.get('accepted_generation')
    selfplay_runtime = selfplay_runtime_config()

    print(
        f"自对弈: {selfplay_runtime['backend']} "
        f"{selfplay_runtime['host']} GPU {selfplay_runtime['gpus']} | "
        f"worker={selfplay_runtime['workers']} × "
        f"{selfplay_runtime['mcts_threads']}线程 | "
        f"训练 GPU: {training_gpus} | "
        f"构建/评估 GPU: {train_gpu} | 精度: {config.TRAIN_PRECISION}"
    )
    print(f"从第 {generation} 代、阶段 {state.get('phase', 'selfplay')} 继续")

    swan_active = False
    completed_generations = 0
    finish_state = 'success'

    try:
        while max_generations is None or completed_generations < max_generations:
            swan_active = initialize_swanlab(
                generation,
                current_epochs,
                accepted_generation,
                swanlab_mode,
            )
            generation_started_at = time.perf_counter()
            phase_times = {}
            print(f"\n{'=' * 20} Generation {generation} {'=' * 20}\n")

            # 1. 自我对弈：只补齐缺少的局数，支持部分文件断点恢复。
            save_loop_state(state_path, generation, 'selfplay', current_epochs, accepted_generation)
            phase_started_at = time.perf_counter()
            generation_data = os.path.join(config.RAW_DATA_DIR, f'gen_{generation}.csv')
            existing_games = count_csv_rows(generation_data)
            remaining_games = max(0, config.GAMES_PER_LOOP - existing_games)
            if remaining_games:
                generation_seed = config.SEED + generation * 100000
                if config.SELFPLAY_BACKEND == 'remote':
                    if config.ASYMMETRIC_SELFPLAY_RATIO > 0:
                        raise ValueError('远端执行器当前只支持纯自对弈，请关闭陪练混合比例')
                    remote_script = os.path.join(
                        config.BASE_DIR,
                        'pipeline',
                        'remote_generate.py',
                    )
                    run_command([
                        PYTHON,
                        remote_script,
                        '--generation', str(generation),
                        '--out', generation_data,
                        '--target-total', str(config.GAMES_PER_LOOP),
                        '--weight', config.CURRENT_MODEL_PTH,
                        '--seed', str(generation_seed),
                    ])
                else:
                    generate_script = os.path.join(config.BASE_DIR, 'pipeline', 'generate.py')
                    command = [
                        PYTHON,
                        generate_script,
                        '--out', generation_data,
                        '--total', str(remaining_games),
                        '--engine', config.CURRENT_ENGINE_PATH,
                        '--seed', str(generation_seed),
                    ]
                    opponent_engine = prepare_opponent_engine(generation, train_gpu)
                    if opponent_engine:
                        command.extend([
                            '--opponent', opponent_engine,
                            '--mix_ratio', str(config.ASYMMETRIC_SELFPLAY_RATIO),
                        ])
                    run_command(command)

            generated_games = count_csv_rows(generation_data)
            if generated_games < config.GAMES_PER_LOOP:
                raise RuntimeError(
                    f"第 {generation} 代只生成 {generated_games}/{config.GAMES_PER_LOOP} 局"
                )
            newly_generated_games = max(0, generated_games - existing_games)
            phase_times['selfplay_seconds'] = time.perf_counter() - phase_started_at
            selfplay_games_per_second = (
                newly_generated_games / max(phase_times['selfplay_seconds'], 1e-9)
            )
            print(
                f"自对弈完成：累计 {generated_games} 局，本次新增 {newly_generated_games} 局，"
                f"耗时 {phase_times['selfplay_seconds']:.1f}s，"
                f"速度 {selfplay_games_per_second:.3f} 局/s"
            )

            # 2. 合并回放池并记录数据分布。
            save_loop_state(state_path, generation, 'buffer', current_epochs, accepted_generation)
            phase_started_at = time.perf_counter()
            buffer_result = manage_buffer(generation_data)
            buffer_path = os.path.join(config.BUFFER_DIR, 'replay_buffer.csv')
            generation_stats = analyze_generation_data(generation_data, generation)
            phase_times['buffer_seconds'] = time.perf_counter() - phase_started_at

            data_log = {
                'generation': generation,
                # SwanLab 标量不接受字符串；1 表示 SSH 远端执行器，0 表示
                # 当前训练主机本地生成。H20 整机部署属于后者。
                'system/selfplay_remote': int(selfplay_runtime['backend'] == 'remote'),
                'data/buffer_size': buffer_result['size'],
                'data/new_unique_games': buffer_result['added'],
                'data/duplicate_games': buffer_result['duplicates'],
                'data/selfplay_games_generated': newly_generated_games,
                'performance/selfplay_games_per_second': selfplay_games_per_second,
                'timing/selfplay_seconds': phase_times['selfplay_seconds'],
                'timing/buffer_seconds': phase_times['buffer_seconds'],
            }
            for key in ('black_win_rate', 'white_win_rate', 'avg_game_length', 'opening_diversity'):
                if key in generation_stats:
                    data_log[f'data/generation/{key}'] = generation_stats[key]
            if generation_stats.get('chart_path'):
                data_log['data/generation_chart'] = swanlab.Image(generation_stats['chart_path'])
            swanlab_log(swan_active, data_log, step=0)

            if buffer_result['size'] < config.HOT_START_MIN_BUFFER:
                print(
                    f"回放池 {buffer_result['size']} < {config.HOT_START_MIN_BUFFER}，"
                    "本代只积累数据，不训练"
                )
                total_seconds = time.perf_counter() - generation_started_at
                phase_times['total_seconds'] = total_seconds
                save_generation_summary(generation, {
                    'version': 1,
                    'generation': generation,
                    'selfplay_backend': selfplay_runtime['backend'],
                    'trained': False,
                    'accepted': None,
                    'accepted_generation': accepted_generation,
                    'buffer_size': buffer_result['size'],
                    'current_epochs': current_epochs,
                    'selfplay_games_per_second': selfplay_games_per_second,
                    'generation_stats': {
                        key: generation_stats[key]
                        for key in (
                            'black_win_rate',
                            'white_win_rate',
                            'avg_game_length',
                            'opening_diversity',
                        )
                        if key in generation_stats
                    },
                    'phase_times': phase_times,
                })
                finish_swanlab(swan_active, 'success')
                swan_active = False
                generation += 1
                completed_generations += 1
                save_loop_state(state_path, generation, 'selfplay', current_epochs, accepted_generation)
                continue

            # 3. 从已通过门控的 best.pth 训练独立候选模型。
            save_loop_state(state_path, generation, 'training', current_epochs, accepted_generation)
            phase_started_at = time.perf_counter()
            candidate_pth = os.path.join(config.CHECKPOINT_DIR, f'candidate_gen_{generation}.pth')
            train_script = os.path.join(config.BASE_DIR, 'pipeline', 'train.py')
            run_command(
                [
                    PYTHON,
                    train_script,
                    '--resume', config.CURRENT_MODEL_PTH,
                    '--data', buffer_path,
                    '--run_name', f'gen_{generation}',
                    '--epochs', str(current_epochs),
                    '--generation', str(generation),
                    '--precision', config.TRAIN_PRECISION,
                    '--output', candidate_pth,
                ],
                env_vars={'CUDA_VISIBLE_DEVICES': training_gpus},
            )
            if not os.path.exists(candidate_pth):
                raise RuntimeError(f"候选权重未生成: {candidate_pth}")
            phase_times['training_seconds'] = time.perf_counter() - phase_started_at

            # model_gen_X 保留全部候选，accepted_gen_X 只保留门控通过的版本。
            candidate_archive = os.path.join(config.CHECKPOINT_DIR, f'model_gen_{generation}.pth')
            atomic_copy(candidate_pth, candidate_archive)
            train_metrics_path = os.path.join(config.LOG_DIR, 'train_metrics.json')
            train_metrics = load_json(train_metrics_path, {})
            epoch_metrics = train_metrics.get('epoch_metrics', [])
            if not isinstance(epoch_metrics, list):
                epoch_metrics = []
            training_log_steps = max(1, len(epoch_metrics))
            if epoch_metrics:
                for epoch_index, epoch_metric in enumerate(epoch_metrics, start=1):
                    train_log = {
                        'generation': generation,
                        'train/epoch': epoch_metric.get('epoch', epoch_index),
                        'train/loss': epoch_metric.get('loss', 0),
                        'train/policy_loss': epoch_metric.get('policy_loss', 0),
                        'train/value_loss': epoch_metric.get('value_loss', 0),
                        'train/accuracy_top1': epoch_metric.get('accuracy_top1', 0),
                        'train/accuracy_top5': epoch_metric.get('accuracy_top5', 0),
                        'train/policy_entropy': epoch_metric.get('policy_entropy', 0),
                        'train/value_mae': epoch_metric.get('value_mae', 0),
                        'train/lr': epoch_metric.get('lr', 0),
                        'train/samples_per_second': epoch_metric.get('samples_per_second', 0),
                    }
                    if epoch_index == len(epoch_metrics):
                        train_log.update({
                            'train/overall_samples_per_second': train_metrics.get(
                                'overall_samples_per_second', 0
                            ),
                            'timing/training_seconds': phase_times['training_seconds'],
                        })
                    swanlab_log(swan_active, train_log, step=epoch_index)
            else:
                train_log = {
                    'generation': generation,
                    'train/loss': train_metrics.get('loss', 0),
                    'train/policy_loss': train_metrics.get('policy_loss', 0),
                    'train/value_loss': train_metrics.get('value_loss', 0),
                    'train/accuracy_top1': train_metrics.get('accuracy_top1', 0),
                    'train/accuracy_top5': train_metrics.get('accuracy_top5', 0),
                    'train/policy_entropy': train_metrics.get('policy_entropy', 0),
                    'train/samples_per_second': train_metrics.get(
                        'overall_samples_per_second', 0
                    ),
                    'timing/training_seconds': phase_times['training_seconds'],
                }
                swanlab_log(swan_active, train_log, step=1)

            # 4. 候选模型导出并编译 TensorRT 引擎。
            save_loop_state(state_path, generation, 'engine', current_epochs, accepted_generation)
            phase_started_at = time.perf_counter()
            candidate_engine = os.path.join(config.CHECKPOINT_DIR, f'model_gen_{generation}.engine')
            build_engine_from_checkpoint(
                candidate_pth,
                candidate_engine,
                train_gpu,
                f'candidate_{generation}',
            )
            phase_times['engine_seconds'] = time.perf_counter() - phase_started_at

            # 5. 与当前 incumbent 进行成对开局评估，评估异常一律不晋升。
            save_loop_state(state_path, generation, 'evaluation', current_epochs, accepted_generation)
            phase_started_at = time.perf_counter()
            evaluate_script = os.path.join(config.BASE_DIR, 'pipeline', 'evaluate.py')
            run_command(
                [
                    PYTHON,
                    evaluate_script,
                    '--current_engine', candidate_engine,
                    '--incumbent_engine', config.CURRENT_ENGINE_PATH,
                    '--generation', str(generation),
                    '--gpu', train_gpu,
                    '--games', str(config.EVAL_GAMES),
                    '--simulations', str(config.EVAL_SIMULATIONS),
                    '--seed', str(config.EVAL_SEED),
                    '--opening_stones', str(config.EVAL_OPENING_STONES),
                    '--save_data',
                ],
            )
            phase_times['evaluation_seconds'] = time.perf_counter() - phase_started_at

            results_path = os.path.join(config.LOG_DIR, 'eval_results.json')
            results = load_json(results_path, {})
            incumbent_stats = results.get('incumbent')
            if not incumbent_stats:
                raise RuntimeError("评估结果缺少 incumbent，对候选模型执行失败关闭")

            score_rate = incumbent_stats.get('score_rate', incumbent_stats.get('win_rate', 0))
            white_win_rate = incumbent_stats.get('white_win_rate', 0)
            passed_gating = (
                score_rate >= config.GATING_MIN_WIN_RATE
                and white_win_rate >= config.GATING_MIN_WHITE_WIN_RATE
            )

            if passed_gating:
                atomic_copy(candidate_pth, config.CURRENT_MODEL_PTH)
                atomic_copy(candidate_engine, config.CURRENT_ENGINE_PATH)
                accepted_path = os.path.join(config.CHECKPOINT_DIR, f'accepted_gen_{generation}.pth')
                atomic_copy(candidate_pth, accepted_path)
                accepted_generation = generation
                print(f">>> 门控通过：第 {generation} 代成为新的 incumbent")
            else:
                print(
                    f">>> 门控失败：score={score_rate:.2%}, white_win={white_win_rate:.2%}；"
                    "保留上一版 best.pth 和 current_model.engine"
                )

            if score_rate < 0.4:
                current_epochs = min(current_epochs + 1, 10)
            elif score_rate > 0.8:
                current_epochs = max(current_epochs - 1, 1)

            # 仅把已通过候选的评估棋谱加入训练池，避免弱候选污染数据。
            evaluation_data = os.path.join(config.RAW_DATA_DIR, f'eval_data_{generation}.csv')
            if passed_gating and os.path.exists(evaluation_data):
                manage_buffer(evaluation_data)

            history = load_json(history_path, [])
            history_entry = {
                'generation': generation,
                'incumbent_score_rate': score_rate,
                'white_win_rate': white_win_rate,
                'accepted': passed_gating,
                'timestamp': time.time(),
            }
            existing_index = next(
                (index for index, entry in enumerate(history) if entry.get('generation') == generation),
                None,
            )
            if existing_index is None:
                history.append(history_entry)
            else:
                history[existing_index] = history_entry
            history.sort(key=lambda entry: entry['generation'])
            atomic_json_dump(history, history_path)
            trend_path = os.path.join(config.LOG_DIR, 'eval_trend.png')
            update_history_plot(history_path, trend_path)

            total_seconds = time.perf_counter() - generation_started_at
            phase_times['total_seconds'] = total_seconds
            eval_log = {
                'generation': generation,
                'eval/gating_passed': int(passed_gating),
                'eval/incumbent_score_rate': score_rate,
                'eval/incumbent_white_win_rate': white_win_rate,
                'train/next_epochs': current_epochs,
                **flatten_eval_results(results),
                **{f'timing/{key}': value for key, value in phase_times.items()},
            }
            chart_path = os.path.join(config.LOG_DIR, 'eval_chart.png')
            if os.path.exists(chart_path):
                eval_log['eval/summary_chart'] = swanlab.Image(chart_path)
            swanlab_log(swan_active, eval_log, step=training_log_steps + 1)

            save_generation_summary(generation, {
                'version': 1,
                'generation': generation,
                'selfplay_backend': selfplay_runtime['backend'],
                'trained': True,
                'accepted': passed_gating,
                'accepted_generation': accepted_generation,
                'buffer_size': count_csv_rows(buffer_path),
                'current_epochs': current_epochs,
                'selfplay_games_per_second': selfplay_games_per_second,
                'train': {
                    'loss': train_metrics.get('loss', 0),
                    'policy_loss': train_metrics.get('policy_loss', 0),
                    'value_loss': train_metrics.get('value_loss', 0),
                    'accuracy_top1': train_metrics.get('accuracy_top1', 0),
                    'accuracy_top5': train_metrics.get('accuracy_top5', 0),
                    'overall_samples_per_second': train_metrics.get(
                        'overall_samples_per_second', 0
                    ),
                },
                'eval': {
                    'incumbent_score_rate': score_rate,
                    'incumbent_white_win_rate': white_win_rate,
                },
                'generation_stats': {
                    key: generation_stats[key]
                    for key in (
                        'black_win_rate',
                        'white_win_rate',
                        'avg_game_length',
                        'opening_diversity',
                    )
                    if key in generation_stats
                },
                'phase_times': phase_times,
            })

            print(f"第 {generation} 代完成，总耗时 {total_seconds:.1f}s")
            finish_swanlab(swan_active, 'success')
            swan_active = False
            generation += 1
            completed_generations += 1
            save_loop_state(state_path, generation, 'selfplay', current_epochs, accepted_generation)

    except KeyboardInterrupt:
        finish_state = 'aborted'
        print("\n收到退出信号，已保留当前阶段，重新启动会从该代继续。")
    except Exception:
        finish_state = 'crashed'
        raise
    finally:
        finish_swanlab(swan_active, finish_state)


def main():
    parser = argparse.ArgumentParser(description='Nebula Zero 强化学习主循环')
    parser.add_argument('--max-generations', type=int, default=None, help='最多运行几代；用于冒烟测试')
    parser.add_argument(
        '--swanlab-mode',
        choices=['online', 'offline', 'local', 'disabled'],
        default=config.SWANLAB_MODE,
    )
    args = parser.parse_args()
    run_training_loop(max_generations=args.max_generations, swanlab_mode=args.swanlab_mode)


if __name__ == '__main__':
    main()
